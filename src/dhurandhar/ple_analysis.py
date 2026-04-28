"""Per-Layer Embedding (PLE) memory footprint analysis.

For models with PLE (e.g. Gemma 4): the PLE embedding table is typically
*larger than the text decoder itself*, so whether PLE can be memory-mapped
from flash at acceptable decode throughput is the single highest-risk item
in any edge deployment plan.

For standard models without PLE: the analysis degenerates to a conventional
embedding-table + decoder memory breakdown. The mmap question simply doesn't
arise — all weights are loaded resident.

Usage
-----
    from dhurandhar.models import get_model
    from dhurandhar.ple_analysis import PLEFootprintAnalyzer

    analyzer = PLEFootprintAnalyzer(get_model("gemma4-e2b"))
    bd = analyzer.compute_breakdown(context_tokens=32768, quant_bits=4)
    print(analyzer.format_breakdown(bd))

    f = analyzer.assess_device("mid_tier_mobile_ufs3")
    print(f.mode, f.rationale)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .config import DEVICE_PROFILES, DeploymentProfile
from .models._base import ModelArchitecture

BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024


class MemoryBreakdown(BaseModel):
    """Per-component memory footprint for a given deployment configuration."""

    model_config = ConfigDict(frozen=True)

    decoder_mb:              float
    embedding_mb:            float
    kv_cache_mb:             float
    vision_encoder_mb:       float
    audio_encoder_mb:        float
    activations_overhead_mb: float
    runtime_overhead_mb:     float
    context_tokens:          int
    quant_bits:              int
    strip_audio:             bool
    has_ple:                 bool

    @property
    def ple_table_mb(self) -> float:
        """Backward-compatible alias — returns embedding_mb (PLE table + token embeddings).

        For PLE models the embedding table is dominated by the PLE component.
        For non-PLE models this equals the token embedding footprint.
        """
        return self.embedding_mb

    @property
    def resident_total_mb(self) -> float:
        audio = 0.0 if self.strip_audio else self.audio_encoder_mb
        return (
            self.decoder_mb
            + self.embedding_mb
            + self.kv_cache_mb
            + self.vision_encoder_mb
            + audio
            + self.activations_overhead_mb
            + self.runtime_overhead_mb
        )

    @property
    def mmap_total_mb(self) -> float:
        """RAM when PLE is memory-mapped. Equals resident_total_mb for non-PLE models."""
        if not self.has_ple:
            return self.resident_total_mb
        audio = 0.0 if self.strip_audio else self.audio_encoder_mb
        return (
            self.decoder_mb
            + 64.0          # PLE page-cache working set
            + self.kv_cache_mb
            + self.vision_encoder_mb
            + audio
            + self.activations_overhead_mb
            + self.runtime_overhead_mb
        )


class DeviceFeasibility(BaseModel):
    """Per-device assessment of whether the model can run on a given device."""

    model_config = ConfigDict(frozen=True)

    device:                       DeploymentProfile
    breakdown:                    MemoryBreakdown
    mode:                         str              # "resident" | "mmap" | "infeasible"
    headroom_mb:                  float
    flash_read_bound_tok_per_sec: float | None
    rationale:                    str


class PLEFootprintAnalyzer:
    """Analyze the memory footprint of any model on edge devices.

    Parameters
    ----------
    arch
        A ModelArchitecture instance. Use dhurandhar.models.get_model() or
        construct directly.

    Examples
    --------
    >>> from dhurandhar.models import get_model
    >>> analyzer = PLEFootprintAnalyzer(get_model("gemma4-e2b"))
    >>> analyzer = PLEFootprintAnalyzer(get_model("llama-3.2-1b"))
    """

    def __init__(self, arch: ModelArchitecture) -> None:
        self.arch = arch

    def compute_breakdown(
        self,
        *,
        context_tokens: int = 32_768,
        quant_bits: int = 4,
        kv_bits: int = 4,
        strip_audio: bool = True,
    ) -> MemoryBreakdown:
        arch = self.arch

        decoder_mb   = arch.decoder_params()   * (quant_bits / 8.0) / BYTES_PER_MB
        embedding_mb = arch.embedding_params() * (quant_bits / 8.0) / BYTES_PER_MB
        kv_mb        = arch.kv_cache_bytes(context_tokens, kv_bits) / BYTES_PER_MB

        if quant_bits == 4 and arch.published_decoder_gb > 0:
            decoder_mb = self._best_estimate(decoder_mb, arch.published_decoder_gb * 1024)
        if quant_bits == 4 and arch.published_embeddings_gb > 0:
            embedding_mb = self._best_estimate(embedding_mb, arch.published_embeddings_gb * 1024)

        activations_mb = (
            arch.hidden_size
            * min(context_tokens, 4096)
            * 2
            * (arch.weight_dtype_bits / 8.0)
            / BYTES_PER_MB
        )

        return MemoryBreakdown(
            decoder_mb              = round(decoder_mb, 2),
            embedding_mb            = round(embedding_mb, 2),
            kv_cache_mb             = round(kv_mb, 2),
            vision_encoder_mb       = arch.vision_encoder_mb,
            audio_encoder_mb        = arch.audio_encoder_mb,
            activations_overhead_mb = round(activations_mb, 2),
            runtime_overhead_mb     = arch.runtime_overhead_mb,
            context_tokens          = context_tokens,
            quant_bits              = quant_bits,
            strip_audio             = strip_audio,
            has_ple                 = arch.has_ple,
        )

    @staticmethod
    def _best_estimate(computed_mb: float, published_mb: float) -> float:
        if published_mb <= 0:
            return computed_mb
        ratio = computed_mb / published_mb
        return published_mb if 0.7 <= ratio <= 1.3 else computed_mb

    def assess_device(
        self,
        device_key: str,
        *,
        context_tokens: int = 32_768,
        quant_bits: int = 4,
        kv_bits: int = 4,
        strip_audio: bool = True,
        decode_tokens_per_sec_target: float = 15.0,
    ) -> DeviceFeasibility:
        if device_key not in DEVICE_PROFILES:
            raise KeyError(
                f"Unknown device {device_key!r}. "
                f"Available: {sorted(DEVICE_PROFILES.keys())}"
            )

        device    = DEVICE_PROFILES[device_key]
        breakdown = self.compute_breakdown(
            context_tokens=context_tokens,
            quant_bits=quant_bits,
            kv_bits=kv_bits,
            strip_audio=strip_audio,
        )

        resident_headroom = device.ram_budget_mb - breakdown.resident_total_mb
        mmap_headroom     = device.ram_budget_mb - breakdown.mmap_total_mb

        ple_bytes_per_token = self.arch.ple_bytes_per_decode_token(quant_bits)
        if ple_bytes_per_token > 0:
            flash_bps       = device.flash_read_gbps * BYTES_PER_GB
            flash_bound_tps = flash_bps / ple_bytes_per_token
        else:
            flash_bound_tps = float("inf")

        if resident_headroom >= 0:
            mode, headroom = "resident", resident_headroom
            rationale = (
                f"Fits resident with {headroom:.0f} MB headroom."
                + (" mmap not required." if self.arch.has_ple else "")
            )
        elif not self.arch.has_ple:
            mode, headroom = "infeasible", resident_headroom
            rationale = (
                f"Insufficient RAM. Short by {-headroom:.0f} MB. "
                "Need smaller model or larger device."
            )
        elif mmap_headroom >= 0 and flash_bound_tps >= decode_tokens_per_sec_target:
            mode, headroom = "mmap", mmap_headroom
            rationale = (
                f"PLE must be mmap'd ({headroom:.0f} MB headroom). "
                f"Flash bound = {flash_bound_tps:.1f} tok/s "
                f"(target {decode_tokens_per_sec_target:.1f}). Viable but measure on device."
            )
        elif mmap_headroom >= 0:
            mode, headroom = "infeasible", mmap_headroom
            rationale = (
                f"PLE mmap fits RAM but flash bound = {flash_bound_tps:.1f} tok/s "
                f"< target {decode_tokens_per_sec_target:.1f}. Need tighter quant or better flash."
            )
        else:
            mode, headroom = "infeasible", mmap_headroom
            rationale = f"Insufficient RAM even with mmap. Short by {-headroom:.0f} MB."

        return DeviceFeasibility(
            device                       = device,
            breakdown                    = breakdown,
            mode                         = mode,
            headroom_mb                  = round(headroom, 1),
            flash_read_bound_tok_per_sec = round(flash_bound_tps, 1) if flash_bound_tps != float("inf") else None,
            rationale                    = rationale,
        )

    def format_breakdown(self, breakdown: MemoryBreakdown) -> str:
        from tabulate import tabulate

        arch       = self.arch
        audio_mb   = 0.0 if breakdown.strip_audio else breakdown.audio_encoder_mb
        audio_note = " (STRIPPED)" if breakdown.strip_audio else ""
        emb_label  = "PLE + token embeddings" if arch.has_ple else "Token embeddings"

        rows = [
            ["Text decoder weights",          f"{breakdown.decoder_mb:,.0f} MB",              f"Q{breakdown.quant_bits}"],
            [emb_label,                        f"{breakdown.embedding_mb:,.0f} MB",             f"Q{breakdown.quant_bits}"],
            [f"KV cache @ {breakdown.context_tokens:,} tokens",
                                               f"{breakdown.kv_cache_mb:,.0f} MB",              "GQA + TurboQuant"],
            ["Vision encoder",                 f"{breakdown.vision_encoder_mb:,.0f} MB",        "bf16"],
            [f"Audio encoder{audio_note}",     f"{audio_mb:,.0f} MB",                           "bf16"],
            ["Activations (peak)",             f"{breakdown.activations_overhead_mb:,.0f} MB",  ""],
            ["Runtime overhead",               f"{breakdown.runtime_overhead_mb:,.0f} MB",      ""],
        ]
        table = tabulate(rows, headers=["Component", "Size", "Notes"], tablefmt="simple")
        lines = [
            "",
            f"Total (resident):   {breakdown.resident_total_mb:,.0f} MB",
        ]
        if arch.has_ple:
            lines.append(f"Total (PLE mmap'd): {breakdown.mmap_total_mb:,.0f} MB")
            lines.append(
                f"Embedding/Decoder ratio: "
                f"{breakdown.embedding_mb / max(breakdown.decoder_mb, 1):.2f}x"
            )
        return table + "\n".join(lines)
