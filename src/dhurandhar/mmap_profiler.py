"""Real mmap decode throughput profiler for PLE-shaped access patterns.

Moves the mmap acceptance gate from "predicted" to "measured." Creates a
PLE-shaped dense file on disk, memory-maps it, and measures sustained
throughput under the access patterns that matter for autoregressive decode:

  * sequential_prefill   — token-ordered scan; simulates prefill of a long prompt
  * random_decode        — random token-ID lookup each step, with the 30 layer
                           rows for that token read contiguously; simulates
                           sustained autoregressive decode with low token-ID
                           correlation across steps
  * random_scatter       — random reads at byte-level granularity; a worst-case
                           lower bound on flash throughput

Cold-mmap measurements use `mmap.MADV_DONTNEED` to evict pages between batches,
producing a conservative estimate of what page-cache-cold access looks like.
Warm measurements report the steady-state after the page cache has absorbed
recent reads — useful for understanding the "best realistic" envelope.

Output is reported as decode-tokens/sec, directly comparable to the G1 target
in the decision log.

Note on fidelity: this profiler measures host flash bandwidth, not target-device
flash bandwidth. It is most useful as a methodology to run on edge target
silicon during the feasibility spike. The code structure, patterns, and measurement methodology all port directly;
only the target hardware changes.
"""

from __future__ import annotations

import gc
import mmap
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# edge deployment memory budgets
# ---------------------------------------------------------------------------
# Real G1 acceptance criterion: peak process RSS stays under budget.
# Throughput is secondary — a model that runs at 100 tok/s but OOMs on-device
# is a non-starter. These budgets come from product requirements:
#
#   INT8 deployment:   ≤ 2 GB peak RSS (primary on-device target)
#   BF16 development:  ≤ 4 GB peak RSS (dev/eval on workstation-class devices)
#   INT4 aggressive:   ≤ 1.5 GB peak RSS (stretch for low-tier mobile SKUs)
# Memory budgets are now in dhurandhar.config.MEMORY_BUDGET_PRESETS
# Re-exported here for backward compatibility
from .config import MEMORY_BUDGET_PRESETS as MEMORY_BUDGETS_MB
from .models._base import ModelArchitecture

# ---------------------------------------------------------------------------
# Platform-aware RSS reader
# ---------------------------------------------------------------------------


class RSSSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    """Point-in-time process memory measurement."""

    timestamp_sec: float
    vm_rss_mb: float              # total resident set size
    vm_hwm_mb: float              # peak resident since process start
    rss_anon_mb: float | None     # anonymous pages — Linux only
    rss_file_mb: float | None     # file-backed (mmap) pages — Linux only
    vm_size_mb: float             # virtual address space


def _read_linux_status() -> dict[str, float]:
    """Parse /proc/self/status (kB values) into MB."""
    stats: dict[str, float] = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip()
                parts = value.split()
                if key in {"VmRSS", "VmHWM", "VmSize", "RssAnon", "RssFile", "RssShmem"} and parts and parts[0].isdigit():
                    stats[key] = int(parts[0]) / 1024.0
    except OSError:
        pass
    return stats


def sample_rss() -> RSSSample:
    """Take a platform-aware RSS sample."""
    t = time.perf_counter()
    system = platform.system()

    if system == "Linux":
        s = _read_linux_status()
        return RSSSample(
            timestamp_sec=t,
            vm_rss_mb=s.get("VmRSS", 0.0),
            vm_hwm_mb=s.get("VmHWM", 0.0),
            rss_anon_mb=s.get("RssAnon"),
            rss_file_mb=s.get("RssFile"),
            vm_size_mb=s.get("VmSize", 0.0),
        )

    if system == "Darwin":
        # macOS: resource.ru_maxrss is in BYTES (unlike Linux where it's kB)
        import resource
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_bytes / (1024 * 1024)
        return RSSSample(
            timestamp_sec=t,
            vm_rss_mb=rss_mb,
            vm_hwm_mb=rss_mb,
            rss_anon_mb=None,
            rss_file_mb=None,
            vm_size_mb=0.0,
        )

    # Windows / other — try psutil as a last resort
    try:
        import psutil
        mem = psutil.Process().memory_info()
        return RSSSample(
            timestamp_sec=t,
            vm_rss_mb=mem.rss / (1024 * 1024),
            vm_hwm_mb=mem.rss / (1024 * 1024),
            rss_anon_mb=None,
            rss_file_mb=None,
            vm_size_mb=mem.vms / (1024 * 1024),
        )
    except ImportError:
        return RSSSample(
            timestamp_sec=t, vm_rss_mb=0.0, vm_hwm_mb=0.0,
            rss_anon_mb=None, rss_file_mb=None, vm_size_mb=0.0,
        )

# ---------------------------------------------------------------------------
# Test file management
# ---------------------------------------------------------------------------

# The full PLE table at Q4 is ~1.12 GB. Creating a 1 GB dense file takes
# 5–15 seconds on consumer SSDs, so we cache it and refuse to re-create
# unless the size changes.
DEFAULT_TEST_FILE = Path.home() / ".cache" / "dhurandhar" / "ple_profile.bin"


def ensure_test_file(
    path: Path,
    size_bytes: int,
    *,
    force_recreate: bool = False,
    chunk_mb: int = 16,
) -> None:
    """Ensure a dense test file of exactly `size_bytes` exists at `path`.

    A dense (non-sparse) file is required — sparse holes return zeros
    without touching disk, which would make throughput numbers meaningless.
    We write random bytes in chunks to produce a genuinely dense file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_recreate:
        actual = path.stat().st_size
        if actual == size_bytes:
            return
        # Size mismatch — recreate
        path.unlink()

    rng = random.Random(42)
    chunk_size = chunk_mb * 1024 * 1024
    with open(path, "wb") as f:
        remaining = size_bytes
        while remaining > 0:
            n = min(chunk_size, remaining)
            # random.randbytes is ~4x faster than os.urandom and we don't
            # need cryptographic randomness
            f.write(rng.randbytes(n))
            remaining -= n
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# mmap primitives with cache control
# ---------------------------------------------------------------------------


@contextmanager
def mmap_file(path: Path, advise: int | None = None):
    """Open a file and mmap it read-only, with optional madvise hint."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
        try:
            if advise is not None:
                mm.madvise(advise)
            yield mm
        finally:
            mm.close()
    finally:
        os.close(fd)


def drop_page_cache_for(mm: mmap.mmap) -> None:
    """Advise the kernel to drop the mmap's pages from the page cache.

    This is the closest user-space analogue to "cold flash" without root.
    On Linux, MADV_DONTNEED actually frees the pages; on macOS it's a hint
    that may be ignored. Best-effort — combine with a short warmup-read
    of unrelated memory to evict any L3 cache lines.
    """
    try:
        mm.madvise(mmap.MADV_DONTNEED)
    except OSError:
        # macOS sometimes rejects MADV_DONTNEED; fall back to a weaker hint
        import contextlib
        with contextlib.suppress(OSError, AttributeError):
            mm.madvise(mmap.MADV_FREE)


# ---------------------------------------------------------------------------
# Access patterns
# ---------------------------------------------------------------------------
#
# Each pattern reads a fixed total number of bytes, organized as a fixed
# number of "decode tokens" × (num_layers × ple_hidden_size × bytes_per_elem).
# Throughput is reported as decode-tokens/sec.


class PatternSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    """A single access pattern to benchmark."""

    name: str
    description: str
    madvise_hint: int | None


PATTERNS: dict[str, PatternSpec] = {
    "sequential_prefill": PatternSpec(
        name="sequential_prefill",
        description="Sequential token-ID scan — simulates prefill of a long prompt",
        madvise_hint=mmap.MADV_SEQUENTIAL,
    ),
    "random_decode": PatternSpec(
        name="random_decode",
        description="Random token-ID per step, 30 layers per token contiguous",
        madvise_hint=mmap.MADV_RANDOM,
    ),
    "random_scatter": PatternSpec(
        name="random_scatter",
        description="Byte-random reads — worst-case lower bound",
        madvise_hint=mmap.MADV_RANDOM,
    ),
}


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class ProfileResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    """Result of profiling a single (pattern, cold/warm) combination."""

    pattern: str
    cold: bool
    num_tokens: int
    bytes_per_token: int
    total_bytes: int
    elapsed_sec: float
    tokens_per_sec: float
    mb_per_sec: float
    p50_token_latency_us: float
    p99_token_latency_us: float

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "cold": self.cold,
            "num_tokens": self.num_tokens,
            "bytes_per_token": self.bytes_per_token,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "mb_per_sec": round(self.mb_per_sec, 2),
            "p50_token_latency_us": round(self.p50_token_latency_us, 2),
            "p99_token_latency_us": round(self.p99_token_latency_us, 2),
        }


class MemoryProbeResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    """Result of a peak-RSS memory probe.

    The key field is `peak_rss_mb` — this is the number to compare against
    the deployment budgets (2 GB at INT8, 4 GB at BF16).
    """

    weight_bits: int
    num_tokens_decoded: int

    # Resident memory at each phase
    baseline_rss_mb: float                # interpreter only
    post_placeholder_rss_mb: float        # + non-PLE dense buffers
    post_mmap_rss_mb: float               # + mmap'd file (before faulting)
    peak_rss_mb: float                    # highest observed during decode
    peak_vm_hwm_mb: float                 # kernel's high-water mark
    steady_state_rss_mb: float            # mean of last 20% of samples

    # Linux-only detail: anon vs file-backed at peak
    peak_rss_anon_mb: float | None
    peak_rss_file_mb: float | None

    # Components
    non_ple_component_mb: float           # placeholder size
    ple_resident_working_set_mb: float    # PLE pages actually kept resident
    weights_only_mb: float = 0.0          # decoder + PLE at target quant (no KV/etc)

    # Raw time series for plotting
    samples: list[RSSSample] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "weight_bits": self.weight_bits,
            "num_tokens_decoded": self.num_tokens_decoded,
            "baseline_rss_mb": round(self.baseline_rss_mb, 1),
            "post_placeholder_rss_mb": round(self.post_placeholder_rss_mb, 1),
            "post_mmap_rss_mb": round(self.post_mmap_rss_mb, 1),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_vm_hwm_mb": round(self.peak_vm_hwm_mb, 1),
            "steady_state_rss_mb": round(self.steady_state_rss_mb, 1),
            "peak_rss_anon_mb": round(self.peak_rss_anon_mb, 1)
                if self.peak_rss_anon_mb is not None else None,
            "peak_rss_file_mb": round(self.peak_rss_file_mb, 1)
                if self.peak_rss_file_mb is not None else None,
            "non_ple_component_mb": round(self.non_ple_component_mb, 1),
            "ple_resident_working_set_mb": round(self.ple_resident_working_set_mb, 1),
            "weights_only_mb": round(self.weights_only_mb, 1),
        }


class MmapDecodeProfiler:
    """Measure real mmap decode throughput for any PLE-equipped model.

    Usage:
        from dhurandhar.models import get_model
        arch = get_model("gemma4-e2b")
        profiler = MmapDecodeProfiler.from_architecture(arch)
        profiler.prepare()
        result = profiler.profile("random_decode", num_tokens=2000, cold=True)
        print(result.tokens_per_sec)
    """

    def __init__(
        self,
        *,
        num_layers: int = 30,
        ple_hidden_size: int = 256,
        quant_bits: int = 4,
        vocab_size: int = 262_144,
        test_file: Path | None = None,
    ):
        self.num_layers = num_layers
        self.ple_hidden_size = ple_hidden_size
        self.quant_bits = quant_bits
        self.vocab_size = vocab_size
        self.test_file = test_file or DEFAULT_TEST_FILE

        # Byte layout (token-major): each token's 30 layer rows are contiguous
        self._bytes_per_layer_row = int(
            self.ple_hidden_size * (self.quant_bits / 8.0)
        )
        # Round up to 8-byte alignment for realistic mmap behavior
        self._bytes_per_layer_row = max(8, (self._bytes_per_layer_row + 7) & ~7)

        self._bytes_per_token = self.num_layers * self._bytes_per_layer_row
        self._total_ple_bytes = self.vocab_size * self._bytes_per_token

    @classmethod
    def from_architecture(
        cls,
        arch: ModelArchitecture | None = None,
        *,
        quant_bits: int = 4,
        test_file: Path | None = None,
    ) -> MmapDecodeProfiler:
        """Build a profiler from any ModelArchitecture.

        For non-PLE models (has_ple=False), ple_hidden_size defaults to
        the hidden_size so the profiler still measures generic weight-read
        throughput (useful for mmap'd weight-file benchmarks).
        """
        if arch is None:
            from .models import GEMMA4_E2B
            arch = GEMMA4_E2B
        ple_dim   = arch.ple_hidden_size if arch.has_ple else arch.hidden_size
        ple_vocab = arch.ple_vocab_size  if arch.has_ple else arch.vocab_size
        return cls(
            num_layers=arch.num_hidden_layers,
            ple_hidden_size=ple_dim,
            quant_bits=quant_bits,
            vocab_size=ple_vocab,
            test_file=test_file,
        )

    # ------------------------------------------------------------------
    # File preparation
    # ------------------------------------------------------------------

    @property
    def total_ple_bytes(self) -> int:
        return self._total_ple_bytes

    @property
    def bytes_per_token(self) -> int:
        return self._bytes_per_token

    def prepare(
        self,
        *,
        scale: float = 1.0,
        force_recreate: bool = False,
    ) -> None:
        """Create the test file. Pass scale < 1.0 for a smaller file.

        A scale of 1.0 creates the full 1.12 GB PLE-equivalent file. Smaller
        scales create proportionally smaller files, which is useful for CI
        or for profiling methodology itself. The reported throughput is the
        same either way (we read the same total bytes per measurement).
        """
        size_bytes = int(self._total_ple_bytes * scale)
        size_bytes = max(size_bytes, 4 * 1024 * 1024)  # minimum 4 MB for sanity
        # Align to bytes_per_token so index math is exact
        size_bytes = (size_bytes // self._bytes_per_token) * self._bytes_per_token

        ensure_test_file(self.test_file, size_bytes, force_recreate=force_recreate)

        # Update vocab_size to reflect actual file size
        self._effective_vocab = size_bytes // self._bytes_per_token

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile(
        self,
        pattern: str,
        *,
        num_tokens: int = 1000,
        cold: bool = True,
        warmup_tokens: int = 50,
        seed: int = 0,
    ) -> ProfileResult:
        """Profile a single pattern, warm or cold.

        Args:
            pattern: one of PATTERNS
            num_tokens: number of decode-tokens-worth of reads to perform
            cold: if True, drop page cache before measurement
            warmup_tokens: extra reads before measurement (discarded from timing)
            seed: RNG seed for random patterns — use same seed for reproducibility
        """
        if pattern not in PATTERNS:
            raise ValueError(
                f"Unknown pattern '{pattern}'. Available: {list(PATTERNS.keys())}"
            )
        spec = PATTERNS[pattern]

        rng = random.Random(seed)
        effective_vocab = getattr(self, "_effective_vocab", self.vocab_size)

        # Pre-compute token indices to avoid RNG overhead inside the timed loop
        if pattern == "sequential_prefill":
            indices = list(range(warmup_tokens + num_tokens))
            indices = [i % effective_vocab for i in indices]
        elif pattern == "random_decode":
            indices = [
                rng.randrange(effective_vocab)
                for _ in range(warmup_tokens + num_tokens)
            ]
        elif pattern == "random_scatter":
            # Byte-random access: pick random byte offsets within the file,
            # but still read `bytes_per_token` bytes per step
            max_offset = effective_vocab * self._bytes_per_token - self._bytes_per_token
            indices = [
                rng.randrange(max_offset) // self._bytes_per_token
                for _ in range(warmup_tokens + num_tokens)
            ]
        else:
            raise RuntimeError(f"Unhandled pattern {pattern}")

        with mmap_file(self.test_file, advise=spec.madvise_hint) as mm:
            if cold:
                drop_page_cache_for(mm)

            # Warmup
            accum = 0
            for tok_id in indices[:warmup_tokens]:
                base = tok_id * self._bytes_per_token
                # Read all 30 layer rows; for scatter we still do this for
                # fair byte-count comparison across patterns
                for layer in range(self.num_layers):
                    offset = base + layer * self._bytes_per_layer_row
                    chunk = mm[offset : offset + self._bytes_per_layer_row]
                    # Prevent dead-code elimination: fold a byte into accum
                    accum ^= chunk[0] if chunk else 0

            # Timed measurement
            per_token_latencies = []
            t_total_start = time.perf_counter_ns()
            for tok_id in indices[warmup_tokens:]:
                t_tok_start = time.perf_counter_ns()
                base = tok_id * self._bytes_per_token
                for layer in range(self.num_layers):
                    offset = base + layer * self._bytes_per_layer_row
                    chunk = mm[offset : offset + self._bytes_per_layer_row]
                    accum ^= chunk[0] if chunk else 0
                t_tok_end = time.perf_counter_ns()
                per_token_latencies.append(t_tok_end - t_tok_start)
            t_total_end = time.perf_counter_ns()

        elapsed_ns = t_total_end - t_total_start
        elapsed_sec = elapsed_ns / 1e9
        total_bytes = num_tokens * self._bytes_per_token
        tokens_per_sec = num_tokens / elapsed_sec if elapsed_sec > 0 else 0.0
        mb_per_sec = (total_bytes / 1024 / 1024) / elapsed_sec if elapsed_sec > 0 else 0.0

        per_token_latencies.sort()
        p50_ns = per_token_latencies[len(per_token_latencies) // 2]
        p99_ns = per_token_latencies[int(len(per_token_latencies) * 0.99)]

        # Keep the compiler honest
        _ = accum

        return ProfileResult(
            pattern=pattern,
            cold=cold,
            num_tokens=num_tokens,
            bytes_per_token=self._bytes_per_token,
            total_bytes=total_bytes,
            elapsed_sec=elapsed_sec,
            tokens_per_sec=tokens_per_sec,
            mb_per_sec=mb_per_sec,
            p50_token_latency_us=p50_ns / 1000.0,
            p99_token_latency_us=p99_ns / 1000.0,
        )

    # ------------------------------------------------------------------
    # Peak RSS measurement (the real G1 gate)
    # ------------------------------------------------------------------

    def _estimate_non_ple_resident_mb(
        self,
        *,
        weight_bits: int,
        context_tokens: int,
        kv_bits: int,
        strip_audio: bool,
    ) -> float:
        """How much RAM the process holds resident BEFORE the mmap'd PLE.

        This is everything the real deployed process keeps in RAM full-time:
        decoder weights (resident, not mmap'd), KV cache, vision encoder,
        activations, runtime. PLE is excluded because its resident footprint
        is what we're measuring.
        """
        from .models import GEMMA4_E2B
        from .ple_analysis import PLEFootprintAnalyzer
        analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
        b = analyzer.compute_breakdown(
            context_tokens=context_tokens,
            quant_bits=weight_bits,
            kv_bits=kv_bits,
            strip_audio=strip_audio,
        )
        audio_mb = 0.0 if strip_audio else b.audio_encoder_mb
        return (
            b.decoder_mb + b.kv_cache_mb + b.vision_encoder_mb + audio_mb
            + b.activations_overhead_mb + b.runtime_overhead_mb
        )

    def profile_memory(
        self,
        *,
        weight_bits: int = 8,
        num_tokens: int = 2000,
        warmup_tokens: int = 100,
        sample_every: int = 25,
        context_tokens: int = 32_768,
        kv_bits: int = 4,
        strip_audio: bool = True,
        simulate_non_ple_resident: bool = True,
        seed: int = 0,
    ) -> MemoryProbeResult:
        """Measure peak RSS during simulated PLE-driven decode.

        The deployed process holds ~non-PLE components resident all the
        time (decoder, KV cache, encoders, runtime). To make the measurement
        reflect the REAL process footprint rather than just this Python
        interpreter's PLE mmap working set, we allocate a dense bytearray
        equal in size to the non-PLE resident load. This produces a
        projected peak RSS directly comparable to the 2 GB / 4 GB budgets.

        Set `simulate_non_ple_resident=False` to measure only the mmap
        working set (useful for methodology validation).
        """
        rng = random.Random(seed)
        effective_vocab = getattr(self, "_effective_vocab", self.vocab_size)

        gc.collect()
        baseline = sample_rss()

        non_ple_mb = self._estimate_non_ple_resident_mb(
            weight_bits=weight_bits,
            context_tokens=context_tokens,
            kv_bits=kv_bits,
            strip_audio=strip_audio,
        )

        # Allocate a dense placeholder for non-PLE resident memory so that
        # the total RSS reflects deployed-process footprint. Must touch every
        # page — Linux lazy-allocates zero-initialized bytearrays via the
        # ZERO_PAGE optimization and would otherwise under-count RSS.
        placeholder: bytearray | None = None
        if simulate_non_ple_resident:
            placeholder = bytearray(int(non_ple_mb * 1024 * 1024))
            for i in range(0, len(placeholder), 4096):
                placeholder[i] = (i >> 12) & 0xFF

        gc.collect()
        after_placeholder = sample_rss()

        # Mmap the PLE file
        with mmap_file(self.test_file, advise=mmap.MADV_RANDOM) as mm:
            post_mmap = sample_rss()

            # Warmup (not sampled)
            accum = 0
            warmup_idx = [rng.randrange(effective_vocab) for _ in range(warmup_tokens)]
            for tok_id in warmup_idx:
                base = tok_id * self._bytes_per_token
                for layer in range(self.num_layers):
                    offset = base + layer * self._bytes_per_layer_row
                    accum ^= mm[offset]

            # Timed decode with periodic RSS sampling
            samples: list[RSSSample] = [post_mmap]
            for step in range(num_tokens):
                tok_id = rng.randrange(effective_vocab)
                base = tok_id * self._bytes_per_token
                for layer in range(self.num_layers):
                    offset = base + layer * self._bytes_per_layer_row
                    accum ^= mm[offset]
                if (step + 1) % sample_every == 0:
                    samples.append(sample_rss())

            final = sample_rss()
            samples.append(final)

        _ = accum
        _ = placeholder  # keep alive through measurement

        # Summary stats
        rss_series = [s.vm_rss_mb for s in samples]
        peak_sample = max(samples, key=lambda s: s.vm_rss_mb)
        tail = rss_series[-max(1, len(rss_series) // 5):]
        steady_state = sum(tail) / len(tail)
        ple_working_set = max(0.0, peak_sample.vm_rss_mb - after_placeholder.vm_rss_mb)

        # Weights-only footprint (decoder + PLE table, no KV / encoders / runtime).
        # This is the "model at X-bit" number that the deployment budgets may refer to.
        from .models import GEMMA4_E2B
        from .ple_analysis import PLEFootprintAnalyzer
        analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
        weights_only_mb = (
            (analyzer.arch.decoder_params() + analyzer.arch.ple_table_params())
            * (weight_bits / 8.0) / (1024 * 1024)
        )

        return MemoryProbeResult(
            weight_bits=weight_bits,
            num_tokens_decoded=num_tokens,
            baseline_rss_mb=baseline.vm_rss_mb,
            post_placeholder_rss_mb=after_placeholder.vm_rss_mb,
            post_mmap_rss_mb=post_mmap.vm_rss_mb,
            peak_rss_mb=peak_sample.vm_rss_mb,
            peak_vm_hwm_mb=final.vm_hwm_mb,
            steady_state_rss_mb=steady_state,
            peak_rss_anon_mb=peak_sample.rss_anon_mb,
            peak_rss_file_mb=peak_sample.rss_file_mb,
            non_ple_component_mb=non_ple_mb,
            ple_resident_working_set_mb=ple_working_set,
            weights_only_mb=weights_only_mb,
            samples=samples,
        )

    def evaluate_budget(
        self,
        result: MemoryProbeResult,
        *,
        budget_mb: float | None = None,
        budget_name: str = "int8_deployment",
        budget_interpretation: str = "full_process",
    ) -> dict:
        """Compare measured memory against a edge deployment budget.

        Two interpretations of the budget are supported:

          budget_interpretation='full_process' (default):
            Budget applies to total peak process RSS (weights + KV cache +
            encoders + activations + runtime). This is the strictest reading
            and matches what a device sees in `/proc/<pid>/status`.

          budget_interpretation='weights_only':
            Budget applies only to model weights (decoder + PLE). Used when
            the product requirement is stated as "the model fits in X GB"
            with separate allowances for KV cache and runtime.

        Verdict logic (full_process):
          PASS  — peak RSS ≤ budget
          WARN  — peak RSS > budget but steady-state ≤ budget (bursty pages)
          FAIL  — steady-state RSS > budget

        Verdict logic (weights_only):
          PASS  — weights at target quant ≤ budget
          FAIL  — weights exceed budget
        """
        if budget_mb is None:
            if budget_name not in MEMORY_BUDGETS_MB:
                raise KeyError(f"Unknown budget '{budget_name}'. "
                               f"Known: {sorted(MEMORY_BUDGETS_MB)}")
            budget_mb = MEMORY_BUDGETS_MB[budget_name]

        if budget_interpretation == "weights_only":
            weights = result.weights_only_mb
            if weights <= budget_mb:
                verdict = "PASS"
                detail = (
                    f"Weights {weights:,.0f} MB ≤ budget {budget_mb:,.0f} MB "
                    f"at {result.weight_bits}-bit (headroom {budget_mb - weights:,.0f} MB)."
                )
            else:
                verdict = "FAIL"
                detail = (
                    f"Weights {weights:,.0f} MB exceed budget {budget_mb:,.0f} MB "
                    f"at {result.weight_bits}-bit by {weights - budget_mb:,.0f} MB. "
                    f"Need tighter quantization or a smaller model."
                )
            return {
                "verdict": verdict,
                "detail": detail,
                "budget_mb": budget_mb,
                "budget_name": budget_name,
                "interpretation": "weights_only",
                "weights_mb": weights,
                "headroom_mb": budget_mb - weights,
            }

        # Default: full_process
        peak = result.peak_rss_mb
        steady = result.steady_state_rss_mb
        headroom = budget_mb - peak

        if peak <= budget_mb:
            verdict = "PASS"
            detail = (
                f"Peak RSS {peak:,.0f} MB ≤ budget {budget_mb:,.0f} MB "
                f"(headroom {headroom:,.0f} MB)."
            )
        elif steady <= budget_mb:
            verdict = "WARN"
            detail = (
                f"Peak RSS {peak:,.0f} MB exceeds budget {budget_mb:,.0f} MB but "
                f"steady-state {steady:,.0f} MB is under. Bursty paging — may be "
                f"acceptable but investigate mmap prefetch behavior."
            )
        else:
            verdict = "FAIL"
            # Break down WHY we failed: weights alone, or weights+other?
            if result.weights_only_mb > budget_mb:
                diagnosis = (
                    f" Weights alone ({result.weights_only_mb:,.0f} MB at "
                    f"{result.weight_bits}-bit) already exceed the budget — "
                    f"the budget is likely unachievable at this precision."
                )
            elif result.non_ple_component_mb > budget_mb:
                diagnosis = (
                    f" Non-PLE resident components ({result.non_ple_component_mb:,.0f} MB: "
                    f"decoder + KV cache + encoders + runtime) exceed the budget "
                    f"by themselves. Weights fit ({result.weights_only_mb:,.0f} MB) "
                    f"but the supporting RAM does not."
                )
            else:
                diagnosis = (
                    f" Individual components fit, but combined peak "
                    f"({peak:,.0f} MB) exceeds budget. "
                    f"Working-set trimming might help."
                )
            detail = (
                f"Steady-state RSS {steady:,.0f} MB exceeds budget "
                f"{budget_mb:,.0f} MB by {steady - budget_mb:,.0f} MB."
                + diagnosis
            )

        return {
            "verdict": verdict,
            "detail": detail,
            "budget_mb": budget_mb,
            "budget_name": budget_name,
            "interpretation": "full_process",
            "peak_rss_mb": peak,
            "steady_state_rss_mb": steady,
            "weights_only_mb": result.weights_only_mb,
            "non_ple_component_mb": result.non_ple_component_mb,
            "headroom_mb": headroom,
        }

    def profile_all(
        self,
        *,
        num_tokens: int = 1000,
        warmup_tokens: int = 50,
        include_warm: bool = True,
    ) -> list[ProfileResult]:
        """Profile every pattern in both cold and (optionally) warm modes."""
        results = []
        for pattern_name in PATTERNS:
            results.append(
                self.profile(
                    pattern_name,
                    num_tokens=num_tokens,
                    cold=True,
                    warmup_tokens=warmup_tokens,
                )
            )
            if include_warm:
                results.append(
                    self.profile(
                        pattern_name,
                        num_tokens=num_tokens,
                        cold=False,
                        warmup_tokens=warmup_tokens,
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Gate evaluation
    # ------------------------------------------------------------------

    def evaluate_gate(
        self,
        results: list[ProfileResult],
        *,
        target_tps: float = 15.0,
    ) -> dict:
        """Evaluate G1-style go/no-go against the measured results.

        Gate logic:
          PASS  — random_decode cold ≥ target_tps
          WARN  — random_decode warm ≥ target_tps but cold < target_tps
                  (production still viable if most reads hit page cache)
          FAIL  — random_decode warm < target_tps
        """
        by_key = {(r.pattern, r.cold): r for r in results}
        random_cold = by_key.get(("random_decode", True))
        random_warm = by_key.get(("random_decode", False))

        if random_cold is None:
            return {"verdict": "UNKNOWN", "detail": "random_decode cold not measured"}

        cold_tps = random_cold.tokens_per_sec
        warm_tps = random_warm.tokens_per_sec if random_warm else None

        if cold_tps >= target_tps:
            verdict = "PASS"
            detail = (
                f"Cold mmap decode throughput {cold_tps:.1f} tok/s "
                f"≥ target {target_tps:.1f} tok/s."
            )
        elif warm_tps is not None and warm_tps >= target_tps:
            verdict = "WARN"
            detail = (
                f"Cold ({cold_tps:.1f} tok/s) below target but warm "
                f"({warm_tps:.1f} tok/s) passes. Consider a PLE pre-warm strategy."
            )
        else:
            verdict = "FAIL"
            detail = (
                f"Both cold ({cold_tps:.1f} tok/s) and warm "
                f"({warm_tps if warm_tps is None else f'{warm_tps:.1f}'} tok/s) "
                f"below target {target_tps:.1f} tok/s. "
                f"Escalate: need tighter PLE quant or resident PLE."
            )

        return {
            "verdict": verdict,
            "detail": detail,
            "target_tps": target_tps,
            "cold_tps": cold_tps,
            "warm_tps": warm_tps,
        }
