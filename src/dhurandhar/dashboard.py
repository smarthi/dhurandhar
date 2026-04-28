"""Gradio dashboard for dhurandhar.

Five tabs aligned to the decision log's acceptance gates:

  1. PLE Memory Analysis       — adjust context, quant, audio-strip; see breakdown
                                 (component table + stacked bar chart)
  2. Device Feasibility        — per-device resident/mmap/infeasible verdicts
                                 across all profiles, plus custom device input
  3. TurboQuant KV             — compression quality sweep across residual bits
  4. Mmap Profiler             — real mmap benchmark + peak RSS probe
  5. TurboQuant vs RotorQuant  — codec comparison: quality vs arithmetic cost

Launch:
    dhurandhar-dashboard                         # localhost:7860
    dhurandhar-dashboard --server-name 0.0.0.0   # LAN access
    dhurandhar-dashboard --share                 # public Gradio URL (external)

Windows compatibility:
    This module forces the non-interactive 'Agg' matplotlib backend BEFORE
    importing pyplot. On Windows the default TkAgg backend tries to spawn a
    GUI window from inside Gradio's event loop, which hangs the page at the
    "page is still loading" screen. Don't import matplotlib before this
    module or the backend won't stick.
"""

from __future__ import annotations

# CRITICAL: set matplotlib backend BEFORE any pyplot import anywhere in the
# process. On Windows, the default (TkAgg) tries to create a GUI window from
# inside Gradio's asyncio loop and freezes the page.
import matplotlib

matplotlib.use("Agg", force=True)

import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from .config import DEVICE_PROFILES
from .mmap_profiler import PATTERNS, MmapDecodeProfiler
from .models import get_model, list_models
from .ple_analysis import PLEFootprintAnalyzer
from .rotorquant import RotorQuantCodec, RotorQuantConfig, fma_cost_comparison
from .turboquant import TurboQuantCodec, TurboQuantConfig, synthesize_kv_tensor

# ---------------------------------------------------------------------------
# Tab 1: PLE Memory Analysis
# ---------------------------------------------------------------------------


def analyze_ple(
    model_name: str,
    context_tokens: int,
    quant_bits: int,
    kv_bits: int,
    strip_audio: bool,
):
    arch     = get_model(model_name)
    analyzer = PLEFootprintAnalyzer(arch)
    breakdown = analyzer.compute_breakdown(
        context_tokens=context_tokens,
        quant_bits=quant_bits,
        kv_bits=kv_bits,
        strip_audio=strip_audio,
    )

    # Build component table
    audio_mb = 0.0 if strip_audio else breakdown.audio_encoder_mb
    audio_label = "Audio encoder (STRIPPED)" if strip_audio else "Audio encoder"
    rows = [
        ["Text decoder weights", f"{breakdown.decoder_mb:,.0f} MB", f"Q{quant_bits}"],
        ["PLE + token embeddings", f"{breakdown.ple_table_mb:,.0f} MB", f"Q{quant_bits}"],
        [
            f"KV cache @ {context_tokens:,} tokens",
            f"{breakdown.kv_cache_mb:,.0f} MB",
            "shared + GQA + TurboQuant",
        ],
        ["Vision encoder", f"{breakdown.vision_encoder_mb:,.0f} MB", "bf16"],
        [audio_label, f"{audio_mb:,.0f} MB", "bf16"],
        ["Activations (peak)", f"{breakdown.activations_overhead_mb:,.0f} MB", ""],
        ["Runtime overhead", f"{breakdown.runtime_overhead_mb:,.0f} MB", "LiteRT-LM + misc"],
    ]

    summary = (
        f"**PLE resident total:** {breakdown.resident_total_mb:,.0f} MB\n\n"
        f"**PLE mmap'd total:** {breakdown.mmap_total_mb:,.0f} MB\n\n"
        f"**PLE / Decoder ratio:** {breakdown.ple_table_mb / breakdown.decoder_mb:.2f}×  "
        f"(PLE is {'larger' if breakdown.ple_table_mb > breakdown.decoder_mb else 'smaller'} "
        f"than the text decoder)"
    )

    # Build stacked bar chart: resident vs mmap
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["PLE resident", "PLE mmap'd (1.5 GB target)"]
    components_labels = [
        "Decoder",
        "PLE table",
        "KV cache",
        "Vision encoder",
        audio_label,
        "Activations",
        "Runtime",
    ]
    resident_vals = [
        breakdown.decoder_mb,
        breakdown.ple_table_mb,
        breakdown.kv_cache_mb,
        breakdown.vision_encoder_mb,
        audio_mb,
        breakdown.activations_overhead_mb,
        breakdown.runtime_overhead_mb,
    ]
    mmap_vals = [
        breakdown.decoder_mb,
        64.0,  # PLE mmap working set
        breakdown.kv_cache_mb,
        breakdown.vision_encoder_mb,
        audio_mb,
        breakdown.activations_overhead_mb,
        breakdown.runtime_overhead_mb,
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(components_labels)))

    bottoms = [0, 0]
    for i, comp in enumerate(components_labels):
        vals = [resident_vals[i], mmap_vals[i]]
        ax.bar(labels, vals, bottom=bottoms, label=comp, color=colors[i])
        bottoms = [bottoms[j] + vals[j] for j in range(2)]

    # Draw the 1.5 GB target line
    ax.axhline(y=1536, linestyle="--", color="red", alpha=0.6, label="1.5 GB target")

    ax.set_ylabel("RAM (MB)")
    ax.set_title(
        f"{arch.name} footprint — {context_tokens:,} ctx, Q{quant_bits}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    return rows, summary, fig


# ---------------------------------------------------------------------------
# Tab 2: Device Feasibility
# ---------------------------------------------------------------------------


def assess_devices(
    model_name: str,
    context_tokens: int,
    quant_bits: int,
    strip_audio: bool,
    target_tps: float,
    custom_name: str,
    custom_ram_mb: float,
    custom_flash_gbps: float,
):
    arch     = get_model(model_name)
    analyzer = PLEFootprintAnalyzer(arch)

    rows = []
    for dev_key in sorted(DEVICE_PROFILES.keys()):
        f = analyzer.assess_device(
            dev_key,
            context_tokens=context_tokens,
            quant_bits=quant_bits,
            strip_audio=strip_audio,
            decode_tokens_per_sec_target=target_tps,
        )
        mode_icon = {"resident": "🟢", "mmap": "🟡", "infeasible": "🔴"}[f.mode]
        rows.append([
            f.device.name,
            f"{f.device.ram_budget_mb:,.0f} MB",
            f"{f.device.flash_read_gbps:.2f} GB/s",
            "yes" if f.device.supports_npu else "no",
            f"{mode_icon} {f.mode}",
            f"{f.headroom_mb:,.1f} MB",
            f.rationale,
        ])

    # Custom device evaluation
    if custom_name and custom_ram_mb > 0 and custom_flash_gbps > 0:
        # Temporarily inject a custom profile
        from .config import DeploymentProfile

        profile = DeploymentProfile(
            name=custom_name,
            ram_budget_mb=custom_ram_mb,
            flash_read_gbps=custom_flash_gbps,
        )
        breakdown = analyzer.compute_breakdown(
            context_tokens=context_tokens,
            quant_bits=quant_bits,
            strip_audio=strip_audio,
        )
        resident_headroom = custom_ram_mb - breakdown.resident_total_mb
        mmap_headroom = custom_ram_mb - breakdown.mmap_total_mb

        ple_bytes_per_token = 30 * 256 * (quant_bits / 8.0)
        flash_bps = custom_flash_gbps * 1024**3
        flash_bound = flash_bps / ple_bytes_per_token

        if resident_headroom >= 0:
            mode, headroom = "resident", resident_headroom
            rationale = "PLE fits resident; mmap not required."
        elif mmap_headroom >= 0 and flash_bound >= target_tps:
            mode, headroom = "mmap", mmap_headroom
            rationale = f"mmap'd; flash-bound {flash_bound:,.0f} tok/s."
        elif mmap_headroom >= 0:
            mode, headroom = "infeasible", mmap_headroom
            rationale = f"Flash-bound {flash_bound:,.1f} below target."
        else:
            mode, headroom = "infeasible", mmap_headroom
            rationale = f"RAM short by {-mmap_headroom:,.0f} MB."

        mode_icon = {"resident": "🟢", "mmap": "🟡", "infeasible": "🔴"}[mode]
        rows.append([
            profile.name + " (custom)",
            f"{custom_ram_mb:,.0f} MB",
            f"{custom_flash_gbps:.2f} GB/s",
            "-",
            f"{mode_icon} {mode}",
            f"{headroom:,.1f} MB",
            rationale,
        ])

    # Summary
    infeasible = sum(1 for r in rows if "🔴" in r[4])
    mmap_req = sum(1 for r in rows if "🟡" in r[4])
    resident = sum(1 for r in rows if "🟢" in r[4])
    summary = (
        f"**{resident}** profile(s) can run PLE resident • "
        f"**{mmap_req}** require mmap • "
        f"**{infeasible}** infeasible without mitigation"
    )
    return rows, summary


# ---------------------------------------------------------------------------
# Tab 3: TurboQuant benchmark
# ---------------------------------------------------------------------------


def benchmark_turboquant(
    head_dim: int,
    num_kv_heads: int,
    seq_len: int,
    residual_bits: int,
    distribution: str,
):
    codec = TurboQuantCodec(
        head_dim=head_dim,
        config=TurboQuantConfig(residual_bits=residual_bits),
    )
    # Use a reasonable sample size for the quality measurement
    sample_seq = min(seq_len, 1024)
    kv = synthesize_kv_tensor(
        seq_len=sample_seq,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        distribution=distribution,
    )
    metrics = codec.reconstruction_error(kv)

    # Sweep residual bits for a quality-vs-bits plot
    bits_sweep = [2, 3, 4, 6, 8]
    sweep_results = []
    for b in bits_sweep:
        c = TurboQuantCodec(head_dim=head_dim, config=TurboQuantConfig(residual_bits=b))
        m = c.reconstruction_error(kv)
        sweep_results.append((b, m["cos_sim"], m["mse"], m["effective_bits"]))

    # Build quality/bits plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bs = [s[0] for s in sweep_results]
    coses = [s[1] for s in sweep_results]
    mses = [s[2] for s in sweep_results]

    ax1.plot(bs, coses, marker="o", color="tab:blue")
    ax1.axvline(x=residual_bits, linestyle="--", color="red", alpha=0.4,
                label=f"current: {residual_bits}")
    ax1.set_xlabel("Residual bits")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("Reconstruction quality vs residual bits")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(bs, mses, marker="s", color="tab:orange")
    ax2.axvline(x=residual_bits, linestyle="--", color="red", alpha=0.4)
    ax2.set_xlabel("Residual bits")
    ax2.set_ylabel("MSE")
    ax2.set_title("Reconstruction error vs residual bits")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    # Memory savings estimate
    from .turboquant import KVCacheCompressor

    compressor = KVCacheCompressor(
        num_layers=30, head_dim=head_dim, shared_kv_last_n=6,
        config=TurboQuantConfig(residual_bits=residual_bits),
    )
    savings = compressor.memory_savings_estimate(
        seq_len=seq_len, num_kv_heads=num_kv_heads
    )

    summary = (
        f"### Quality @ residual_bits={residual_bits}\n"
        f"- Cosine similarity: **{metrics['cos_sim']:.4f}**\n"
        f"- MSE: **{metrics['mse']:.6f}**\n"
        f"- Norm preservation: **{metrics['norm_ratio']:.4f}**\n"
        f"- Effective bits/channel: **{metrics['effective_bits']:.2f}**\n"
        f"- Compression ratio vs bf16: **{metrics['compression_ratio']:.2f}×**\n\n"
        f"### KV cache footprint @ {seq_len:,} tokens\n"
        f"- Baseline (bf16): **{savings['baseline_mb']:,.1f} MB**\n"
        f"- TurboQuant: **{savings['quantized_mb']:,.1f} MB**\n"
        f"- Savings: **{savings['savings_mb']:,.1f} MB** "
        f"({savings['savings_ratio']}× reduction)\n"
        f"- {savings['fresh_kv_layers']} fresh-KV layers compressed; "
        f"{savings['shared_kv_layers']} shared-KV layers skipped"
    )

    return summary, fig


# ---------------------------------------------------------------------------
# Tab 4: Mmap Profiler
# ---------------------------------------------------------------------------


def run_mmap_profile(
    scale: float,
    num_tokens: int,
    warmup_tokens: int,
    quant_bits: int,
    target_tps: float,
    measure_memory: bool,
    weight_bits: str,
    context_tokens: int,
    progress=gr.Progress(),  # noqa: B008
):
    progress(0.05, desc="Configuring profiler...")
    with tempfile.TemporaryDirectory() as td:
        tf = Path(td) / "ple_profile.bin"
        profiler = MmapDecodeProfiler.from_architecture(
            quant_bits=quant_bits, test_file=tf
        )

        progress(
            0.1,
            desc=f"Creating {profiler.total_ple_bytes*scale/1024/1024:.0f} MB test file...",
        )
        profiler.prepare(scale=scale)

        results = []
        pattern_list = list(PATTERNS.keys())
        n_pattern = len(pattern_list)
        # Reserve 30% of progress for memory probe if requested
        throughput_budget = 0.55 if measure_memory else 0.85
        step = throughput_budget / (n_pattern * 2)
        current_progress = 0.1

        for pat in pattern_list:
            current_progress += step
            progress(current_progress, desc=f"Profiling {pat} cold...")
            results.append(profiler.profile(
                pat, num_tokens=num_tokens, cold=True, warmup_tokens=warmup_tokens
            ))
            current_progress += step
            progress(current_progress, desc=f"Profiling {pat} warm...")
            results.append(profiler.profile(
                pat, num_tokens=num_tokens, cold=False, warmup_tokens=warmup_tokens
            ))

        tput_gate = profiler.evaluate_gate(results, target_tps=target_tps)

        memory_result = None
        memory_gate = None
        if measure_memory:
            progress(0.7, desc="Running memory probe (allocating placeholder)...")
            w_bits = int(weight_bits)
            budget_name = {
                4: "int4_aggressive",
                8: "int8_deployment",
                16: "bf16_development",
            }[w_bits]
            memory_result = profiler.profile_memory(
                weight_bits=w_bits,
                num_tokens=num_tokens,
                warmup_tokens=warmup_tokens,
                context_tokens=context_tokens,
            )
            memory_gate = profiler.evaluate_budget(
                memory_result, budget_name=budget_name
            )

        progress(1.0, desc="Done")

    # Build throughput table
    rows = [
        [r.pattern, "cold" if r.cold else "warm",
         f"{r.tokens_per_sec:,.0f}", f"{r.mb_per_sec:,.0f}",
         f"{r.p50_token_latency_us:.1f}", f"{r.p99_token_latency_us:.1f}"]
        for r in results
    ]

    # Build verdict markdown
    verdict_color = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴", "UNKNOWN": "⚪"}
    verdict_md = (
        f"### Throughput gate {verdict_color.get(tput_gate['verdict'], '⚪')} "
        f"**{tput_gate['verdict']}**\n\n"
        f"{tput_gate['detail']}\n\n"
        f"**Target:** {tput_gate['target_tps']} tok/s  •  "
        f"**Cold:** {tput_gate['cold_tps']:,.0f} tok/s  •  "
        f"**Warm:** {tput_gate['warm_tps']:,.0f} tok/s\n\n"
    )
    if memory_gate is not None and memory_result is not None:
        verdict_md += (
            f"---\n\n"
            f"### Memory gate {verdict_color.get(memory_gate['verdict'], '⚪')} "
            f"**{memory_gate['verdict']}**  (the real G1 criterion)\n\n"
            f"{memory_gate['detail']}\n\n"
            f"| Component | MB |\n|---|---:|\n"
            f"| Baseline RSS | {memory_result.baseline_rss_mb:,.0f} |\n"
            f"| + Non-PLE resident buffers | "
            f"{memory_result.post_placeholder_rss_mb:,.0f} |\n"
            f"| + PLE mmap'd | {memory_result.post_mmap_rss_mb:,.0f} |\n"
            f"| **Peak during decode** | **{memory_result.peak_rss_mb:,.0f}** |\n"
            f"| Steady-state | {memory_result.steady_state_rss_mb:,.0f} |\n"
            f"| PLE resident working set | "
            f"{memory_result.ple_resident_working_set_mb:,.0f} |\n\n"
            f"**Budget:** {memory_gate['budget_mb']:,.0f} MB  •  "
            f"**Headroom:** {memory_gate['headroom_mb']:,.0f} MB"
        )

    # Throughput bar chart
    fig, axes = plt.subplots(
        1, 2 if memory_result else 1,
        figsize=(13 if memory_result else 8, 4.5),
    )
    ax_tp = axes[0] if memory_result else axes
    cold_rs = [r for r in results if r.cold]
    warm_rs = [r for r in results if not r.cold]
    patterns = [r.pattern for r in cold_rs]
    cold_tps = [r.tokens_per_sec for r in cold_rs]
    warm_tps = [r.tokens_per_sec for r in warm_rs]
    x = np.arange(len(patterns))
    width = 0.35
    ax_tp.bar(x - width/2, cold_tps, width, label="Cold (mmap)",
              color="tab:blue", alpha=0.8)
    ax_tp.bar(x + width/2, warm_tps, width, label="Warm (page cache)",
              color="tab:orange", alpha=0.8)
    ax_tp.axhline(y=target_tps, linestyle="--", color="red", alpha=0.6,
                  label=f"Target {target_tps} tok/s")
    ax_tp.set_xticks(x)
    ax_tp.set_xticklabels(patterns, rotation=15, ha="right")
    ax_tp.set_ylabel("Decode tokens/sec")
    ax_tp.set_yscale("log")
    ax_tp.set_title("Throughput (higher is better)")
    ax_tp.legend()
    ax_tp.grid(True, alpha=0.3)

    # Memory chart (if measured)
    if memory_result is not None:
        ax_mem = axes[1]
        components = [
            ("Baseline", memory_result.baseline_rss_mb, "#cccccc"),
            ("+ Non-PLE", memory_result.post_placeholder_rss_mb
                - memory_result.baseline_rss_mb, "tab:blue"),
            ("+ PLE working set",
                memory_result.peak_rss_mb
                - memory_result.post_placeholder_rss_mb, "tab:orange"),
        ]
        bottom = 0
        for label, val, color in components:
            ax_mem.bar(["Process RSS"], [val], bottom=bottom,
                       label=label, color=color, alpha=0.85)
            bottom += val
        if memory_gate is not None:
            ax_mem.axhline(y=memory_gate["budget_mb"], linestyle="--",
                           color="red", alpha=0.7,
                           label=f"Budget {memory_gate['budget_mb']:.0f} MB")
        ax_mem.set_ylabel("RAM (MB)")
        ax_mem.set_title("Peak RSS vs budget")
        ax_mem.legend(loc="upper right", fontsize=8)
        ax_mem.grid(True, alpha=0.3)

    fig.tight_layout()

    return rows, verdict_md, fig


# ---------------------------------------------------------------------------
# Dashboard assembly
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tab 5: Codec Comparison (TurboQuant vs RotorQuant)
# ---------------------------------------------------------------------------


def compare_codecs(
    head_dim: int,
    num_kv_heads: int,
    seq_len: int,
    distribution: str,
):
    """Run both codecs across residual-bits sweep and return comparison plot + table."""
    kv = synthesize_kv_tensor(
        seq_len=min(seq_len, 1024),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        distribution=distribution,
    )

    bits_sweep = [2, 3, 4, 6, 8]
    tq_cos, tq_mse = [], []
    rq_cos, rq_mse = [], []
    table_rows = []
    for b in bits_sweep:
        tq = TurboQuantCodec(head_dim=head_dim, config=TurboQuantConfig(residual_bits=b))
        rq = RotorQuantCodec(head_dim=head_dim, config=RotorQuantConfig(residual_bits=b))
        mt = tq.reconstruction_error(kv)
        mr = rq.reconstruction_error(kv)
        tq_cos.append(mt["cos_sim"])
        rq_cos.append(mr["cos_sim"])
        tq_mse.append(mt["mse"])
        rq_mse.append(mr["mse"])
        delta = mr["cos_sim"] - mt["cos_sim"]
        table_rows.append([
            b,
            f"{mt['cos_sim']:.4f}",
            f"{mt['mse']:.5f}",
            f"{mr['cos_sim']:.4f}",
            f"{mr['mse']:.5f}",
            f"{delta:+.4f}",
        ])

    # FMA comparison table
    fma_rows = []
    for d in sorted({64, 128, head_dim, 256, 512}):
        c = fma_cost_comparison(d)
        fma_rows.append([
            d,
            f"{c['turboquant_fmas']:,}",
            f"{c['rotorquant_fmas']:,}",
            f"{c['speedup_ratio']:.2f}×",
        ])

    # Plot: side-by-side quality + FMA cost
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(bits_sweep, tq_cos, marker="o", label="TurboQuant", color="tab:blue")
    ax1.plot(bits_sweep, rq_cos, marker="s", label="RotorQuant", color="tab:orange")
    ax1.set_xlabel("Residual bits")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("Reconstruction quality")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    all_dims = sorted({64, 128, head_dim, 256, 512})
    tq_fmas = [fma_cost_comparison(d)["turboquant_fmas"] for d in all_dims]
    rq_fmas = [fma_cost_comparison(d)["rotorquant_fmas"] for d in all_dims]
    x = np.arange(len(all_dims))
    width = 0.35
    ax2.bar(x - width/2, tq_fmas, width, label="TurboQuant", color="tab:blue", alpha=0.8)
    ax2.bar(x + width/2, rq_fmas, width, label="RotorQuant", color="tab:orange", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in all_dims])
    ax2.set_xlabel("head_dim")
    ax2.set_ylabel("FMAs per KV vector")
    ax2.set_title("Stage-1 rotation arithmetic cost")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    summary = (
        "### Quality vs arithmetic cost\n"
        "TurboQuant uses a dense Hadamard rotation (O(d·log d) via FWHT); "
        "RotorQuant uses blockwise 3D Clifford rotors (O(d)). At 4-bit residual, "
        "TurboQuant typically has a slight quality edge on synthetic KV; "
        "RotorQuant wins on kernel simplicity and parallelizability — both "
        "important on NPU/SIMD edge silicon.\n\n"
        "**Caveat:** these are reference Python implementations on synthetic "
        "heavy-tail KV. The RotorQuant paper's published PPL wins on real LLMs "
        "depend on an optimized kernel and real model activations. Real-model "
        "quality comparison requires the DynamicCache integration (next ADR)."
    )

    return table_rows, fma_rows, summary, fig


def build_dashboard() -> gr.Blocks:
    with gr.Blocks(
        title="dhurandhar — Decision Support",
    ) as app:
        gr.Markdown(
            """
            # 🤖 dhurandhar — Decision Support Dashboard

            Interactive analysis framework for edge model deployment decisions.
            Each tab maps to an acceptance criterion in the
            [decision log](docs/DECISION_LOG.md).
            """
        )

        with gr.Tabs():
            # -------------- Tab 1 --------------
            with gr.Tab("📊 PLE Memory Analysis"):
                gr.Markdown(
                    "Interactive memory footprint breakdown for the selected model. "
                    "For PLE models, the embedding table may exceed the decoder — "
                    "this is the central fact driving the mmap-vs-resident decision."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dd = gr.Dropdown(
                            choices=list_models(),
                            value="gemma4-e2b",
                            label="Model",
                        )
                        ctx_slider = gr.Slider(
                            512, 128_000, value=32768, step=512, label="Context tokens"
                        )
                        qbits_slider = gr.Slider(2, 8, value=4, step=1, label="Weight bits")
                        kvbits_slider = gr.Slider(2, 16, value=4, step=1, label="KV bits")
                        strip_audio_cb = gr.Checkbox(
                            value=True,
                            label="Strip audio encoder",
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=2):
                        breakdown_table = gr.Dataframe(
                            headers=["Component", "Size", "Notes"],
                            label="Component breakdown",
                            wrap=True,
                            value=[["Click 'Analyze'", "→", "to populate"]],
                        )
                        summary_md = gr.Markdown(
                            "_Adjust sliders on the left and click **Analyze**._"
                        )
                        breakdown_plot = gr.Plot(label="Resident vs mmap footprint")
                analyze_btn.click(
                    analyze_ple,
                    inputs=[model_dd, ctx_slider, qbits_slider, kvbits_slider, strip_audio_cb],
                    outputs=[breakdown_table, summary_md, breakdown_plot],
                )

            # -------------- Tab 2 --------------
            with gr.Tab("📱 Device Feasibility"):
                gr.Markdown(
                    "Per-device resident / mmap / infeasible verdicts. "
                    "The low-end eMMC profile is expected to fail — use the custom "
                    "row to model your exact target SKU."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dd2 = gr.Dropdown(
                            choices=list_models(),
                            value="gemma4-e2b",
                            label="Model",
                        )
                        ctx2 = gr.Slider(
                            512, 128_000, value=32768, step=512, label="Context tokens"
                        )
                        qbits2 = gr.Slider(2, 8, value=4, step=1, label="Weight bits")
                        strip2 = gr.Checkbox(value=True, label="Strip audio encoder")
                        target_tps = gr.Slider(
                            1, 50, value=15, step=1, label="Target decode tok/s"
                        )
                        gr.Markdown("### Custom device (optional)")
                        custom_name = gr.Textbox(label="Device name", value="")
                        custom_ram = gr.Number(value=0, label="RAM budget (MB)")
                        custom_flash = gr.Number(value=0, label="Flash read bw (GB/s)")
                        assess_btn = gr.Button("Assess devices", variant="primary")
                    with gr.Column(scale=2):
                        device_table = gr.Dataframe(
                            headers=[
                                "Device", "RAM", "Flash BW", "NPU",
                                "Mode", "Headroom", "Notes",
                            ],
                            label="Device feasibility",
                            wrap=True,
                            value=[["Click 'Assess devices'", "→", "to populate",
                                    "", "", "", ""]],
                        )
                        device_summary = gr.Markdown(
                            "_Click **Assess devices** to evaluate all profiles._"
                        )
                assess_btn.click(
                    assess_devices,
                    inputs=[model_dd2, ctx2, qbits2, strip2, target_tps,
                            custom_name, custom_ram, custom_flash],
                    outputs=[device_table, device_summary],
                )

            # -------------- Tab 3 --------------
            with gr.Tab("🗜️ TurboQuant KV"):
                gr.Markdown(
                    "TurboQuant (arXiv:2504.19874) — two-stage KV cache compression "
                    "via randomized Hadamard rotation + sign bit + residual. "
                    "Adapted to respect Gemma 4's Shared KV Cache."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        head_dim_s = gr.Slider(
                            64, 512, value=256, step=32, label="Head dim"
                        )
                        n_kv_s = gr.Slider(1, 16, value=4, step=1, label="Num KV heads")
                        seq_len_s = gr.Slider(
                            512, 131_072, value=32768, step=512, label="Seq length"
                        )
                        res_bits_s = gr.Slider(
                            2, 8, value=4, step=1, label="Residual bits"
                        )
                        dist_s = gr.Dropdown(
                            choices=["gaussian_heavy_tail", "gaussian"],
                            value="gaussian_heavy_tail",
                            label="KV distribution (heavy-tail is realistic)",
                        )
                        tq_btn = gr.Button("Run benchmark", variant="primary")
                    with gr.Column(scale=2):
                        tq_summary = gr.Markdown(
                            "_Click **Run benchmark** to measure quality and "
                            "compression ratio._"
                        )
                        tq_plot = gr.Plot(label="Quality vs residual bits")
                tq_btn.click(
                    benchmark_turboquant,
                    inputs=[head_dim_s, n_kv_s, seq_len_s, res_bits_s, dist_s],
                    outputs=[tq_summary, tq_plot],
                )

            # -------------- Tab 4 --------------
            with gr.Tab("⚡ Mmap Profiler (G1)"):
                gr.Markdown(
                    "**Run this on the actual target device** for a real G1 reading. "
                    "Throughput numbers are interesting; **peak RSS vs budget is the "
                    "real gate**. Enable *Measure peak RSS* to get the memory verdict."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        scale_s = gr.Slider(
                            0.01, 1.0, value=0.1, step=0.01,
                            label="Test file scale (1.0 ≈ 1 GB)",
                        )
                        num_tok_s = gr.Slider(
                            100, 5000, value=1000, step=100,
                            label="Tokens to profile",
                        )
                        warmup_s = gr.Slider(
                            0, 500, value=50, step=10, label="Warmup tokens",
                        )
                        qbits_prof = gr.Slider(
                            2, 8, value=4, step=1,
                            label="PLE-file quant bits (bytes-per-token)",
                        )
                        target_prof = gr.Slider(
                            1, 50, value=15, step=1,
                            label="Throughput target (tok/s)",
                        )
                        gr.Markdown("### Memory probe (the real G1 gate)")
                        measure_mem_cb = gr.Checkbox(
                            value=True,
                            label="Measure peak RSS vs budget",
                        )
                        weight_bits_dd = gr.Dropdown(
                            choices=["4", "8", "16"],
                            value="8",
                            label="Deployed weights — selects budget "
                                  "(4=1.5 GB, 8=2 GB, 16=4 GB)",
                        )
                        ctx_prof = gr.Slider(
                            512, 128_000, value=32768, step=512,
                            label="Context tokens (for non-PLE sizing)",
                        )
                        prof_btn = gr.Button(
                            "▶  Run profile", variant="primary"
                        )
                    with gr.Column(scale=2):
                        prof_verdict = gr.Markdown(
                            "_Click 'Run profile' to measure._"
                        )
                        prof_table = gr.Dataframe(
                            headers=[
                                "Pattern", "Mode", "tok/s", "MB/s",
                                "p50 µs", "p99 µs",
                            ],
                            label="Throughput — per-pattern results",
                            wrap=True,
                        )
                        prof_plot = gr.Plot(
                            label="Throughput and (optional) peak-RSS vs budget",
                        )
                prof_btn.click(
                    run_mmap_profile,
                    inputs=[
                        scale_s, num_tok_s, warmup_s, qbits_prof, target_prof,
                        measure_mem_cb, weight_bits_dd, ctx_prof,
                    ],
                    outputs=[prof_table, prof_verdict, prof_plot],
                )

            # -------------- Tab 5 --------------
            with gr.Tab("🔄 TurboQuant vs RotorQuant"):
                gr.Markdown(
                    "Side-by-side comparison of two KV cache compression codecs. "
                    "Both use the same two-stage pipeline; they differ in the "
                    "stage-1 rotation: TurboQuant uses a dense Hadamard butterfly, "
                    "RotorQuant uses blockwise 3D Clifford rotors (sparse, more "
                    "NPU-friendly). For edge deployment, kernel complexity "
                    "matters as much as quality."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        cc_head_dim = gr.Slider(
                            63, 512, value=255, step=3,
                            label="Head dim (RotorQuant needs multiple of 3)",
                        )
                        cc_n_kv = gr.Slider(
                            1, 16, value=4, step=1, label="Num KV heads"
                        )
                        cc_seq_len = gr.Slider(
                            128, 4096, value=1024, step=128,
                            label="Sample size for quality",
                        )
                        cc_dist = gr.Dropdown(
                            choices=["gaussian_heavy_tail", "gaussian"],
                            value="gaussian_heavy_tail",
                            label="KV distribution",
                        )
                        cc_btn = gr.Button("▶  Compare codecs", variant="primary")
                    with gr.Column(scale=2):
                        cc_summary = gr.Markdown(
                            "_Click **Compare codecs** to run the sweep._"
                        )
                        cc_quality = gr.Dataframe(
                            headers=["Res.bits",
                                     "TQ cos_sim", "TQ mse",
                                     "RQ cos_sim", "RQ mse",
                                     "Δ cos_sim"],
                            label="Quality at each residual bit width",
                            value=[["click →", "", "", "", "", ""]],
                            wrap=True,
                        )
                        cc_fma = gr.Dataframe(
                            headers=["head_dim",
                                     "TurboQuant FMAs",
                                     "RotorQuant FMAs",
                                     "Speedup"],
                            label="Stage-1 arithmetic cost per KV vector",
                            value=[["click →", "", "", ""]],
                            wrap=True,
                        )
                        cc_plot = gr.Plot(label="Quality + FMA cost")
                cc_btn.click(
                    compare_codecs,
                    inputs=[cc_head_dim, cc_n_kv, cc_seq_len, cc_dist],
                    outputs=[cc_quality, cc_fma, cc_summary, cc_plot],
                )

        gr.Markdown(
            """
            ---
            **dhurandhar** • edge AI • Apache 2.0
            """
        )
    return app


def launch(port: int = 7860, share: bool = False, server_name: str = "127.0.0.1") -> None:
    """Launch the Gradio dashboard.

    Windows notes:
      * If '127.0.0.1' hangs on page load, try 'localhost' or '0.0.0.0'
      * If the browser still hangs, close it, kill the Python process,
        and re-launch — Gradio occasionally caches a broken page.
      * Firewall prompts should be allowed for Python.exe.
    """
    app = build_dashboard()

    # Some Windows environments have issues with Gradio's default queue
    # under Python 3.13+. Disable it if the DHURANDHAR_GRADIO_NOQUEUE env var
    # is set.
    import os
    if os.environ.get("DHURANDHAR_GRADIO_NOQUEUE"):
        app.queue(default_concurrency_limit=None, max_size=None, status_update_rate="auto")

    print(f"\n  Launching dhurandhar dashboard on http://{server_name}:{port}")
    print("  (If the page hangs on Windows, try: dhurandhar-dashboard --server-name localhost)")
    print()

    app.launch(
        server_port=port,
        share=share,
        server_name=server_name,
        theme=gr.themes.Soft(),
        show_error=True,  # Show tracebacks in the UI, not just in terminal
        quiet=False,
    )


if __name__ == "__main__":
    launch()
