"""Command-line entry points.

Installed as console scripts via pyproject.toml:
    dhurandhar-analyze-ple   — run PLE footprint analysis across device profiles
    dhurandhar-benchmark-kv  — run TurboQuant KV compression benchmark
    dhurandhar-train-lora    — launch a LoRA fine-tuning job
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .config import DEVICE_PROFILES
from .mmap_profiler import MEMORY_BUDGETS_MB, PATTERNS, MmapDecodeProfiler
from .models import GEMMA4_E2B, get_model
from .ple_analysis import PLEFootprintAnalyzer
from .rotorquant import RotorQuantCodec, RotorQuantConfig, fma_cost_comparison
from .turboquant import (
    KVCacheCompressor,
    TurboQuantCodec,
    TurboQuantConfig,
    synthesize_kv_tensor,
)

# ---------------------------------------------------------------------------
# dhurandhar-analyze-ple
# ---------------------------------------------------------------------------


@click.command(name="analyze-ple")
@click.option(
    "--model", "model_name", default="gemma4-e2b",
    help="Model name (built-in slug or path to YAML). "
         "Built-ins: gemma4-e2b, gemma4-e4b, qwen2.5-0.5b, qwen2.5-1.5b, "
         "qwen2.5-3b, granite-3.3-2b, llama-3.2-1b, llama-3.2-3b."
)
@click.option(
    "--context-tokens",
    type=int,
    default=32768,
    help="Context length for KV cache sizing.",
)
@click.option("--quant-bits", type=int, default=4, help="Weight quantization bits.")
@click.option("--kv-bits", type=int, default=4, help="KV cache quantization bits.")
@click.option(
    "--strip-audio/--keep-audio",
    default=True,
    help="Strip the audio encoder (Default: strip).",
)
@click.option(
    "--device",
    type=click.Choice(sorted(DEVICE_PROFILES.keys())),
    default=None,
    help="Assess a specific device; omit to print breakdown for all profiles.",
)
@click.option(
    "--target-decode-tps",
    type=float,
    default=15.0,
    help="Target decode tokens/sec for the go/no-go threshold.",
)
@click.option("--json-out", type=click.Path(), default=None, help="Also write JSON report.")
def analyze_ple_cmd(
    model_name: str,
    context_tokens: int,
    quant_bits: int,
    kv_bits: int,
    strip_audio: bool,
    device: str | None,
    target_decode_tps: float,
    json_out: str | None,
) -> None:
    """Analyze memory footprint and device feasibility for the given model."""
    arch = get_model(model_name)
    analyzer = PLEFootprintAnalyzer(arch)

    click.echo("=" * 78)
    click.echo(f" {arch.name} memory breakdown — context={context_tokens:,} tokens,"
               f" Q{quant_bits} weights, Q{kv_bits} KV")
    click.echo("=" * 78)

    breakdown = analyzer.compute_breakdown(
        context_tokens=context_tokens,
        quant_bits=quant_bits,
        kv_bits=kv_bits,
        strip_audio=strip_audio,
    )
    click.echo(analyzer.format_breakdown(breakdown))
    click.echo("")

    report: dict = {
        "breakdown": {
            "context_tokens": context_tokens,
            "quant_bits": quant_bits,
            "kv_bits": kv_bits,
            "strip_audio": strip_audio,
            "decoder_mb": breakdown.decoder_mb,
            "ple_table_mb": breakdown.ple_table_mb,
            "kv_cache_mb": breakdown.kv_cache_mb,
            "resident_total_mb": breakdown.resident_total_mb,
            "mmap_total_mb": breakdown.mmap_total_mb,
            "ple_to_decoder_ratio": round(breakdown.ple_table_mb / breakdown.decoder_mb, 3),
        },
        "device_assessments": [],
    }

    devices_to_check = [device] if device else sorted(DEVICE_PROFILES.keys())
    click.echo("=" * 78)
    click.echo(" Per-device feasibility (PLE mmap vs resident)")
    click.echo("=" * 78)

    for dev_key in devices_to_check:
        f = analyzer.assess_device(
            dev_key,
            context_tokens=context_tokens,
            quant_bits=quant_bits,
            kv_bits=kv_bits,
            strip_audio=strip_audio,
            decode_tokens_per_sec_target=target_decode_tps,
        )
        status_color = {"resident": "green", "mmap": "yellow", "infeasible": "red"}[f.mode]
        click.echo("")
        click.echo(f"[{dev_key}] {f.device.name}")
        click.echo(f"  RAM budget:       {f.device.ram_budget_mb:>6.0f} MB")
        click.echo(f"  Flash bandwidth:  {f.device.flash_read_gbps:>6.2f} GB/s")
        click.echo(f"  NPU:              {'yes' if f.device.supports_npu else 'no'}")
        click.secho(f"  Mode:             {f.mode}", fg=status_color)
        click.echo(f"  Headroom:         {f.headroom_mb:>6.1f} MB")
        click.echo(f"  Flash-bound t/s:  {f.flash_read_bound_tok_per_sec:>6.1f}")
        click.echo(f"  Notes:            {f.rationale}")
        report["device_assessments"].append(
            {
                "device": dev_key,
                "device_name": f.device.name,
                "mode": f.mode,
                "headroom_mb": f.headroom_mb,
                "flash_read_bound_tps": f.flash_read_bound_tok_per_sec,
                "rationale": f.rationale,
            }
        )

    click.echo("")
    click.echo("=" * 78)
    click.echo(" Go/No-go summary")
    click.echo("=" * 78)
    infeasible = [a for a in report["device_assessments"] if a["mode"] == "infeasible"]
    if infeasible:
        click.secho(
            f"❌ {len(infeasible)} device profile(s) infeasible without mitigation:",
            fg="red",
        )
        for a in infeasible:
            click.echo(f"   - {a['device_name']}: {a['rationale']}")
    else:
        click.secho("✅ All assessed profiles viable (measurement required on-device).", fg="green")

    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=2))
        click.echo(f"\nJSON report written to {json_out}")


# ---------------------------------------------------------------------------
# dhurandhar-benchmark-kv
# ---------------------------------------------------------------------------


@click.command(name="benchmark-kv")
@click.option("--head-dim", type=int, default=256)
@click.option("--num-kv-heads", type=int, default=4)
@click.option("--seq-len", type=int, default=32768)
@click.option("--num-layers", type=int, default=30)
@click.option("--shared-kv-last-n", type=int, default=6)
@click.option("--residual-bits", type=int, default=4)
@click.option(
    "--distribution",
    type=click.Choice(["gaussian", "gaussian_heavy_tail"]),
    default="gaussian_heavy_tail",
)
@click.option("--samples", type=int, default=2048, help="Per-head sample count for quality eval.")
def benchmark_kv_cmd(
    head_dim: int,
    num_kv_heads: int,
    seq_len: int,
    num_layers: int,
    shared_kv_last_n: int,
    residual_bits: int,
    distribution: str,
    samples: int,
) -> None:
    """Benchmark TurboQuant KV compression on synthetic KV tensors."""
    click.echo("=" * 78)
    click.echo(" TurboQuant KV compression benchmark")
    click.echo("=" * 78)
    click.echo(f"  head_dim={head_dim}, num_kv_heads={num_kv_heads}, seq_len={seq_len:,}")
    click.echo(f"  num_layers={num_layers} ({shared_kv_last_n} shared KV), "
               f"residual_bits={residual_bits}")
    click.echo("")

    # Quality measurement on synthetic realistic tensor
    codec = TurboQuantCodec(
        head_dim=head_dim,
        config=TurboQuantConfig(residual_bits=residual_bits),
    )
    kv = synthesize_kv_tensor(
        seq_len=samples,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        distribution=distribution,
    )
    metrics = codec.reconstruction_error(kv)

    click.echo("Quality (synthetic KV reconstruction):")
    click.echo(f"  MSE:                 {metrics['mse']:.6f}")
    click.echo(f"  Cosine similarity:   {metrics['cos_sim']:.4f}")
    click.echo(f"  Norm preservation:   {metrics['norm_ratio']:.4f}")
    click.echo(f"  Effective bits/chan: {metrics['effective_bits']:.2f}")
    click.echo(f"  Compression ratio:   {metrics['compression_ratio']:.2f}x vs bf16")
    click.echo("")

    # Whole-model savings estimate
    compressor = KVCacheCompressor(
        num_layers=num_layers,
        head_dim=head_dim,
        shared_kv_last_n=shared_kv_last_n,
    )
    savings = compressor.memory_savings_estimate(seq_len=seq_len, num_kv_heads=num_kv_heads)

    click.echo(f"KV cache footprint @ {seq_len:,} tokens:")
    click.echo(f"  Baseline (bf16):     {savings['baseline_mb']:>8.1f} MB")
    click.echo(f"  TurboQuant:          {savings['quantized_mb']:>8.1f} MB")
    click.echo(f"  Savings:             {savings['savings_mb']:>8.1f} MB "
               f"({savings['savings_ratio']}x reduction)")
    click.echo(f"  Fresh-KV layers:     {savings['fresh_kv_layers']}")
    click.echo(f"  Shared-KV layers:    {savings['shared_kv_layers']} (skipped)")


# ---------------------------------------------------------------------------
# dhurandhar-train-lora
# ---------------------------------------------------------------------------


@click.command(name="train-lora")
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--dry-run", is_flag=True, help="Build model and trainer but don't train.")
def train_lora_cmd(config_path: str, dry_run: bool) -> None:
    """Launch a LoRA fine-tuning job from a YAML config file."""
    from .finetune import (
        FinetuneJobConfig,
        build_model_and_tokenizer,
        build_trainer,
        count_parameters,
    )

    cfg = FinetuneJobConfig.from_yaml(config_path)
    click.echo("=" * 78)
    click.echo(f" LoRA fine-tuning — base model: {cfg.base_model}")
    click.echo(f" Dataset: {cfg.dataset_name}")
    click.echo("=" * 78)

    model, tokenizer = build_model_and_tokenizer(cfg)
    params = count_parameters(model)
    click.echo(f"Total params:     {params['total']:>14,}")
    click.echo(f"Trainable params: {params['trainable']:>14,} ({params['trainable_pct']}%)")

    if not cfg.dataset_name:
        click.secho("\nNo dataset_name set in config. Use --dry-run or provide dataset.", fg="red")
        sys.exit(1)

    # Load dataset
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name)
    train_ds = ds[cfg.dataset_split_train]
    eval_ds = ds.get(cfg.dataset_split_eval)

    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds)

    if dry_run:
        click.echo("\n[--dry-run] Trainer constructed successfully. Exiting.")
        return

    click.echo("\nStarting training...")
    trainer.train()
    trainer.save_model()
    click.echo(f"\nAdapter saved to: {cfg.training.output_dir}")


# ---------------------------------------------------------------------------
# dhurandhar-profile-mmap
# ---------------------------------------------------------------------------


@click.command(name="profile-mmap")
@click.option(
    "--scale",
    type=float,
    default=1.0,
    help="Test file size as a fraction of full PLE (~1 GB at 1.0). "
         "Use 0.1 for quick tests, 1.0 for realistic measurement.",
)
@click.option("--num-tokens", type=int, default=2000)
@click.option("--warmup-tokens", type=int, default=100)
@click.option("--quant-bits", type=int, default=4,
              help="PLE-file quantization bits (affects bytes-per-token).")
@click.option(
    "--target-tps",
    type=float,
    default=15.0,
    help="Target decode tokens/sec for the throughput gate.",
)
@click.option(
    "--test-file",
    type=click.Path(),
    default=None,
    help="Test file path. Defaults to ~/.cache/dhurandhar/ple_profile.bin",
)
@click.option("--warm-only", is_flag=True, help="Skip cold throughput measurements.")
@click.option(
    "--measure-memory",
    is_flag=True,
    help="Also measure peak RSS against deployment budgets (the real G1 gate).",
)
@click.option(
    "--weight-bits",
    type=click.Choice(["4", "8", "16"]),
    default="8",
    help="Deployed weight precision for the memory probe — selects budget: "
         "4=INT4 (1.5 GB), 8=INT8 (2 GB), 16=BF16 (4 GB).",
)
@click.option(
    "--context-tokens",
    type=int,
    default=32768,
    help="Context size used when sizing the non-PLE resident placeholder.",
)
@click.option(
    "--budget-interpretation",
    type=click.Choice(["full_process", "weights_only", "both"]),
    default="both",
    help="Interpret budget as full process RSS, weights-only (model size), "
         "or evaluate both (default).",
)
@click.option("--json-out", type=click.Path(), default=None)
def profile_mmap_cmd(
    scale: float,
    num_tokens: int,
    warmup_tokens: int,
    quant_bits: int,
    target_tps: float,
    test_file: str | None,
    warm_only: bool,
    measure_memory: bool,
    weight_bits: str,
    context_tokens: int,
    budget_interpretation: str,
    json_out: str | None,
) -> None:
    """Profile real mmap decode throughput AND/OR peak RSS on this host."""
    from pathlib import Path

    tf = Path(test_file) if test_file else None
    profiler = MmapDecodeProfiler.from_architecture(GEMMA4_E2B, quant_bits=quant_bits, test_file=tf)

    click.echo("=" * 78)
    click.echo(" Mmap profiler — measuring on THIS host")
    click.echo("=" * 78)
    click.echo(f"  Full PLE size at model defaults: {profiler.total_ple_bytes/1024/1024:,.0f} MB")
    click.echo(f"  Bytes per decode token:        {profiler.bytes_per_token} B")
    click.echo(f"  Test file scale:               {scale} "
               f"(~{profiler.total_ple_bytes*scale/1024/1024:,.0f} MB)")
    click.echo("")

    click.echo("Preparing test file (first run takes a few seconds)...")
    profiler.prepare(scale=scale)

    # --------- Throughput measurement ---------
    click.echo(f"Running throughput profile ({len(PATTERNS)} patterns × "
               f"{'cold+warm' if not warm_only else 'warm only'})...")
    click.echo("")
    results = profiler.profile_all(
        num_tokens=num_tokens,
        warmup_tokens=warmup_tokens,
        include_warm=True,
    )
    if warm_only:
        results = [r for r in results if not r.cold]

    from tabulate import tabulate as tab
    rows = [
        [r.pattern, "cold" if r.cold else "warm",
         f"{r.tokens_per_sec:,.0f}", f"{r.mb_per_sec:,.0f}",
         f"{r.p50_token_latency_us:.1f}", f"{r.p99_token_latency_us:.1f}"]
        for r in results
    ]
    click.echo(tab(rows,
        headers=["Pattern", "Mode", "tok/s", "MB/s", "p50 µs", "p99 µs"],
        tablefmt="simple"))
    click.echo("")

    throughput_gate = profiler.evaluate_gate(results, target_tps=target_tps)
    click.echo("=" * 78)
    click.echo(f" Throughput gate (target {target_tps} tok/s)")
    click.echo("=" * 78)
    color_map = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "UNKNOWN": "cyan"}
    click.secho(f"  {throughput_gate['verdict']}",
                fg=color_map.get(throughput_gate['verdict'], 'white'), bold=True)
    click.echo(f"  {throughput_gate['detail']}")

    # --------- Memory probe (the actual G1 gate) ---------
    memory_result = None
    memory_gate = None
    if measure_memory:
        click.echo("")
        click.echo("=" * 78)
        click.echo(" Peak RSS memory probe — the real G1 gate")
        click.echo("=" * 78)

        w_bits = int(weight_bits)
        budget_name = {
            4: "int4_aggressive",
            8: "int8_deployment",
            16: "bf16_development",
        }[w_bits]
        budget_mb = MEMORY_BUDGETS_MB[budget_name]
        click.echo(f"  Weight precision:    {weight_bits}-bit ({budget_name})")
        click.echo(f"  Budget:              {budget_mb:,.0f} MB peak RSS")
        click.echo(f"  Context tokens:      {context_tokens:,}")
        click.echo("")
        click.echo("Running memory probe (decoding with PLE mmap'd + "
                   "dense non-PLE placeholder)...")

        memory_result = profiler.profile_memory(
            weight_bits=w_bits,
            num_tokens=num_tokens,
            warmup_tokens=warmup_tokens,
            context_tokens=context_tokens,
        )

        click.echo("")
        click.echo("RSS progression:")
        click.echo(f"  Baseline (interpreter):       "
                   f"{memory_result.baseline_rss_mb:>8,.0f} MB")
        click.echo(f"  + Non-PLE resident buffers:   "
                   f"{memory_result.post_placeholder_rss_mb:>8,.0f} MB")
        click.echo(f"  + PLE mmap'd (pre-fault):     "
                   f"{memory_result.post_mmap_rss_mb:>8,.0f} MB")
        click.echo(f"  Peak during decode:           "
                   f"{memory_result.peak_rss_mb:>8,.0f} MB  ← full-process")
        click.echo(f"  Steady-state (last 20% mean): "
                   f"{memory_result.steady_state_rss_mb:>8,.0f} MB")
        if memory_result.peak_rss_file_mb is not None:
            click.echo(f"  Peak file-backed (PLE pages): "
                       f"{memory_result.peak_rss_file_mb:>8,.0f} MB")
            click.echo(f"  Peak anonymous (heap/stack):  "
                       f"{memory_result.peak_rss_anon_mb:>8,.0f} MB")
        click.echo(f"  Non-PLE component size:       "
                   f"{memory_result.non_ple_component_mb:>8,.0f} MB")
        click.echo(f"  PLE resident working set:     "
                   f"{memory_result.ple_resident_working_set_mb:>8,.0f} MB")
        click.echo(f"  Weights-only (decoder+PLE):   "
                   f"{memory_result.weights_only_mb:>8,.0f} MB  "
                   f"← model size at {weight_bits}-bit")
        click.echo("")

        # Evaluate budget under selected interpretation(s)
        interpretations = (
            [budget_interpretation] if budget_interpretation != "both"
            else ["weights_only", "full_process"]
        )
        all_gates = []
        for interp in interpretations:
            gate = profiler.evaluate_budget(
                memory_result,
                budget_name=budget_name,
                budget_interpretation=interp,
            )
            all_gates.append(gate)
            label = {
                "weights_only": "Weights-only budget check",
                "full_process": "Full-process RSS budget check",
            }[interp]
            click.echo(f"  {label}:")
            click.secho(f"    {gate['verdict']}",
                        fg=color_map.get(gate['verdict'], 'white'), bold=True)
            click.echo(f"    {gate['detail']}")
            click.echo("")

        # Use the strictest verdict as the primary memory_gate for JSON output
        verdict_severity = {"PASS": 0, "WARN": 1, "FAIL": 2, "UNKNOWN": 3}
        memory_gate = max(all_gates, key=lambda g: verdict_severity.get(g["verdict"], 0))

    if json_out:
        report = {
            "throughput": {
                "target_tps": target_tps,
                "host_info": {
                    "ple_size_mb": profiler.total_ple_bytes / 1024 / 1024,
                    "bytes_per_token": profiler.bytes_per_token,
                    "scale": scale,
                },
                "results": [r.to_dict() for r in results],
                "gate": throughput_gate,
            },
            "memory": (
                {
                    "weight_bits": int(weight_bits),
                    "result": memory_result.to_dict() if memory_result else None,
                    "gate": memory_gate,
                }
                if measure_memory else None
            ),
        }
        import json
        Path(json_out).write_text(json.dumps(report, indent=2))
        click.echo(f"\nJSON report written to {json_out}")


# ---------------------------------------------------------------------------
# dhurandhar-compare-codecs
# ---------------------------------------------------------------------------


@click.command(name="compare-codecs")
@click.option("--head-dim", type=int, default=255,
              help="Must be divisible by 3 for clean RotorQuant blocks.")
@click.option("--num-kv-heads", type=int, default=4)
@click.option("--seq-len", type=int, default=2048,
              help="Sample size for quality measurement.")
@click.option(
    "--residual-bits",
    type=str,
    default="2,3,4,6,8",
    help="Comma-separated list of residual bit widths to sweep.",
)
@click.option(
    "--distribution",
    type=click.Choice(["gaussian", "gaussian_heavy_tail"]),
    default="gaussian_heavy_tail",
)
@click.option("--json-out", type=click.Path(), default=None)
def compare_codecs_cmd(
    head_dim: int,
    num_kv_heads: int,
    seq_len: int,
    residual_bits: str,
    distribution: str,
    json_out: str | None,
) -> None:
    """Compare TurboQuant vs RotorQuant on the same synthetic KV.

    RotorQuant is expected to offer comparable (sometimes slightly worse)
    reconstruction quality at a meaningful arithmetic cost reduction — the
    trade is relevant when the bottleneck is the stage-1 rotation kernel
    rather than the residual quantization.
    """
    from tabulate import tabulate as tab

    bits_list = [int(b.strip()) for b in residual_bits.split(",")]

    click.echo("=" * 78)
    click.echo(" TurboQuant vs RotorQuant codec comparison")
    click.echo("=" * 78)
    click.echo(f"  head_dim={head_dim}, num_kv_heads={num_kv_heads}, seq_len={seq_len}")
    click.echo(f"  distribution={distribution}")
    click.echo("")

    # Generate one tensor, compare both codecs on it
    kv = synthesize_kv_tensor(
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        distribution=distribution,
    )

    rows = []
    all_results = {}
    for bits in bits_list:
        tq = TurboQuantCodec(head_dim=head_dim, config=TurboQuantConfig(residual_bits=bits))
        rq = RotorQuantCodec(head_dim=head_dim, config=RotorQuantConfig(residual_bits=bits))
        mt = tq.reconstruction_error(kv)
        mr = rq.reconstruction_error(kv)
        rows.append([
            bits,
            f"{mt['cos_sim']:.4f}",
            f"{mt['mse']:.5f}",
            f"{mr['cos_sim']:.4f}",
            f"{mr['mse']:.5f}",
            f"{mr['cos_sim'] - mt['cos_sim']:+.4f}",
        ])
        all_results[bits] = {"turboquant": mt, "rotorquant": mr}

    click.echo(tab(
        rows,
        headers=["Res.bits", "TQ cos_sim", "TQ mse",
                 "RQ cos_sim", "RQ mse", "Δ cos_sim"],
        tablefmt="simple",
    ))
    click.echo("")

    # Arithmetic cost comparison
    click.echo("Arithmetic cost per KV vector (stage-1 rotation only):")
    cost_rows = []
    for d in [64, 128, head_dim, 256, 512]:
        c = fma_cost_comparison(d)
        cost_rows.append([
            d,
            c["turboquant_fmas"],
            c["rotorquant_fmas"],
            f"{c['speedup_ratio']:.2f}x",
        ])
    # Dedupe head_dim
    seen = set()
    cost_rows_unique = []
    for r in cost_rows:
        if r[0] not in seen:
            seen.add(r[0])
            cost_rows_unique.append(r)
    cost_rows_unique.sort(key=lambda r: r[0])

    click.echo(tab(
        cost_rows_unique,
        headers=["head_dim", "TurboQuant FMAs", "RotorQuant FMAs", "Speedup"],
        tablefmt="simple",
    ))
    click.echo("")

    click.echo("Notes:")
    click.echo("  • TurboQuant FMA cost assumes Fast Walsh-Hadamard Transform (d·log2(d)),")
    click.echo("    not naive dense matmul (d²).")
    click.echo("  • RotorQuant's real wins are in kernel simplicity and parallelizability,")
    click.echo("    not FMA count alone. Measure on target silicon for the full picture.")
    click.echo("  • Quality numbers here are on synthetic heavy-tail KV. Real-model")
    click.echo("    perplexity comparison requires the DynamicCache integration.")

    if json_out:
        import json
        from pathlib import Path
        report = {
            "config": {
                "head_dim": head_dim,
                "num_kv_heads": num_kv_heads,
                "seq_len": seq_len,
                "distribution": distribution,
            },
            "quality_sweep": {str(b): r for b, r in all_results.items()},
            "fma_costs": [fma_cost_comparison(d) for d in sorted({64, 128, head_dim, 256, 512})],
        }
        Path(json_out).write_text(json.dumps(report, indent=2))
        click.echo(f"\nJSON report written to {json_out}")



# ---------------------------------------------------------------------------
# dhurandhar-dashboard
# ---------------------------------------------------------------------------


@click.command(name="dashboard")
@click.option("--port", type=int, default=7860)
@click.option("--share", is_flag=True, help="Create a public Gradio share link.")
@click.option(
    "--server-name",
    default="127.0.0.1",
    help="Bind address. Use 0.0.0.0 to expose on LAN.",
)
def dashboard_cmd(port: int, share: bool, server_name: str) -> None:
    """Launch the Gradio dashboard for interactive PLE/TurboQuant/mmap analysis."""
    try:
        from .dashboard import launch
    except ImportError as e:
        click.secho(
            f"Gradio not installed: {e}\n"
            "Install with: uv sync --extra dashboard",
            fg="red",
        )
        raise SystemExit(1) from e

    launch(port=port, share=share, server_name=server_name)


# ---------------------------------------------------------------------------
# Grouped root (if invoked via `python -m dhurandhar.cli`)
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """dhurandhar — edge deployment analysis toolkit."""


main.add_command(analyze_ple_cmd)
main.add_command(benchmark_kv_cmd)
main.add_command(train_lora_cmd)
main.add_command(profile_mmap_cmd)
main.add_command(compare_codecs_cmd)
main.add_command(dashboard_cmd)


if __name__ == "__main__":
    main()
