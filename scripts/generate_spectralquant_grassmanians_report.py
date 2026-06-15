"""Subspace-drift recalibration trigger (Tier A: controlled synthetic).

Question
--------
A SpectralQuant calibration (eigenbasis U, spectrum lambda, effective rank
d_eff, water-filled bit vector b) is fit on one data distribution and then
*assumed* valid as the serving distribution drifts. This harness tests whether
two INFERENCE-FREE geometric quantities predict the fidelity penalty of running
a stale calibration on drifted data:

  (a) principal-angle (Grassmannian) distance between the top-k calibration
      eigenbasis and the target-domain eigenbasis  ->  d_G = ||theta||_2 ;
  (b) L1 distance between the two water-filled bit allocations  ->  ||b_cal - b_tgt||_1 .

Hypothesis: d_G predicts the ORDERING of the penalty; the allocation-L1 term
sharpens its MAGNITUDE (water-filling acts on eigenvalue magnitudes, not just
the span, so a corpus can share the subspace yet mis-allocate bits).

Tier A (this script): controlled drift via an SO(d) geodesic rotation
U_t = U0 @ expm(t*A) of a fixed base eigenbasis, plus optional spectrum jitter.
No model required. Reuses dhurandhar's SpectralQuantCodec verbatim.

Two methodological points this Tier-A study established empirically and that
carry forward to Tier B:
  * CUT k AT THE SPECTRAL KNEE, not the participation-ratio round. When the
    d_eff cut lands inside a degenerate noise floor, the boundary eigenvector
    is interchangeable across samples and saturates d_G even at zero drift
    (Spearman collapses ~0.98 -> ~0.4). Report d_G across a small k-window
    around the knee to show stability.
  * USE MSE, NOT COSINE, as the dependent variable. Reconstruction cosine
    saturates near 1.0 and cannot resolve drift; the MSE ratio does.

Metric note: dhurandhar's codec reports RECONSTRUCTION fidelity on K, not
attention-output cosine. Reconstruction MSE is the available proxy; attention-
output cosine (the SpectralQuant paper's metric, and Idea 3's true dependent
variable) needs the real-activation engine and is the Tier-B / v2 metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm, subspace_angles
from scipy.stats import spearmanr

from dhurandhar.spectralquant import SpectralQuantConfig, SpectralQuantCodec


# --------------------------------------------------------------------------- #
# Controlled-drift generators
# --------------------------------------------------------------------------- #
def base_spectrum(head_dim: int, signal_eigs: list[float], floor: float) -> torch.Tensor:
    """Clean-gap eigenspectrum: well-separated signal dims, hard gap to floor."""
    lam = torch.full((head_dim,), float(floor))
    lam[: len(signal_eigs)] = torch.tensor(signal_eigs, dtype=torch.float32)
    return lam


def random_orthogonal(head_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=g))
    return q


def unit_skew(head_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((head_dim, head_dim))
    a = m - m.T
    return a / (np.linalg.norm(a) + 1e-12)


def covariance_sample(U: torch.Tensor, lam: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, lam.numel(), generator=g)
    return (z * lam.sqrt().unsqueeze(0)) @ U.T


# --------------------------------------------------------------------------- #
# Subspace cut + inference-free predictors
# --------------------------------------------------------------------------- #
def knee_cut(eigenvalues: torch.Tensor, search_frac: float = 0.25) -> int:
    """Cut k at the largest log-eigenvalue gap in the top region (the knee).

    This is the fix for the degenerate-d_eff failure: the participation-ratio
    round can land the cut inside a flat noise floor, where the boundary
    eigenvector is interchangeable across samples.
    """
    lam = eigenvalues.clamp(min=1e-12).log()
    top = max(2, int(len(lam) * search_frac))
    gaps = (lam[: top - 1] - lam[1:top])  # consecutive log drops
    return int(torch.argmax(gaps).item()) + 1


def grassmann_distance(U_cal: torch.Tensor, U_tgt: torch.Tensor, k: int) -> tuple[float, float]:
    theta = subspace_angles(U_cal[:, :k].numpy(), U_tgt[:, :k].numpy())
    return float(np.linalg.norm(theta)), float(theta.max())


def allocation_l1(b_cal: torch.Tensor, b_tgt: torch.Tensor) -> float:
    return float((b_cal - b_tgt).abs().sum().item())


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict:
    d = args.head_dim
    cfg = SpectralQuantConfig(avg_bits=args.bits)
    lam0 = base_spectrum(d, args.signal_eigs, args.floor)
    U0 = random_orthogonal(d, seed=0)
    A = unit_skew(d, seed=7)
    expA = {t: torch.from_numpy(expm(t * A)).float() for t in args.drift}

    X_base = covariance_sample(U0, lam0, args.n_calib, seed=1000)
    codec_cal = SpectralQuantCodec(head_dim=d, config=cfg)
    codec_cal.calibrate(X_base)
    U_cal, b_cal = codec_cal.eigenbasis, codec_cal.bit_alloc

    # Cut at the knee unless asked to reproduce the participation-ratio failure.
    k = (knee_cut(codec_cal.eigenvalues) if args.cut == "knee" else int(codec_cal.d_eff))
    k_window = [kk for kk in range(max(1, k - 2), k + 3) if kk <= d]

    rows: list[dict] = []
    for t in args.drift:
        U_t = U0 @ expA[t]
        for js in range(args.jitter_seeds):
            rng = np.random.default_rng(100 + js)
            jit = 1.0 if js == 0 else np.exp(rng.normal(0.0, args.jitter, size=d))
            lam_t = lam0 * torch.from_numpy(np.asarray(jit, dtype=np.float32))
            for ss in range(args.sample_seeds):
                X_t = covariance_sample(U_t, lam_t, args.n_calib, seed=2000 + 31 * js + ss)
                stale = codec_cal.reconstruction_error(X_t)
                codec_t = SpectralQuantCodec(head_dim=d, config=cfg)
                codec_t.calibrate(X_t)
                matched = codec_t.reconstruction_error(X_t)
                d_G, theta_max = grassmann_distance(U_cal, codec_t.eigenbasis, k)
                d_G_window = {kk: grassmann_distance(U_cal, codec_t.eigenbasis, kk)[0]
                              for kk in k_window}
                mse_ratio = stale["mse"] / max(matched["mse"], 1e-12)
                rows.append(
                    {
                        "t": t, "jitter_seed": js, "sample_seed": ss,
                        "k_cut": k, "d_G": d_G, "theta_max": theta_max,
                        "d_G_window": d_G_window,
                        "alloc_l1": allocation_l1(b_cal, codec_t.bit_alloc),
                        "d_eff_participation_tgt": int(codec_t.d_eff),
                        "cos_stale": stale["cos_sim"], "cos_matched": matched["cos_sim"],
                        "mse_stale": stale["mse"], "mse_matched": matched["mse"],
                        "mse_ratio": mse_ratio,
                        "log_mse_ratio": float(np.log(mse_ratio)),
                        "delta_cos": matched["cos_sim"] - stale["cos_sim"],
                    }
                )
    meta = {key: (str(val) if isinstance(val, Path) else val) for key, val in vars(args).items()}
    return {"meta": meta, "k_cut": k, "k_window": k_window,
            "d_eff_participation_cal": int(codec_cal.d_eff), "rows": rows}


def summarize(result: dict) -> str:
    r = result["rows"]
    dG = np.array([x["d_G"] for x in r])
    al = np.array([x["alloc_l1"] for x in r])
    y = np.array([x["log_mse_ratio"] for x in r])
    Xd = np.column_stack([dG, al, np.ones_like(dG)])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ beta
    r2 = 1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    import collections
    by_t = collections.defaultdict(list)
    for x in r:
        by_t[x["t"]].append(x["mse_ratio"])
    tlines = "\n".join(f"     t={t:<5} mse_ratio={np.mean(v):.3f}" for t, v in sorted(by_t.items()))
    return "\n".join([
        f"n points                       : {len(r)}",
        f"subspace cut k                 : {result['k_cut']}  "
        f"(participation-ratio d_eff = {result['d_eff_participation_cal']})",
        f"Spearman log(mse_ratio)~d_G    : {spearmanr(dG, y).statistic:+.3f}",
        f"Spearman log(mse_ratio)~allocL1: {spearmanr(al, y).statistic:+.3f}",
        f"2-term OLS R^2                 : {float(r2):.3f}",
        "mse_ratio by drift t (stale/matched; >1 => stale worse):",
        tlines,
    ])


def plot(result: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = result["rows"]
    dG = [x["d_G"] for x in r]
    mr = [x["mse_ratio"] for x in r]
    al = [x["alloc_l1"] for x in r]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(dG, mr, c=al, cmap="viridis", s=30, edgecolor="k", linewidth=0.3)
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel(r"Grassmannian distance $\|\theta\|_2$  (top-$k$ subspaces, $k$ at knee)")
    ax.set_ylabel(r"stale-calibration MSE penalty  (stale / matched)")
    ax.set_title("Idea 3 (Tier A): subspace drift predicts the recalibration penalty")
    fig.colorbar(sc, label=r"allocation $\|b_{cal}-b_{tgt}\|_1$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--signal-eigs", type=float, nargs="+", default=[60., 36., 22., 13., 8.])
    p.add_argument("--floor", type=float, default=0.02)
    p.add_argument("--bits", type=float, default=3.0)
    p.add_argument("--n-calib", type=int, default=6000)
    p.add_argument("--drift", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6])
    p.add_argument("--cut", choices=["knee", "participation"], default="knee",
                   help="subspace cut; 'participation' reproduces the degeneracy failure")
    p.add_argument("--jitter", type=float, default=0.25, help="spectrum log-jitter sigma")
    p.add_argument("--jitter-seeds", type=int, default=3)
    p.add_argument("--sample-seeds", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = p.parse_args()

    result = run(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "idea3_drift_synthetic.json").write_text(json.dumps(result, indent=2))
    plot(result, args.out_dir / "idea3_drift_synthetic.png")
    print("=" * 72)
    print(" Idea 3 - Tier A controlled-synthetic drift study")
    print("=" * 72)
    print(summarize(result))
    print(f"\nJSON  -> {args.out_dir / 'idea3_drift_synthetic.json'}")
    print(f"Figure-> {args.out_dir / 'idea3_drift_synthetic.png'}")


if __name__ == "__main__":
    main()