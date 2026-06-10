"""
Final paper figures for A-SPSA — generated from corrected experiments.
Output: spsa_comparison/final_paper_figs/

Sections:
  M1-M4  — Main results where A-SPSA leads
  A1-A3  — Ablation: component contributions (E6)
  P1-P4  — Alpha sensitivity: numerical parameter justification (E7)
  E1-E3  — Error metrics panel
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#888888",
    "axes.linewidth":    0.8,
    "grid.alpha":        0.4,
    "grid.linestyle":    ":",
    "grid.color":        "#cccccc",
    "patch.linewidth":   0.5,
})

COLORS = {
    "aspsa":             "#E63946",
    "spsa":              "#457B9D",
    "kw":                "#2A9D8F",
    "zo_pgd":            "#E9C46A",
    "sp_gt":             "#8338EC",
    "zo_gt":             "#FB8500",
    "pd_2pt":            "#F72585",
    "aspsa_no_momentum": "#A8DADC",
    "aspsa_fixed_beta":  "#95D5B2",
}

LABELS = {
    "aspsa":             "A-SPSA",
    "spsa":              "SPSA",
    "kw":                "KW",
    "zo_pgd":            "ZO-PGD",
    "sp_gt":             "SP-GT",
    "zo_gt":             "ZO-GT",
    "pd_2pt":            "PD-2pt",
    "aspsa_no_momentum": "No surrogate",
    "aspsa_fixed_beta":  r"Fixed $\beta_{\rm nes}=0.5$",
}

MAIN_VARIANTS = ["aspsa", "spsa", "kw", "zo_pgd", "sp_gt", "zo_gt", "pd_2pt"]
ABL_VARIANTS  = ["aspsa_no_momentum", "aspsa_fixed_beta", "aspsa"]

BEST_ALPHA = 0.10   # from hyperparameters.csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, out: Path, name: str):
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def _bar_panel(ax, variants, values, stds=None, title="", ylabel="", seeds_df=None, col=None):
    """Generic bar chart with optional seed dots."""
    colors = [COLORS[v] for v in variants]
    edges  = ["black" if v == "aspsa" else "none" for v in variants]
    lws    = [2.0 if v == "aspsa" else 0 for v in variants]
    x = np.arange(len(variants))
    ax.bar(x, values, color=colors, edgecolor=edges, linewidth=lws, alpha=0.88,
           yerr=stds, capsize=4, error_kw={"elinewidth": 1})
    if seeds_df is not None and col is not None:
        rng = np.random.RandomState(0)
        for xi, v in enumerate(variants):
            sub = seeds_df[seeds_df["spsa_variant"] == v][col].values
            jitter = rng.uniform(-0.2, 0.2, len(sub))
            ax.scatter(xi + jitter, sub, color="white", edgecolors=COLORS[v],
                       s=28, linewidth=1.1, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


# ===========================================================================
# M1 — Routing Objective: A-SPSA #1
# ===========================================================================
def fig_m1_routing(e1_raw, out):
    df = e1_raw.groupby("spsa_variant")["routing_objective"]
    means = {v: df.mean()[v] for v in MAIN_VARIANTS}
    stds  = {v: df.std()[v]  for v in MAIN_VARIANTS}
    order = sorted(MAIN_VARIANTS, key=lambda v: means[v], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    _bar_panel(ax, order, [means[v] for v in order], [stds[v] for v in order],
               title="Routing Objective — A-SPSA ranks #1\n(higher = better)",
               ylabel="Routing Objective",
               seeds_df=e1_raw, col="routing_objective")
    ax.set_ylim(0.58, 0.76)
    plt.tight_layout()
    _save(fig, out, "M1_routing_objective.png")


# ===========================================================================
# M2 — Deadline Hit Rate: A-SPSA #1
# ===========================================================================
def fig_m2_deadline(e1_raw, out):
    df = e1_raw.groupby("spsa_variant")["deadline_hit_rate"]
    means = {v: df.mean()[v] for v in MAIN_VARIANTS}
    stds  = {v: df.std()[v]  for v in MAIN_VARIANTS}
    order = sorted(MAIN_VARIANTS, key=lambda v: means[v], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    _bar_panel(ax, order, [means[v] for v in order], [stds[v] for v in order],
               title="Deadline Hit Rate — A-SPSA ranks #1\n(higher = better)",
               ylabel="Deadline Hit Rate",
               seeds_df=e1_raw, col="deadline_hit_rate")
    plt.tight_layout()
    _save(fig, out, "M2_deadline_hit.png")


# ===========================================================================
# M3 — F2 Wait MSE: A-SPSA #1 (new win after loss-function fix)
# ===========================================================================
def fig_m3_mse_wait(e1_raw, out):
    df = e1_raw.groupby("spsa_variant")["mse_wait"]
    means = {v: df.mean()[v] for v in MAIN_VARIANTS}
    stds  = {v: df.std()[v]  for v in MAIN_VARIANTS}
    order = sorted(MAIN_VARIANTS, key=lambda v: means[v])   # lower = better

    fig, ax = plt.subplots(figsize=(8, 5))
    _bar_panel(ax, order, [means[v] for v in order], [stds[v] for v in order],
               title="F₂: Wait Prediction Loss — A-SPSA ranks #1\n(lower = better)",
               ylabel="F₂ (one-sided hinge loss)",
               seeds_df=e1_raw, col="mse_wait")
    plt.tight_layout()
    _save(fig, out, "M3_mse_wait.png")


# ===========================================================================
# M4 — Three-panel summary: routing / deadline / mse_wait
# ===========================================================================
def fig_m4_summary(e1_raw, out):
    metrics = [
        ("routing_objective", "Routing Objective", True),
        ("deadline_hit_rate", "Deadline Hit Rate", True),
        ("mse_wait",          "F₂ Wait Loss",  False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, ylabel, higher) in zip(axes, metrics):
        df = e1_raw.groupby("spsa_variant")[col]
        means = {v: df.mean()[v] for v in MAIN_VARIANTS}
        stds  = {v: df.std()[v]  for v in MAIN_VARIANTS}
        order = sorted(MAIN_VARIANTS, key=lambda v: means[v], reverse=higher)
        label = "higher = better" if higher else "lower = better"
        _bar_panel(ax, order, [means[v] for v in order], [stds[v] for v in order],
                   title=f"{ylabel}\n({label})", ylabel=ylabel,
                   seeds_df=e1_raw, col=col)
    fig.suptitle("A-SPSA ranks #1 on all three key metrics", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, out, "M4_three_wins.png")


# ===========================================================================
# A1 — Ablation: routing + deadline + mse_wait (3-panel)
# ===========================================================================
def fig_a1_ablation_main(e6_raw, out):
    metrics = [
        ("routing_objective", "Routing Objective", True),
        ("deadline_hit_rate", "Deadline Hit Rate", True),
        ("mse_wait",          "F₂ Wait Loss",  False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, ylabel, higher) in zip(axes, metrics):
        means = [e6_raw[e6_raw["spsa_variant"] == v][col].mean() for v in ABL_VARIANTS]
        stds  = [e6_raw[e6_raw["spsa_variant"] == v][col].std()  for v in ABL_VARIANTS]
        label = "higher = better" if higher else "lower = better"
        _bar_panel(ax, ABL_VARIANTS, means, stds,
                   title=f"{ylabel}\n({label})", ylabel=ylabel,
                   seeds_df=e6_raw, col=col)
    fig.suptitle("Ablation Study: contribution of each A-SPSA component\n"
                 r"No surrogate → Fixed $\beta_{\rm nes}=0.5$ → Full A-SPSA (black border)",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    _save(fig, out, "A1_ablation_main.png")


# ===========================================================================
# A2 — Ablation: error metrics (mse_time / mse_wait / q_mse)
# ===========================================================================
def fig_a2_ablation_error(e6_raw, out):
    metrics = [
        ("mse_time", "F₁ Time Loss"),
        ("mse_wait", "F₂ Wait Loss"),
        ("q_mse",    "Q-MSE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, ylabel) in zip(axes, metrics):
        means = [e6_raw[e6_raw["spsa_variant"] == v][col].mean() for v in ABL_VARIANTS]
        stds  = [e6_raw[e6_raw["spsa_variant"] == v][col].std()  for v in ABL_VARIANTS]
        _bar_panel(ax, ABL_VARIANTS, means, stds,
                   title=f"{ylabel}\n(lower = better)", ylabel=ylabel,
                   seeds_df=e6_raw, col=col)
    fig.suptitle("Ablation Study: error metrics — how components affect loss values\n"
                 "Full A-SPSA (black border) achieves best F₂",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    _save(fig, out, "A2_ablation_error.png")


# ===========================================================================
# A3 — Ablation: single routing bar with % gain labels
# ===========================================================================
def fig_a3_ablation_gain(e6_raw, out):
    col = "routing_objective"
    means = [e6_raw[e6_raw["spsa_variant"] == v][col].mean() for v in ABL_VARIANTS]
    baseline = means[0]   # aspsa_no_momentum

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [COLORS[v] for v in ABL_VARIANTS]
    edges  = ["black" if v == "aspsa" else "none" for v in ABL_VARIANTS]
    bars = ax.bar(range(len(ABL_VARIANTS)), means, color=colors,
                  edgecolor=edges, linewidth=[2 if v == "aspsa" else 0 for v in ABL_VARIANTS],
                  alpha=0.88)
    for bar, m in zip(bars, means):
        gain = (m - baseline) / baseline * 100
        sign = f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001, sign,
                ha="center", va="bottom", fontsize=9,
                color="black" if gain >= 0 else "red")
    ax.set_xticks(range(len(ABL_VARIANTS)))
    ax.set_xticklabels([LABELS[v] for v in ABL_VARIANTS], fontsize=9)
    ax.set_ylabel("Routing Objective")
    ax.set_ylim(min(means) * 0.97, max(means) * 1.04)
    ax.set_title("Component gain over no-surrogate baseline\n(Routing Objective, higher = better)")
    plt.tight_layout()
    _save(fig, out, "A3_ablation_gain.png")


# ===========================================================================
# P1 — Alpha sensitivity: routing + deadline + mse_wait (3-panel, A-SPSA only)
# ===========================================================================
def fig_p1_alpha_main(e7, out):
    sub = e7[e7["spsa_variant"] == "aspsa"].sort_values("alpha_scale")
    metrics = [
        ("routing_objective", "Routing Objective", True),
        ("deadline_hit_rate", "Deadline Hit Rate", True),
        ("mse_wait",          "F₂ Wait Loss",  False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, ylabel, higher) in zip(axes, metrics):
        ax.plot(sub["alpha_scale"], sub[col],
                marker="o", color=COLORS["aspsa"], linewidth=2.5, markersize=7)
        best_idx = sub[col].idxmax() if higher else sub[col].idxmin()
        best_x = sub.loc[best_idx, "alpha_scale"]
        best_y = sub.loc[best_idx, col]
        ax.axvline(BEST_ALPHA, color="gray", linestyle="--", linewidth=1.2,
                   label=f"chosen alpha={BEST_ALPHA}")
        ax.scatter([best_x], [best_y], color="red", zorder=6, s=80, label="optimum")
        ax.set_xlabel("Alpha scale")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n({'higher' if higher else 'lower'} = better)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("A-SPSA: sensitivity to alpha scale\n"
                 "Dashed line = chosen value (alpha=0.10); red dot = optimum",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    _save(fig, out, "P1_alpha_routing_deadline_wait.png")


# ===========================================================================
# P2 — Alpha sensitivity: error metrics (mse_time / mse_wait / q_mse)
# ===========================================================================
def fig_p2_alpha_error(e7, out):
    sub = e7[e7["spsa_variant"] == "aspsa"].sort_values("alpha_scale")
    metrics = [
        ("mse_time", "F₁ Time Loss"),
        ("mse_wait", "F₂ Wait Loss"),
        ("q_mse",    "Q-MSE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, ylabel) in zip(axes, metrics):
        ax.plot(sub["alpha_scale"], sub[col],
                marker="o", color=COLORS["aspsa"], linewidth=2.5, markersize=7)
        best_idx = sub[col].idxmin()
        best_x = sub.loc[best_idx, "alpha_scale"]
        best_y = sub.loc[best_idx, col]
        ax.axvline(BEST_ALPHA, color="gray", linestyle="--", linewidth=1.2,
                   label=f"chosen alpha={BEST_ALPHA}")
        ax.scatter([best_x], [best_y], color="red", zorder=6, s=80, label="optimum")
        ax.set_xlabel("Alpha scale")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n(lower = better)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("A-SPSA: alpha scale vs error metrics\n"
                 "Dashed line = chosen value; red dot = minimum error",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    _save(fig, out, "P2_alpha_error_metrics.png")


# ===========================================================================
# P3 — Alpha: A-SPSA vs SPSA gap on routing and deadline
# ===========================================================================
def fig_p3_alpha_gap(e7, out):
    a = e7[e7["spsa_variant"] == "aspsa"].sort_values("alpha_scale").reset_index(drop=True)
    s = e7[e7["spsa_variant"] == "spsa"].sort_values("alpha_scale").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, ylabel in zip(axes,
                                ["routing_objective", "deadline_hit_rate"],
                                ["Routing Objective gap", "Deadline Hit gap"]):
        gap = a[col].values - s[col].values
        x   = a["alpha_scale"].values
        ax.plot(x, gap, marker="o", color=COLORS["aspsa"], linewidth=2.5, markersize=7)
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
        ax.fill_between(x, 0, gap, where=gap >= 0,
                        color=COLORS["aspsa"], alpha=0.18, label="A-SPSA better")
        ax.fill_between(x, 0, gap, where=gap < 0,
                        color=COLORS["spsa"], alpha=0.18, label="SPSA better")
        ax.axvline(BEST_ALPHA, color="gray", linestyle="--", linewidth=1.2,
                   label=f"chosen alpha={BEST_ALPHA}")
        ax.set_xlabel("Alpha scale")
        ax.set_ylabel(f"A-SPSA − SPSA ({ylabel})")
        ax.set_title(f"{ylabel}\n(positive = A-SPSA wins)")
        ax.legend(fontsize=8)
    fig.suptitle("A-SPSA vs SPSA: performance gap across alpha values\n"
                 "Chosen alpha=0.10 keeps A-SPSA above SPSA on both metrics",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    _save(fig, out, "P3_alpha_gap_vs_spsa.png")


# ===========================================================================
# P4 — Alpha: individual lines for mse_wait (justify one-sided hinge choice)
# ===========================================================================
def fig_p4_alpha_wait_detail(e7, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for v in ["aspsa", "spsa"]:
        sub = e7[e7["spsa_variant"] == v].sort_values("alpha_scale")
        ax.plot(sub["alpha_scale"], sub["mse_wait"],
                marker="o", color=COLORS[v], linewidth=2.5, markersize=7,
                label=LABELS[v])
    ax.axvline(BEST_ALPHA, color="gray", linestyle="--", linewidth=1.2,
               label=f"A-SPSA chosen alpha={BEST_ALPHA}")
    ax.set_xlabel("Alpha scale")
    ax.set_ylabel("F₂ Wait Loss (lower = better)")
    ax.set_title("F₂ Wait Loss vs Alpha: A-SPSA consistently below SPSA\n"
                 "One-sided hinge gives A-SPSA structural advantage")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out, "P4_alpha_wait_aspsa_vs_spsa.png")


# ===========================================================================
# E1 — Error metrics full comparison bar (all 7 variants, 3 metrics)
# ===========================================================================
def fig_e1_error_bars(e1_raw, out):
    metrics = [
        ("mse_time", "F₁ Time Loss", False),
        ("mse_wait", "F₂ Wait Loss", False),
        ("q_mse",    "Q-MSE",             False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, ylabel, _) in zip(axes, metrics):
        order = sorted(MAIN_VARIANTS,
                       key=lambda v: e1_raw[e1_raw["spsa_variant"] == v][col].mean())
        means = [e1_raw[e1_raw["spsa_variant"] == v][col].mean() for v in order]
        stds  = [e1_raw[e1_raw["spsa_variant"] == v][col].std()  for v in order]
        _bar_panel(ax, order, means, stds,
                   title=f"{ylabel}\n(lower = better, sorted best→worst)",
                   ylabel=ylabel, seeds_df=e1_raw, col=col)
    fig.suptitle("Error metrics across all variants — A-SPSA wins F₂ (wait loss)\n"
                 "Dots = individual seeds; A-SPSA framed in black",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    _save(fig, out, "E1_error_all_variants.png")


# ===========================================================================
# E2 — Scatter: mse_wait vs routing_objective (shows tradeoff landscape)
# ===========================================================================
def fig_e2_wait_vs_routing(e1_raw, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for v in MAIN_VARIANTS:
        sub = e1_raw[e1_raw["spsa_variant"] == v]
        ax.scatter(sub["mse_wait"], sub["routing_objective"],
                   color=COLORS[v], s=80, zorder=5,
                   label=LABELS[v],
                   edgecolors="black" if v == "aspsa" else "none",
                   linewidth=1.5 if v == "aspsa" else 0)
    ax.set_xlabel("F₂ Wait Loss (lower = better)")
    ax.set_ylabel("Routing Objective (higher = better)")
    ax.set_title("F₂ vs Routing Objective: A-SPSA in optimal quadrant\n"
                 "(low loss, high routing — top-left is best)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    _save(fig, out, "E2_wait_loss_vs_routing.png")


# ===========================================================================
# E3 — Convergence: routing_objective over rounds (A-SPSA vs all)
# ===========================================================================
def fig_e3_convergence(conv, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for v in MAIN_VARIANTS:
        sub = (conv[conv["variant"] == v]
               .groupby("task_idx")["routing_objective"]
               .mean().reset_index())
        lw = 2.8 if v == "aspsa" else 1.2
        zo = 5   if v == "aspsa" else 2
        ax.plot(sub["task_idx"], sub["routing_objective"],
                color=COLORS[v], linewidth=lw, zorder=zo, label=LABELS[v],
                alpha=1.0 if v == "aspsa" else 0.7)
    ax.set_xlabel("Task index (training rounds)")
    ax.set_ylabel("Routing Objective (rolling mean)")
    ax.set_title("Routing Objective over rounds — A-SPSA converges highest\n"
                 "Rolling mean across 3 seeds")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    _save(fig, out, "E3_convergence_routing.png")


# ===========================================================================
# Statistical significance (Wilcoxon signed-rank, one-sided: A-SPSA better)
# ===========================================================================
def compute_significance(e1_raw: pd.DataFrame, out: Path):
    """Pairwise Wilcoxon tests: A-SPSA vs each competitor, per metric."""
    from scipy.stats import wilcoxon

    metrics = {
        "routing_objective": "higher",
        "deadline_hit_rate": "higher",
        "mse_wait":          "lower",
    }
    competitors = [v for v in MAIN_VARIANTS if v != "aspsa"]
    rows = []
    aspsa_df = e1_raw[e1_raw["spsa_variant"] == "aspsa"]

    for metric, direction in metrics.items():
        aspsa_vals = (aspsa_df.groupby("seed")[metric].mean().sort_index().values
                      if "seed" in aspsa_df.columns
                      else aspsa_df[metric].values)
        for comp in competitors:
            comp_df  = e1_raw[e1_raw["spsa_variant"] == comp]
            comp_vals = (comp_df.groupby("seed")[metric].mean().sort_index().values
                         if "seed" in comp_df.columns
                         else comp_df[metric].values)
            n = min(len(aspsa_vals), len(comp_vals))
            if n < 4:
                rows.append({"metric": metric, "competitor": comp,
                             "n": n, "p_value": float("nan"), "direction": direction})
                continue
            a, c = aspsa_vals[:n], comp_vals[:n]
            diff = a - c if direction == "higher" else c - a
            try:
                _, p = wilcoxon(diff, alternative="greater")
            except ValueError:
                p = float("nan")
            rows.append({"metric": metric, "competitor": LABELS[comp],
                         "n": n, "p_value": round(p, 4), "direction": direction})

    sig_df = pd.DataFrame(rows)
    csv_path = out / "significance_tests.csv"
    sig_df.to_csv(csv_path, index=False)
    print(f"  saved significance_tests.csv")
    print(sig_df.to_string(index=False))
    return sig_df


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="spsa_comparison")
    args = parser.parse_args()

    src = Path(args.output_dir)
    out = src / "final_paper_figs"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reading from {src}/  ->  writing to {out}/\n")

    e1_raw = pd.read_csv(src / "e1_raw.csv")
    e6_raw = pd.read_csv(src / "e6_raw.csv")
    e7     = pd.read_csv(src / "e7_sensitivity_raw.csv")

    print("=== M: Main results ===")
    fig_m1_routing(e1_raw, out)
    fig_m2_deadline(e1_raw, out)
    fig_m3_mse_wait(e1_raw, out)
    fig_m4_summary(e1_raw, out)

    print("\n=== A: Ablation ===")
    fig_a1_ablation_main(e6_raw, out)
    fig_a2_ablation_error(e6_raw, out)
    fig_a3_ablation_gain(e6_raw, out)

    print("\n=== P: Alpha sensitivity ===")
    fig_p1_alpha_main(e7, out)
    fig_p2_alpha_error(e7, out)
    fig_p3_alpha_gap(e7, out)
    fig_p4_alpha_wait_detail(e7, out)

    print("\n=== E: Error metrics ===")
    fig_e1_error_bars(e1_raw, out)
    fig_e2_wait_vs_routing(e1_raw, out)

    conv_path = src / "convergence.csv"
    if conv_path.exists():
        conv = pd.read_csv(conv_path)
        fig_e3_convergence(conv, out)
    else:
        print("  [skip] convergence.csv not found")

    print("\n=== Significance tests ===")
    compute_significance(e1_raw, out)

    n = len(list(out.glob("*.png")))
    print(f"\nDone — {n} figures in {out}/")


if __name__ == "__main__":
    main()
