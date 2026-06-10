"""
Top-3 publication-quality figures for the ApPLIED '26 paper.

Figure 1 — E6 Ablation:      component contribution bar chart + MSE inset
Figure 2 — E3 Non-stationarity: recovery drop bars
Figure 3 — E1 Pareto scatter:  routing_objective vs DHR

Run:
    python -m src.plot_paper_top3 [--data spsa_comparison_v5] [--out viz/paper_top3]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── shared style ──────────────────────────────────────────────────────────────

COLORS = {
    "aspsa":             "#d62728",
    "spsa":              "#1f77b4",
    "kw":                "#2ca02c",
    "zo_pgd":            "#ff7f0e",
    "sp_gt":             "#9467bd",
    "zo_gt":             "#8c564b",
    "pd_2pt":            "#17becf",
    "aspsa_no_momentum": "#e377c2",
    "aspsa_fixed_beta":  "#bcbd22",
}
LABELS = {
    "aspsa":             "A-SPSA (full)",
    "spsa":              "SPSA",
    "kw":                "KW",
    "zo_pgd":            "ZO-PGD",
    "sp_gt":             "SP-GT",
    "zo_gt":             "ZO-GT",
    "pd_2pt":            "PD-2pt",
    "aspsa_no_momentum": "A-SPSA\n(no momentum)",
    "aspsa_fixed_beta":  "A-SPSA\n(fixed β)",
}

DPI = 220
FONT = {"family": "DejaVu Sans", "size": 10}


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.5, zorder=0)


# ── Figure 1: E6 Ablation ─────────────────────────────────────────────────────

def plot_ablation(e6_path: Path, out_path: Path) -> None:
    df = pd.read_csv(e6_path)
    order = ["aspsa", "aspsa_no_momentum", "aspsa_fixed_beta", "spsa"]
    agg = (df.groupby("variant")[["routing_objective", "deadline_hit_rate", "mse_time"]]
             .agg(["mean", "std"])
             .loc[order])

    labels = [LABELS[v] for v in order]
    ro_m  = agg["routing_objective"]["mean"].values
    ro_s  = agg["routing_objective"]["std"].values
    dhr_m = agg["deadline_hit_rate"]["mean"].values
    dhr_s = agg["deadline_hit_rate"]["std"].values
    mse_m = agg["mse_time"]["mean"].values
    mse_s = agg["mse_time"]["std"].values
    colors = [COLORS[v] for v in order]

    plt.rc("font", **FONT)
    fig = plt.figure(figsize=(11, 4.8))
    # left panel: routing + DHR grouped bars
    ax1 = fig.add_axes([0.07, 0.14, 0.52, 0.72])
    x = np.arange(len(order))
    w = 0.32
    bars1 = ax1.bar(x - w/2, ro_m,  w, yerr=ro_s,  color=colors,
                    alpha=0.90, capsize=4, label="Routing obj. ↑",
                    error_kw={"elinewidth": 1.1},
                    edgecolor=["#111" if v=="aspsa" else c for v,c in zip(order,colors)],
                    linewidth=[2.2 if v=="aspsa" else 0.8 for v in order])
    bars2 = ax1.bar(x + w/2, dhr_m, w, yerr=dhr_s, color=colors,
                    alpha=0.45, capsize=4, label="DHR ↑",
                    error_kw={"elinewidth": 1.1}, hatch="///",
                    edgecolor=["#111" if v=="aspsa" else c for v,c in zip(order,colors)],
                    linewidth=[2.2 if v=="aspsa" else 0.8 for v in order])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Score", labelpad=4)
    ax1.set_ylim(0.55, 0.88)
    ax1.set_title("Ablation: component contribution", pad=7, fontsize=10.5)
    _despine(ax1)

    solid_patch = mpatches.Patch(facecolor="#999", alpha=0.90, label="Routing obj. ↑")
    hatch_patch = mpatches.Patch(facecolor="#999", alpha=0.45, hatch="///", label="DHR ↑")
    ax1.legend(handles=[solid_patch, hatch_patch], fontsize=8, loc="lower right")

    # right panel: MSE time (log scale — fixed_beta explodes)
    ax2 = fig.add_axes([0.67, 0.14, 0.30, 0.72])
    bars3 = ax2.bar(x, mse_m, 0.55, yerr=mse_s, color=colors,
                    capsize=4, alpha=0.85,
                    error_kw={"elinewidth": 1.1},
                    edgecolor=["#111" if v=="aspsa" else c for v,c in zip(order,colors)],
                    linewidth=[2.2 if v=="aspsa" else 0.8 for v in order])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("MSE (time prediction) ↓", labelpad=4)
    ax2.set_yscale("log")
    ax2.set_title("Timing MSE (log scale)", pad=7, fontsize=10.5)
    _despine(ax2)
    ax2.grid(axis="y", which="both", linewidth=0.35, alpha=0.5)

    # annotate the catastrophic value
    idx_fb = order.index("aspsa_fixed_beta")
    ax2.annotate(f"{mse_m[idx_fb]:.1f}",
                 xy=(idx_fb, mse_m[idx_fb]),
                 xytext=(idx_fb + 0.5, mse_m[idx_fb] * 1.3),
                 fontsize=8, color=COLORS["aspsa_fixed_beta"],
                 arrowprops=dict(arrowstyle="->", lw=0.9,
                                 color=COLORS["aspsa_fixed_beta"]))

    fig.suptitle("Fig. 1 — Ablation Study: each A-SPSA component matters",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Figure 1 (ablation) ->", out_path)


# ── Figure 2: E3 Non-stationarity recovery ────────────────────────────────────

def plot_recovery(e3_path: Path, out_path: Path) -> None:
    df = pd.read_csv(e3_path)
    order = sorted(df["variant"].unique(),
                   key=lambda v: df.loc[df.variant==v, "drop"].values[0])
    colors = [COLORS.get(v, "#888") for v in order]
    labels = [LABELS.get(v, v) for v in order]
    drops  = [df.loc[df.variant==v, "drop"].values[0] for v in order]
    post   = [df.loc[df.variant==v, "post_shift_sr_100"].values[0] for v in order]

    plt.rc("font", **FONT)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: drop bars (smaller = better)
    ax = axes[0]
    bars = ax.barh(labels, drops, color=colors, alpha=0.85,
                   edgecolor=["#111" if v=="aspsa" else c for v,c in zip(order,colors)],
                   linewidth=[2.2 if v=="aspsa" else 0.8 for v in order])
    # annotate best
    aspsa_idx = order.index("aspsa")
    ax.annotate("best", xy=(drops[aspsa_idx], aspsa_idx),
                xytext=(drops[aspsa_idx] + 0.003, aspsa_idx),
                fontsize=8, color=COLORS["aspsa"], va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=COLORS["aspsa"]))
    ax.set_xlabel("Performance drop after distribution shift ↓", labelpad=4)
    ax.set_title("A-SPSA: smallest performance drop", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linewidth=0.35, alpha=0.5)

    # right: post-shift success rate
    ax2 = axes[1]
    ax2.barh(labels, post, color=colors, alpha=0.85,
             edgecolor=["#111" if v=="aspsa" else c for v,c in zip(order,colors)],
             linewidth=[2.2 if v=="aspsa" else 0.8 for v in order])
    ax2.set_xlabel("Success rate 100 tasks after shift ↑", labelpad=4)
    ax2.set_title("Post-shift recovery", pad=6)
    ax2.set_xlim(0.83, 0.91)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", linewidth=0.35, alpha=0.5)
    ax2.set_yticklabels([])

    fig.suptitle("Fig. 2 — Non-Stationarity: A-SPSA recovers fastest after distribution shift",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Figure 2 (recovery) ->", out_path)


# ── Figure 3: E1 Pareto scatter ───────────────────────────────────────────────

# Label placement: (ha, x_offset, y_offset) per variant
_LABEL_POS = {
    "aspsa":   ("right", -0.0005, +0.013),
    "spsa":    ("left",  +0.0005, -0.018),
    "kw":      ("left",  +0.0005, +0.013),
    "zo_pgd":  ("left",  +0.0005, -0.018),
    "sp_gt":   ("left",  +0.0005, -0.018),
    "zo_gt":   ("right", -0.0005, +0.013),
    "pd_2pt":  ("right", -0.0005, -0.018),
}


def plot_pareto(e1_path: Path, out_path: Path) -> None:
    df = pd.read_csv(e1_path)
    agg = df.groupby("variant")[
        ["routing_objective", "deadline_hit_rate"]
    ].agg(["mean", "std"])

    plt.rc("font", **FONT)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # compute axis limits with margin before placing labels
    ro_all  = agg["routing_objective"]["mean"]
    dhr_all = agg["deadline_hit_rate"]["mean"]
    ro_std  = agg["routing_objective"]["std"]
    dhr_std = agg["deadline_hit_rate"]["std"]
    ax.set_xlim(ro_all.min()  - ro_std.max()  - 0.010,
                ro_all.max()  + ro_std.max()  + 0.018)
    ax.set_ylim(dhr_all.min() - dhr_std.max() - 0.04,
                dhr_all.max() + dhr_std.max() + 0.06)

    for v in agg.index:
        ro    = agg.loc[v, ("routing_objective", "mean")]
        dhr   = agg.loc[v, ("deadline_hit_rate",  "mean")]
        ro_s  = agg.loc[v, ("routing_objective", "std")]
        dhr_s = agg.loc[v, ("deadline_hit_rate",  "std")]
        c     = COLORS.get(v, "#888")
        lbl   = LABELS.get(v, v).replace("\n", " ")
        is_aspsa = (v == "aspsa")

        ax.errorbar(ro, dhr, xerr=ro_s, yerr=dhr_s,
                    fmt="o", color=c,
                    markersize=13 if is_aspsa else 8,
                    markeredgecolor="#111" if is_aspsa else c,
                    markeredgewidth=2.0 if is_aspsa else 0.8,
                    elinewidth=1.0, capsize=3, alpha=0.9, zorder=3)

        ha, dx, dy = _LABEL_POS.get(v, ("left", 0.001, 0.012))
        ax.text(ro + dx, dhr + dy, lbl,
                fontsize=8.5 if is_aspsa else 7.5,
                color=c, fontweight="bold" if is_aspsa else "normal",
                ha=ha, va="center")

    # shaded dominance region for A-SPSA
    best_ro  = agg.loc["aspsa", ("routing_objective", "mean")]
    best_dhr = agg.loc["aspsa", ("deadline_hit_rate",  "mean")]
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.fill_betweenx([best_dhr, ylim[1]], best_ro, xlim[1],
                     color=COLORS["aspsa"], alpha=0.06, zorder=0,
                     label="A-SPSA dominance region")
    ax.axvline(best_ro,  color=COLORS["aspsa"], lw=0.7, ls="--", alpha=0.4)
    ax.axhline(best_dhr, color=COLORS["aspsa"], lw=0.7, ls="--", alpha=0.4)

    ax.set_xlabel("Routing Objective (higher is better)", labelpad=5)
    ax.set_ylabel("Deadline Hit Rate (higher is better)", labelpad=5)
    ax.set_title("Routing Objective vs. Deadline Hit Rate\n(mean +/- std, 3 seeds x 800 tasks)",
                 pad=8, fontsize=10.5)
    _despine(ax)
    ax.grid(linewidth=0.35, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Figure 3 (Pareto scatter) ->", out_path)


# ── Figure 4: Convergence — routing objective over tasks ─────────────────────

def _smooth(series: pd.Series, w: int) -> pd.Series:
    return series.rolling(w, min_periods=1).mean()


def plot_convergence_routing(conv_path: Path, out_path: Path, window: int = 30) -> None:
    df = pd.read_csv(conv_path).dropna(subset=["routing_objective"])
    order = ["aspsa", "spsa", "kw", "zo_pgd", "zo_gt", "pd_2pt", "sp_gt"]

    plt.rc("font", **FONT)
    fig, ax = plt.subplots(figsize=(10, 4.8))

    for v in order:
        sub = df[df.variant == v].sort_values("task_idx")
        if sub.empty:
            continue
        by_seed = sub.groupby("seed")["routing_objective"].apply(list)
        curves = [_smooth(pd.Series(vals), window).values for vals in by_seed]
        max_len = max(len(c) for c in curves)
        mat = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            mat[i, :len(c)] = c
        mean = np.nanmean(mat, 0)
        std  = np.nanstd(mat,  0)
        xs   = np.arange(len(mean))
        c    = COLORS.get(v, "#888")
        lw   = 2.5 if v == "aspsa" else 1.1
        zo   = 4   if v == "aspsa" else 2
        ax.plot(xs, mean, color=c, lw=lw, label=LABELS.get(v, v), zorder=zo)
        ax.fill_between(xs, mean - std, mean + std, color=c, alpha=0.09, zorder=zo-1)

    ax.set_xlabel("Task index", labelpad=4)
    ax.set_ylabel(f"Rolling routing objective (w={window})", labelpad=4)
    ax.set_title("Convergence of Routing Objective over Tasks (3 seeds)", pad=7)
    ax.legend(ncol=4, fontsize=8, loc="lower right", framealpha=0.9)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Figure 4 (convergence routing) ->", out_path)


# ── Figure 5: Convergence — 3-panel DHR / MSE / latency ──────────────────────

def plot_convergence_multi(conv_path: Path, out_path: Path, window: int = 30) -> None:
    df = pd.read_csv(conv_path).dropna(subset=["routing_objective"])
    order = ["aspsa", "spsa", "kw", "zo_pgd", "zo_gt", "pd_2pt", "sp_gt"]

    panels = [
        ("deadline_hit",      "Deadline Hit Rate", True),
        ("q_mse",             "Queue MSE (lower is better)", False),
        ("success_rate",      "Success Rate", True),
    ]

    plt.rc("font", **FONT)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, (col, ylabel, higher_better) in zip(axes, panels):
        sub_df = df.dropna(subset=[col])
        for v in order:
            sub = sub_df[sub_df.variant == v].sort_values("task_idx")
            if sub.empty:
                continue
            by_seed = sub.groupby("seed")[col].apply(list)
            curves = [_smooth(pd.Series(vals), window).values for vals in by_seed]
            max_len = max(len(c) for c in curves)
            mat = np.full((len(curves), max_len), np.nan)
            for i, c in enumerate(curves):
                mat[i, :len(c)] = c
            mean = np.nanmean(mat, 0)
            std  = np.nanstd(mat,  0)
            xs   = np.arange(len(mean))
            c    = COLORS.get(v, "#888")
            lw   = 2.5 if v == "aspsa" else 1.0
            zo   = 4   if v == "aspsa" else 2
            ax.plot(xs, mean, color=c, lw=lw, label=LABELS.get(v, v), zorder=zo)
            ax.fill_between(xs, mean - std, mean + std, color=c, alpha=0.08, zorder=zo-1)

        arrow = "up" if higher_better else "down"
        ax.set_title(f"{ylabel} ({arrow})", pad=5, fontsize=9.5)
        ax.set_xlabel("Task index", fontsize=8)
        _despine(ax)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=7, fontsize=7.5,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Per-Task Convergence Dynamics (rolling mean +/- std, w=30)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Figure 5 (convergence multi) ->", out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="spsa_comparison_v5",
                    help="Simulation output directory")
    ap.add_argument("--out",  default="viz/paper_top3",
                    help="Output directory for figures")
    args = ap.parse_args()

    data = Path(args.data)
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    conv = Path("spsa_comparison") / "convergence.csv"

    plot_ablation          (data / "e6_raw.csv",      out / "fig1_ablation.pdf")
    plot_ablation          (data / "e6_raw.csv",      out / "fig1_ablation.png")
    plot_recovery          (data / "e3_recovery.csv", out / "fig2_recovery.pdf")
    plot_recovery          (data / "e3_recovery.csv", out / "fig2_recovery.png")
    plot_pareto            (data / "e1_raw.csv",      out / "fig3_pareto.pdf")
    plot_pareto            (data / "e1_raw.csv",      out / "fig3_pareto.png")
    plot_convergence_routing(conv,                    out / "fig4_convergence_routing.pdf")
    plot_convergence_routing(conv,                    out / "fig4_convergence_routing.png")
    plot_convergence_multi  (conv,                    out / "fig5_convergence_multi.pdf")
    plot_convergence_multi  (conv,                    out / "fig5_convergence_multi.png")

    print(f"\nAll figures saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
