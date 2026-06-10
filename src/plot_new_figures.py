"""
Three new paper figures — same style as plot_final_figures.py.

N1 — Non-stationarity: performance drop after distribution shift (E3 data)
N2 — DHR convergence over tasks (convergence.csv)
N3 — SPSA update scaling: interval=1 vs interval=5 (real-LLM Groq runs)

Run:
    python -m src.plot_new_figures [--out viz/paper_new]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── exact same style as plot_final_figures.py ─────────────────────────────────
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
    "aspsa":   "#E63946",
    "spsa":    "#457B9D",
    "kw":      "#2A9D8F",
    "zo_pgd":  "#E9C46A",
    "sp_gt":   "#8338EC",
    "zo_gt":   "#FB8500",
    "pd_2pt":  "#F72585",
}
LABELS = {
    "aspsa":   "A-SPSA",
    "spsa":    "SPSA",
    "kw":      "KW",
    "zo_pgd":  "ZO-PGD",
    "sp_gt":   "SP-GT",
    "zo_gt":   "ZO-GT",
    "pd_2pt":  "PD-2pt",
}
MAIN_VARIANTS = ["aspsa", "spsa", "kw", "zo_pgd", "sp_gt", "zo_gt", "pd_2pt"]


def _save(fig, out: Path, name: str):
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def _bar_panel(ax, variants, values, stds=None, title="", ylabel="", seeds=None):
    colors = [COLORS[v] for v in variants]
    edges  = ["black" if v == "aspsa" else "none" for v in variants]
    lws    = [2.0     if v == "aspsa" else 0       for v in variants]
    x = np.arange(len(variants))
    ax.bar(x, values, color=colors, edgecolor=edges, linewidth=lws, alpha=0.88,
           yerr=stds, capsize=4, error_kw={"elinewidth": 1})
    if seeds is not None:
        rng = np.random.RandomState(0)
        for xi, (v, vals) in enumerate(zip(variants, seeds)):
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(xi + jitter, vals, color="white", edgecolors=COLORS[v],
                       s=28, linewidth=1.1, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


# =============================================================================
# N1 — Non-stationarity: drop after distribution shift
# =============================================================================

def fig_n1_nonstationarity(e3_path: Path, out: Path) -> None:
    df = pd.read_csv(e3_path)
    # sort by drop ascending (lower = more robust)
    order = list(df.sort_values("drop")["variant"])

    drops   = [df.loc[df.variant == v, "drop"].values[0]           for v in order]
    post    = [df.loc[df.variant == v, "post_shift_sr_100"].values[0] for v in order]
    pre     = [df.loc[df.variant == v, "pre_shift_sr"].values[0]    for v in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # left: drop bars
    ax = axes[0]
    colors = [COLORS.get(v, "#888") for v in order]
    edges  = ["black" if v == "aspsa" else "none" for v in order]
    lws    = [2.0     if v == "aspsa" else 0       for v in order]
    x = np.arange(len(order))
    ax.bar(x, drops, color=colors, edgecolor=edges, linewidth=lws, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(v, v) for v in order], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Performance drop (success rate)")
    ax.set_title("Drop after distribution shift\n(lower = more robust)", fontsize=10)
    # annotate A-SPSA
    idx = order.index("aspsa")
    ax.annotate("best", xy=(idx, drops[idx]),
                xytext=(idx + 0.5, drops[idx] + 0.004),
                fontsize=8.5, color=COLORS["aspsa"],
                arrowprops=dict(arrowstyle="->", lw=0.9, color=COLORS["aspsa"]))

    # right: pre vs post comparison for all variants
    ax2 = axes[1]
    width = 0.35
    x2 = np.arange(len(MAIN_VARIANTS))
    sorted_main = sorted(MAIN_VARIANTS,
                         key=lambda v: df.loc[df.variant == v, "drop"].values[0]
                         if v in df.variant.values else 99)
    pre2  = [df.loc[df.variant == v, "pre_shift_sr"].values[0]
             if v in df.variant.values else 0 for v in sorted_main]
    post2 = [df.loc[df.variant == v, "post_shift_sr_100"].values[0]
             if v in df.variant.values else 0 for v in sorted_main]
    colors2 = [COLORS.get(v, "#888") for v in sorted_main]

    bars_pre  = ax2.bar(x2 - width/2, pre2,  width, color=colors2, alpha=0.45,
                        label="Pre-shift", hatch="///",
                        edgecolor=["black" if v == "aspsa" else c
                                   for v, c in zip(sorted_main, colors2)],
                        linewidth=[2.0 if v == "aspsa" else 0.5 for v in sorted_main])
    bars_post = ax2.bar(x2 + width/2, post2, width, color=colors2, alpha=0.88,
                        label="Post-shift (100 tasks)",
                        edgecolor=["black" if v == "aspsa" else c
                                   for v, c in zip(sorted_main, colors2)],
                        linewidth=[2.0 if v == "aspsa" else 0.5 for v in sorted_main])
    ax2.set_xticks(x2)
    ax2.set_xticklabels([LABELS.get(v, v) for v in sorted_main],
                        rotation=25, ha="right", fontsize=8.5)
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Pre-shift vs. post-shift performance\n(higher = better recovery)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.82, 0.94)

    fig.suptitle("N1 — Non-Stationarity: A-SPSA is most robust to distribution shift",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out, "N1_nonstationarity.png")
    _save(fig, out, "N1_nonstationarity.pdf") if False else None
    fig2 = plt.figure()
    plt.close(fig2)


# =============================================================================
# N2 — DHR convergence over tasks
# =============================================================================

def fig_n2_dhr_convergence(conv_path: Path, out: Path, window: int = 40) -> None:
    df = pd.read_csv(conv_path).dropna(subset=["deadline_hit"])
    order_draw = ["sp_gt", "zo_gt", "pd_2pt", "kw", "zo_pgd", "spsa", "aspsa"]

    fig, ax = plt.subplots(figsize=(10, 4.8))

    for v in order_draw:
        sub = df[df.variant == v].sort_values("task_idx")
        if sub.empty:
            continue
        by_seed = sub.groupby("seed")["deadline_hit"].apply(list)
        curves = [pd.Series(vals).rolling(window, min_periods=1).mean().values
                  for vals in by_seed]
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
        ax.fill_between(xs, mean - std, mean + std, color=c, alpha=0.10, zorder=zo - 1)

    ax.set_xlabel("Task index (training rounds)")
    ax.set_ylabel(f"Deadline Hit Rate (rolling mean, w={window})")
    ax.set_title("N2 — Deadline Hit Rate over rounds — A-SPSA converges highest",
                 fontsize=11)
    ax.legend(ncol=4, fontsize=8.5, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    _save(fig, out, "N2_dhr_convergence.png")


# =============================================================================
# N3 — SPSA update frequency scaling (real-LLM Groq experiment)
# =============================================================================

def fig_n3_scaling(
    interval5_path: Path,
    interval1_path: Path,
    out: Path,
) -> None:
    df5 = pd.read_csv(interval5_path)
    df1 = pd.read_csv(interval1_path)

    variants = ["aspsa", "spsa", "kw", "zo_pgd", "zo_gt", "pd_2pt", "sp_gt"]

    ro5 = {row["variant"]: row["routing_objective_mean"] for _, row in df5.iterrows()
           if row["variant"] in variants}
    ro1 = {row["variant"]: row["routing_objective_mean"] for _, row in df1.iterrows()
           if row["variant"] in variants}

    # delta = interval1 - interval5 (positive = improves with more steps)
    deltas = {v: ro1.get(v, 0) - ro5.get(v, 0) for v in variants if v in ro5}
    order  = sorted(deltas, key=lambda v: deltas[v], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # left: absolute values grouped
    ax = axes[0]
    x = np.arange(len(order))
    w = 0.32
    colors = [COLORS.get(v, "#888") for v in order]
    edges  = ["black" if v == "aspsa" else "none" for v in order]
    lws    = [2.0 if v == "aspsa" else 0 for v in order]
    ax.bar(x - w/2, [ro5[v] for v in order], w, color=colors, alpha=0.50,
           edgecolor=edges, linewidth=lws, label="38 SPSA steps (interval=5)", hatch="///")
    ax.bar(x + w/2, [ro1[v] for v in order], w, color=colors, alpha=0.88,
           edgecolor=edges, linewidth=lws, label="186 SPSA steps (interval=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(v, v) for v in order],
                       rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Routing Objective")
    ax.set_title("Routing objective: 38 vs. 186 SPSA steps\n(real LLM, 200 tasks, seed 11)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.72, 0.87)

    # right: delta bars — who benefits from more steps
    ax2 = axes[1]
    delta_vals = [deltas[v] for v in order]
    bar_colors = [COLORS.get(v, "#888") for v in order]
    bar_edges  = ["black" if v == "aspsa" else "none" for v in order]
    bar_lws    = [2.0 if v == "aspsa" else 0 for v in order]
    bars = ax2.bar(x, delta_vals, color=bar_colors, alpha=0.88,
                   edgecolor=bar_edges, linewidth=bar_lws)
    # zero line
    ax2.axhline(0, color="#555", lw=0.9, ls="--")
    # annotate A-SPSA
    idx = order.index("aspsa")
    ax2.annotate("only A-SPSA\nimproves", xy=(idx, delta_vals[idx]),
                 xytext=(idx + 0.6, delta_vals[idx] + 0.003),
                 fontsize=8, color=COLORS["aspsa"],
                 arrowprops=dict(arrowstyle="->", lw=0.9, color=COLORS["aspsa"]))
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS.get(v, v) for v in order],
                        rotation=25, ha="right", fontsize=8.5)
    ax2.set_ylabel("Delta routing objective (interval=1 - interval=5)")
    ax2.set_title("Change with more SPSA steps\n(positive = benefits from more updates)", fontsize=10)

    fig.suptitle("N3 — SPSA Update Frequency: only A-SPSA scales with more gradient steps",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out, "N3_scaling.png")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim",  default="spsa_comparison_v5",
                    help="Simulation data directory")
    ap.add_argument("--conv", default="spsa_comparison/convergence.csv",
                    help="Convergence CSV path")
    ap.add_argument("--i5",   default="real_llm_outputs_groq/summary_real_llm.csv",
                    help="Real-LLM summary (interval=5)")
    ap.add_argument("--i1",   default="real_llm_outputs_groq_interval1/summary_real_llm.csv",
                    help="Real-LLM summary (interval=1)")
    ap.add_argument("--out",  default="viz/paper_new",
                    help="Output directory")
    args = ap.parse_args()

    sim = Path(args.sim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating N1 — Non-stationarity...")
    fig_n1_nonstationarity(sim / "e3_recovery.csv", out)

    print("Generating N2 — DHR convergence...")
    fig_n2_dhr_convergence(Path(args.conv), out)

    print("Generating N3 — SPSA scaling...")
    fig_n3_scaling(Path(args.i5), Path(args.i1), out)

    print("All figures saved to:", out.resolve())


if __name__ == "__main__":
    main()
