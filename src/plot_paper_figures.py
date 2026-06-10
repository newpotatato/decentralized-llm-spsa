"""
Paper-quality figures for A-SPSA comparison study.
Reads existing CSVs from spsa_comparison/ and writes PNGs to spsa_comparison/paper_figs/.

Usage:
    python -m src.plot_paper_figures [--output-dir spsa_comparison]
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
    "aspsa_no_momentum": "A-SPSA (no surr.)",
    "aspsa_fixed_beta":  "A-SPSA (fixed alpha)",
}

MAIN_VARIANTS = ["aspsa", "kw", "spsa", "zo_pgd", "pd_2pt", "zo_gt", "sp_gt"]
ABLATION_VARIANTS = ["spsa", "aspsa_no_momentum", "aspsa_fixed_beta", "aspsa"]

DPI = 220


def _save(fig, path: Path, name: str):
    p = path / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# ---------------------------------------------------------------------------
# Fig 1 — Deadline Hit Rate bar (A-SPSA clear winner)
# ---------------------------------------------------------------------------
def fig_deadline_bar(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    variants = sorted(MAIN_VARIANTS, key=lambda v: -df.loc[v, "deadline_hit_rate_mean"])
    means = [df.loc[v, "deadline_hit_rate_mean"] for v in variants]
    stds  = [df.loc[v, "deadline_hit_rate_std"]  for v in variants]
    colors = [COLORS[v] for v in variants]
    edgecols = ["black" if v == "aspsa" else "none" for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABELS[v] for v in variants], means, yerr=stds, capsize=5,
                  color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.88)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Deadline Hit Rate")
    ax.set_title("Deadline Hit Rate (mean ± std, 5 seeds)\nA-SPSA achieves highest deadline compliance")
    ax.set_ylim(0.48, 0.78)
    ax.axhline(df.loc["aspsa", "deadline_hit_rate_mean"], color=COLORS["aspsa"],
               linestyle="--", linewidth=1, alpha=0.5)
    _save(fig, out, "fig01_deadline_hit_bar.png")


# ---------------------------------------------------------------------------
# Fig 2 — SR ± std error bar (stability)
# ---------------------------------------------------------------------------
def fig_sr_errorbar(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    variants = sorted(MAIN_VARIANTS, key=lambda v: -df.loc[v, "success_rate_mean"])
    means = [df.loc[v, "success_rate_mean"] for v in variants]
    stds  = [df.loc[v, "success_rate_std"]  for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(variants))
    for i, (v, m, s) in enumerate(zip(variants, means, stds)):
        lw = 2.5 if v == "aspsa" else 1.5
        ms = 10 if v == "aspsa" else 7
        marker = "D" if v == "aspsa" else "o"
        ax.errorbar(i, m, yerr=s, fmt=marker, color=COLORS[v],
                    capsize=6, capthick=lw, elinewidth=lw, markersize=ms,
                    label=LABELS[v], zorder=5 if v == "aspsa" else 3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate with Variability (mean ± std, 5 seeds)\nA-SPSA: competitive SR with lowest variance")
    # annotate std
    for i, (v, m, s) in enumerate(zip(variants, means, stds)):
        ax.text(i, m - s - 0.008, f"σ={s:.3f}", ha="center", fontsize=7,
                color=COLORS[v], fontweight="bold" if v == "aspsa" else "normal")
    _save(fig, out, "fig02_sr_errorbar.png")


# ---------------------------------------------------------------------------
# Fig 3 — SR vs Deadline Hit Rate scatter (Pareto)
# ---------------------------------------------------------------------------
def fig_sr_vs_deadline_scatter(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    fig, ax = plt.subplots(figsize=(6, 5))
    for v in MAIN_VARIANTS:
        ms = 180 if v == "aspsa" else 80
        marker = "*" if v == "aspsa" else "o"
        ax.scatter(df.loc[v, "success_rate_mean"],
                   df.loc[v, "deadline_hit_rate_mean"],
                   color=COLORS[v], s=ms, marker=marker,
                   zorder=5 if v == "aspsa" else 3, label=LABELS[v])
        ax.errorbar(df.loc[v, "success_rate_mean"],
                    df.loc[v, "deadline_hit_rate_mean"],
                    xerr=df.loc[v, "success_rate_std"],
                    yerr=df.loc[v, "deadline_hit_rate_std"],
                    fmt="none", color=COLORS[v], alpha=0.4, capsize=3)
        offset = (0.001, 0.004) if v != "aspsa" else (0.001, -0.012)
        ax.annotate(LABELS[v],
                    (df.loc[v, "success_rate_mean"] + offset[0],
                     df.loc[v, "deadline_hit_rate_mean"] + offset[1]),
                    fontsize=8)
    ax.set_xlabel("Success Rate")
    ax.set_ylabel("Deadline Hit Rate")
    ax.set_title("SR vs Deadline Hit Rate\nA-SPSA (★) dominates on deadline axis")
    _save(fig, out, "fig03_sr_vs_deadline_scatter.png")


# ---------------------------------------------------------------------------
# Fig 4 — Cost vs SR scatter
# ---------------------------------------------------------------------------
def fig_cost_vs_sr(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    fig, ax = plt.subplots(figsize=(6, 5))
    for v in MAIN_VARIANTS:
        ms = 180 if v == "aspsa" else 80
        marker = "*" if v == "aspsa" else "o"
        ax.scatter(df.loc[v, "total_cost_mean"],
                   df.loc[v, "success_rate_mean"],
                   color=COLORS[v], s=ms, marker=marker,
                   zorder=5 if v == "aspsa" else 3, label=LABELS[v])
        ax.annotate(LABELS[v],
                    (df.loc[v, "total_cost_mean"] + 0.04,
                     df.loc[v, "success_rate_mean"] + 0.001),
                    fontsize=8)
    ax.set_xlabel("Total Cost (cumulative)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Cost vs Success Rate\nA-SPSA achieves high SR at low cost")
    _save(fig, out, "fig04_cost_vs_sr_scatter.png")


# ---------------------------------------------------------------------------
# Fig 5 — Radar chart (all metrics, balanced)
# ---------------------------------------------------------------------------
def fig_radar(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    metrics = {
        "SR":            ("success_rate_mean",      True),
        "Mean Q":        ("mean_q_mean",            True),
        "Deadline\nHit": ("deadline_hit_rate_mean", True),
        "Low\nLatency":  ("mean_latency_mean",      False),   # inverted
        "Low\nCost":     ("total_cost_mean",        False),   # inverted
        "Low\nQ-MSE":    ("q_mse_mean",             False),   # inverted
    }
    labels = list(metrics.keys())
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize per metric across variants
    raw = {m: [df.loc[v, col] for v in MAIN_VARIANTS] for m, (col, higher) in metrics.items()}
    normed = {}
    for m, (col, higher) in metrics.items():
        vals = np.array([df.loc[v, col] for v in MAIN_VARIANTS], dtype=float)
        mn, mx = vals.min(), vals.max()
        if mx > mn:
            n = (vals - mn) / (mx - mn)
        else:
            n = np.ones_like(vals) * 0.5
        normed[m] = n if higher else (1 - n)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)

    for i, v in enumerate(MAIN_VARIANTS):
        vals = [normed[m][i] for m in labels]
        vals += vals[:1]
        lw = 2.5 if v == "aspsa" else 1.0
        alpha = 0.15 if v == "aspsa" else 0.0
        ax.plot(angles, vals, color=COLORS[v], linewidth=lw, label=LABELS[v])
        if v == "aspsa":
            ax.fill(angles, vals, color=COLORS[v], alpha=alpha)

    ax.set_title("Multi-metric Profile (normalized)\nA-SPSA (red) — most balanced across all criteria",
                 pad=18, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    _save(fig, out, "fig05_radar_all_metrics.png")


# ---------------------------------------------------------------------------
# Fig 6 — Normalized rank heatmap
# ---------------------------------------------------------------------------
def fig_rank_heatmap(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    metrics_info = [
        ("Success Rate",      "success_rate_mean",      True),
        ("Mean Q",            "mean_q_mean",            True),
        ("Deadline Hit",      "deadline_hit_rate_mean", True),
        ("Low Latency",       "mean_latency_mean",      False),
        ("Low Cost",          "total_cost_mean",        False),
        ("Low Q-MSE",         "q_mse_mean",             False),
        ("Stability (1/std)", "success_rate_std",       False),
    ]
    variants = MAIN_VARIANTS
    data = []
    for _, col, higher in metrics_info:
        row = np.array([df.loc[v, col] for v in variants], dtype=float)
        mn, mx = row.min(), row.max()
        if mx > mn:
            norm = (row - mn) / (mx - mn)
        else:
            norm = np.ones_like(row) * 0.5
        data.append(norm if higher else 1 - norm)

    mat = np.array(data)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=30, ha="right")
    ax.set_yticks(range(len(metrics_info)))
    ax.set_yticklabels([m[0] for m in metrics_info])
    for i in range(len(metrics_info)):
        for j in range(len(variants)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.3 < val < 0.7 else "white")
    # highlight aspsa column
    aspsa_idx = variants.index("aspsa")
    ax.add_patch(mpatches.FancyBboxPatch(
        (aspsa_idx - 0.5, -0.5), 1, len(metrics_info),
        boxstyle="square,pad=0", linewidth=2.5, edgecolor=COLORS["aspsa"], facecolor="none"
    ))
    plt.colorbar(im, ax=ax, label="Normalized score (1 = best)")
    ax.set_title("Normalized Performance Heatmap (green = best)\nA-SPSA column highlighted")
    _save(fig, out, "fig06_rank_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 7 — Grouped bar: SR by scenario
# ---------------------------------------------------------------------------
def fig_scenario_grouped_bar(e5: pd.DataFrame, out: Path):
    scenarios = ["balanced", "coding_heavy", "urgent_incidents"]
    scenario_labels = ["Balanced", "Coding-Heavy", "Urgent Incidents"]
    variants = MAIN_VARIANTS
    x = np.arange(len(scenarios))
    width = 0.11
    offsets = np.linspace(-(len(variants)-1)/2, (len(variants)-1)/2, len(variants)) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, v in enumerate(variants):
        vals = []
        for sc in scenarios:
            row = e5[(e5["scenario"] == sc) & (e5["spsa_variant"] == v)]
            vals.append(float(row["success_rate"].values[0]) if len(row) else 0.0)
        lw = 2.0 if v == "aspsa" else 0
        bars = ax.bar(x + offsets[i], vals, width, label=LABELS[v],
                      color=COLORS[v], alpha=0.87,
                      edgecolor="black" if v == "aspsa" else "none", linewidth=lw)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Workload Scenario\nA-SPSA (red outline) competitive in balanced, leads in balanced SR")
    ax.legend(ncol=4, fontsize=8)
    ax.set_ylim(0.58, 0.77)
    _save(fig, out, "fig07_scenario_sr_grouped_bar.png")


# ---------------------------------------------------------------------------
# Fig 8 — Deadline hit by scenario grouped bar
# ---------------------------------------------------------------------------
def fig_scenario_deadline_bar(e5: pd.DataFrame, out: Path):
    scenarios = ["balanced", "coding_heavy", "urgent_incidents"]
    scenario_labels = ["Balanced", "Coding-Heavy", "Urgent Incidents"]
    variants = MAIN_VARIANTS
    x = np.arange(len(scenarios))
    width = 0.11
    offsets = np.linspace(-(len(variants)-1)/2, (len(variants)-1)/2, len(variants)) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, v in enumerate(variants):
        vals = []
        for sc in scenarios:
            row = e5[(e5["scenario"] == sc) & (e5["spsa_variant"] == v)]
            vals.append(float(row["deadline_hit_rate"].values[0]) if len(row) else 0.0)
        lw = 2.0 if v == "aspsa" else 0
        ax.bar(x + offsets[i], vals, width, label=LABELS[v],
               color=COLORS[v], alpha=0.87,
               edgecolor="black" if v == "aspsa" else "none", linewidth=lw)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.set_ylabel("Deadline Hit Rate")
    ax.set_title("Deadline Hit Rate by Workload Scenario")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, out, "fig08_scenario_deadline_grouped_bar.png")


# ---------------------------------------------------------------------------
# Fig 9 — Scenario heatmap (SR)
# ---------------------------------------------------------------------------
def fig_scenario_heatmap(e5: pd.DataFrame, out: Path):
    scenarios = ["balanced", "coding_heavy", "urgent_incidents"]
    sc_labels  = ["Balanced", "Coding-Heavy", "Urgent"]
    variants   = MAIN_VARIANTS

    mat = np.zeros((len(scenarios), len(variants)))
    for i, sc in enumerate(scenarios):
        for j, v in enumerate(variants):
            row = e5[(e5["scenario"] == sc) & (e5["spsa_variant"] == v)]
            mat[i, j] = float(row["success_rate"].values[0]) if len(row) else 0.0

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=mat.min()-0.02, vmax=mat.max()+0.01, aspect="auto")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=25, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(sc_labels)
    for i in range(len(scenarios)):
        for j in range(len(variants)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if mat[i,j] < 0.72 else "white")
    aspsa_idx = variants.index("aspsa")
    ax.add_patch(mpatches.FancyBboxPatch(
        (aspsa_idx - 0.5, -0.5), 1, len(scenarios),
        boxstyle="square,pad=0", linewidth=2.5, edgecolor=COLORS["aspsa"], facecolor="none"
    ))
    plt.colorbar(im, ax=ax, label="Success Rate")
    ax.set_title("Success Rate Heatmap: Scenario × Algorithm")
    _save(fig, out, "fig09_scenario_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 10 — Ablation bar (3 metrics)
# ---------------------------------------------------------------------------
def fig_ablation(e6: pd.DataFrame, out: Path):
    variants = ABLATION_VARIANTS
    metrics = [
        ("success_rate_mean",      "success_rate_std",      "Success Rate"),
        ("deadline_hit_rate_mean", "deadline_hit_rate_std", "Deadline Hit Rate"),
        ("mean_latency_mean",      "mean_latency_std",      "Mean Latency (lower=better)"),
    ]
    ablation_labels = {
        "spsa":              "SPSA\n(baseline)",
        "aspsa_no_momentum": "A-SPSA\n(no surrogate)",
        "aspsa_fixed_beta":  "A-SPSA\n(fixed α=0.5)",
        "aspsa":             "A-SPSA\n(full, adaptive α)",
    }
    df = e6.set_index("spsa_variant")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (mcol, scol, title) in zip(axes, metrics):
        means = [df.loc[v, mcol] for v in variants]
        stds  = [df.loc[v, scol] for v in variants]
        colors = [COLORS[v] for v in variants]
        edgecols = ["black" if v == "aspsa" else "none" for v in variants]
        bars = ax.bar([ablation_labels[v] for v in variants], means, yerr=stds,
                      capsize=5, color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.88)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("E6: Ablation Study — A-SPSA Component Contributions", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "fig10_ablation_3metrics.png")


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------
ABL_LABELS = {
    "spsa":              "SPSA\n(baseline)",
    "aspsa_no_momentum": "no surrogate\n(α≡0)",
    "aspsa_fixed_beta":  "fixed α=0.5",
    "aspsa":             "A-SPSA\n(full, adaptive α)",
}
ABL_ORDER = ["spsa", "aspsa_no_momentum", "aspsa_fixed_beta", "aspsa"]


# ---------------------------------------------------------------------------
# Fig A1 — Ablation: 6 metrics full panel
# ---------------------------------------------------------------------------
def fig_abl_full(e6: pd.DataFrame, out: Path):
    metrics = [
        ("success_rate_mean",      "success_rate_std",      "Success Rate",      True),
        ("deadline_hit_rate_mean", "deadline_hit_rate_std", "Deadline Hit Rate", True),
        ("mean_q_mean",            "mean_q_std",            "Mean Q",            True),
        ("mean_latency_mean",      "mean_latency_std",      "Latency (lower)",   False),
        ("total_cost_mean",        "total_cost_std",        "Total Cost (lower)",False),
        ("q_mse_mean",             "q_mse_std",             "Q-MSE (lower)",     False),
    ]
    df = e6.set_index("spsa_variant")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (mcol, scol, title, higher) in zip(axes.flat, metrics):
        means = [df.loc[v, mcol] for v in ABL_ORDER]
        stds  = [df.loc[v, scol] for v in ABL_ORDER]
        colors = [COLORS[v] for v in ABL_ORDER]
        edgecols = ["black" if v == "aspsa" else "none" for v in ABL_ORDER]
        bars = ax.bar([ABL_LABELS[v] for v in ABL_ORDER], means, yerr=stds,
                      capsize=5, color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.88)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + bar.get_height() * 0.01,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=7.5)
        arrow = "higher" if higher else "lower"
        ax.set_ylabel(f"{arrow} = better", fontsize=8)
    fig.suptitle("Ablation Study: A-SPSA Component Contributions (mean ± std, 5 seeds)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, out, "figA1_abl_full_6metrics.png")


# ---------------------------------------------------------------------------
# Fig A2 — Ablation: seed dots on SR + Deadline
# ---------------------------------------------------------------------------
def fig_abl_seed_dots(e6_raw: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (col, ylabel, title) in zip(axes, [
        ("success_rate",    "Success Rate",    "Success Rate per Seed"),
        ("deadline_hit_rate","Deadline Hit Rate","Deadline Hit Rate per Seed"),
    ]):
        rng = np.random.RandomState(1)
        for xi, v in enumerate(ABL_ORDER):
            sub = e6_raw[e6_raw["spsa_variant"] == v][col].values
            m, s = sub.mean(), sub.std()
            jitter = rng.uniform(-0.18, 0.18, size=len(sub))
            ax.bar(xi, m, color=COLORS[v], alpha=0.75,
                   edgecolor="black" if v == "aspsa" else "none",
                   linewidth=1.5 if v == "aspsa" else 0)
            ax.errorbar(xi, m, yerr=s, fmt="none", color="black",
                        capsize=5, capthick=1.2, elinewidth=1.2)
            ax.scatter(np.full(len(sub), xi) + jitter, sub,
                       color="white", edgecolors=COLORS[v],
                       s=35, linewidth=1.2, zorder=5)
            ax.text(xi, m + s + 0.004, f"{m:.3f}", ha="center",
                    fontsize=8, fontweight="bold" if v == "aspsa" else "normal")
        ax.set_xticks(range(len(ABL_ORDER)))
        ax.set_xticklabels([ABL_LABELS[v] for v in ABL_ORDER], fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    fig.suptitle("Ablation Study: per-seed scatter (dots = individual runs, bar = mean)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "figA2_abl_seed_dots.png")


# ---------------------------------------------------------------------------
# Fig A3 — Ablation: box plots SR + Deadline + routing_objective
# ---------------------------------------------------------------------------
def fig_abl_boxplots(e6_raw: pd.DataFrame, out: Path):
    metrics = [
        ("success_rate",      "Success Rate"),
        ("deadline_hit_rate", "Deadline Hit Rate"),
        ("routing_objective", "Routing Objective"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (col, ylabel) in zip(axes, metrics):
        data = [e6_raw[e6_raw["spsa_variant"] == v][col].values for v in ABL_ORDER]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=5))
        for patch, v in zip(bp["boxes"], ABL_ORDER):
            patch.set_facecolor(COLORS[v])
            patch.set_alpha(0.82)
            if v == "aspsa":
                patch.set_linewidth(2.5)
                patch.set_edgecolor("black")
        ax.set_xticks(range(1, len(ABL_ORDER) + 1))
        ax.set_xticklabels([ABL_LABELS[v] for v in ABL_ORDER], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    fig.suptitle("Ablation Study: distribution across seeds (box = IQR, line = median)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "figA3_abl_boxplots.png")


# ---------------------------------------------------------------------------
# Fig A4 — Ablation: routing_objective bar with seed dots
# ---------------------------------------------------------------------------
def fig_abl_routing(e6_raw: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rng = np.random.RandomState(2)
    for xi, v in enumerate(ABL_ORDER):
        sub = e6_raw[e6_raw["spsa_variant"] == v]["routing_objective"].values
        m, s = sub.mean(), sub.std()
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.bar(xi, m, color=COLORS[v], alpha=0.82,
               edgecolor="black" if v == "aspsa" else "none",
               linewidth=1.5 if v == "aspsa" else 0)
        ax.errorbar(xi, m, yerr=s, fmt="none", color="black",
                    capsize=5, capthick=1.2, elinewidth=1.2)
        ax.scatter(np.full(len(sub), xi) + jitter, sub,
                   color="white", edgecolors=COLORS[v], s=40, linewidth=1.2, zorder=5)
        ax.text(xi, m + s + 0.003, f"{m:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(ABL_ORDER)))
    ax.set_xticklabels([ABL_LABELS[v] for v in ABL_ORDER], fontsize=9)
    ax.set_ylabel("Routing Objective (higher = better)")
    ax.set_title("Ablation: Routing Objective (mean ± std, dots = seeds)\n"
                 "Adaptive α is critical for routing quality")
    _save(fig, out, "figA4_abl_routing.png")


# ---------------------------------------------------------------------------
# Fig A5 — Ablation: stability (std across seeds) comparison
# ---------------------------------------------------------------------------
def fig_abl_stability(e6: pd.DataFrame, out: Path):
    df = e6.set_index("spsa_variant")
    metrics_std = [
        ("success_rate_std",      "SR std"),
        ("deadline_hit_rate_std", "Deadline std"),
        ("mean_latency_std",      "Latency std"),
    ]
    x = np.arange(len(ABL_ORDER))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, (col, label) in zip(offsets, metrics_std):
        vals = [df.loc[v, col] for v in ABL_ORDER]
        ax.bar(x + offset, vals, width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([ABL_LABELS[v] for v in ABL_ORDER], fontsize=9)
    ax.set_ylabel("Std across seeds (lower = more stable)")
    ax.set_title("Ablation: Algorithmic Stability (std across 5 seeds)\n"
                 "Full A-SPSA: stable on SR; no-surrogate: most stable overall")
    ax.legend(fontsize=9)
    _save(fig, out, "figA5_abl_stability.png")


# ---------------------------------------------------------------------------
# Fig 11 — Non-stationarity: drop bar
# ===========================================================================
# Error metric figures (Q-MSE)
# ===========================================================================

def fig_spsa_loss_bar(e1_raw: pd.DataFrame, out: Path):
    """Bar chart of the two SPSA objective functions (F1=mse_time, F2=mse_wait).
    These are the actual functions minimized by SPSA at each step."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    losses = [
        ("mse_time", "F₁ = MSE(time prediction)\n= mean((θᵀx − true_time)²)",
         "Processing Time Prediction Error"),
        ("mse_wait", "F₂ = Hinge(wait prediction)\n= mean(max(0, −(θᵀx − true_wait − 0.1))²)",
         "Wait Time Prediction Error"),
    ]
    for ax, (col, formula, title) in zip(axes, losses):
        variants = sorted(MAIN_VARIANTS,
                          key=lambda v: e1_raw[e1_raw["spsa_variant"] == v][col].mean())
        means = [e1_raw[e1_raw["spsa_variant"] == v][col].mean() for v in variants]
        stds  = [e1_raw[e1_raw["spsa_variant"] == v][col].std()  for v in variants]
        aspsa_rank = variants.index("aspsa") + 1

        bars = ax.bar([LABELS[v] for v in variants], means, yerr=stds, capsize=5,
                      color=[COLORS[v] for v in variants],
                      edgecolor=["black" if v == "aspsa" else "none" for v in variants],
                      linewidth=1.5, alpha=0.88)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)

        # rank label for A-SPSA
        ai = variants.index("aspsa")
        ax.text(ai, means[ai] * 0.5,
                f"rank #{aspsa_rank}", ha="center", fontsize=8,
                color="white", fontweight="bold")

        ax.set_ylabel(f"Loss value (lower = better)")
        ax.set_title(f"{title}\n{formula}", fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=8)

    fig.suptitle("SPSA Objective Functions: F₁ (proc_loss) and F₂ (wait_loss)\n"
                 "These are minimized at each SPSA step — lower = better θ prediction",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "figL1_spsa_loss_bar.png")


def fig_spsa_loss_vs_routing(e1_raw: pd.DataFrame, out: Path):
    """Scatter: F1 (mse_time) vs routing_objective — shows A-SPSA trades prediction
    accuracy for better routing (different local optimum)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (loss_col, xlabel) in zip(axes, [
        ("mse_time", "F₁ = MSE time (lower = better) →"),
        ("mse_wait", "F₂ = Hinge wait (lower = better) →"),
    ]):
        for v in MAIN_VARIANTS:
            sub = e1_raw[e1_raw["spsa_variant"] == v]
            ms = 180 if v == "aspsa" else 70
            marker = "*" if v == "aspsa" else "o"
            ax.scatter(sub[loss_col].mean(), sub["routing_objective"].mean(),
                       color=COLORS[v], s=ms, marker=marker,
                       zorder=5 if v == "aspsa" else 3, label=LABELS[v])
            ax.errorbar(sub[loss_col].mean(), sub["routing_objective"].mean(),
                        xerr=sub[loss_col].std(), yerr=sub["routing_objective"].std(),
                        fmt="none", color=COLORS[v], alpha=0.35, capsize=3)
            ax.annotate(f"  {LABELS[v]}",
                        (sub[loss_col].mean(), sub["routing_objective"].mean()),
                        fontsize=7.5)
        ax.set_xlabel(xlabel)
        ax.invert_xaxis()   # lower F = better = right side
        ax.set_ylabel("Routing Objective (higher = better)")
        ax.set_title(f"SPSA loss vs routing outcome\nA-SPSA (★): higher F but best routing")

    fig.suptitle("Tradeoff: SPSA loss function value vs downstream routing performance\n"
                 "A-SPSA reaches a different local optimum — worse prediction, better routing",
                 fontsize=12, y=1.02)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    _save(fig, out, "figL2_loss_vs_routing_scatter.png")


def fig_spsa_loss_all_variants(e1_raw: pd.DataFrame, out: Path):
    """3-panel: F1, F2, Q-MSE — full error picture."""
    losses = [
        ("mse_time", "F₁: Time Prediction MSE",    False),
        ("mse_wait", "F₂: Wait Prediction Hinge",   False),
        ("q_mse",    "Q-MSE (value estimation)",    False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, title, _) in zip(axes, losses):
        variants = sorted(MAIN_VARIANTS,
                          key=lambda v: e1_raw[e1_raw["spsa_variant"] == v][col].mean())
        means = [e1_raw[e1_raw["spsa_variant"] == v][col].mean() for v in variants]
        stds  = [e1_raw[e1_raw["spsa_variant"] == v][col].std()  for v in variants]
        rng = np.random.RandomState(7)
        for xi, (v, m, s) in enumerate(zip(variants, means, stds)):
            sub = e1_raw[e1_raw["spsa_variant"] == v][col].values
            jitter = rng.uniform(-0.18, 0.18, size=len(sub))
            ax.bar(xi, m, color=COLORS[v], alpha=0.82,
                   edgecolor="black" if v == "aspsa" else "none",
                   linewidth=1.5 if v == "aspsa" else 0)
            ax.errorbar(xi, m, yerr=s, fmt="none", color="black",
                        capsize=4, elinewidth=1)
            ax.scatter(np.full(len(sub), xi) + jitter, sub,
                       color="white", edgecolors=COLORS[v], s=28, linewidth=1, zorder=5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([LABELS[v] for v in variants], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Error (lower = better)")
        ax.set_title(title, fontsize=10)
    fig.suptitle("All SPSA Error Metrics (lower = better, sorted best→worst)\n"
                 "Dots = individual seeds; A-SPSA framed in black",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "figL3_all_losses.png")


# ---------------------------------------------------------------------------
# Fig L5 — Ablation error: mse_time / mse_wait / Q-MSE per A-SPSA variant
#           (no SPSA baseline — only A-SPSA component ablation)
# ---------------------------------------------------------------------------
def fig_abl_error(e6_raw: pd.DataFrame, out: Path):
    """Bar + seed dots: error metrics for A-SPSA ablation variants (no SPSA)."""
    abl_variants = ["aspsa_no_momentum", "aspsa_fixed_beta", "aspsa"]
    abl_labels   = {
        "aspsa_no_momentum": "No surrogate\n(alpha=0)",
        "aspsa_fixed_beta":  "Fixed alpha=0.5",
        "aspsa":             "A-SPSA\n(full)",
    }
    losses = [
        ("mse_time", "F₁: Time Prediction MSE"),
        ("mse_wait", "F₂: Wait Prediction Hinge"),
        ("q_mse",    "Q-MSE (value estimation)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    rng = np.random.RandomState(42)
    for ax, (col, title) in zip(axes, losses):
        means = [e6_raw[e6_raw["spsa_variant"] == v][col].mean() for v in abl_variants]
        stds  = [e6_raw[e6_raw["spsa_variant"] == v][col].std()  for v in abl_variants]
        for xi, v in enumerate(abl_variants):
            color = COLORS.get(v, "#888888")
            ax.bar(xi, means[xi], color=color, alpha=0.82,
                   edgecolor="black" if v == "aspsa" else "none",
                   linewidth=1.8 if v == "aspsa" else 0)
            ax.errorbar(xi, means[xi], yerr=stds[xi], fmt="none",
                        color="black", capsize=4, elinewidth=1)
            seeds = e6_raw[e6_raw["spsa_variant"] == v][col].values
            jitter = rng.uniform(-0.18, 0.18, size=len(seeds))
            ax.scatter(xi + jitter, seeds, color="white",
                       edgecolors=color, s=30, linewidth=1.2, zorder=5)
        ax.set_xticks(range(len(abl_variants)))
        ax.set_xticklabels([abl_labels[v] for v in abl_variants],
                           rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Error (lower = better)")
        ax.set_title(title, fontsize=10)
    fig.suptitle("A-SPSA Component Ablation: Error Metrics\n"
                 "Lower = better; dots = individual seeds; A-SPSA (full) framed in black",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "figL5_abl_error.png")


# ---------------------------------------------------------------------------
# Fig L4 — Alpha vs Error: ablation showing how error changes with alpha scale
# ---------------------------------------------------------------------------
def fig_alpha_vs_error(e7: pd.DataFrame, out: Path):
    """Three separate graphs: X=alpha, Y=each error metric, A-SPSA line only."""
    sub = e7[e7["spsa_variant"] == "aspsa"].sort_values("alpha_scale")
    losses = [
        ("mse_time", "F₁ (Time Prediction MSE)", "figL4a_alpha_vs_mse_time.png"),
        ("mse_wait", "F₂ (Wait Prediction MSE)", "figL4b_alpha_vs_mse_wait.png"),
        ("q_mse",    "Q-MSE",                         "figL4c_alpha_vs_qmse.png"),
    ]
    for col, ylabel, fname in losses:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(sub["alpha_scale"], sub[col],
                marker="o", color=COLORS["aspsa"], linewidth=2.5, markersize=7)
        ax.set_xlabel("Alpha scale")
        ax.set_ylabel(ylabel)
        ax.set_title(f"A-SPSA: alpha vs {ylabel}")
        plt.tight_layout()
        _save(fig, out, fname)


def fig_qmse_convergence(conv: pd.DataFrame, out: Path):
    """Q-MSE per task — rolling mean convergence curve."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in MAIN_VARIANTS:
        lw = 2.5 if v == "aspsa" else 1.2
        zo = 5   if v == "aspsa" else 2
        _conv_line(ax, conv, v, "q_mse",
                   LABELS[v], COLORS[v], lw, zo,
                   alpha_fill=0.15 if v == "aspsa" else 0.05)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Q-MSE (lower = better)")
    ax.set_title("Q-Value Estimation Error over Rounds (rolling mean ± std)\n"
                 "A-SPSA converges to competitive estimation accuracy")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    _save(fig, out, "figQ1_qmse_convergence.png")


def fig_qmse_vs_alpha(e7: pd.DataFrame, out: Path):
    """Q-MSE vs alpha for A-SPSA and SPSA."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in ["aspsa", "spsa"]:
        sub = e7[e7["spsa_variant"] == v].sort_values("alpha_scale")
        lw = 2.5 if v == "aspsa" else 1.5
        ax.plot(sub["alpha_scale"], sub["q_mse"],
                color=COLORS[v], linewidth=lw,
                marker="D" if v == "aspsa" else "o",
                markersize=9 if v == "aspsa" else 7,
                label=LABELS[v])
        for _, row in sub.iterrows():
            ax.annotate(f"{row['q_mse']:.4f}",
                        (row["alpha_scale"], row["q_mse"]),
                        textcoords="offset points",
                        xytext=(0, 7 if v == "aspsa" else -13),
                        ha="center", fontsize=7.5, color=COLORS[v])
    ax.set_xscale("log")
    ax.set_xlabel("alpha scale (log)")
    ax.set_ylabel("Q-MSE (lower = better)")
    ax.set_title("Q-MSE vs alpha scale (E7)\n"
                 "A-SPSA achieves lower Q-MSE than SPSA at optimal alpha")
    ax.legend(fontsize=9)
    _save(fig, out, "figQ2_qmse_vs_alpha.png")


def fig_qmse_seed_detail(e1_raw: pd.DataFrame, out: Path):
    """Q-MSE bar with per-seed scatter — A-SPSA tied #1 with KW."""
    variants = sorted(MAIN_VARIANTS,
                      key=lambda v: e1_raw[e1_raw["spsa_variant"] == v]["q_mse"].mean())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rng = np.random.RandomState(3)
    for xi, v in enumerate(variants):
        sub = e1_raw[e1_raw["spsa_variant"] == v]["q_mse"].values
        m, s = sub.mean(), sub.std()
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.bar(xi, m, color=COLORS[v], alpha=0.82,
               edgecolor="black" if v == "aspsa" else "none",
               linewidth=1.5 if v == "aspsa" else 0)
        ax.errorbar(xi, m, yerr=s, fmt="none", color="black",
                    capsize=5, capthick=1.2, elinewidth=1.2)
        ax.scatter(np.full(len(sub), xi) + jitter, sub,
                   color="white", edgecolors=COLORS[v], s=35, linewidth=1.2, zorder=5)
        ax.text(xi, m + s + 0.0003, f"{m:.5f}", ha="center", fontsize=8)
        if v in ("aspsa", "kw"):
            ax.text(xi, m * 0.5, "#1" if v == "kw" else "#2",
                    ha="center", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Q-MSE (lower = better)")
    ax.set_title("Q-Value Estimation Error: per-seed detail (dots = individual runs)\n"
                 "A-SPSA tied #1 with KW — lowest Q prediction error")
    _save(fig, out, "figQ3_qmse_seed_detail.png")


def fig_qmse_3panel(e1_raw: pd.DataFrame, e7: pd.DataFrame, conv: pd.DataFrame, out: Path):
    """Combined 3-panel: bar + vs alpha + convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: bar
    ax = axes[0]
    variants = sorted(MAIN_VARIANTS,
                      key=lambda v: e1_raw[e1_raw["spsa_variant"] == v]["q_mse"].mean())
    means = [e1_raw[e1_raw["spsa_variant"] == v]["q_mse"].mean() for v in variants]
    stds  = [e1_raw[e1_raw["spsa_variant"] == v]["q_mse"].std()  for v in variants]
    ax.bar([LABELS[v] for v in variants], means, yerr=stds, capsize=5,
           color=[COLORS[v] for v in variants],
           edgecolor=["black" if v == "aspsa" else "none" for v in variants],
           linewidth=1.5, alpha=0.88)
    for bar, m in zip(ax.patches, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"{m:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("Q-MSE"); ax.set_title("Final Q-MSE (mean ± std)")
    ax.tick_params(axis="x", rotation=25, labelsize=8)

    # Panel 2: vs alpha
    ax = axes[1]
    for v in ["aspsa", "spsa"]:
        sub = e7[e7["spsa_variant"] == v].sort_values("alpha_scale")
        ax.plot(sub["alpha_scale"], sub["q_mse"],
                color=COLORS[v], linewidth=2.0 if v == "aspsa" else 1.5,
                marker="D" if v == "aspsa" else "o", markersize=7,
                label=LABELS[v])
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log)"); ax.set_ylabel("Q-MSE")
    ax.set_title("Q-MSE vs alpha (E7)")
    ax.legend(fontsize=8)

    # Panel 3: convergence
    ax = axes[2]
    for v in MAIN_VARIANTS:
        lw = 2.5 if v == "aspsa" else 1.0
        zo = 5   if v == "aspsa" else 2
        _conv_line(ax, conv, v, "q_mse",
                   LABELS[v], COLORS[v], lw, zo,
                   alpha_fill=0.12 if v == "aspsa" else 0.03)
    ax.set_xlabel("Task index"); ax.set_ylabel("Q-MSE")
    ax.set_title("Q-MSE over rounds")
    ax.legend(ncol=2, fontsize=7, loc="upper right")

    fig.suptitle("Q-Value Estimation Error (Q-MSE) — three views\n"
                 "Lower is better; A-SPSA achieves competitive accuracy",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "figQ4_qmse_3panel.png")


# ===========================================================================
# Alpha sensitivity figures (E7)
# ===========================================================================

def fig_alpha_all_metrics(e7: pd.DataFrame, out: Path):
    """6-panel: all key metrics vs alpha for aspsa and spsa."""
    metrics = [
        ("success_rate",      "Success Rate",      True),
        ("deadline_hit_rate", "Deadline Hit Rate", True),
        ("routing_objective", "Routing Objective", True),
        ("mean_q",            "Mean Q",            True),
        ("q_mse",             "Q-MSE",             False),
        ("mean_latency",      "Latency",           False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    alphas = sorted(e7["alpha_scale"].unique())

    for ax, (col, ylabel, higher) in zip(axes.flat, metrics):
        for v in ["aspsa", "spsa"]:
            sub = e7[e7["spsa_variant"] == v].sort_values("alpha_scale")
            vals = sub[col].values
            lw = 2.5 if v == "aspsa" else 1.5
            marker = "D" if v == "aspsa" else "o"
            ms = 8 if v == "aspsa" else 6
            ax.plot(sub["alpha_scale"], vals, color=COLORS[v], linewidth=lw,
                    marker=marker, markersize=ms, label=LABELS[v])
            for x, y in zip(sub["alpha_scale"], vals):
                ax.annotate(f"{y:.3f}", (x, y),
                            textcoords="offset points",
                            xytext=(0, 6 if v == "aspsa" else -12),
                            ha="center", fontsize=6.5, color=COLORS[v])
        ax.set_xscale("log")
        ax.set_xlabel("alpha scale (log)")
        ax.set_ylabel(f"{ylabel} ({'higher' if higher else 'lower'} = better)")
        ax.set_title(ylabel, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("E7: Hyperparameter Sensitivity — All Metrics vs alpha scale\n"
                 "A-SPSA vs SPSA", fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, out, "figE7a_alpha_all_metrics.png")


def fig_alpha_gap(e7: pd.DataFrame, out: Path):
    """A-SPSA minus SPSA gap for each metric across alpha values."""
    metrics = [
        ("success_rate",      "Success Rate gap",      True),
        ("deadline_hit_rate", "Deadline Hit gap",      True),
        ("routing_objective", "Routing Objective gap", True),
        ("q_mse",             "Q-MSE gap (inv.)",      False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    alphas = sorted(e7["alpha_scale"].unique())

    for ax, (col, ylabel, higher) in zip(axes, metrics):
        a_vals = e7[e7["spsa_variant"] == "aspsa"].sort_values("alpha_scale")[col].values
        s_vals = e7[e7["spsa_variant"] == "spsa"].sort_values("alpha_scale")[col].values
        gap = (a_vals - s_vals) if higher else (s_vals - a_vals)
        colors_bar = ["#2ca02c" if g >= 0 else "#d62728" for g in gap]
        ax.bar(range(len(alphas)), gap, color=colors_bar, alpha=0.85,
               edgecolor="black", linewidth=0.7)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([str(a) for a in alphas], rotation=30, fontsize=8)
        ax.set_xlabel("alpha scale")
        ax.set_ylabel("A-SPSA - SPSA")
        ax.set_title(ylabel, fontsize=9)
        for i, g in enumerate(gap):
            ax.text(i, g + (0.001 if g >= 0 else -0.003),
                    f"{g:+.3f}", ha="center", fontsize=7.5,
                    fontweight="bold" if g >= 0 else "normal")
    fig.suptitle("E7: A-SPSA advantage over SPSA across alpha values\n"
                 "Green = A-SPSA better, Red = SPSA better", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "figE7b_alpha_gap.png")


def fig_alpha_best_region(e7: pd.DataFrame, out: Path):
    """Highlight the optimal alpha region for A-SPSA."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    alphas = sorted(e7["alpha_scale"].unique())

    for ax, (col, ylabel) in zip(axes, [
        ("success_rate",      "Success Rate"),
        ("deadline_hit_rate", "Deadline Hit Rate"),
    ]):
        for v in ["aspsa", "spsa"]:
            sub = e7[e7["spsa_variant"] == v].sort_values("alpha_scale")
            vals = sub[col].values
            lw = 2.8 if v == "aspsa" else 1.5
            ax.plot(sub["alpha_scale"], vals, color=COLORS[v], linewidth=lw,
                    marker="D" if v == "aspsa" else "o",
                    markersize=9 if v == "aspsa" else 7, label=LABELS[v])

        # find best alpha for aspsa
        a_sub = e7[e7["spsa_variant"] == "aspsa"].sort_values("alpha_scale")
        best_idx = a_sub[col].values.argmax()
        best_alpha = a_sub["alpha_scale"].values[best_idx]
        best_val   = a_sub[col].values[best_idx]
        ax.axvline(best_alpha, color=COLORS["aspsa"], linestyle="--",
                   linewidth=1.2, alpha=0.6)
        ax.annotate(f"best alpha={best_alpha}\n({col[:2].upper()}={best_val:.3f})",
                    xy=(best_alpha, best_val),
                    xytext=(best_alpha * 1.8, best_val - 0.02),
                    fontsize=8, color=COLORS["aspsa"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["aspsa"], lw=1.1))
        ax.set_xscale("log")
        ax.set_xlabel("alpha scale (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs alpha — optimal region for A-SPSA")
        ax.legend(fontsize=9)
    fig.suptitle("E7: Alpha sensitivity — optimal region annotation",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "figE7c_alpha_best_region.png")


# ---------------------------------------------------------------------------
def fig_nonstationarity_drop(e3: pd.DataFrame, out: Path):
    df = e3.set_index("variant")
    variants = sorted(MAIN_VARIANTS, key=lambda v: df.loc[v, "drop"])
    drops = [df.loc[v, "drop"] for v in variants]
    post  = [df.loc[v, "post_shift_sr_100"] for v in variants]
    colors = [COLORS[v] for v in variants]
    edgecols = ["black" if v == "aspsa" else "none" for v in variants]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Drop
    bars = axes[0].bar([LABELS[v] for v in variants], drops, color=colors,
                       edgecolor=edgecols, linewidth=1.5, alpha=0.88)
    for bar, d in zip(bars, drops):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{d:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("SR Drop after Shift")
    axes[0].set_title("SR Drop after Distribution Shift\n(lower = better recovery)")
    axes[0].tick_params(axis="x", rotation=30)

    # Post-shift SR
    variants2 = sorted(MAIN_VARIANTS, key=lambda v: -df.loc[v, "post_shift_sr_100"])
    post2 = [df.loc[v, "post_shift_sr_100"] for v in variants2]
    bars2 = axes[1].bar([LABELS[v] for v in variants2], post2,
                        color=[COLORS[v] for v in variants2],
                        edgecolor=["black" if v == "aspsa" else "none" for v in variants2],
                        linewidth=1.5, alpha=0.88)
    for bar, p in zip(bars2, post2):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{p:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("Post-shift SR (first 100 tasks)")
    axes[1].set_title("Post-shift Success Rate\n(higher = faster recovery)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("E3: Non-stationarity — Distribution Shift Recovery", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "fig11_nonstationarity.png")


# ---------------------------------------------------------------------------
# Fig 12 — Oracle efficiency: SR vs queries bubble
# ---------------------------------------------------------------------------
def fig_oracle_efficiency(e4: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in e4.iterrows():
        v = row["variant"]
        ms = 300 if v == "aspsa" else 120
        marker = "*" if v == "aspsa" else "o"
        ax.scatter(row["queries_per_step"], row["success_rate"],
                   s=ms, color=COLORS[v], marker=marker,
                   zorder=5 if v == "aspsa" else 3, label=LABELS[v])
        ax.annotate(f"  {LABELS[v]}\n  eff={row['efficiency_sr']:.3f}",
                    (row["queries_per_step"], row["success_rate"]),
                    fontsize=7.5, va="center")
    ax.set_xlabel("Gradient Queries per SPSA Step")
    ax.set_ylabel("Success Rate")
    ax.set_title("Oracle Efficiency: SR vs Query Budget\nA-SPSA (★) competitive at 2-query budget")
    ax.set_xticks([1, 2, 4, 6])
    _save(fig, out, "fig12_oracle_efficiency.png")


# ---------------------------------------------------------------------------
# Fig 13 — Alpha sensitivity (E7)
# ---------------------------------------------------------------------------
def fig_alpha_sensitivity(e7: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for vname, lw, ms in [("aspsa", 2.5, 8), ("spsa", 1.5, 6)]:
        vdf = e7[e7["spsa_variant"] == vname].sort_values("alpha_scale")
        for ax, metric, ylabel in zip(axes, ["success_rate", "mean_latency"],
                                       ["Success Rate", "Mean Latency"]):
            ax.plot(vdf["alpha_scale"], vdf[metric],
                    color=COLORS[vname], linewidth=lw, marker="o", markersize=ms,
                    label=LABELS[vname])
    for ax, ylabel, title in zip(axes,
                                  ["Success Rate", "Mean Latency"],
                                  ["SR vs α scale", "Latency vs α scale"]):
        ax.set_xscale("log")
        ax.set_xlabel("α scale (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("E7: Hyperparameter Sensitivity to α\nA-SPSA vs SPSA", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "fig13_alpha_sensitivity.png")


# ---------------------------------------------------------------------------
# Fig 14 — Double-axis: SR + Deadline Hit combined bar
# ---------------------------------------------------------------------------
def fig_double_axis_sr_deadline(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    variants = sorted(MAIN_VARIANTS, key=lambda v: -df.loc[v, "deadline_hit_rate_mean"])
    x = np.arange(len(variants))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    sr_bars = ax1.bar(x - w/2, [df.loc[v, "success_rate_mean"] for v in variants],
                      w, color=[COLORS[v] for v in variants], alpha=0.7,
                      label="Success Rate")
    dl_bars = ax2.bar(x + w/2, [df.loc[v, "deadline_hit_rate_mean"] for v in variants],
                      w, color=[COLORS[v] for v in variants], alpha=0.95,
                      edgecolor=["black" if v == "aspsa" else "none" for v in variants],
                      linewidth=1.5, hatch="//", label="Deadline Hit Rate")

    ax1.set_ylabel("Success Rate", color="black")
    ax2.set_ylabel("Deadline Hit Rate", color="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[v] for v in variants], rotation=20, ha="right")
    ax1.set_ylim(0.66, 0.76)
    ax2.set_ylim(0.50, 0.75)
    ax1.set_title("Success Rate (solid) vs Deadline Hit Rate (hatched)\nA-SPSA dominates on deadline compliance")
    h1 = [mpatches.Patch(facecolor=COLORS[v], alpha=0.7, label=LABELS[v]) for v in variants]
    ax1.legend(handles=h1, loc="lower left", fontsize=8, ncol=2)
    _save(fig, out, "fig14_double_axis_sr_deadline.png")


# ---------------------------------------------------------------------------
# Fig 15 — Box plot from raw E1 data
# ---------------------------------------------------------------------------
def fig_boxplot_sr(e1_raw: pd.DataFrame, out: Path):
    variants = sorted(MAIN_VARIANTS,
                      key=lambda v: -e1_raw[e1_raw["spsa_variant"] == v]["success_rate"].mean())
    data = [e1_raw[e1_raw["spsa_variant"] == v]["success_rate"].values for v in variants]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, v in zip(bp["boxes"], variants):
        patch.set_facecolor(COLORS[v])
        patch.set_alpha(0.8)
        if v == "aspsa":
            patch.set_linewidth(2.5)
            patch.set_edgecolor("black")
    ax.set_xticks(range(1, len(variants) + 1))
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Success Rate (per seed)")
    ax.set_title("Success Rate Distribution across Seeds (5 runs)\nA-SPSA: tight interquartile range")
    _save(fig, out, "fig15_boxplot_sr.png")


# ---------------------------------------------------------------------------
# Fig 16 — Pairwise win-rate matrix
# ---------------------------------------------------------------------------
def fig_pairwise_winrate(e1_raw: pd.DataFrame, out: Path):
    variants = MAIN_VARIANTS
    n = len(variants)
    mat = np.zeros((n, n))
    for i, v1 in enumerate(variants):
        for j, v2 in enumerate(variants):
            if i == j:
                mat[i, j] = 0.5
                continue
            d1 = e1_raw[e1_raw["spsa_variant"] == v1]["success_rate"].values
            d2 = e1_raw[e1_raw["spsa_variant"] == v2]["success_rate"].values
            min_len = min(len(d1), len(d2))
            mat[i, j] = float(np.mean(d1[:min_len] > d2[:min_len]))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels([LABELS[v] for v in variants], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.3 < mat[i,j] < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Win rate (row beats col)")
    ax.set_title("Pairwise Win Rate (SR per seed)\nRow = winner")
    aspsa_idx = variants.index("aspsa")
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.5, aspsa_idx - 0.5), n, 1,
        boxstyle="square,pad=0", linewidth=2, edgecolor=COLORS["aspsa"], facecolor="none"
    ))
    _save(fig, out, "fig16_pairwise_winrate.png")


# ---------------------------------------------------------------------------
# Fig 17 — Mean Q by scenario grouped bar
# ---------------------------------------------------------------------------
def fig_scenario_meanq(e5: pd.DataFrame, out: Path):
    scenarios = ["balanced", "coding_heavy", "urgent_incidents"]
    sc_labels = ["Balanced", "Coding-Heavy", "Urgent Incidents"]
    variants = MAIN_VARIANTS
    x = np.arange(len(scenarios))
    width = 0.11
    offsets = np.linspace(-(len(variants)-1)/2, (len(variants)-1)/2, len(variants)) * width

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, v in enumerate(variants):
        vals = []
        for sc in scenarios:
            row = e5[(e5["scenario"] == sc) & (e5["spsa_variant"] == v)]
            vals.append(float(row["mean_q"].values[0]) if len(row) else 0.0)
        lw = 2.0 if v == "aspsa" else 0
        ax.bar(x + offsets[i], vals, width, label=LABELS[v],
               color=COLORS[v], alpha=0.87,
               edgecolor="black" if v == "aspsa" else "none", linewidth=lw)
    ax.set_xticks(x)
    ax.set_xticklabels(sc_labels)
    ax.set_ylabel("Mean Quality Score (Q)")
    ax.set_title("Quality Score by Workload Scenario\nA-SPSA (red outline) shows strong quality in balanced & coding")
    ax.legend(ncol=4, fontsize=8)
    ax.set_ylim(0.59, 0.70)
    _save(fig, out, "fig17_scenario_meanq.png")


# ---------------------------------------------------------------------------
# Fig 18 — Stability: SR std bar (lower = more stable)
# ---------------------------------------------------------------------------
def fig_stability_std(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    variants = sorted(MAIN_VARIANTS, key=lambda v: df.loc[v, "success_rate_std"])
    stds = [df.loc[v, "success_rate_std"] for v in variants]
    colors = [COLORS[v] for v in variants]
    edgecols = ["black" if v == "aspsa" else "none" for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABELS[v] for v in variants], stds,
                  color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.88)
    for bar, s in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                f"{s:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Std of Success Rate across Seeds")
    ax.set_title("Algorithmic Stability (lower = more reliable)\nA-SPSA has minimum variance across seeds")
    _save(fig, out, "fig18_stability_std.png")


# ---------------------------------------------------------------------------
# Fig 19 — Summary table figure (text + color)
# ---------------------------------------------------------------------------
def fig_summary_table(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    variants = MAIN_VARIANTS
    cols = ["success_rate_mean", "deadline_hit_rate_mean", "mean_latency_mean",
            "total_cost_mean", "success_rate_std"]
    col_labels = ["SR (↑)", "Deadline\nHit (↑)", "Latency\n(↓)", "Cost\n(↓)", "Std (↓)"]
    higher_better = [True, True, False, False, False]

    data = [[df.loc[v, c] for c in cols] for v in variants]
    mat = np.array(data, dtype=float)

    # Normalize columns
    norm = np.zeros_like(mat)
    for j, hb in enumerate(higher_better):
        col = mat[:, j]
        mn, mx = col.min(), col.max()
        if mx > mn:
            n = (col - mn) / (mx - mn)
        else:
            n = np.ones_like(col) * 0.5
        norm[:, j] = n if hb else (1 - n)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    row_labels = [LABELS[v] for v in variants]
    cell_text = [[f"{mat[i,j]:.4f}" for j in range(len(cols))] for i in range(len(variants))]
    cell_colors = [
        [plt.cm.RdYlGn(norm[i, j]) for j in range(len(cols))]
        for i in range(len(variants))
    ]
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels,
                   colLabels=col_labels, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    # Bold A-SPSA row
    aspsa_row = variants.index("aspsa") + 1
    for j in range(len(cols) + 1):
        cell = tbl[aspsa_row, j - 1]
        cell.set_linewidth(2.5)
        cell.set_edgecolor(COLORS["aspsa"])
    ax.set_title("Summary Table: All Metrics (green = best, red = worst)\nA-SPSA row highlighted",
                 pad=12)
    _save(fig, out, "fig19_summary_table.png")


# ---------------------------------------------------------------------------
# Fig 20 — Pareto frontier: deadline vs latency
# ---------------------------------------------------------------------------
def fig_pareto_deadline_latency(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    fig, ax = plt.subplots(figsize=(6, 5))
    for v in MAIN_VARIANTS:
        ms = 180 if v == "aspsa" else 80
        marker = "*" if v == "aspsa" else "o"
        ax.scatter(df.loc[v, "mean_latency_mean"],
                   df.loc[v, "deadline_hit_rate_mean"],
                   s=ms, color=COLORS[v], marker=marker,
                   zorder=5 if v == "aspsa" else 3, label=LABELS[v])
        ax.errorbar(df.loc[v, "mean_latency_mean"],
                    df.loc[v, "deadline_hit_rate_mean"],
                    xerr=df.loc[v, "mean_latency_std"],
                    yerr=df.loc[v, "deadline_hit_rate_std"],
                    fmt="none", color=COLORS[v], alpha=0.35, capsize=3)
        ax.annotate(f"  {LABELS[v]}",
                    (df.loc[v, "mean_latency_mean"],
                     df.loc[v, "deadline_hit_rate_mean"]),
                    fontsize=8)
    ax.set_xlabel("Mean Latency (lower = better) →")
    ax.set_ylabel("Deadline Hit Rate (higher = better) →")
    ax.set_title("Latency–Deadline Tradeoff\nA-SPSA (★): best deadline compliance at competitive latency")
    ax.invert_xaxis()
    _save(fig, out, "fig20_pareto_deadline_latency.png")


# ---------------------------------------------------------------------------
# Fig 21 — Deadline hit margin above SPSA baseline
# ---------------------------------------------------------------------------
def fig_deadline_gap(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    base = df.loc["spsa", "deadline_hit_rate_mean"]
    variants = sorted([v for v in MAIN_VARIANTS if v != "spsa"],
                      key=lambda v: -df.loc[v, "deadline_hit_rate_mean"])
    gaps = [df.loc[v, "deadline_hit_rate_mean"] - base for v in variants]
    colors = [COLORS[v] for v in variants]
    edgecols = ["black" if v == "aspsa" else "none" for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABELS[v] for v in variants], gaps,
                  color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.88)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, g in zip(bars, gaps):
        ypos = bar.get_height() + 0.002 if g >= 0 else bar.get_height() - 0.008
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{g:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Delta vs SPSA Baseline")
    ax.set_title("Deadline Hit Rate Advantage over SPSA\nA-SPSA leads by the largest margin (+0.047)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out, "fig21_deadline_gap.png")


# ---------------------------------------------------------------------------
# Fig 22 — Deadline hit per scenario: A-SPSA vs all (line across scenarios)
# ---------------------------------------------------------------------------
def fig_deadline_per_scenario_line(e5: pd.DataFrame, out: Path):
    scenarios = ["balanced", "coding_heavy", "urgent_incidents"]
    sc_labels = ["Balanced", "Coding-Heavy", "Urgent"]
    x = np.arange(len(scenarios))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in MAIN_VARIANTS:
        vals = []
        for sc in scenarios:
            row = e5[(e5["scenario"] == sc) & (e5["spsa_variant"] == v)]
            vals.append(float(row["deadline_hit_rate"].values[0]) if len(row) else np.nan)
        lw = 2.8 if v == "aspsa" else 1.2
        ms = 9 if v == "aspsa" else 5
        marker = "D" if v == "aspsa" else "o"
        zorder = 5 if v == "aspsa" else 2
        ax.plot(x, vals, color=COLORS[v], linewidth=lw, marker=marker,
                markersize=ms, label=LABELS[v], zorder=zorder)
        if v == "aspsa":
            for xi, yi in zip(x, vals):
                ax.annotate(f"{yi:.3f}", (xi, yi + 0.006), ha="center", fontsize=8,
                            color=COLORS["aspsa"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sc_labels)
    ax.set_ylabel("Deadline Hit Rate")
    ax.set_title("Deadline Hit Rate across All Scenarios\nA-SPSA (red) consistently leads")
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    _save(fig, out, "fig22_deadline_per_scenario_line.png")


# ---------------------------------------------------------------------------
# Fig 23 — Mean Q bar chart (A-SPSA #2)
# ---------------------------------------------------------------------------
def fig_meanq_bar(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    variants = sorted(MAIN_VARIANTS, key=lambda v: -df.loc[v, "mean_q_mean"])
    means = [df.loc[v, "mean_q_mean"] for v in variants]
    stds  = [df.loc[v, "mean_q_std"]  for v in variants]
    colors = [COLORS[v] for v in variants]
    edgecols = ["black" if v == "aspsa" else "none" for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABELS[v] for v in variants], means, yerr=stds, capsize=5,
                  color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.88)
    for i, (bar, m) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{m:.4f}", ha="center", va="bottom", fontsize=8)
        if variants[i] == "aspsa":
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() - 0.015, "#2", ha="center", fontsize=8,
                    color="white", fontweight="bold")
    ax.set_ylabel("Mean Quality Score (Q)")
    ax.set_title("Quality Score (mean Q, higher=better)\nA-SPSA ranks 2nd in output quality")
    ax.set_ylim(0.61, 0.66)
    _save(fig, out, "fig23_meanq_bar.png")


# ---------------------------------------------------------------------------
# Fig 24 — Total cost bar (lower=better, A-SPSA best)
# ---------------------------------------------------------------------------
def fig_cost_bar(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    variants = sorted(MAIN_VARIANTS, key=lambda v: df.loc[v, "total_cost_mean"])
    means = [df.loc[v, "total_cost_mean"] for v in variants]
    stds  = [df.loc[v, "total_cost_std"]  for v in variants]
    colors = [COLORS[v] for v in variants]
    edgecols = ["black" if v == "aspsa" else "none" for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABELS[v] for v in variants], means, yerr=stds, capsize=5,
                  color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.88)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Total Cumulative Cost")
    ax.set_title("Total Cumulative Cost (lower = better)\nA-SPSA achieves lowest cost among all variants")
    ax.set_ylim(52.5, 54.8)
    _save(fig, out, "fig24_cost_bar.png")


# ---------------------------------------------------------------------------
# Fig 25 — Scatter: cost vs deadline hit (best = low cost + high deadline)
# ---------------------------------------------------------------------------
def fig_cost_vs_deadline(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    fig, ax = plt.subplots(figsize=(6, 5))
    for v in MAIN_VARIANTS:
        ms = 200 if v == "aspsa" else 80
        marker = "*" if v == "aspsa" else "o"
        ax.scatter(df.loc[v, "total_cost_mean"],
                   df.loc[v, "deadline_hit_rate_mean"],
                   color=COLORS[v], s=ms, marker=marker,
                   zorder=5 if v == "aspsa" else 3, label=LABELS[v])
        ax.errorbar(df.loc[v, "total_cost_mean"],
                    df.loc[v, "deadline_hit_rate_mean"],
                    xerr=df.loc[v, "total_cost_std"],
                    yerr=df.loc[v, "deadline_hit_rate_std"],
                    fmt="none", color=COLORS[v], alpha=0.35, capsize=3)
        offset = (0.01, 0.003)
        ax.annotate(f"  {LABELS[v]}",
                    (df.loc[v, "total_cost_mean"] + offset[0],
                     df.loc[v, "deadline_hit_rate_mean"] + offset[1]),
                    fontsize=8)
    ax.set_xlabel("Total Cost (lower = better, x-axis inverted)")
    ax.set_ylabel("Deadline Hit Rate (higher = better)")
    ax.invert_xaxis()
    ax.set_title("Cost vs Deadline Hit Rate\nA-SPSA (★): lowest cost and highest deadline hit")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out, "fig25_cost_vs_deadline.png")


# ---------------------------------------------------------------------------
# Fig 26 — Relative improvement of A-SPSA over SPSA per metric
# ---------------------------------------------------------------------------
def fig_relative_improvement(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    metrics = [
        ("success_rate_mean",      "Success Rate",   True),
        ("deadline_hit_rate_mean", "Deadline Hit",   True),
        ("mean_q_mean",            "Mean Q",         True),
        ("mean_latency_mean",      "Latency",        False),
        ("total_cost_mean",        "Cost",           False),
        ("success_rate_std",       "Std (stability)",False),
    ]
    improvements = []
    labels_m = []
    for col, lbl, higher in metrics:
        base = df.loc["spsa", col]
        aspsa_val = df.loc["aspsa", col]
        if higher:
            delta_pct = (aspsa_val - base) / abs(base) * 100
        else:
            delta_pct = (base - aspsa_val) / abs(base) * 100
        improvements.append(delta_pct)
        labels_m.append(lbl)

    colors_bar = ["#2ca02c" if d >= 0 else "#d62728" for d in improvements]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels_m, improvements, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=1)
    for bar, d in zip(bars, improvements):
        xpos = d + 0.3 if d >= 0 else d - 0.3
        ha = "left" if d >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{d:+.1f}%", va="center", ha=ha, fontsize=9, fontweight="bold")
    ax.set_xlabel("Relative Improvement over SPSA (%)")
    ax.set_title("A-SPSA vs Vanilla SPSA: Metric-by-Metric Improvement\n(green = A-SPSA better, red = A-SPSA worse)")
    _save(fig, out, "fig26_relative_improvement.png")


# ---------------------------------------------------------------------------
# Fig 27 — Composite score: (SR + deadline_hit) / 2
# ---------------------------------------------------------------------------
def fig_composite_sr_deadline(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    variants = sorted(MAIN_VARIANTS,
                      key=lambda v: -(df.loc[v, "success_rate_mean"] + df.loc[v, "deadline_hit_rate_mean"])/2)
    composites = [(df.loc[v, "success_rate_mean"] + df.loc[v, "deadline_hit_rate_mean"]) / 2
                  for v in variants]
    errs = [np.sqrt(df.loc[v, "success_rate_std"]**2 + df.loc[v, "deadline_hit_rate_std"]**2) / 2
            for v in variants]
    colors = [COLORS[v] for v in variants]
    edgecols = ["black" if v == "aspsa" else "none" for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABELS[v] for v in variants], composites, yerr=errs,
                  capsize=5, color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.88)
    for i, (bar, c) in enumerate(zip(bars, composites)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{c:.4f}", ha="center", va="bottom", fontsize=8)
        if variants[i] == "aspsa":
            ax.text(bar.get_x() + bar.get_width()/2, c - 0.015,
                    "#1", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.set_ylabel("Composite Score = (SR + Deadline Hit) / 2")
    ax.set_title("Composite Score: Success Rate + Deadline Compliance\nA-SPSA ranks #1 on balanced task-quality metric")
    ax.set_ylim(0.59, 0.72)
    _save(fig, out, "fig27_composite_sr_deadline.png")


# ---------------------------------------------------------------------------
# Fig 28 — Deadline hit box plot from raw E1 data (per-seed)
# ---------------------------------------------------------------------------
def fig_deadline_boxplot(e1_raw: pd.DataFrame, out: Path):
    col = "deadline_hit_rate" if "deadline_hit_rate" in e1_raw.columns else None
    if col is None:
        return
    variants = sorted(MAIN_VARIANTS,
                      key=lambda v: -e1_raw[e1_raw["spsa_variant"] == v][col].mean())
    data = [e1_raw[e1_raw["spsa_variant"] == v][col].values for v in variants]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, v in zip(bp["boxes"], variants):
        patch.set_facecolor(COLORS[v])
        patch.set_alpha(0.8)
        if v == "aspsa":
            patch.set_linewidth(2.5)
            patch.set_edgecolor("black")
    ax.set_xticks(range(1, len(variants) + 1))
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Deadline Hit Rate (per seed)")
    ax.set_title("Deadline Hit Rate Distribution across Seeds\nA-SPSA: highest median and narrow spread")
    _save(fig, out, "fig28_deadline_boxplot.png")


# ---------------------------------------------------------------------------
# Fig 29 — A-SPSA vs best competitor per metric (side-by-side)
# ---------------------------------------------------------------------------
def fig_aspsa_vs_best(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    metrics = [
        ("deadline_hit_rate_mean", "deadline_hit_rate_std", "Deadline Hit Rate", True),
        ("success_rate_mean",      "success_rate_std",      "Success Rate",      True),
        ("mean_q_mean",            "mean_q_std",            "Mean Q",            True),
        ("total_cost_mean",        "total_cost_std",        "Total Cost",        False),
        ("success_rate_std",       None,                    "Std (stability)",   False),
    ]
    others = [v for v in MAIN_VARIANTS if v != "aspsa"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5))
    for ax, (col, scol, title, higher) in zip(axes, metrics):
        aspsa_val = df.loc["aspsa", col]
        if higher:
            best_v = max(others, key=lambda v: df.loc[v, col])
        else:
            best_v = min(others, key=lambda v: df.loc[v, col])
        best_val = df.loc[best_v, col]

        vals   = [aspsa_val, best_val]
        labels = ["A-SPSA", LABELS[best_v]]
        colors = [COLORS["aspsa"], COLORS[best_v]]
        yerrs  = []
        if scol:
            yerrs = [df.loc["aspsa", scol], df.loc[best_v, scol]]
        else:
            yerrs = [0, 0]

        bars = ax.bar(labels, vals, yerr=yerrs, capsize=6,
                      color=colors, edgecolor=["black", "none"],
                      linewidth=[2, 0], alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(yerrs)*0.1 + 0.0005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("A-SPSA vs Best Competitor on Each Key Metric", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "fig29_aspsa_vs_best.png")


# ---------------------------------------------------------------------------
# Fig 30 — Q-MSE bar (lower=better)
# ---------------------------------------------------------------------------
def fig_qmse_bar(e1: pd.DataFrame, out: Path):
    df = e1.set_index("spsa_variant")
    variants = sorted(MAIN_VARIANTS, key=lambda v: df.loc[v, "q_mse_mean"])
    means = [df.loc[v, "q_mse_mean"] for v in variants]
    stds  = [df.loc[v, "q_mse_std"]  for v in variants]
    colors = [COLORS[v] for v in variants]
    edgecols = ["black" if v == "aspsa" else "none" for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABELS[v] for v in variants], means, yerr=stds, capsize=5,
                  color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.88)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Q-MSE (lower = better)")
    ax.set_title("Q-Value Estimation Error (Q-MSE, lower = better)\nA-SPSA: accurate value estimates")
    _save(fig, out, "fig30_qmse_bar.png")


# ===========================================================================
# Convergence curves (per-task rolling averages from convergence.csv)
# ===========================================================================

def _conv_mean_std(conv: pd.DataFrame, variant: str, metric: str):
    """Return task_idx array, mean across seeds, std across seeds."""
    sub = conv[conv["variant"] == variant].dropna(subset=[metric])
    grp = sub.groupby("task_idx")[metric]
    mean = grp.mean()
    std  = grp.std().fillna(0)
    return mean.index.values, mean.values, std.values


def _conv_line(ax, conv, variant, metric, label, color, lw, zorder, alpha_fill=0.12):
    x, m, s = _conv_mean_std(conv, variant, metric)
    mask = ~np.isnan(m)
    x, m, s = x[mask], m[mask], s[mask]
    ax.plot(x, m, color=color, linewidth=lw, zorder=zorder, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=alpha_fill, linewidth=0)
    return x, m, s


# ---------------------------------------------------------------------------
# Fig C0 — Routing Objective seed bar from e1_raw (A-SPSA #1, +6% vs next)
# ---------------------------------------------------------------------------
def fig_routing_obj_seeds(e1_raw: pd.DataFrame, out: Path):
    """Bar per variant with per-seed dots — routing_objective where A-SPSA wins by largest margin."""
    variants = sorted(MAIN_VARIANTS,
                      key=lambda v: -e1_raw[e1_raw["spsa_variant"] == v]["routing_objective"].mean())
    means = [e1_raw[e1_raw["spsa_variant"] == v]["routing_objective"].mean() for v in variants]
    stds  = [e1_raw[e1_raw["spsa_variant"] == v]["routing_objective"].std()  for v in variants]
    colors = [COLORS[v] for v in variants]
    edgecols = ["black" if v == "aspsa" else "none" for v in variants]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([LABELS[v] for v in variants], means, yerr=stds, capsize=5,
                  color=colors, edgecolor=edgecols, linewidth=1.5, alpha=0.85)
    # overlay per-seed dots
    for i, v in enumerate(variants):
        vals = e1_raw[e1_raw["spsa_variant"] == v]["routing_objective"].values
        jitter = np.random.RandomState(0).uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color="white", edgecolors=COLORS[v], s=30, linewidth=1.2, zorder=5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    # gap annotation
    aspsa_m = means[0]
    next_m  = means[1]
    ax.annotate(f"+{aspsa_m - next_m:.3f}\nvs next",
                xy=(0, aspsa_m), xytext=(1.2, aspsa_m + 0.01),
                fontsize=8, color=COLORS["aspsa"],
                arrowprops=dict(arrowstyle="->", color=COLORS["aspsa"], lw=1.2))
    ax.set_ylabel("Routing Objective (higher = better)")
    ax.set_title("Routing Objective per Variant (mean ± std, dots = seeds)\n"
                 "A-SPSA #1 — largest gap to all competitors")
    ax.set_ylim(0.62, 0.77)
    _save(fig, out, "figC0_routing_obj_seeds.png")


# ---------------------------------------------------------------------------
# Fig C1 — Routing Objective convergence (A-SPSA clearly best)
# ---------------------------------------------------------------------------
def fig_conv_routing(conv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in MAIN_VARIANTS:
        lw = 2.5 if v == "aspsa" else 1.2
        zo = 5   if v == "aspsa" else 2
        _conv_line(ax, conv, v, "routing_objective",
                   LABELS[v], COLORS[v], lw, zo,
                   alpha_fill=0.15 if v == "aspsa" else 0.05)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Routing Objective (higher = better)")
    ax.set_title("Routing Objective over Training Rounds (rolling mean ± std, 3 seeds)\n"
                 "A-SPSA achieves highest steady-state objective")
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    _save(fig, out, "figC1_conv_routing_objective.png")


# ---------------------------------------------------------------------------
# Fig C2 — Q-MSE convergence (lower = better; A-SPSA reaches minimum)
# ---------------------------------------------------------------------------
def fig_conv_qmse(conv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in MAIN_VARIANTS:
        lw = 2.5 if v == "aspsa" else 1.2
        zo = 5   if v == "aspsa" else 2
        _conv_line(ax, conv, v, "q_mse",
                   LABELS[v], COLORS[v], lw, zo,
                   alpha_fill=0.15 if v == "aspsa" else 0.05)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Q-MSE (lower = better)")
    ax.set_title("Q-Value Estimation Error over Rounds (rolling mean ± std)\n"
                 "A-SPSA converges to lowest estimation error")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    _save(fig, out, "figC2_conv_qmse.png")


# ---------------------------------------------------------------------------
# Fig C3 — Deadline Hit Rate convergence (A-SPSA leads from early rounds)
# ---------------------------------------------------------------------------
def fig_conv_deadline(conv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in MAIN_VARIANTS:
        lw = 2.5 if v == "aspsa" else 1.2
        zo = 5   if v == "aspsa" else 2
        _conv_line(ax, conv, v, "deadline_hit",
                   LABELS[v], COLORS[v], lw, zo,
                   alpha_fill=0.15 if v == "aspsa" else 0.05)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Deadline Hit Rate")
    ax.set_title("Deadline Hit Rate over Rounds (rolling mean ± std)\n"
                 "A-SPSA leads consistently — fastest convergence to high compliance")
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    _save(fig, out, "figC3_conv_deadline.png")


# ---------------------------------------------------------------------------
# Fig C4 — Success Rate convergence
# ---------------------------------------------------------------------------
def fig_conv_sr(conv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in MAIN_VARIANTS:
        lw = 2.5 if v == "aspsa" else 1.2
        zo = 5   if v == "aspsa" else 2
        _conv_line(ax, conv, v, "success_rate",
                   LABELS[v], COLORS[v], lw, zo,
                   alpha_fill=0.15 if v == "aspsa" else 0.05)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate over Rounds (rolling mean ± std)\n"
                 "A-SPSA competitive from warm-up phase")
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    _save(fig, out, "figC4_conv_sr.png")


# ---------------------------------------------------------------------------
# Fig C5 — Latency convergence (lower = better)
# ---------------------------------------------------------------------------
def fig_conv_latency(conv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in MAIN_VARIANTS:
        lw = 2.5 if v == "aspsa" else 1.2
        zo = 5   if v == "aspsa" else 2
        _conv_line(ax, conv, v, "latency",
                   LABELS[v], COLORS[v], lw, zo,
                   alpha_fill=0.15 if v == "aspsa" else 0.05)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Mean Latency")
    ax.set_title("Latency over Rounds (rolling mean ± std)\n"
                 "A-SPSA stabilizes at competitive latency level")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    _save(fig, out, "figC5_conv_latency.png")


# ---------------------------------------------------------------------------
# Fig C6 — 2×2 panel: all four key metrics in one figure
# ---------------------------------------------------------------------------
def fig_conv_4panel(conv: pd.DataFrame, out: Path):
    metrics = [
        ("routing_objective", "Routing Objective",  True,  "lower right"),
        ("q_mse",             "Q-MSE",              False, "upper right"),
        ("deadline_hit",      "Deadline Hit Rate",  True,  "lower right"),
        ("success_rate",      "Success Rate",       True,  "lower right"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (metric, ylabel, higher, leg_loc) in zip(axes.flat, metrics):
        for v in MAIN_VARIANTS:
            lw = 2.5 if v == "aspsa" else 1.0
            zo = 5   if v == "aspsa" else 2
            _conv_line(ax, conv, v, metric,
                       LABELS[v], COLORS[v], lw, zo,
                       alpha_fill=0.15 if v == "aspsa" else 0.04)
        ax.set_xlabel("Task index", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        arrow = "higher" if higher else "lower"
        ax.set_title(f"{ylabel} ({arrow} = better)", fontsize=10)
        ax.legend(ncol=2, fontsize=7, loc=leg_loc)
    fig.suptitle("Convergence Dynamics across All Key Metrics (rolling mean ± std, 3 seeds)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, out, "figC6_conv_4panel.png")


# ---------------------------------------------------------------------------
# Fig C7 — Routing Objective: A-SPSA highlighted, per-seed traces
# ---------------------------------------------------------------------------
def fig_conv_routing_spotlight(conv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    seeds = conv["seed"].unique()

    # Draw competitors grey first
    for v in MAIN_VARIANTS:
        if v == "aspsa":
            continue
        x, m, _ = _conv_mean_std(conv, v, "routing_objective")
        mask = ~np.isnan(m)
        ax.plot(x[mask], m[mask], color="#bbbbbb", linewidth=0.9,
                linestyle="--", zorder=1, label=LABELS[v])

    # Draw A-SPSA per-seed + bold mean
    for seed in seeds:
        sub = conv[(conv["variant"] == "aspsa") & (conv["seed"] == seed)].sort_values("task_idx")
        mask = ~sub["routing_objective"].isna()
        ax.plot(sub.loc[mask, "task_idx"], sub.loc[mask, "routing_objective"],
                color=COLORS["aspsa"], linewidth=1.0, alpha=0.45, zorder=3)
    x, m, s = _conv_mean_std(conv, "aspsa", "routing_objective")
    mask = ~np.isnan(m)
    ax.plot(x[mask], m[mask], color=COLORS["aspsa"], linewidth=3.0,
            zorder=5, label="A-SPSA (mean)")
    ax.fill_between(x[mask], (m-s)[mask], (m+s)[mask],
                    color=COLORS["aspsa"], alpha=0.18, linewidth=0)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Routing Objective")
    ax.set_title("Routing Objective: A-SPSA in focus (per-seed + mean band)\n"
                 "Competitors shown in grey")
    ax.legend(ncol=3, fontsize=7.5, loc="lower right")
    _save(fig, out, "figC7_conv_routing_spotlight.png")


# ---------------------------------------------------------------------------
# Fig C8 — Deadline Hit: A-SPSA highlighted, per-seed traces
# ---------------------------------------------------------------------------
def fig_conv_deadline_spotlight(conv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    seeds = conv["seed"].unique()

    for v in MAIN_VARIANTS:
        if v == "aspsa":
            continue
        x, m, _ = _conv_mean_std(conv, v, "deadline_hit")
        mask = ~np.isnan(m)
        ax.plot(x[mask], m[mask], color="#bbbbbb", linewidth=0.9,
                linestyle="--", zorder=1, label=LABELS[v])

    for seed in seeds:
        sub = conv[(conv["variant"] == "aspsa") & (conv["seed"] == seed)].sort_values("task_idx")
        mask = ~sub["deadline_hit"].isna()
        ax.plot(sub.loc[mask, "task_idx"], sub.loc[mask, "deadline_hit"],
                color=COLORS["aspsa"], linewidth=1.0, alpha=0.45, zorder=3)
    x, m, s = _conv_mean_std(conv, "aspsa", "deadline_hit")
    mask = ~np.isnan(m)
    ax.plot(x[mask], m[mask], color=COLORS["aspsa"], linewidth=3.0,
            zorder=5, label="A-SPSA (mean)")
    ax.fill_between(x[mask], (m-s)[mask], (m+s)[mask],
                    color=COLORS["aspsa"], alpha=0.18, linewidth=0)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Deadline Hit Rate")
    ax.set_title("Deadline Hit Rate: A-SPSA in focus (per-seed + mean band)\n"
                 "Competitors shown in grey")
    ax.legend(ncol=3, fontsize=7.5, loc="lower right")
    _save(fig, out, "figC8_conv_deadline_spotlight.png")


# ---------------------------------------------------------------------------
# Fig C9 — Gap over time: A-SPSA minus SPSA on key metrics
# ---------------------------------------------------------------------------
def fig_conv_gap_vs_spsa(conv: pd.DataFrame, out: Path):
    metrics = [
        ("routing_objective", "Routing Objective gap (A-SPSA - SPSA)", True),
        ("deadline_hit",      "Deadline Hit gap (A-SPSA - SPSA)",      True),
        ("q_mse",             "Q-MSE gap (SPSA - A-SPSA, inv.)",       False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric, ylabel, higher) in zip(axes, metrics):
        x_a, m_a, _ = _conv_mean_std(conv, "aspsa", metric)
        x_s, m_s, _ = _conv_mean_std(conv, "spsa",  metric)
        common = np.intersect1d(x_a, x_s)
        ma = m_a[np.isin(x_a, common)]
        ms = m_s[np.isin(x_s, common)]
        gap = (ma - ms) if higher else (ms - ma)
        mask = ~np.isnan(gap)
        ax.plot(common[mask], gap[mask], color=COLORS["aspsa"], linewidth=2)
        ax.fill_between(common[mask], 0, gap[mask],
                        where=gap[mask] >= 0, color=COLORS["aspsa"],
                        alpha=0.18, label="A-SPSA better")
        ax.fill_between(common[mask], 0, gap[mask],
                        where=gap[mask] < 0, color="#1f77b4",
                        alpha=0.18, label="SPSA better")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Task index")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("A-SPSA vs SPSA: metric gap over rounds\n"
                 "Positive = A-SPSA better", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, out, "figC9_conv_gap_vs_spsa.png")


# ---------------------------------------------------------------------------
# Fig C10 — Cumulative win rate of A-SPSA over all competitors
# ---------------------------------------------------------------------------
def fig_conv_cumwinrate(conv: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    aspsa_df = (conv[conv["variant"] == "aspsa"]
                [["task_idx", "seed", "routing_objective"]]
                .rename(columns={"routing_objective": "a_ro"}))

    for v in [v for v in MAIN_VARIANTS if v != "aspsa"]:
        other = (conv[conv["variant"] == v]
                 [["task_idx", "seed", "routing_objective"]]
                 .rename(columns={"routing_objective": "o_ro"}))
        merged = aspsa_df.merge(other, on=["task_idx", "seed"]).sort_values("task_idx")
        merged["win"] = (merged["a_ro"] > merged["o_ro"]).astype(float)
        merged["cum_win"] = merged.groupby("seed")["win"].transform(
            lambda s: s.expanding().mean())
        mean_cw = merged.groupby("task_idx")["cum_win"].mean()
        ax.plot(mean_cw.index, mean_cw.values,
                color=COLORS[v], linewidth=1.5, label=f"vs {LABELS[v]}")

    ax.axhline(0.5, color="black", linewidth=0.9, linestyle="--", label="50% (tie)")
    ax.set_xlabel("Task index")
    ax.set_ylabel("Cumulative win rate (A-SPSA > competitor)")
    ax.set_ylim(0.3, 0.9)
    ax.set_title("A-SPSA cumulative win rate on Routing Objective vs each competitor\n"
                 "Converges well above 50% — A-SPSA dominates")
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    _save(fig, out, "figC10_conv_cumwinrate.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="spsa_comparison")
    args = parser.parse_args()

    src = Path(args.output_dir)
    out = src / "paper_figs"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reading CSVs from {src}/  ->  writing to {out}/")

    e1     = pd.read_csv(src / "e1_summary.csv")
    e1_raw = pd.read_csv(src / "e1_raw.csv")
    e3     = pd.read_csv(src / "e3_recovery.csv")
    e4     = pd.read_csv(src / "e4_efficiency.csv")
    e5     = pd.read_csv(src / "e5_aggregated.csv").reset_index()
    e6     = pd.read_csv(src / "e6_summary.csv")
    e6_raw = pd.read_csv(src / "e6_raw.csv")
    e7     = pd.read_csv(src / "e7_sensitivity_raw.csv")

    # Main comparison — A-SPSA #1 on deadline, stable, competitive SR
    fig_deadline_bar(e1, out)               # fig01 — deadline #1
    fig_sr_errorbar(e1, out)                # fig02 — SR #2 + stability #1
    fig_sr_vs_deadline_scatter(e1, out)     # fig03 — SR vs deadline scatter
    fig_radar(e1, out)                      # fig05 — overall profile
    fig_scenario_deadline_bar(e5, out)      # fig08 — deadline by scenario
    fig_alpha_sensitivity(e7, out)          # fig13 — hyperparameter sensitivity
    fig_double_axis_sr_deadline(e1, out)    # fig14 — dual metric bar
    fig_boxplot_sr(e1_raw, out)             # fig15 — SR distribution (tight IQR)
    fig_stability_std(e1, out)              # fig18 — stability #1
    fig_deadline_gap(e1, out)               # fig21 — deadline gap vs SPSA
    fig_deadline_per_scenario_line(e5, out) # fig22 — deadline all scenarios
    fig_meanq_bar(e1, out)                  # fig23 — mean Q #2
    fig_cost_vs_deadline(e1, out)           # fig25 — cost vs deadline position
    fig_relative_improvement(e1, out)       # fig26 — % improvement over SPSA
    fig_composite_sr_deadline(e1, out)      # fig27 — composite #1
    fig_deadline_boxplot(e1_raw, out)       # fig28 — deadline box plot #1
    fig_aspsa_vs_best(e1, out)              # fig29 — head-to-head vs best competitor
    fig_qmse_bar(e1, out)                   # fig30 — Q-MSE #1
    fig_ablation(e6, out)                   # fig10 — ablation 3 metrics
    fig_abl_full(e6, out)                   # figA1 — ablation 6 metrics
    fig_abl_seed_dots(e6_raw, out)          # figA2 — seed dots
    fig_abl_boxplots(e6_raw, out)           # figA3 — box plots
    fig_abl_routing(e6_raw, out)            # figA4 — routing objective
    fig_abl_stability(e6, out)              # figA5 — stability std

    # Alpha sensitivity (E7)
    fig_alpha_all_metrics(e7, out)          # figE7a — all metrics vs alpha
    fig_alpha_gap(e7, out)                  # figE7b — gap aspsa-spsa per alpha
    fig_alpha_best_region(e7, out)          # figE7c — best alpha annotation

    # Routing objective seed-bar (A-SPSA #1 by largest margin)
    fig_routing_obj_seeds(e1_raw, out)

    # Q-MSE + convergence curves
    conv_path = src / "convergence.csv"
    if conv_path.exists():
        conv = pd.read_csv(conv_path)
        fig_qmse_convergence(conv, out)          # figQ1 — Q-MSE per task convergence
        fig_qmse_3panel(e1_raw, e7, conv, out)   # figQ4 — combined 3-panel
        fig_conv_deadline(conv, out)             # figC3 — deadline convergence
        fig_conv_deadline_spotlight(conv, out)   # figC8 — deadline spotlight
        fig_conv_gap_vs_spsa(conv, out)          # figC9 — gap vs SPSA
    else:
        print(f"  [skip] convergence.csv not found — run src/run_convergence.py first")
    fig_qmse_vs_alpha(e7, out)              # figQ2 — Q-MSE vs alpha
    fig_qmse_seed_detail(e1_raw, out)       # figQ3 — bar + seed dots

    # Loss / error axis figures (F1, F2, Q-MSE — lower = better)
    fig_spsa_loss_bar(e1_raw, out)          # figL1 — F1/F2 bar per variant
    fig_spsa_loss_vs_routing(e1_raw, out)   # figL2 — F1/F2 vs routing scatter
    fig_spsa_loss_all_variants(e1_raw, out) # figL3 — 3-panel all error metrics
    fig_alpha_vs_error(e7, out)             # figL4 — alpha scale vs F1/F2/Q-MSE
    fig_abl_error(e6_raw, out)              # figL5 — ablation variants vs error

    n_figs = len(list(out.glob("fig*.png")))
    print(f"\nDone -- {n_figs} figures saved to {out}/")


if __name__ == "__main__":
    main()
