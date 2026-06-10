"""
Box-plot / strip-plot figures for the paper.

Fig A — Simulation (5 seeds): proper box plots, 3 metrics
Fig B — Real-LLM (3 seeds): strip plots with mean line, 3 metrics

Run:
    python -m src.plot_boxplots [--out new_results/figures/boxplots]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

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
    "grid.alpha":        0.35,
    "grid.linestyle":    ":",
    "grid.color":        "#cccccc",
})

VARIANTS = ["aspsa", "spsa", "kw", "zo_pgd", "zo_gt", "pd_2pt", "sp_gt"]
LABELS   = {
    "aspsa":  "A-SPSA",
    "spsa":   "SPSA",
    "kw":     "KW",
    "zo_pgd": "ZO-PGD",
    "zo_gt":  "ZO-GT",
    "pd_2pt": "PD-2pt",
    "sp_gt":  "SP-GT",
}
COLORS = {
    "aspsa":  "#E63946",
    "spsa":   "#457B9D",
    "kw":     "#2A9D8F",
    "zo_pgd": "#E9C46A",
    "zo_gt":  "#FB8500",
    "pd_2pt": "#F72585",
    "sp_gt":  "#8338EC",
}

METRICS_SIM = [
    ("routing_objective", "Routing Objective ↑"),
    ("deadline_hit_rate", "Deadline Hit Rate ↑"),
    ("mse_wait",          "Wait MSE ↓"),
]

METRICS_LLM = [
    ("routing_objective", "Routing Objective ↑"),
    ("deadline_hit_rate", "Deadline Hit Rate ↑"),
    ("f2_wait_loss",      "F₂ Wait Loss ↓"),
]


def _save(fig, out: Path, name: str):
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved", p)


# ─── Fig A: Simulation box plots ─────────────────────────────────────────────

def fig_simulation_boxplots(raw_csv: Path, out: Path) -> None:
    df = pd.read_csv(raw_csv)
    # use spsa_variant if variant column is missing
    if "variant" not in df.columns and "spsa_variant" in df.columns:
        df = df.rename(columns={"spsa_variant": "variant"})

    # keep only VARIANTS present in data
    present = [v for v in VARIANTS if v in df["variant"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (col, ylabel) in zip(axes, METRICS_SIM):
        data_per_variant = []
        for v in present:
            vals = df.loc[df["variant"] == v, col].dropna().values
            data_per_variant.append(vals)

        x_pos = np.arange(len(present))
        bp = ax.boxplot(
            data_per_variant,
            positions=x_pos,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, linestyle="none"),
            zorder=3,
        )

        rng = np.random.RandomState(0)
        for i, (v, vals) in enumerate(zip(present, data_per_variant)):
            color = COLORS.get(v, "#888888")
            # box fill
            bp["boxes"][i].set_facecolor(color)
            bp["boxes"][i].set_alpha(0.55)
            if v == "aspsa":
                bp["boxes"][i].set_linewidth(2.2)
                bp["boxes"][i].set_edgecolor("black")
            else:
                bp["boxes"][i].set_edgecolor(color)
            # strip jitter
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(i + jitter, vals, color=color, s=28, zorder=5,
                       edgecolors="white", linewidths=0.6, alpha=0.9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([LABELS.get(v, v) for v in present],
                           rotation=28, ha="right", fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10)

    fig.suptitle(
        "Algorithm comparison — simulation (N=500, 5 seeds)\n"
        "Box: Q1–Q3, whiskers: 1.5×IQR, dots: individual seeds",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, out, "boxplot_simulation.png")


# ─── Fig B: Real-LLM strip plots ─────────────────────────────────────────────

def _load_real_llm(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        p = d / "summaries_per_seed.json"
        if not p.exists():
            print(f"  missing {p}, skipping")
            continue
        with open(p) as f:
            data = json.load(f)
        for variant_list in data.values():
            if isinstance(variant_list, list):
                rows.extend(variant_list)
            else:
                rows.append(variant_list)
    return pd.DataFrame(rows)


def fig_real_llm_stripplot(dirs: list[Path], out: Path) -> None:
    df = _load_real_llm(dirs)
    if df.empty:
        print("  no real-LLM data found, skipping Fig B")
        return

    present = [v for v in VARIANTS if v in df["variant"].unique()]
    seeds = sorted(df["seed"].unique())
    n_seeds = len(seeds)
    seed_markers = ["o", "s", "^", "D", "v"][:n_seeds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (col, ylabel) in zip(axes, METRICS_LLM):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        x_pos = np.arange(len(present))
        rng = np.random.RandomState(1)

        for i, v in enumerate(present):
            color = COLORS.get(v, "#888888")
            sub = df[df["variant"] == v].sort_values("seed")
            vals = sub[col].values
            s_ids = sub["seed"].values

            # mean line
            mean_val = np.mean(vals)
            ax.hlines(mean_val, i - 0.28, i + 0.28,
                      colors=color, linewidths=2.2,
                      linestyles="-", zorder=4,
                      alpha=0.85 if v != "aspsa" else 1.0)

            # dots per seed
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            for j, (val, sid) in enumerate(zip(vals, s_ids)):
                marker = seed_markers[seeds.index(sid)]
                msize  = 60 if v == "aspsa" else 44
                ew     = 1.8 if v == "aspsa" else 0.8
                ax.scatter(i + jitter[j], val, marker=marker,
                           s=msize, color=color, zorder=5,
                           edgecolors="black" if v == "aspsa" else "white",
                           linewidths=ew, alpha=0.95)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([LABELS.get(v, v) for v in present],
                           rotation=28, ha="right", fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10)

    # legend for seeds
    handles = [
        plt.scatter([], [], marker=seed_markers[i], color="#555555",
                    s=40, label=f"seed {s}")
        for i, s in enumerate(seeds)
    ]
    handles.append(plt.Line2D([0], [0], color="#555555", lw=2.2, label="mean"))
    axes[-1].legend(handles=handles, fontsize=8, loc="upper right",
                    framealpha=0.85, title="Seeds", title_fontsize=8)

    seed_str = ", ".join(str(s) for s in seeds)
    fig.suptitle(
        f"Algorithm comparison — real-LLM validation (N=200, seeds {{{seed_str}}})\n"
        "Horizontal line: mean across seeds; markers: individual seeds",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, out, "stripplot_real_llm.png")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim",  default="spsa_comparison_v5/e1_raw.csv")
    ap.add_argument("--llm",  nargs="+",
                    default=[
                        "real_llm_outputs_groq_interval1",
                        "real_llm_outputs_groq_s42",
                    ],
                    help="Directories with summaries_per_seed.json")
    ap.add_argument("--out",  default="new_results/figures/boxplots")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Fig A — simulation box plots...")
    fig_simulation_boxplots(Path(args.sim), out)

    print("Fig B — real-LLM strip plots...")
    fig_real_llm_stripplot([Path(d) for d in args.llm], out)

    print("Done →", out.resolve())


if __name__ == "__main__":
    main()
