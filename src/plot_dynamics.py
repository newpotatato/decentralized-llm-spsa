"""
Dynamic metric curves + clean box plots.

Output (new_results/figures/dynamics/):
  dynamics_sim.png          — rolling curves, simulation, linear scale
  dynamics_sim_log.png      — rolling curves, simulation, log scale
  dynamics_llm.png          — rolling curves, real-LLM, linear scale
  dynamics_llm_log.png      — rolling curves, real-LLM, log scale
  boxplot_sim.png           — box plots, simulation, no dots

Run:
    python -m src.plot_dynamics
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── style ─────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "grid.color": "#ccc",
})

VARIANTS = ["aspsa", "spsa", "kw", "zo_pgd", "zo_gt", "pd_2pt", "sp_gt"]
LABELS = {
    "aspsa": "A-SPSA", "spsa": "SPSA", "kw": "KW",
    "zo_pgd": "ZO-PGD", "zo_gt": "ZO-GT",
    "pd_2pt": "PD-2pt", "sp_gt": "SP-GT",
}
COLORS = {
    "aspsa": "#E63946", "spsa": "#457B9D", "kw": "#2A9D8F",
    "zo_pgd": "#E9C46A", "zo_gt": "#FB8500",
    "pd_2pt": "#F72585", "sp_gt": "#8338EC",
}

WINDOW = 30


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path.name)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rolling_mean_across_seeds(df, variant, col, window):
    sub = df[(df["variant"] == variant) & df[col].notna()].sort_values("task_idx")
    by_seed = sub.groupby("seed")[col].apply(list)
    if by_seed.empty:
        return None, None, None
    curves = [
        pd.Series(v).rolling(window, min_periods=max(1, window // 4)).mean().values
        for v in by_seed
    ]
    max_len = max(len(c) for c in curves)
    mat = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        mat[i, :len(c)] = c
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    xs = np.arange(max_len)
    return xs, mean, std


def _draw_panel(ax, df, col, ylabel, title, variants, log_scale=False):
    order = ["sp_gt", "zo_gt", "pd_2pt", "kw", "zo_pgd", "spsa", "aspsa"]
    for v in order:
        if v not in variants:
            continue
        xs, mean, std = _rolling_mean_across_seeds(df, v, col, WINDOW)
        if xs is None:
            continue
        c = COLORS[v]
        lw = 2.5 if v == "aspsa" else 1.1
        zo = 4 if v == "aspsa" else 2
        ax.plot(xs, mean, color=c, lw=lw, label=LABELS[v], zorder=zo)
        ax.fill_between(xs, mean - std, mean + std, color=c, alpha=0.12, zorder=zo - 1)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Task index")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


# ── Fig 1-2: simulation dynamics ─────────────────────────────────────────────

SIM_METRICS = [
    ("routing_objective", "Routing Objective", "Routing Objective ↑"),
    ("deadline_hit",      "Deadline Hit Rate", "Deadline Hit Rate ↑"),
    ("success_rate",      "Success Rate",      "Success Rate ↑"),
]


def fig_dynamics_sim(conv_csv: Path, out: Path, log_scale: bool):
    df = pd.read_csv(conv_csv)
    present = [v for v in VARIANTS if v in df["variant"].unique()]
    scale_tag = "log" if log_scale else "linear"
    for col, title, ylabel in SIM_METRICS:
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_panel(ax, df, col, ylabel, "", present, log_scale=log_scale)
        ax.legend(ncol=1, loc="lower right", framealpha=0.9)
        ax.set_title(
            f"Simulation — {title} (rolling mean w={WINDOW}, seeds {{11, 42, 123}}, N=800)\n"
            f"Shaded band = ±1 std  |  scale: {scale_tag}",
            fontsize=10,
        )
        plt.tight_layout()
        slug = col.replace("_", "-")
        fname = f"sim_{slug}{'_log' if log_scale else ''}.png"
        _save(fig, out / fname)


# ── Fig 3-4: real-LLM dynamics ────────────────────────────────────────────────

LLM_METRICS = [
    ("routing_utility", "Routing Objective", "Routing Objective ↑"),
    ("deadline_hit",    "Deadline Hit Rate", "Deadline Hit Rate ↑"),
    ("f2",              "F₂ Wait Loss",  "F₂ Wait Loss ↓"),
]


def _load_llm_records(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for fname in os.listdir(d):
            if not fname.startswith("records_") or not fname.endswith(".csv"):
                continue
            df = pd.read_csv(d / fname)
            # parse variant and seed from filename: records_aspsa_seed11.csv
            parts = fname.replace(".csv", "").split("_seed")
            variant = parts[0].replace("records_", "")
            seed = int(parts[1]) if len(parts) > 1 else -1
            df["variant"] = variant
            df["seed"] = seed
            df["task_idx"] = df["parent_id"]
            df["routing_utility"] = df["routing_utility"]
            df["f2"] = np.maximum(0, df["true_wait"] + 0.1 - df["predicted_wait"]) ** 2
            rows.append(df[["variant", "seed", "task_idx",
                             "routing_utility", "deadline_hit", "f2"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fig_dynamics_llm(llm_dirs: list[Path], out: Path, log_scale: bool):
    df = _load_llm_records(llm_dirs)
    if df.empty:
        print("  no real-LLM records found, skipping")
        return
    present = [v for v in VARIANTS if v in df["variant"].unique()]
    seeds = sorted(df["seed"].unique())
    seed_str = "{" + ", ".join(str(s) for s in seeds) + "}"
    scale_tag = "log" if log_scale else "linear"
    for col, title, ylabel in LLM_METRICS:
        # F2 with log scale: use symlog to handle zeros gracefully
        use_log = log_scale
        use_symlog = log_scale and col == "f2"
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_panel(ax, df, col, ylabel, "", present, log_scale=False)
        if use_symlog:
            ax.set_yscale("symlog", linthresh=1e-4)
        elif use_log:
            ax.set_yscale("log")
        ax.legend(ncol=1, loc="upper right", framealpha=0.9)
        ax.set_title(
            f"Real-LLM — {title} (rolling mean w={WINDOW}, seeds {seed_str})\n"
            f"Shaded band = ±1 std  |  scale: {'symlog' if use_symlog else scale_tag}",
            fontsize=10,
        )
        plt.tight_layout()
        slug = col.replace("_", "-")
        fname = f"llm_{slug}{'_log' if log_scale else ''}.png"
        _save(fig, out / fname)


# ── Fig 5: simulation box plots, no dots ─────────────────────────────────────

BOX_METRICS = [
    ("routing_objective", "Routing Objective ↑"),
    ("deadline_hit_rate", "Deadline Hit Rate ↑"),
    ("mse_wait",          "Wait MSE ↓"),
]


def fig_boxplot_sim(raw_csv: Path, out: Path):
    df = pd.read_csv(raw_csv)
    if "variant" not in df.columns and "spsa_variant" in df.columns:
        df = df.rename(columns={"spsa_variant": "variant"})
    present = [v for v in VARIANTS if v in df["variant"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, ylabel) in zip(axes, BOX_METRICS):
        data = [df.loc[df["variant"] == v, col].dropna().values for v in present]
        bp = ax.boxplot(
            data,
            positions=np.arange(len(present)),
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="", markersize=0),  # hide outlier dots
            showfliers=False,
            zorder=3,
        )
        for i, v in enumerate(present):
            c = COLORS.get(v, "#888")
            bp["boxes"][i].set_facecolor(c)
            bp["boxes"][i].set_alpha(0.60)
            if v == "aspsa":
                bp["boxes"][i].set_linewidth(2.2)
                bp["boxes"][i].set_edgecolor("black")
            else:
                bp["boxes"][i].set_edgecolor(c)
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels(
            [LABELS.get(v, v) for v in present],
            rotation=25, ha="right", fontsize=8.5,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10)

    fig.suptitle(
        "Algorithm comparison — simulation (N=800, 7 seeds)\n"
        "Box: Q1–Q3 | whiskers: 1.5×IQR | line: median",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, out / "boxplot_sim.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv",   default="spsa_comparison/convergence.csv")
    ap.add_argument("--raw",    default="spsa_comparison_v5/e1_raw.csv")
    ap.add_argument("--llm",    nargs="+",
                    default=["real_llm_outputs_groq_interval1",
                             "real_llm_outputs_groq_s42"])
    ap.add_argument("--out",    default="new_results/figures/dynamics")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("dynamics_sim (linear)...")
    fig_dynamics_sim(Path(args.conv), out, log_scale=False)
    print("dynamics_sim (log)...")
    fig_dynamics_sim(Path(args.conv), out, log_scale=True)
    print("dynamics_llm (linear)...")
    fig_dynamics_llm([Path(d) for d in args.llm], out, log_scale=False)
    print("dynamics_llm (log)...")
    fig_dynamics_llm([Path(d) for d in args.llm], out, log_scale=True)
    print("boxplot_sim...")
    fig_boxplot_sim(Path(args.raw), out)
    print("All done ->", out.resolve())


if __name__ == "__main__":
    main()
