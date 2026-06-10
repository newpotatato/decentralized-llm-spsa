"""
Experiments to compare SPSA variants and demonstrate A-SPSA superiority.

Tuning + seven experiments (tuning always runs first, E6 ablation second):
  E6  Ablation of A-SPSA components (momentum on/off, adaptive vs fixed β)
  E1  Main performance table (mean ± std over seeds, all metrics)
  E2  Convergence curves (rolling success rate / Q over task index)
  E3  Non-stationarity test (mid-run distribution shift, recovery speed)
  E4  Oracle efficiency (performance vs gradient queries per SPSA step)
  E5  Scenario robustness (workload scenarios)
  E7  α sensitivity grid + θ-trajectory export (Robbins-Monro decay)

Usage:
  python -m src.run_spsa_comparison [--tasks 500] [--seeds 11,42,123,7,99]
                                    [--controllers 3] [--agents 5]
                                    [--output-dir spsa_comparison]
                                    [--experiments E6,E1,E2,E3,E4,E5,E7]
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .main import ExactOrchestrator, SimulationVariant, SemanticFeatureExtractor, Task, TaskType
    from .spsa_variants import VARIANT_DESCRIPTIONS
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from main import ExactOrchestrator, SimulationVariant, SemanticFeatureExtractor, Task, TaskType
    from spsa_variants import VARIANT_DESCRIPTIONS

# ---------------------------------------------------------------------------
# SPSA variants to compare
# ---------------------------------------------------------------------------

SPSA_VARIANTS = ["spsa", "aspsa", "kw", "zo_pgd", "sp_gt", "zo_gt", "pd_2pt"]

# Gradient queries per SPSA step for the oracle-efficiency plot
QUERY_COUNT = {
    "spsa":    2,
    "aspsa":   2,
    "kw":      6,    # 2 * dim (dim=3 theta parameters per controller)
    "zo_pgd":  2,
    "sp_gt":   2,
    "zo_gt":   2,
    "pd_2pt":  2,
}

# Ablation configs: what each variant has enabled
ABLATION_VARIANTS = ["spsa", "aspsa_no_momentum", "aspsa_fixed_beta", "aspsa"]
ABLATION_LABELS = {
    "spsa":              "SPSA\n(baseline)",
    "aspsa_no_momentum": "A-SPSA\nβ≡0",
    "aspsa_fixed_beta":  "A-SPSA\nβ=0.5 fixed",
    "aspsa":             "A-SPSA\n(full)",
}

COLORS = {
    "spsa":              "#457B9D",
    "aspsa":             "#E63946",
    "aspsa_no_momentum": "#A8DADC",
    "aspsa_fixed_beta":  "#95D5B2",
    "kw":                "#2A9D8F",
    "zo_pgd":            "#E9C46A",
    "sp_gt":             "#8338EC",
    "zo_gt":             "#FB8500",
    "pd_2pt":            "#F72585",
}

SCENARIO_TEMPLATES = {
    "balanced": {
        TaskType.PROGRAMMING: "Build ML pipeline code with feature engineering and model training.",
        TaskType.QA: "Answer factual question and explain reasoning with references.",
        TaskType.SUMMARIZATION: "Summarize the report into concise bullet points.",
        TaskType.TRANSLATION: "Translate customer message from Russian to English.",
        TaskType.TOOL_USE: "Call CRM API tool to create ticket and update status.",
    },
    "coding_heavy": {
        TaskType.PROGRAMMING: "Debug a production ML pipeline, optimize training code, fix data leakage, benchmark latency and prepare deployment notes.",
        TaskType.QA: "Explain benchmark regression root cause and answer engineering postmortem questions.",
        TaskType.SUMMARIZATION: "Summarize repository refactor progress and open technical risks for the team.",
        TaskType.TRANSLATION: "Translate code review comments and release notes from Russian to English.",
        TaskType.TOOL_USE: "Use CI and issue tracker APIs to open incidents, update tickets, and attach logs.",
    },
    "urgent_incidents": {
        TaskType.PROGRAMMING: "Urgent production incident: debug failing pipeline, patch code, restore service before deadline.",
        TaskType.QA: "Critical customer question: explain outage status, impact, workaround and ETA immediately.",
        TaskType.SUMMARIZATION: "Summarize incident bridge updates into short executive bullets for the on-call lead.",
        TaskType.TRANSLATION: "Translate incident updates and customer-facing notices between Russian and English ASAP.",
        TaskType.TOOL_USE: "Critical incident: call ticketing, CRM and monitoring APIs to escalate, acknowledge and close tasks.",
    },
}

SCENARIO_PROBS = {
    "balanced":         [0.25, 0.20, 0.20, 0.15, 0.20],
    "coding_heavy":     [0.55, 0.10, 0.10, 0.05, 0.20],
    "urgent_incidents": [0.20, 0.20, 0.15, 0.10, 0.35],
}


# ---------------------------------------------------------------------------
# Task generation helpers
# ---------------------------------------------------------------------------

def _estimate_complexity(text: str, phi: np.ndarray) -> float:
    text_low = text.lower()
    length_f = min(len(text_low) / 800.0, 1.5)
    kw_bonus = sum(0.4 for kw in ("optimize", "pipeline", "debug", "production", "integration", "benchmark") if kw in text_low)
    return float(np.clip(2.0 + 4.0 * float(np.mean(phi)) + 2.0 * length_f + kw_bonus, 1.0, 10.0))


def _estimate_urgency(text: str, h: float) -> float:
    hot = ("urgent", "asap", "critical", "prod", "incident", "deadline")
    bonus = 0.25 if any(w in text.lower() for w in hot) else 0.0
    return float(np.clip(0.25 + 0.05 * h + bonus, 0.05, 1.0))


def generate_tasks(num_tasks: int, seed: int, scenario: str = "balanced") -> list:
    try:
        from .dataset_loader import get_task_pool
    except ImportError:
        from dataset_loader import get_task_pool

    rng = np.random.RandomState(seed)
    probs = SCENARIO_PROBS[scenario]
    types = list(TaskType)

    # Load real task pools once per type (memory-cached after first call).
    pools = {t: get_task_pool(t.name) for t in TaskType}

    tasks = []
    current_arrival = 0.0
    for i in range(num_tasks):
        task_type = types[rng.choice(len(types), p=probs)]
        pool = pools[task_type]
        text = pool[rng.randint(0, len(pool))]
        phi = SemanticFeatureExtractor.extract(text)
        h = _estimate_complexity(text, phi)
        urgency = _estimate_urgency(text, h)
        current_arrival += float(rng.exponential(0.8))
        tasks.append(Task(i, task_type, current_arrival, h, phi, urgency, text))
    return tasks


def generate_nonstationary_tasks(num_tasks: int, seed: int) -> list:
    """Two-phase stream: first half balanced, second half coding_heavy."""
    try:
        from .dataset_loader import get_task_pool
    except ImportError:
        from dataset_loader import get_task_pool

    half = num_tasks // 2
    t1 = generate_tasks(half, seed, "balanced")

    rng = np.random.RandomState(seed + 1000)
    probs2 = SCENARIO_PROBS["coding_heavy"]
    types = list(TaskType)
    pools = {t: get_task_pool(t.name) for t in TaskType}

    base_arrival = t1[-1].t_arrival if t1 else 0.0
    t2 = []
    for i in range(num_tasks - half):
        task_type = types[rng.choice(len(types), p=probs2)]
        pool = pools[task_type]
        text = pool[rng.randint(0, len(pool))]
        phi = SemanticFeatureExtractor.extract(text)
        h = _estimate_complexity(text, phi)
        urgency = _estimate_urgency(text, h)
        base_arrival += float(rng.exponential(0.6))
        t2.append(Task(half + i, task_type, base_arrival, h, phi, urgency, text))
    return t1 + t2


# Best (alpha, beta) per variant — populated by run_tuning(), used by all experiments.
BEST_PARAMS: dict[str, dict] = {}

# Hyperparameters from Table 1 of the paper (verified on 500-task, n=5, m=3 runs).
# Use --use-paper-params to skip grid search and load these directly.
PAPER_PARAMS: dict[str, dict] = {
    "aspsa":             dict(alpha=0.10, beta=0.30, beta_nes_max=0.30),
    "spsa":              dict(alpha=0.70, beta=0.05, beta_nes_max=1.0),
    "kw":                dict(alpha=1.00, beta=0.30, beta_nes_max=1.0),
    "zo_pgd":            dict(alpha=0.50, beta=0.10, beta_nes_max=1.0),
    "sp_gt":             dict(alpha=0.10, beta=0.30, beta_nes_max=1.0),
    "zo_gt":             dict(alpha=0.10, beta=0.30, beta_nes_max=1.0),
    "pd_2pt":            dict(alpha=0.05, beta=0.10, beta_nes_max=1.0),
    "aspsa_no_momentum": dict(alpha=0.10, beta=0.30, beta_nes_max=1.0),
    "aspsa_fixed_beta":  dict(alpha=0.10, beta=0.30, beta_nes_max=1.0),
}

ALPHA_TUNE_GRID    = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0]

# ---------------------------------------------------------------------------
# Task cache — generate once per (num_tasks, seed, scenario), reuse everywhere
# ---------------------------------------------------------------------------
_TASK_CACHE: dict = {}


def _get_tasks(num_tasks: int, seed: int, scenario: str = "balanced") -> list:
    key = (num_tasks, seed, scenario)
    if key not in _TASK_CACHE:
        _TASK_CACHE[key] = generate_tasks(num_tasks, seed, scenario)
    return _TASK_CACHE[key]


def _get_ns_tasks(num_tasks: int, seed: int) -> list:
    key = ("ns", num_tasks, seed)
    if key not in _TASK_CACHE:
        _TASK_CACHE[key] = generate_nonstationary_tasks(num_tasks, seed)
    return _TASK_CACHE[key]


# ---------------------------------------------------------------------------
# Parallel execution helpers
# ---------------------------------------------------------------------------

def _worker_run_one(args: tuple) -> dict:
    """Worker for ThreadPoolExecutor — numpy/CatBoost release the GIL for real concurrency."""
    variant_name, tasks, num_ctrl, num_agents, seed, _best_params, alpha, beta, beta_nes_max = args
    return _run_one(variant_name, tasks, num_ctrl, num_agents, seed,
                    alpha=alpha, beta=beta, beta_nes_max=beta_nes_max)


def _parallel_runs(jobs: list, max_workers: int | None = None) -> list:
    n = max_workers or max(2, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(_worker_run_one, jobs))
BETA_TUNE_GRID     = [0.05, 0.1, 0.2, 0.3]
BETA_NES_MAX_GRID  = [0.3, 0.5, 0.7]   # cap on Nesterov momentum for aspsa


def _get_params(variant_name: str) -> tuple[float, float, float]:
    p = BEST_PARAMS.get(variant_name, {})
    return p.get("alpha", 0.1), p.get("beta", 0.1), p.get("beta_nes_max", 1.0)


def _build_variant(
    name: str,
    alpha: float | None = None,
    beta: float | None = None,
    beta_nes_max: float | None = None,
) -> SimulationVariant:
    a, b, bnm = _get_params(name)
    return SimulationVariant(
        name=name,
        routing_mode="adaptive",
        use_spsa=True,
        spsa_variant=name,
        spsa_alpha=alpha if alpha is not None else a,
        spsa_beta=beta  if beta  is not None else b,
        spsa_beta_nes_max=beta_nes_max if beta_nes_max is not None else bnm,
    )


def _run_one(variant_name: str, tasks: list, num_ctrl: int, num_agents: int, seed: int,
             plot_path: str | None = None,
             alpha: float | None = None, beta: float | None = None,
             beta_nes_max: float | None = None) -> dict:
    np.random.seed(seed)
    orch = ExactOrchestrator(
        num_ctrl=num_ctrl,
        num_agents=num_agents,
        variant=_build_variant(variant_name, alpha=alpha, beta=beta, beta_nes_max=beta_nes_max),
        plot_output_path=plot_path,
        classifier_seed=seed,
    )
    orch.run(list(tasks))
    return orch.collect_summary()


def _run_one_with_records(variant_name: str, tasks: list, num_ctrl: int, num_agents: int,
                           seed: int, alpha: float | None = None,
                           beta: float | None = None,
                           beta_nes_max: float | None = None) -> tuple[dict, list, list]:
    np.random.seed(seed)
    orch = ExactOrchestrator(
        num_ctrl=num_ctrl,
        num_agents=num_agents,
        variant=_build_variant(variant_name, alpha=alpha, beta=beta, beta_nes_max=beta_nes_max),
        classifier_seed=seed,
    )
    orch.run(list(tasks))
    return orch.collect_summary(), list(orch.task_records), list(orch.theta_records)


# ---------------------------------------------------------------------------
# Tuning: grid search over (α, β) on a validation split
# ---------------------------------------------------------------------------

def tune_variant(variant_name: str, val_tasks: list, num_ctrl: int,
                 num_agents: int, seed: int) -> tuple[float, float, float, float]:
    """Grid-search best (alpha, beta[, beta_nes_max]) for one variant on val_tasks.

    For aspsa also searches beta_nes_max in BETA_NES_MAX_GRID.
    Returns (best_alpha, best_beta, best_beta_nes_max, best_success_rate).
    """
    best_sr, best_a, best_b, best_bnm = -1.0, 0.1, 0.1, 1.0
    bnm_grid = BETA_NES_MAX_GRID if variant_name == "aspsa" else [1.0]
    for a in ALPHA_TUNE_GRID:
        for b in BETA_TUNE_GRID:
            for bnm in bnm_grid:
                summary = _run_one(variant_name, val_tasks, num_ctrl, num_agents,
                                   seed, alpha=a, beta=b, beta_nes_max=bnm)
                sr = float(summary.get("success_rate", 0.0))
                if sr > best_sr:
                    best_sr, best_a, best_b, best_bnm = sr, a, b, bnm
    return best_a, best_b, best_bnm, best_sr


def run_tuning(num_tasks: int, seeds: list, num_ctrl: int, num_agents: int,
               output_dir: Path) -> dict[str, dict]:
    """Tune all SPSA variants + ablation variants on a shared val split (parallel grid search).

    IMPORTANT: Uses a separate validation seed (999) to ensure independence from evaluation seeds.
    This prevents data leakage and ensures fair hyperparameter selection.
    """
    global BEST_PARAMS
    print("\n[TUNE] Grid search over (α, β) on validation split (independent seed 999, first 20% of tasks)...")

    # Use a dedicated validation seed INDEPENDENT of evaluation seeds
    val_seed = 999
    all_tasks = _get_tasks(num_tasks, val_seed)
    val_size  = max(30, num_tasks // 5)
    val_tasks = all_tasks[:val_size]

    all_variants = SPSA_VARIANTS + [v for v in ABLATION_VARIANTS if v not in SPSA_VARIANTS]

    # Build all grid jobs across all variants at once
    jobs: list = []
    job_meta: list = []
    for vname in all_variants:
        bnm_grid = BETA_NES_MAX_GRID if vname == "aspsa" else [1.0]
        for a in ALPHA_TUNE_GRID:
            for b in BETA_TUNE_GRID:
                for bnm in bnm_grid:
                    jobs.append((vname, val_tasks, num_ctrl, num_agents, val_seed, {}, a, b, bnm))
                    job_meta.append((vname, a, b, bnm))

    results = _parallel_runs(jobs)

    # Find best per variant
    variant_best: dict = {v: (-1.0, 0.1, 0.1, 1.0) for v in all_variants}
    for (vname, a, b, bnm), summary in zip(job_meta, results):
        sr = float(summary.get("success_rate", 0.0))
        if sr > variant_best[vname][0]:
            variant_best[vname] = (sr, a, b, bnm)

    rows = []
    for vname in all_variants:
        best_sr, best_a, best_b, best_bnm = variant_best[vname]
        BEST_PARAMS[vname] = {"alpha": best_a, "beta": best_b, "beta_nes_max": best_bnm}
        desc = VARIANT_DESCRIPTIONS.get(vname, vname)
        rows.append({"variant": vname, "description": desc,
                     "best_alpha": best_a, "best_beta": best_b,
                     "best_beta_nes_max": best_bnm if vname == "aspsa" else None,
                     "val_success_rate": round(best_sr, 4)})
        bnm_str = f"  bnm={best_bnm:.2f}" if vname == "aspsa" else ""
        print(f"  {vname:22s}  alpha={best_a:.3f}  beta={best_b:.3f}{bnm_str}  val_sr={best_sr:.4f}")

    pd.DataFrame(rows).to_csv(output_dir / "hyperparameters.csv", index=False)
    print(f"  -> saved hyperparameters.csv")
    return BEST_PARAMS


# ---------------------------------------------------------------------------
# E6: Ablation — what drives A-SPSA's gain?
# ---------------------------------------------------------------------------

def run_e6(num_tasks: int, seeds: list, num_ctrl: int, num_agents: int, output_dir: Path, window: int = 50):
    print("\n[E6] Ablation of A-SPSA components...")
    metrics = ["success_rate", "mean_q", "deadline_hit_rate", "mean_latency", "total_cost", "q_mse"]

    jobs, job_meta = [], []
    for vname in ABLATION_VARIANTS:
        for seed in seeds:
            jobs.append((vname, _get_tasks(num_tasks, seed), num_ctrl, num_agents, seed, dict(BEST_PARAMS), None, None, None))
            job_meta.append({"spsa_variant": vname, "seed": seed})

    results = _parallel_runs(jobs)
    rows = []
    for meta, summary in zip(job_meta, results):
        summary.update(meta)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "e6_raw.csv", index=False)

    agg = df.groupby("spsa_variant")[metrics].agg(["mean", "std"]).round(4)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["label"] = agg["spsa_variant"].map(ABLATION_LABELS)
    agg.to_csv(output_dir / "e6_summary.csv", index=False)

    # Bar chart: 3 key metrics side by side
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ordered = ABLATION_VARIANTS
    labels = [ABLATION_LABELS[v] for v in ordered]
    bar_colors = [COLORS[v] for v in ordered]
    edge_colors = ["black" if v == "aspsa" else "none" for v in ordered]
    linewidths  = [2 if v == "aspsa" else 0 for v in ordered]

    for ax, metric, title in zip(axes,
                                  ["success_rate", "mean_q", "deadline_hit_rate"],
                                  ["Success Rate", "Mean Q", "Deadline Hit Rate"]):
        means = [df[df["spsa_variant"] == v][metric].mean() for v in ordered]
        stds  = [df[df["spsa_variant"] == v][metric].std() for v in ordered]
        bars = ax.bar(labels, means, yerr=stds, capsize=5,
                      color=bar_colors, edgecolor=edge_colors, linewidth=linewidths, alpha=0.88)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle(
        "E6: Ablation — A-SPSA component contributions\n"
        "β≡0: no momentum  |  β=0.5 fixed: constant momentum  |  full: adaptive β_t=(t−1)/(t+2)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "e6_ablation_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Convergence curves for ablation variants (single seed)
    seed = seeds[0]
    base_tasks = _get_tasks(num_tasks, seed)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for vname in ABLATION_VARIANTS:
        _, records, _thr = _run_one_with_records(vname, base_tasks, num_ctrl, num_agents, seed)
        rdf = pd.DataFrame(records)
        if rdf.empty:
            continue
        x = np.arange(1, len(rdf) + 1)
        lw = 2.5 if vname == "aspsa" else 1.5
        lbl = ABLATION_LABELS[vname].replace("\n", " ")
        if "success" in rdf.columns:
            sr = rdf["success"].astype(float).rolling(window=window, min_periods=1).mean()
            axes[0].plot(x, sr, color=COLORS[vname], linewidth=lw, label=lbl, alpha=0.9)
        if {"q_hat", "q_true"}.issubset(rdf.columns):
            mse = ((rdf["q_hat"].astype(float) - rdf["q_true"].astype(float)) ** 2).rolling(window=window, min_periods=1).mean()
            axes[1].plot(x, mse, color=COLORS[vname], linewidth=lw, label=lbl, alpha=0.9)

    for ax, ylabel, title in zip(axes,
                                  [f"Rolling SR (w={window})", f"Rolling Q-MSE (w={window})"],
                                  ["Convergence: Success Rate", "Convergence: Q-MSE"]):
        ax.set_xlabel("Task index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("E6: Ablation — Convergence Curves (full A-SPSA in bold red)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "e6_ablation_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("  -> saved e6_summary.csv, e6_ablation_bars.png, e6_ablation_convergence.png")
    return agg


# ---------------------------------------------------------------------------
# E1: Main performance table
# ---------------------------------------------------------------------------

def run_e1(num_tasks: int, seeds: list, num_ctrl: int, num_agents: int, output_dir: Path):
    print("\n[E1] Main performance table...")
    jobs, job_meta = [], []
    for vname in SPSA_VARIANTS:
        for seed in seeds:
            jobs.append((vname, _get_tasks(num_tasks, seed), num_ctrl, num_agents, seed, dict(BEST_PARAMS), None, None, None))
            job_meta.append({"spsa_variant": vname, "seed": seed})

    results = _parallel_runs(jobs)
    rows = []
    for meta, summary in zip(job_meta, results):
        summary.update(meta)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "e1_raw.csv", index=False)

    metrics = ["success_rate", "mean_q", "deadline_hit_rate", "mean_latency", "total_cost", "q_mse"]
    agg = df.groupby("spsa_variant")[metrics].agg(["mean", "std"]).round(4)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.insert(0, "description", agg["spsa_variant"].map(lambda v: VARIANT_DESCRIPTIONS.get(v, v)))
    agg.to_csv(output_dir / "e1_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    labels = SPSA_VARIANTS
    for ax, metric in zip(axes, metrics):
        means = [df[df["spsa_variant"] == v][metric].mean() for v in labels]
        stds  = [df[df["spsa_variant"] == v][metric].std() for v in labels]
        colors = [COLORS[v] for v in labels]
        bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=35)
        for bar, v in zip(bars, labels):
            if v == "aspsa":
                bar.set_edgecolor("black")
                bar.set_linewidth(2)
    plt.suptitle("E1: SPSA Variants — Main Performance Table (mean ± std)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "e1_performance_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    _significance_table(df, metrics, output_dir / "e1_significance.csv")
    print(f"  -> saved e1_summary.csv, e1_performance_table.png, e1_significance.csv")
    return agg


def _significance_table(df: pd.DataFrame, metrics: list, path: Path):
    from scipy import stats  # lazy — only used in main process, not in workers
    rows = []
    aspsa_data = df[df["spsa_variant"] == "aspsa"]
    for vname in SPSA_VARIANTS:
        if vname == "aspsa":
            continue
        other = df[df["spsa_variant"] == vname]
        row = {"vs": vname}
        for m in metrics:
            a = aspsa_data[m].dropna().values
            b = other[m].dropna().values
            if len(a) >= 2 and len(b) >= 2:
                t, p = stats.ttest_ind(a, b, equal_var=False)
                row[f"{m}_p"] = round(p, 4)
                row[f"{m}_d"] = round(float(np.mean(a) - np.mean(b)), 5)
            else:
                row[f"{m}_p"] = None
                row[f"{m}_d"] = None
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# E2: Convergence curves
# ---------------------------------------------------------------------------

def run_e2(num_tasks: int, seeds: list, num_ctrl: int, num_agents: int, output_dir: Path, window: int = 50):
    print("\n[E2] Convergence curves...")
    seed = seeds[0]
    base_tasks = _get_tasks(num_tasks, seed)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for vname in SPSA_VARIANTS:
        _, records, _thr = _run_one_with_records(vname, base_tasks, num_ctrl, num_agents, seed)
        df = pd.DataFrame(records)
        if df.empty:
            continue

        x = np.arange(1, len(df) + 1)

        if "success" in df.columns:
            sr = df["success"].astype(float).rolling(window=window, min_periods=1).mean()
            lw = 2.5 if vname == "aspsa" else 1.2
            axes[0].plot(x, sr, color=COLORS[vname], linewidth=lw,
                         label=vname.upper(), alpha=0.9 if vname == "aspsa" else 0.7)

        if {"q_hat", "q_true"}.issubset(df.columns):
            mse = ((df["q_hat"].astype(float) - df["q_true"].astype(float)) ** 2).rolling(window=window, min_periods=1).mean()
            lw = 2.5 if vname == "aspsa" else 1.2
            axes[1].plot(x, mse, color=COLORS[vname], linewidth=lw,
                         label=vname.upper(), alpha=0.9 if vname == "aspsa" else 0.7)

    for ax, ylabel, title in zip(axes,
                                  [f"Rolling Success Rate (w={window})", f"Rolling Q-MSE (w={window})"],
                                  ["Convergence: Success Rate", "Convergence: Q Prediction MSE"]):
        ax.set_xlabel("Task index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    plt.suptitle("E2: Convergence Curves (A-SPSA in bold red)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "e2_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> saved e2_convergence.png")


# ---------------------------------------------------------------------------
# E3: Non-stationarity / distribution shift
# ---------------------------------------------------------------------------

def run_e3(num_tasks: int, seeds: list, num_ctrl: int, num_agents: int, output_dir: Path, window: int = 50):
    print("\n[E3] Non-stationarity test (mid-run distribution shift)...")
    seed = seeds[0]
    ns_tasks = _get_ns_tasks(num_tasks, seed)
    shift_idx = num_tasks // 2

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    recovery_rows = []
    for vname in SPSA_VARIANTS:
        _, records, _thr = _run_one_with_records(vname, ns_tasks, num_ctrl, num_agents, seed)
        df = pd.DataFrame(records)
        if df.empty:
            continue

        x = np.arange(1, len(df) + 1)
        lw = 2.5 if vname == "aspsa" else 1.2

        if "success" in df.columns:
            sr = df["success"].astype(float).rolling(window=window, min_periods=1).mean()
            axes[0].plot(x, sr, color=COLORS[vname], linewidth=lw,
                         label=vname.upper(), alpha=0.9 if vname == "aspsa" else 0.7)
            post_shift = sr.iloc[shift_idx:].values
            pre_shift_mean = float(sr.iloc[:shift_idx].mean()) if shift_idx > 0 else float(sr.mean())
            if len(post_shift) > 10:
                recovery_window = min(100, len(post_shift))
                post_mean = float(post_shift[:recovery_window].mean())
                recovery_rows.append({"variant": vname, "pre_shift_sr": pre_shift_mean, "post_shift_sr_100": post_mean,
                                       "drop": pre_shift_mean - post_mean})

        if {"q_hat", "q_true"}.issubset(df.columns):
            mse = ((df["q_hat"].astype(float) - df["q_true"].astype(float)) ** 2).rolling(window=window, min_periods=1).mean()
            axes[1].plot(x, mse, color=COLORS[vname], linewidth=lw,
                         label=vname.upper(), alpha=0.9 if vname == "aspsa" else 0.7)

    for ax in axes:
        ax.axvline(shift_idx, color="gray", linestyle="--", linewidth=1.2, label="Distribution shift")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Task index")

    axes[0].set_ylabel(f"Rolling Success Rate (w={window})")
    axes[0].set_title("E3: Non-stationarity — Success Rate")
    axes[1].set_ylabel(f"Rolling Q-MSE (w={window})")
    axes[1].set_title("E3: Non-stationarity — Q Prediction MSE")

    plt.suptitle("E3: Distribution Shift at Task {shift_idx} (A-SPSA in bold red)".format(shift_idx=shift_idx), fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "e3_nonstationary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if recovery_rows:
        pd.DataFrame(recovery_rows).to_csv(output_dir / "e3_recovery.csv", index=False)
    print("  -> saved e3_nonstationary.png, e3_recovery.csv")


# ---------------------------------------------------------------------------
# E4: Oracle efficiency (performance per gradient query)
# ---------------------------------------------------------------------------

def run_e4(num_tasks: int, seeds: list, num_ctrl: int, num_agents: int, output_dir: Path):
    print("\n[E4] Oracle efficiency (success rate vs gradient queries per step)...")
    seed = seeds[0]
    base_tasks = _get_tasks(num_tasks, seed)

    jobs = [(vname, base_tasks, num_ctrl, num_agents, seed, dict(BEST_PARAMS), None, None, None)
            for vname in SPSA_VARIANTS]
    results = _parallel_runs(jobs)
    rows = [{"variant": vname, "success_rate": s["success_rate"],
             "mean_q": s["mean_q"], "queries_per_step": QUERY_COUNT[vname]}
            for vname, s in zip(SPSA_VARIANTS, results)]

    df = pd.DataFrame(rows)
    df["efficiency_sr"] = df["success_rate"] / df["queries_per_step"]
    df["efficiency_q"] = df["mean_q"] / df["queries_per_step"]
    df.to_csv(output_dir / "e4_efficiency.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, ylabel in zip(axes,
                                   ["success_rate", "mean_q"],
                                   ["Success Rate", "Mean Q"]):
        for _, row in df.iterrows():
            v = row["variant"]
            ms = 180 if v == "aspsa" else 80
            marker = "*" if v == "aspsa" else "o"
            ax.scatter(row["queries_per_step"], row[metric],
                       color=COLORS[v], s=ms, marker=marker, zorder=5, label=v.upper())
            ax.annotate(f" {v}", (row["queries_per_step"], row[metric]), fontsize=8)
        ax.set_xlabel("Gradient queries per SPSA step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"E4: {ylabel} vs Oracle Budget")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    plt.suptitle("E4: Oracle Efficiency — A-SPSA achieves top performance at 2 queries (★)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "e4_oracle_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> saved e4_efficiency.csv, e4_oracle_efficiency.png")


# ---------------------------------------------------------------------------
# E5: Scenario robustness
# ---------------------------------------------------------------------------

def run_e5(num_tasks: int, seeds: list, num_ctrl: int, num_agents: int, output_dir: Path):
    print("\n[E5] Scenario robustness across workload types...")
    scenarios = list(SCENARIO_TEMPLATES.keys())

    jobs, job_meta = [], []
    for scenario in scenarios:
        for seed in seeds:
            tasks = _get_tasks(num_tasks, seed, scenario)
            for vname in SPSA_VARIANTS:
                jobs.append((vname, tasks, num_ctrl, num_agents, seed, dict(BEST_PARAMS), None, None, None))
                job_meta.append({"scenario": scenario, "spsa_variant": vname, "seed": seed})

    results = _parallel_runs(jobs)
    rows = []
    for meta, summary in zip(job_meta, results):
        summary.update(meta)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "e5_raw.csv", index=False)

    agg = df.groupby(["scenario", "spsa_variant"])[["success_rate", "mean_q", "deadline_hit_rate"]].mean().round(4)
    agg.to_csv(output_dir / "e5_aggregated.csv")

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 5 * len(scenarios)))
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        scen_df = df[df["scenario"] == scenario]
        for vname in SPSA_VARIANTS:
            v_df = scen_df[scen_df["spsa_variant"] == vname]
            mean_sr = v_df["success_rate"].mean()
            std_sr = v_df["success_rate"].std()
            lw = 2 if vname == "aspsa" else 1
            ax.bar(vname, mean_sr, yerr=std_sr, capsize=4, color=COLORS[vname],
                   alpha=0.85, linewidth=lw, edgecolor="black" if vname == "aspsa" else "none")
        ax.set_title(f"Scenario: {scenario}")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("E5: Scenario Robustness — Success Rate per Workload Type", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "e5_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> saved e5_raw.csv, e5_aggregated.csv, e5_robustness.png")


# ---------------------------------------------------------------------------
# E7: Hyperparameter sensitivity (α scale) + θ-trajectory export
# ---------------------------------------------------------------------------

ALPHA_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def _run_with_alpha(variant_name: str, tasks: list, num_ctrl: int, num_agents: int,
                    seed: int, alpha_scale: float) -> tuple[dict, list]:
    """Run a variant with a custom alpha, returning (summary, theta_records)."""
    _, best_b, best_bnm = _get_params(variant_name)
    summary, _, theta = _run_one_with_records(variant_name, tasks, num_ctrl, num_agents,
                                               seed, alpha=alpha_scale, beta=best_b,
                                               beta_nes_max=best_bnm)
    return summary, theta


def run_e7(num_tasks: int, seeds: list, num_ctrl: int, num_agents: int, output_dir: Path):
    print("\n[E7] Hyperparameter sensitivity (α) + θ-trajectory export...")
    focus_variants = ["spsa", "aspsa"]
    seed = seeds[0]
    base_tasks = _get_tasks(num_tasks, seed)

    # Run all (variant, alpha) combos in parallel
    jobs, job_meta = [], []
    for vname in focus_variants:
        _, best_b, best_bnm = _get_params(vname)
        for alpha_val in ALPHA_GRID:
            jobs.append((vname, base_tasks, num_ctrl, num_agents, seed, dict(BEST_PARAMS), alpha_val, best_b, best_bnm))
            job_meta.append((vname, alpha_val))

    # Need theta records too — use _run_one_with_records sequentially (theta only)
    rows = []
    theta_by_config: dict = {}
    for (vname, alpha_val), job in zip(job_meta, jobs):
        _, best_b, best_bnm = _get_params(vname)
        summary, _, theta_recs = _run_one_with_records(vname, base_tasks, num_ctrl, num_agents,
                                                        seed, alpha=alpha_val, beta=best_b,
                                                        beta_nes_max=best_bnm)
        summary["spsa_variant"] = vname
        summary["alpha_scale"]  = alpha_val
        rows.append(summary)
        theta_by_config[(vname, alpha_val)] = theta_recs
        if theta_recs:
            tdf = pd.DataFrame(theta_recs)
            fname = f"e7_theta_{vname}_a{alpha_val:.4f}.csv".replace(".", "p")
            tdf.to_csv(output_dir / fname, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "e7_sensitivity_raw.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in zip(axes, ["success_rate", "mean_q"],
                                   ["Success Rate", "Mean Q"]):
        for vname in focus_variants:
            vdf = df[df["spsa_variant"] == vname].sort_values("alpha_scale")
            lw = 2.5 if vname == "aspsa" else 1.5
            ax.plot(vdf["alpha_scale"], vdf[metric], color=COLORS[vname],
                    linewidth=lw, marker="o", label=vname.upper())
        ax.set_xscale("log")
        ax.set_xlabel("α scale (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"E7: {ylabel} vs α (Robbins-Monro decay)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(
        "E7: Sensitivity to α — A-SPSA vs SPSA\n"
        "α_t = α / (t + 11)^0.602, shown: scale factor α",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "e7_alpha_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # θ-trajectory plot for tuned α — reuse already-collected records
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for vname in focus_variants:
        best_alpha, _, _bnm = _get_params(vname)
        theta_recs = theta_by_config.get((vname, best_alpha), [])
        if not theta_recs:
            continue
        tdf = pd.DataFrame(theta_recs)
        lw = 2.5 if vname == "aspsa" else 1.5
        for col in [c for c in tdf.columns if c.startswith("theta_time_")]:
            axes[0].plot(tdf["task_idx"], tdf[col], color=COLORS[vname], linewidth=lw,
                         alpha=0.7, label=f"{vname} {col}" if col == "theta_time_0" else "_")
        for col in [c for c in tdf.columns if c.startswith("theta_wait_")]:
            axes[1].plot(tdf["task_idx"], tdf[col], color=COLORS[vname], linewidth=lw,
                         alpha=0.7, label=f"{vname} {col}" if col == "theta_wait_0" else "_")

    for ax, title in zip(axes, ["θ_time trajectory", "θ_wait trajectory"]):
        ax.set_xlabel("Task index")
        ax.set_ylabel("θ value")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("E7: θ-trajectory (tuned α per variant, A-SPSA in red)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "e7_theta_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  -> saved e7_sensitivity_raw.csv, e7_alpha_sensitivity.png, "
          f"e7_theta_trajectory.png, e7_theta_*.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def use_paper_params(output_dir: Path) -> None:
    """Skip grid search — load verified paper hyperparameters directly."""
    global BEST_PARAMS
    BEST_PARAMS = {k: dict(v) for k, v in PAPER_PARAMS.items()}
    rows = []
    for vname, p in BEST_PARAMS.items():
        rows.append({
            "variant": vname,
            "description": VARIANT_DESCRIPTIONS.get(vname, vname),
            "best_alpha": p["alpha"],
            "best_beta": p["beta"],
            "best_beta_nes_max": p.get("beta_nes_max"),
            "val_success_rate": "paper",
        })
    pd.DataFrame(rows).to_csv(output_dir / "hyperparameters.csv", index=False)
    print("[PARAMS] Using paper hyperparameters (Table 1) — grid search skipped.")
    for vname, p in BEST_PARAMS.items():
        bnm = f"  bnm={p['beta_nes_max']:.2f}" if vname == "aspsa" else ""
        print(f"  {vname:22s}  alpha={p['alpha']:.3f}  beta={p['beta']:.3f}{bnm}")


def parse_args():
    parser = argparse.ArgumentParser(description="SPSA variant comparison experiments for the paper.")
    parser.add_argument("--tasks", type=int, default=1000)
    parser.add_argument("--seeds", type=str, default="11,42,123,7,99,17,88")
    parser.add_argument("--controllers", type=int, default=5)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="spsa_comparison")
    parser.add_argument(
        "--experiments", type=str, default="E6,E1,E2,E3,E4,E5,E7",
        help="Comma-separated subset, e.g. E6,E1,E7"
    )
    parser.add_argument(
        "--use-paper-params", action="store_true",
        help="Skip hyperparameter tuning and use verified paper params (Table 1)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    experiments = {e.strip().upper() for e in args.experiments.split(",")}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(num_tasks=args.tasks, seeds=seeds, num_ctrl=args.controllers, num_agents=args.agents, output_dir=output_dir)

    if args.use_paper_params:
        use_paper_params(output_dir)
    else:
        # Tuning always runs first — its results flow into all experiments via BEST_PARAMS.
        run_tuning(**kw)

    if "E6" in experiments:
        run_e6(**kw)
    if "E1" in experiments:
        run_e1(**kw)
    if "E2" in experiments:
        run_e2(**kw)
    if "E3" in experiments:
        run_e3(**kw)
    if "E4" in experiments:
        run_e4(**kw)
    if "E5" in experiments:
        run_e5(**kw)
    if "E7" in experiments:
        run_e7(**kw)

    print(f"\nAll done. Results saved to: {output_dir}/")
