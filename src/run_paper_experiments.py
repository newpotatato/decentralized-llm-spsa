"""Run the controlled ablation, optimizer, architecture and load experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .paper_experiment import (
    ControlledReplayExperiment, DOMAIN_ORDER, ExperimentConfig,
    load_manifest_tasks, make_agent_pool,
)
from .paper_optimizers import (
    ModelOptimizerConfigs, OptimizerConfig, lipschitz_constants,
    ring_laplacian_spectrum, theory_parameters,
)


ABLATIONS = {
    "no_learning": dict(use_learning=False, use_rls=False, optimizer="none"),
    "local_rls": dict(use_learning=True, use_rls=True, optimizer="local_rls"),
    "spsa": dict(use_learning=True, use_rls=True, optimizer="spsa"),
    "aspsa_no_momentum": dict(use_learning=True, use_rls=True, optimizer="aspsa_no_momentum"),
    "aspsa_fixed_beta": dict(use_learning=True, use_rls=True, optimizer="aspsa_fixed_beta"),
    "aspsa_no_consensus": dict(use_learning=True, use_rls=True, optimizer="aspsa_no_consensus"),
    "aspsa": dict(use_learning=True, use_rls=True, optimizer="aspsa"),
}
OPTIMIZERS = ("spsa", "aspsa", "kw", "zo_pgd", "sp_gt", "zo_gt", "pd_2pt")
ARCHITECTURES = (
    "legacy_local_queue", "random", "static_skill", "shortest_queue",
    "global_greedy", "global_exact", "centralized_exact", "oracle",
)
LOAD_LEVELS = (0.50, 0.85, 1.00, 1.15, 1.40, 2.00)
# Network coefficient omega of Phi, shared by the objective and by every
# method's consensus gradient.  Kept in one place so the objective being
# measured and the objective being descended cannot drift apart.
NETWORK_OMEGA = 0.002
# One shared step/perturbation grid: six configurations, identical for every
# method and every parameter block.
#
# The range spans four decades on purpose.  The methods do not share a usable
# step region: on a quadratic with M = 2 the two-point estimators keep
# improving up to the theoretical h = 1/M, A-SPSA is best near 1e-3 and
# diverges above 5e-2 because the accelerated recurrence amplifies oracle
# noise, and the single-point SP-GT estimator only survives at 1e-4 since its
# magnitude scales with f(x) rather than with grad f.  A grid wide enough to
# contain every method's optimum is what makes an equal six-configuration
# budget an equal opportunity; a narrow per-method range chosen by hand is what
# decided the previous comparison.
STEP_GRID = (
    (0.0001, 0.06), (0.0010, 0.06), (0.0100, 0.08),
    (0.0500, 0.08), (0.2000, 0.12), (0.5000, 0.12),
)
FIGURE_NOTE = ""

METRICS = (
    "mean_utility", "success_rate", "mean_quality", "deadline_hit_rate",
    "mean_latency", "p95_wait", "mean_cost", "f1", "f2", "f3",
    "mean_utilization", "jain_fairness", "throughput", "oracle_queries",
    "rho_effective", "sync_rounds", "tuned_parameters",
    "true_processing_rmse", "true_quality_rmse", "domain_mix_total_variation",
    "queries_per_parameter", "worst_block_queries_per_parameter",
    "queries_per_parameter_processing", "queries_per_parameter_waiting",
    "queries_per_parameter_quality", "query_budget_sufficient",
    "phi_p_auc", "phi_w_auc", "phi_q_auc",
    "phi_p_final", "phi_w_final", "phi_q_final",
    "local_p_final", "local_w_final", "local_q_final",
    "data_p_final", "data_w_final", "data_q_final",
    "disagreement_p_auc", "disagreement_w_auc", "disagreement_q_auc",
    "disagreement_p_final", "disagreement_w_final", "disagreement_q_final",
)

# Phi bundles prediction accuracy and parameter agreement into one scalar.
# At the previous omega the agreement penalty was one to two orders of
# magnitude larger than the local objective it regularised, so ranking by Phi
# alone ranked methods by how hard they averaged parameters.  The two
# components are therefore prespecified endpoints in their own right.
OPTIMIZER_PRIMARY_METRICS = (
    "data_p_final", "data_w_final", "data_q_final",
    "true_processing_rmse", "true_quality_rmse",
    "disagreement_p_auc", "disagreement_w_auc", "disagreement_q_auc",
    "phi_p_auc", "phi_w_auc", "phi_q_auc",
)
ARCHITECTURE_PRIMARY_METRICS = (
    "mean_quality", "deadline_hit_rate", "p95_wait",
    "throughput", "mean_cost", "jain_fairness",
)

def _ci95(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def _aggregate(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for name, part in frame.groupby(group, sort=False):
        row = {group: name, "n": len(part)}
        for metric in METRICS:
            if metric in part:
                row[f"{metric}_mean"] = float(part[metric].mean())
                row[f"{metric}_ci95"] = _ci95(part[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def _run_one(manifest: Path, config: ExperimentConfig, split: str):
    agents = make_agent_pool(config.agents)
    tasks = load_manifest_tasks(
        manifest, config.seed, config.rho, agents, config.task_limit, split=split,
        batch_size=config.arrival_batch_size,
    )
    experiment = ControlledReplayExperiment(tasks, config)
    summary = experiment.run()
    summary.update({
        "seed": config.seed, "architecture": config.architecture,
        "optimizer": config.optimizer, "split": split,
        "sync_interval": config.sync_interval,
    })
    return summary, experiment


def _run_group(manifest: Path, output: Path, label: str, configs: Iterable[tuple[str, ExperimentConfig]], split: str):
    summaries, task_rows, round_rows, load_rows = [], [], [], []
    for variant, config in configs:
        summary, experiment = _run_one(manifest, config, split)
        summary["variant"] = variant
        summaries.append(summary)
        for row in experiment.task_records:
            task_rows.append({"variant": variant, "seed": config.seed, **row})
        for row in experiment.round_records:
            round_rows.append({"variant": variant, "seed": config.seed, **row})
        for row in experiment.load_records:
            load_rows.append({"variant": variant, "seed": config.seed, **row})
    summary_df = pd.DataFrame(summaries)
    task_df = pd.DataFrame(task_rows)
    round_df = pd.DataFrame(round_rows)
    load_df = pd.DataFrame(load_rows)
    summary_df.to_csv(output / f"{label}_raw.csv", index=False)
    task_df.to_csv(output / f"{label}_tasks.csv", index=False)
    round_df.to_csv(output / f"{label}_rounds.csv", index=False)
    load_df.to_csv(output / f"{label}_load_events.csv", index=False)
    _aggregate(summary_df, "variant").to_csv(output / f"{label}_summary.csv", index=False)
    return summary_df, task_df, round_df, load_df


def _plot_loss_dynamics(tasks: pd.DataFrame, rounds: pd.DataFrame, output: Path, prefix: str, window: int = 20):
    if tasks.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="row")
    for variant, part in tasks.groupby("variant", sort=False):
        curves = []
        for _, seed_part in part.groupby("seed"):
            seed_part = seed_part.sort_values("completion_index")
            curve = seed_part.set_index("completion_index")[["f1", "f2", "f3"]].rolling(window, min_periods=1).mean()
            curve["seed"] = seed_part["seed"].iloc[0]
            curves.append(curve.reset_index())
        joined = pd.concat(curves)
        for col, ax in zip(("f1", "f2", "f3"), axes[0]):
            grouped = joined.groupby("completion_index")[col]
            x = np.asarray(sorted(grouped.groups), dtype=float)
            mean = grouped.mean().reindex(x).to_numpy(dtype=float)
            ci = grouped.apply(_ci95).reindex(x).to_numpy(dtype=float)
            ax.plot(x, mean, label=variant)
            ax.fill_between(x, mean - ci, mean + ci, alpha=0.12)
            ax.set_title(f"{col.upper()} by completed task")
            ax.set_ylabel(f"rolling loss (w={window})")
            ax.grid(alpha=0.25)
    if not rounds.empty:
        for variant, part in rounds.groupby("variant", sort=False):
            for col, ax in zip(("f1", "f2", "f3"), axes[1]):
                grouped = part.groupby("round")[col]
                x = np.asarray(sorted(grouped.groups), dtype=float)
                mean = grouped.mean().reindex(x).to_numpy(dtype=float)
                ci = grouped.apply(_ci95).reindex(x).to_numpy(dtype=float)
                ax.plot(x, mean, label=variant)
                ax.fill_between(x, mean - ci, mean + ci, alpha=0.12)
                ax.set_title(f"{col.upper()} by synchronization round")
                ax.set_xlabel("event-driven synchronization round")
                ax.set_ylabel("controller-history loss")
                ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)))
    fig.suptitle(f"{FIGURE_NOTE}Manuscript losses: paired seeds, mean and 95% CI")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(output / f"{prefix}_loss_dynamics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_network_dynamics(rounds: pd.DataFrame, output: Path, prefix: str) -> None:
    if rounds.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for variant, part in rounds.groupby("variant", sort=False):
        for short_name, ax in zip(("p", "w", "q"), axes[0]):
            grouped = part.groupby("round")[f"phi_{short_name}"]
            x = np.asarray(sorted(grouped.groups), dtype=float)
            mean = grouped.mean().reindex(x).to_numpy(dtype=float)
            ci = grouped.apply(_ci95).reindex(x).to_numpy(dtype=float)
            ax.plot(x, mean, label=variant)
            ax.fill_between(x, mean - ci, mean + ci, alpha=0.12)
            ax.set_title(f"Network objective Phi {short_name}")
            ax.set_ylabel("rolling-window Phi (lower is better)")
            ax.grid(alpha=0.25)
        for short_name, ax in zip(("p", "w", "q"), axes[1]):
            grouped = part.groupby("round")[f"disagreement_{short_name}"]
            x = np.asarray(sorted(grouped.groups), dtype=float)
            mean = grouped.mean().reindex(x).to_numpy(dtype=float)
            ci = grouped.apply(_ci95).reindex(x).to_numpy(dtype=float)
            ax.plot(x, mean, label=variant)
            ax.fill_between(x, mean - ci, mean + ci, alpha=0.12)
            ax.set_title(f"Controller disagreement {short_name}")
            ax.set_xlabel("event-driven synchronization round")
            ax.set_ylabel("unweighted quadratic disagreement")
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)))
    fig.suptitle(f"{FIGURE_NOTE}Primary optimizer trajectories: paired mean and 95% CI")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(output / f"{prefix}_network_dynamics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def _plot_system_bars(summary: pd.DataFrame, output: Path, prefix: str):
    """Operational outcomes; routing utility is not an optimizer endpoint."""
    if summary.empty:
        return
    metrics = ARCHITECTURE_PRIMARY_METRICS
    variants = list(summary["variant"].drop_duplicates())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for metric, ax in zip(metrics, axes.ravel()):
        groups = [summary.loc[summary.variant == variant, metric] for variant in variants]
        means = [float(group.mean()) for group in groups]
        cis = [_ci95(group) for group in groups]
        ax.bar(variants, means, yerr=cis, capsize=3)
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{FIGURE_NOTE}Operational system outcomes: mean and 95% CI")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_optimizer_bars(summary: pd.DataFrame, output: Path, prefix: str):
    """Primary endpoints for the three independent network problems."""
    if summary.empty:
        return
    metrics = (
        "phi_p_auc", "phi_w_auc", "phi_q_auc",
        "disagreement_p_auc", "disagreement_w_auc", "disagreement_q_auc",
    )
    variants = list(summary["variant"].drop_duplicates())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for metric, ax in zip(metrics, axes.ravel()):
        groups = [summary.loc[summary.variant == variant, metric] for variant in variants]
        means = [float(group.mean()) for group in groups]
        cis = [_ci95(group) for group in groups]
        ax.bar(variants, means, yerr=cis, capsize=3)
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{FIGURE_NOTE}Primary network objectives: mean and 95% CI")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_network_objectives.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def _paired_significance(
    frame: pd.DataFrame,
    reference: str,
    output_path: Path,
    metrics: tuple[str, ...],
) -> None:
    """Paired Wilcoxon tests; Holm correction within each endpoint family."""
    rows = []
    comparisons: dict[str, list[tuple[float, int]]] = {}
    for variant in frame["variant"].unique():
        if variant == reference:
            continue
        for metric in metrics:
            left = frame[frame.variant == reference][["seed", metric]].rename(columns={metric: "left"})
            right = frame[frame.variant == variant][["seed", metric]].rename(columns={metric: "right"})
            paired = left.merge(right, on="seed").dropna()
            difference = paired.left - paired.right
            if len(difference) >= 2 and np.any(np.abs(difference) > 1e-12):
                p_value = float(stats.wilcoxon(difference).pvalue)
            else:
                p_value = 1.0
            # Smallest two-sided p the exact signed-rank test can produce.
            # Zero differences are discarded by the test, so the floor is set
            # by the number of non-zero pairs.  Every "significant" row of a
            # ten-seed design sits on this floor: the test reports only that
            # the sign was consistent and cannot separate a 1e-8 effect from a
            # large one.
            effective_pairs = int(np.count_nonzero(np.abs(difference) > 1e-12))
            resolution_floor = (
                2.0 / (2.0 ** effective_pairs) if effective_pairs else 1.0
            )
            reference_mean = float(paired.left.mean()) if len(paired) else np.nan
            variant_mean = float(paired.right.mean()) if len(paired) else np.nan
            mean_difference = float(difference.mean()) if len(difference) else np.nan
            relative_difference = (
                mean_difference / abs(variant_mean)
                if np.isfinite(variant_mean) and abs(variant_mean) > 1e-12
                else np.nan
            )
            row = {
                "reference": reference, "variant": variant, "metric": metric,
                "pairs": len(difference),
                "reference_mean": reference_mean,
                "variant_mean": variant_mean,
                "mean_paired_difference": mean_difference,
                "relative_difference_vs_variant": relative_difference,
                "p_raw": p_value,
                "effective_pairs": effective_pairs,
                "p_resolution_floor": float(resolution_floor),
                "p_at_resolution_floor": bool(p_value <= resolution_floor * (1.0 + 1e-9)),
            }
            rows.append(row)
            comparisons.setdefault(metric, []).append((p_value, len(rows) - 1))
    adjusted = [1.0] * len(rows)
    for family in comparisons.values():
        running = 0.0
        total = len(family)
        for rank, (p_value, index) in enumerate(sorted(family), start=1):
            corrected = min(1.0, (total - rank + 1) * p_value)
            running = max(running, corrected)
            adjusted[index] = running
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = p_holm
        row["reject_holm_0_05"] = bool(p_holm < 0.05)
        row["practical_effect_ge_1pct"] = bool(
            np.isfinite(row["relative_difference_vs_variant"])
            and abs(row["relative_difference_vs_variant"]) >= 0.01
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)

def _plot_oracle_efficiency(summary: pd.DataFrame, output: Path) -> None:
    grouped = summary.groupby("variant", sort=False).agg(
        queries=("oracle_queries", "mean"),
        phi_p=("phi_p_auc", "mean"),
        phi_w=("phi_w_auc", "mean"),
        phi_q=("phi_q_auc", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for short_name, ax in zip(("p", "w", "q"), axes):
        for row in grouped.itertuples(index=False):
            value = getattr(row, f"phi_{short_name}")
            ax.scatter(row.queries, value, s=65)
            ax.annotate(row.variant, (row.queries, value), xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("cumulative zeroth-order objective queries")
        ax.set_ylabel(f"Phi {short_name} AUC (lower is better)")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{FIGURE_NOTE}Oracle-query efficiency for primary objectives")
    fig.tight_layout()
    fig.savefig(output / "optimizer_oracle_efficiency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def _plot_load(summary: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    metrics = ("mean_utilization", "throughput", "p95_wait", "deadline_hit_rate", "jain_fairness", "mean_quality")
    for metric, ax in zip(metrics, axes.ravel()):
        grouped = summary.groupby("rho")[metric]
        x = np.asarray(sorted(grouped.groups), dtype=float)
        mean = grouped.mean().reindex(x).to_numpy(dtype=float)
        ci = grouped.apply(_ci95).reindex(x).to_numpy(dtype=float)
        ax.plot(x, mean, marker="o")
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.18)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("offered load rho")
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(alpha=0.25)
    fig.suptitle(f"{FIGURE_NOTE}LLM-agent load and saturation")
    fig.tight_layout()
    fig.savefig(output / "load_saturation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_agent_domain_heatmap(tasks: pd.DataFrame, agents: int, output: Path):
    table = pd.crosstab(tasks["agent_id"], tasks["domain"], normalize="columns")
    table = table.reindex(index=range(agents), columns=[skill.value for skill in DOMAIN_ORDER], fill_value=0.0)
    fig, ax = plt.subplots(figsize=(11, 5))
    image = ax.imshow(table.to_numpy(), aspect="auto", vmin=0.0, vmax=max(0.01, float(table.to_numpy().max())), cmap="viridis")
    ax.set_yticks(range(agents), [f"agent {index}" for index in range(agents)])
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_title(f"{FIGURE_NOTE}Assignment share by agent and IT domain")
    fig.colorbar(image, ax=ax, label="share within domain")
    fig.tight_layout()
    fig.savefig(output / "agent_domain_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _tune_methods(
    manifest: Path,
    output: Path,
    base: ExperimentConfig,
    methods: Iterable[str],
    profile: str,
    label: str,
) -> dict[str, ModelOptimizerConfigs]:
    """Tune p/w/q independently on validation Phi AUC with equal grids."""
    def tuning_grid(method: str, block_name: str) -> tuple[OptimizerConfig, ...]:
        """One shared six-configuration grid for every method and block.

        The estimators are dimension-compensated, so a step size means the same
        displacement for every method and a single grid is genuinely
        comparable.  Earlier revisions gave A-SPSA a "dimension aware" range up
        to h = 20 on the processing block while the baselines topped out at
        h = 0.05; that alone decided which method won.

        ``mu_minus_eta`` must satisfy 0 < mu - eta < mu = omega * lambda_2 of
        the averaged ring Laplacian, so it is derived from omega rather than
        tuned.
        """
        # mu is the strong-convexity modulus of the network objective, which
        # on the consensus subspace is the ridge, not omega*lambda_2.
        ridge = {
            "processing": base.loss.lambda_reg_p,
            "waiting": base.loss.lambda_reg_w,
            "quality": base.loss.lambda_reg_q,
        }[block_name]
        mu = theory_parameters(1.0, NETWORK_OMEGA, base.controllers, ridge)["mu"]
        mu_minus_eta = max(0.5 * mu, 1e-6)
        return tuple(
            OptimizerConfig(
                step_size=step,
                perturbation=pert,
                consensus_weight=NETWORK_OMEGA,
                gamma0=1.0,
                mu_minus_eta=mu_minus_eta,
            )
            for step, pert in STEP_GRID
        )

    seeds = (131,) if profile == "pilot" else (131, 149, 163)
    limit = 60 if profile == "pilot" else 120
    block_metrics = {
        "processing": "p",
        "waiting": "w",
        "quality": "q",
    }
    rows = []
    selected: dict[str, ModelOptimizerConfigs] = {}
    for method in methods:
        chosen_blocks: dict[str, OptimizerConfig] = {}
        for block_name, short_name in block_metrics.items():
            grid = tuning_grid(method, block_name)
            block_rows = []
            for grid_id, optimizer_config in enumerate(grid):
                model_configs = ModelOptimizerConfigs(**{block_name: optimizer_config})
                for seed in seeds:
                    config = replace(
                        base,
                        optimizer=method,
                        architecture="fixed_replay",
                        sync_interval=5,
                        optimizer_configs=model_configs,
                        optimizer_blocks=(block_name,),
                        seed=seed,
                        task_limit=limit,
                    )
                    summary, _ = _run_one(manifest, config, "validation")
                    row = {
                        "method": method,
                        "block": block_name,
                        "grid_id": grid_id,
                        **summary,
                    }
                    rows.append(row)
                    block_rows.append(row)
            frame = pd.DataFrame(block_rows)
            aggregate = frame.groupby("grid_id", as_index=False).agg(
                phi_auc=(f"phi_{short_name}_auc", "mean"),
                phi_final=(f"phi_{short_name}_final", "mean"),
                local_final=(f"local_{short_name}_final", "mean"),
                disagreement=(f"disagreement_{short_name}_auc", "mean"),
                oracle_queries=("oracle_queries", "mean"),
            )
            winner = aggregate.sort_values(
                ["phi_auc", "phi_final", "disagreement", "oracle_queries"],
                ascending=[True, True, True, True],
            ).iloc[0]
            chosen_blocks[block_name] = grid[int(winner.grid_id)]
        selected[method] = ModelOptimizerConfigs(**chosen_blocks)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{label}_tuning_raw.csv", index=False)
    payload = {method: asdict(config) for method, config in selected.items()}
    (output / f"{label}_selected.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return selected

def _architecture_search(manifest: Path, output: Path, base: ExperimentConfig, profile: str):
    """Bounded operational search without an arbitrary routing-utility scalar."""
    if profile == "pilot":
        controller_grid, agent_grid, interval_grid, seeds, limit = (3, 5), (3, 6, 9), (5, 10), (11, 23), 60
        validation_seeds = (211, 223)
    else:
        # The agent grid extends past twelve so a selected configuration is no
        # longer pinned to the boundary of its own search space.
        controller_grid, agent_grid, interval_grid, seeds, limit = (3, 4, 5, 6, 7), (3, 4, 6, 8, 10, 12, 15, 18), (5, 10, 20), (11, 23, 37), 120
        validation_seeds = (211, 223, 227, 229, 233)
    rows = []
    for controllers in controller_grid:
        for agents in agent_grid:
            for interval in interval_grid:
                for seed in seeds:
                    config = replace(
                        base, controllers=controllers, agents=agents,
                        sync_interval=interval, seed=seed, task_limit=limit,
                    )
                    summary, _ = _run_one(manifest, config, "train")
                    rows.append(summary)
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "architecture_search_raw.csv", index=False)
    train = raw.groupby(["controllers", "agents", "sync_interval"], as_index=False).agg(
        train_quality=("mean_quality", "mean"),
        train_dhr=("deadline_hit_rate", "mean"),
        train_p95_wait=("p95_wait", "mean"),
        train_throughput=("throughput", "mean"),
        train_cost=("mean_cost", "mean"),
        infrastructure_units=("infrastructure_units", "mean"),
    )
    train_feasible = train[train.train_dhr >= 0.75]
    if train_feasible.empty:
        train_feasible = train
    top_quality = train_feasible.nlargest(5, "train_quality")
    best_per_agent = train_feasible.loc[
        train_feasible.groupby("agents")["train_quality"].idxmax()
    ]
    best_per_controller = train_feasible.loc[
        train_feasible.groupby("controllers")["train_quality"].idxmax()
    ]
    ordered = train_feasible.sort_values(
        ["infrastructure_units", "train_quality"], ascending=[True, False]
    )
    frontier_indices = []
    frontier_best = -np.inf
    for candidate in ordered.itertuples():
        if candidate.train_quality > frontier_best:
            frontier_indices.append(candidate.Index)
            frontier_best = candidate.train_quality
    frontier = train_feasible.loc[frontier_indices]
    shortlist = pd.concat(
        [top_quality, best_per_agent, best_per_controller, frontier], ignore_index=True
    ).drop_duplicates(["controllers", "agents", "sync_interval"])

    validation_rows = []
    for candidate in shortlist.itertuples(index=False):
        for seed in validation_seeds:
            config = replace(
                base, controllers=int(candidate.controllers), agents=int(candidate.agents),
                sync_interval=int(candidate.sync_interval), seed=seed, task_limit=limit,
            )
            summary, _ = _run_one(manifest, config, "validation")
            validation_rows.append(summary)
    validation_raw = pd.DataFrame(validation_rows)
    validation_raw.to_csv(output / "architecture_validation_raw.csv", index=False)
    validation = validation_raw.groupby(
        ["controllers", "agents", "sync_interval"], as_index=False
    ).agg(
        mean_quality=("mean_quality", "mean"),
        deadline_hit_rate=("deadline_hit_rate", "mean"),
        p95_wait=("p95_wait", "mean"),
        throughput=("throughput", "mean"),
        mean_cost=("mean_cost", "mean"),
        infrastructure_units=("infrastructure_units", "mean"),
    )
    feasible = validation[validation.deadline_hit_rate >= 0.80].copy()
    if feasible.empty:
        feasible = validation.copy()

    def select_one(candidates: pd.DataFrame) -> pd.Series:
        best_quality = float(candidates.mean_quality.max())
        near = candidates[candidates.mean_quality >= best_quality - 0.01]
        return near.sort_values(
            ["infrastructure_units", "deadline_hit_rate", "p95_wait", "mean_cost"],
            ascending=[True, False, True, True],
        ).iloc[0]

    distributed_candidates = feasible[feasible.controllers >= 3]
    if distributed_candidates.empty:
        raise RuntimeError("no feasible distributed architecture with at least three controllers")
    selected = select_one(distributed_candidates)
    best_quality = float(distributed_candidates.mean_quality.max())
    validation["pareto_candidate"] = (
        (validation.controllers >= 3)
        & (validation.mean_quality >= best_quality - 0.01)
        & (validation.deadline_hit_rate >= 0.80)
    )
    train.merge(
        validation,
        on=["controllers", "agents", "sync_interval", "infrastructure_units"],
        how="left",
    ).to_csv(output / "architecture_search_summary.csv", index=False)
    selection = {key: int(selected[key]) for key in ("controllers", "agents", "sync_interval")}
    payload = {
        "selected_main": selection,
        "minimum_controllers": 3,
        "minimum_agents": 3,
        "primary_quality_tolerance": 0.01,
        "deadline_hit_constraint": 0.80,
        "routing_utility_used_for_selection": False,
    }
    (output / "selected_architecture.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return replace(base, **selection)

def _git_revision() -> str:
    """Commit the artifacts were produced from, so a result binds to code."""
    import subprocess
    root = Path(__file__).resolve().parent.parent
    prefix = ["git", "-c", f"safe.directory={root.as_posix()}"]
    try:
        completed = subprocess.run(
            prefix + ["rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, cwd=str(root),
        )
        revision = completed.stdout.strip()
        if completed.returncode != 0 or not revision:
            return "unknown"
        dirty = subprocess.run(
            prefix + ["status", "--porcelain"],
            capture_output=True, text=True, timeout=60, cwd=str(root),
        )
        return revision + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def _theory_report(config: ExperimentConfig) -> dict:
    """Manuscript parameter prescription versus what the search may select.

    The manuscript fixes mu = omega*lambda_2 and h = 1/max{M, omega*lambda_max}
    and requires 0 < eta < mu.  Nothing in the search enforces those bounds, so
    they are computed and recorded to make any violation explicit.
    """
    constants = lipschitz_constants(config.loss)
    lambda_2, lambda_max = ring_laplacian_spectrum(config.controllers)
    blocks = {"processing": "M_p", "waiting": "M_w", "quality": "M_q"}
    report = {
        "lambda_2": lambda_2,
        "lambda_max": lambda_max,
        "lipschitz_constants": constants,
        "blocks": {},
    }
    ridges = {
        "processing": config.loss.lambda_reg_p,
        "waiting": config.loss.lambda_reg_w,
        "quality": config.loss.lambda_reg_q,
    }
    for index, (block, key) in enumerate(blocks.items()):
        omega = config.network_omega[index]
        report["blocks"][block] = theory_parameters(
            constants[key], omega, config.controllers, ridges[block]
        )
    return report


def run_suite(args):
    global FIGURE_NOTE
    FIGURE_NOTE = "PILOT — NOT FOR PAPER | " if args.profile == "pilot" else ""
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    seeds = (11, 23) if args.profile == "pilot" else (11, 23, 37, 41, 53, 67, 79, 83, 97, 109)
    limit = 60 if args.profile == "pilot" else None
    base = ExperimentConfig(
        task_limit=limit,
        seed=seeds[0],
        network_omega=(NETWORK_OMEGA, NETWORK_OMEGA, NETWORK_OMEGA),
        diffusion_mixing=float(getattr(args, "diffusion_mixing", 0.0)),
    )
    initial_tuning = _tune_methods(args.manifest, output, base, ("aspsa",), args.profile, "initial_aspsa")
    base = replace(base, optimizer_configs=initial_tuning["aspsa"])
    selected = _architecture_search(args.manifest, output, base, args.profile)
    tuned_names = tuple(dict.fromkeys((*OPTIMIZERS, "aspsa_no_momentum", "aspsa_fixed_beta", "aspsa_no_consensus")))
    tuned = _tune_methods(args.manifest, output, selected, tuned_names, args.profile, "optimizer")
    selected = replace(
        selected, optimizer_configs=tuned["aspsa"],
        optimizer_blocks=("processing", "waiting", "quality"),
    )

    configs = []
    for name, overrides in ABLATIONS.items():
        optimizer_configs = tuned.get(overrides.get("optimizer", "aspsa"), tuned["aspsa"])
        for seed in seeds:
            configs.append((name, replace(
                selected, seed=seed, task_limit=limit,
                architecture="fixed_replay", sync_interval=5,
                optimizer_configs=optimizer_configs, **overrides,
            )))
    ab_summary, ab_tasks, ab_rounds, _ = _run_group(args.manifest, output, "ablation", configs, "test")
    _plot_optimizer_bars(ab_summary, output, "ablation")
    _plot_loss_dynamics(ab_tasks, ab_rounds, output, "ablation")
    _plot_network_dynamics(ab_rounds, output, "ablation")
    _paired_significance(
        ab_summary, "aspsa", output / "ablation_significance.csv",
        OPTIMIZER_PRIMARY_METRICS,
    )

    configs = [
        (method, replace(
            selected, seed=seed, optimizer=method,
            architecture="fixed_replay", sync_interval=5,
            optimizer_configs=tuned[method], task_limit=limit,
        ))
        for method in OPTIMIZERS for seed in seeds
    ]
    op_summary, op_tasks, op_rounds, _ = _run_group(args.manifest, output, "optimizers", configs, "test")
    _plot_optimizer_bars(op_summary, output, "optimizers")
    _plot_loss_dynamics(op_tasks, op_rounds, output, "optimizers")
    _plot_network_dynamics(op_rounds, output, "optimizers")
    _plot_oracle_efficiency(op_summary, output)
    _paired_significance(
        op_summary, "aspsa", output / "optimizer_significance.csv",
        OPTIMIZER_PRIMARY_METRICS,
    )

    configs = []
    for architecture in ARCHITECTURES:
        controllers = selected.controllers
        for seed in seeds:
            configs.append((architecture, replace(selected, controllers=controllers, architecture=architecture, seed=seed, task_limit=limit)))
    ar_summary, _, _, _ = _run_group(args.manifest, output, "architectures", configs, "test")
    _plot_system_bars(ar_summary, output, "architectures")
    _paired_significance(
        ar_summary, "global_exact", output / "architecture_significance.csv",
        ARCHITECTURE_PRIMARY_METRICS,
    )

    stress_configs = [
        (architecture, replace(
            selected, controllers=selected.controllers, architecture=architecture,
            rho=2.0, arrival_batch_size=6, seed=seed, task_limit=limit,
        ))
        for architecture in ARCHITECTURES for seed in seeds
    ]
    stress_summary, _, _, _ = _run_group(
        args.manifest, output, "architectures_stress", stress_configs, "test"
    )
    _plot_system_bars(stress_summary, output, "architectures_stress")
    _paired_significance(
        stress_summary, "global_exact", output / "architecture_stress_significance.csv",
        ARCHITECTURE_PRIMARY_METRICS,
    )

    load_configs = [(f"rho_{rho:.2f}", replace(selected, rho=rho, seed=seed, task_limit=limit)) for rho in LOAD_LEVELS for seed in seeds]
    load_summary, load_tasks, _, _ = _run_group(args.manifest, output, "load", load_configs, "test")
    _plot_load(load_summary, output)
    focus = load_tasks[load_tasks.variant == "rho_0.85"]
    if not focus.empty:
        _plot_agent_domain_heatmap(focus, selected.agents, output)

    manifest = {
        "profile": args.profile, "seeds": list(seeds), "load_levels": list(LOAD_LEVELS),
        "architecture_stress": {"rho": 2.0, "arrival_batch_size": 6},
        "holm_family": "variants within each prespecified outcome metric",
        "architecture_search_budget": {
            "controllers": [3, 5] if args.profile == "pilot" else [3, 4, 5, 6, 7],
            "agents": [3, 6, 9] if args.profile == "pilot" else [3, 4, 6, 8, 10, 12, 15, 18],
            "sync_intervals": [5, 10] if args.profile == "pilot" else [5, 10, 20],
        },
        "selected_architecture": {
            "controllers": selected.controllers, "agents": selected.agents,
            "sync_interval": selected.sync_interval,
        },
        "loss": asdict(selected.loss),
        "network_omega": list(selected.network_omega),
        "optimizer_primary_endpoints": list(OPTIMIZER_PRIMARY_METRICS),
        "architecture_primary_endpoints": list(ARCHITECTURE_PRIMARY_METRICS),
        "routing_utility_role": "secondary diagnostic only; never used for tuning or selection",
        "optimizer_tuning_budget": "six validation configurations per method and parameter block",
        "aspsa_perturbation_protocol": {
            "components": "+/-1/sqrt(d), exactly as in the manuscript",
            "dimension_multiplier": True,
            "dimension_multiplier_note": (
                "E[Delta Delta^T] = I/d makes the raw two-point estimate biased "
                "towards grad F / d.  Every method is compensated by the factor "
                "d so a single step grid is comparable; the perturbation points "
                "themselves remain exactly those of the manuscript."
            ),
            "step_grid": "one shared six-configuration grid for every method and block",
        },
        "consensus_protocol": {
            "form": "omega * sum_l b_jl (theta_j - theta_l), added to the descent direction",
            "identical_across_methods": True,
            "realised_strength": "omega * h for every method",
            "note": (
                "Previously the baselines received a step-independent 5% mixing "
                "while A-SPSA received omega*h; with a tuned h=12 that made "
                "A-SPSA average twelve times harder than the methods it was "
                "compared against, which accounted for most of its reported "
                "advantage on Phi_p."
            ),
        },
        "theory_check": _theory_report(selected),
        "query_budget": {
            "note": (
                "queries_per_parameter below 1 means the zeroth-order methods "
                "received fewer objective evaluations than the block has "
                "parameters; a ranking obtained in that regime reflects the "
                "deterministic parts of each recurrence, not gradient "
                "information.  The ratio is reported per block as well: the "
                "blocks differ in size by an order of magnitude, so the pooled "
                "ratio can look sufficient while the two large blocks are not. "
                "query_budget_sufficient tests the worst block."
            ),
        },
        "code_revision": _git_revision(),
        "optimizer_protocol": {
            "routing": "fixed_replay",
            "sync_interval": 5,
            "feedback": "identical selected task-agent-controller stream across methods",
        },
        "architecture_protocol": {"routing": "event_driven"},
        "selected_optimizer_parameters": {name: asdict(cfg) for name, cfg in tuned.items()},
        "selection_rule": "bounded grid; DHR>=0.80; within 0.01 of best validation quality choose smallest infrastructure, then DHR, p95 wait, and cost; routing utility excluded",
        "replay": "fixed potential outcomes per (task_id, agent_id, seed); selected-pair feedback only",
        "diffusion_mixing": float(getattr(args, "diffusion_mixing", 0.0)),
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/it_routing_benchmark.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_experiment_outputs"))
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--diffusion-mixing", type=float, default=0.0,
                        help="adapt-then-combine strength; 0 keeps the manuscript's "
                             "consensus-only synchronisation")
    return parser.parse_args()


if __name__ == "__main__":
    run_suite(parse_args())


























