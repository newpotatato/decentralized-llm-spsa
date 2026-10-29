"""Measured-outcome evaluation: the architecture against published routers.

Every system -- the learned controllers, the literature baselines and the
per-prompt oracle -- runs on the same measured task stream (``table_oracle``)
under two capacity regimes:

* ``api``: many serving slots per model at low load, the setting of RouterBench,
  RouteLLM, CARROT and RouterArena.  Endpoints are quality, cost, the share of
  the best-single-to-oracle gap recovered, and the cost--quality frontier.
* ``capacity``: one slot per model at high load, the setting of the LLM-serving
  literature.  Endpoints add SLO attainment, goodput and tail latency.

Systems that trade quality against price are swept over the price weight, so
each contributes a frontier rather than one operating point.  Optimizer step
sizes are selected on the validation split by one criterion shared by every
method (held-out counterfactual quality error), then the test split is run on
paired seeds.
"""
from __future__ import annotations

import os

# One BLAS thread per worker: the runs are many small linear-algebra calls, and
# a full OpenBLAS thread pool in every worker process exhausts memory.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
# Arrow's default pool keeps every parquet read resident (~200 MB per worker);
# the system allocator returns it, which halves a worker's memory.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import real_benchmarks, table_oracle
from .paper_experiment import ControlledReplayExperiment, ExperimentConfig
from .paper_losses import PaperLossConfig
from .paper_optimizers import ModelOptimizerConfigs, OptimizerConfig, theory_parameters
from .prompt_embeddings import ENCODER, PromptFeatures, mlp_quality
from .run_paper_experiments import _git_revision, _paired_significance


# name: (architecture, optimizer, sweeps price weight)
SYSTEMS = {
    "random": ("random", "none", False),
    "cost_greedy": ("cost_greedy", "none", False),
    "quality_greedy": ("quality_greedy", "none", False),
    "static_domain": ("static_domain", "none", False),
    "shortest_queue": ("shortest_queue", "none", False),
    "knn_router": ("knn_router", "none", True),
    "local_rls": ("global_exact", "local_rls", True),
    "spsa": ("global_exact", "spsa", True),
    "aspsa": ("global_exact", "aspsa", True),
    "zo_gt": ("global_exact", "zo_gt", True),
    "oracle": ("oracle", "none", False),
    # A trained router (RouterBench's MLP router) as a baseline, and the
    # architecture given the same offline estimate as its quality prior, so the
    # comparison with each router holds the offline estimator fixed.
    "mlp_router": ("mlp_router", "none", True),
    "aspsa_mlp": ("global_exact", "aspsa", True),
    # The architecture with the buffer's weighted priority replaced by a
    # classic SLO-aware rule; routing to agents is unchanged.
    "aspsa_edf": ("global_exact", "aspsa", True),
    "aspsa_llf": ("global_exact", "aspsa", True),
    "aspsa_sjf": ("global_exact", "aspsa", True),
    # The baselines behind the same EDF buffer: the architecture comparison
    # under the classic rule rather than under the weighted priority.
    "knn_router_edf": ("knn_router", "none", True),
    "mlp_router_edf": ("mlp_router", "none", True),
    "static_domain_edf": ("static_domain", "none", False),
}
PRIOR_ROUTER = {"aspsa_mlp": "mlp"}
BUFFER_ORDER = {"aspsa_edf": "edf", "aspsa_llf": "llf", "aspsa_sjf": "sjf",
                "knn_router_edf": "edf", "mlp_router_edf": "edf", "static_domain_edf": "edf"}
LEARNED = ("spsa", "aspsa", "zo_gt", "aspsa_mlp", "aspsa_edf", "aspsa_llf", "aspsa_sjf")
# Sensitivity of the conclusions to the assumed latency profiles: every model
# slower or faster by 2x, or each model's speed drawn log-uniformly in
# [1/2, 2] with a fixed seed, which reorders which models are fast.
LATENCY_VARIANTS = ("default", "slow", "fast", "shuffle1", "shuffle2", "shuffle3")
PRICE_WEIGHTS = (0.0, -0.08, -0.3, -1.0, -3.0)  # includes the default operating point
DEFAULT_PRICE_WEIGHT = -0.08
# Extended below 1e-4 because every method selected the old lower bound.
STEP_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2)
# name: (slots per model, offered load)
# One slot per model is the stress case: a single 40-minute reasoning call
# blocks a model entirely, which no batching server does.  Four slots is a
# small self-hosted pool; sixteen at low load approximates a hosted API.
# name: (slots per model, offered load, latency weight in the routing score).
# In the API regime the absolute arrival rate equals that of 16 slots at 0.30,
# spread over 32 slots so the strongest models are not saturated, and routing
# trades quality against price only, as RouterBench and RouteLLM define it.
REGIMES = {"api": (32, 0.50, 0.0), "capacity": (4, 0.85, -0.16), "stress": (1, 0.85, -0.16)}
# What rho is relative to.  In the API regime it is the utilisation the slowest
# model would reach if it took all traffic, so capacity never binds; elsewhere
# it is relative to the whole pool, so queues form as in a real serving pool.
LOAD_REFERENCE = {"api": "slowest_model", "capacity": "pool", "stress": "pool"}
# Strength of the offline prior in the shared local estimator (RLS).  One value
# is selected per regime on validation with local_rls and then used by every
# learned method, so it is not a per-method tuning dimension.
PRIOR_STRENGTHS = (1.0, 10.0, 100.0)
# Per-block step selection criterion: the manuscript functional of the block,
# F1 / F2 / F3, on predictions made before each task ran (out of sample).
BLOCK_CRITERION = {"processing": "f1", "waiting": "f2", "quality": "f3"}
# The per-block grid extends past 0.05 because the non-accelerated methods
# selected the upper edge of STEP_GRID on the processing and waiting blocks.
BLOCK_STEP_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5, 1.0)

TIME_METRICS = ("mean_latency", "p50_latency", "p95_latency", "p99_latency", "mean_wait", "p95_wait")
RATE_METRICS = ("throughput", "goodput")


def latency_profiles(benchmark: str, variant: str) -> list[list] | None:
    """Per-model [name, TTFT, decode rate] of a sensitivity variant; None is the default."""
    if variant == "default":
        return None
    models = sorted(real_benchmarks.load(benchmark)[1]["model"].unique())
    if variant in ("slow", "fast"):
        speed = np.full(len(models), 0.5 if variant == "slow" else 2.0)
    elif variant.startswith("shuffle"):
        rng = np.random.default_rng(int(variant.removeprefix("shuffle")))
        speed = np.exp(rng.uniform(np.log(0.5), np.log(2.0), len(models)))
    else:
        raise ValueError(f"unknown latency variant: {variant}")
    base = [table_oracle.DEFAULT_LATENCY_PROFILES[m] for m in models]
    return [[m, ttft / s, rate * s] for m, (ttft, rate), s in zip(models, base, speed.tolist())]


@lru_cache(maxsize=4)
def _table(benchmark: str, seed: int, cfg_json: str) -> table_oracle.OutcomeTable:
    cfg = table_oracle.TableConfig(**json.loads(cfg_json))
    return table_oracle.OutcomeTable(benchmark, cfg, seed)


@lru_cache(maxsize=2)
def _prompts(benchmark: str, dim: int) -> PromptFeatures:
    return PromptFeatures(benchmark, dim)


def run_one(job: dict) -> dict:
    """One (system, regime, seed, split, price weight, step) run."""
    table_cfg = job["table_cfg"]
    table = _table(job["benchmark"], job["seed"], json.dumps(table_cfg, sort_keys=True))
    prompts = _prompts(job["benchmark"], job["embedding_dim"])
    slots, rho = job["slots"], job["rho"]
    agents = table_oracle.make_table_agents(table, slots)
    tasks = table_oracle.load_table_tasks(
        table, agents, job["seed"], rho, limit=job["tasks"], split=job["split"],
        per_domain=job["per_domain"], prompt_features=prompts,
        load_reference=LOAD_REFERENCE[job["regime"]])
    architecture, optimizer, _ = SYSTEMS[job["system"]]
    weights = (1.10, 0.40, job["latency_weight"], job["price_weight"])
    # Per-model step sizes h_p, h_w, h_q, as the manuscript allows; blocks not
    # listed in "blocks" are left to RLS alone.
    block_steps = job.get("block_steps")
    # The manuscript requires 0 < eta < mu with mu = omega * lambda_2(L_av);
    # with the ridge that supplies strong convexity on the consensus subspace
    # (PAPER_EXPERIMENTS.md) mu is computed by theory_parameters.  The
    # simulated protocol takes mu - eta = mu / 2; the optimizer default 0.01
    # exceeds mu = 0.002 here and would put eta below zero.
    loss = PaperLossConfig()
    omega = ExperimentConfig().network_omega[0]
    mu_minus_eta = 0.5 * theory_parameters(1.0, omega, job["controllers"], loss.lambda_reg_p)["mu"]
    base_optimizer = replace(OptimizerConfig(), step_size=job["step_size"], mu_minus_eta=mu_minus_eta)
    per_block = {}
    if block_steps:
        per_block = dict(
            optimizer_configs=ModelOptimizerConfigs(**{
                name: replace(base_optimizer, step_size=float(step))
                for name, step in block_steps.items()}),
            optimizer_blocks=tuple(job.get("blocks", ("processing", "waiting", "quality"))),
        )
    config = ExperimentConfig(
        **per_block,
        controllers=job["controllers"], agents=len(agents), rho=rho, seed=job["seed"],
        architecture=architecture, optimizer=optimizer,
        use_learning=optimizer != "none", use_rls=optimizer != "none",
        base_score_weights=weights, embedding_dim=job["embedding_dim"],
        optimizer_config=base_optimizer,
        controller_assignment=job["controller_assignment"],
        max_processing_units=job["max_processing_units"],
        max_waiting_units=job["max_waiting_units"],
        profile_prior=True,
        rls_prior_strength=job.get("rls_prior_strength", 1.0),
        pair_quality_feature=job.get("pair_quality_feature", False),
        diffusion_mixing=job.get("diffusion_mixing", 0.0),
        specialization=job.get("specialization", 0.5),
        buffer_order=BUFFER_ORDER.get(job["system"], "weighted"),
    )
    oracle = table_oracle.TableOracle(table, agents, prompts, prior=PRIOR_ROUTER.get(job["system"], "knn"))
    summary = ControlledReplayExperiment(tasks, config, agents=agents, oracle=oracle).run()
    unit = table.cfg.time_unit_seconds
    for key in TIME_METRICS:
        summary[f"{key}_s"] = summary[key] * unit
    for key in RATE_METRICS:
        summary[f"{key}_per_s"] = summary[key] / unit
    summary.update({k: job[k] for k in ("system", "regime", "seed", "split", "price_weight", "step_size")})
    summary["block_steps"] = json.dumps(block_steps) if block_steps else ""
    summary["blocks"] = ",".join(job.get("blocks", ())) if block_steps else ""
    return summary


def frontier_metrics(points: pd.DataFrame, lo: float, hi: float, best_q: float, best_c: float) -> dict:
    """Cost--quality frontier of one system from its price-weight sweep.

    ``aiq`` is RouterBench's average improvement in quality: the area under
    the non-decreasing hull of the system's points over the pool's cost range,
    divided by that range.  The hull is a step function from the system's own
    points and is zero below its cheapest point, so a system is not credited
    for costs it cannot reach.  ``quality_at_best_cost`` and
    ``cost_at_best_quality`` compare with the best single model at iso-cost and
    iso-quality, the two readings RouteLLM reports.
    """
    ordered = points.sort_values("mean_cost")
    costs = ordered["mean_cost"].to_numpy()
    quality = np.maximum.accumulate(ordered["mean_quality"].to_numpy())
    grid = np.linspace(lo, hi, 512)
    curve = np.array([quality[costs <= c].max() if np.any(costs <= c) else 0.0 for c in grid])
    reach = quality >= best_q
    return {
        "aiq": float(np.trapezoid(curve, grid) / max(hi - lo, 1e-12)),
        "quality_at_best_cost": float(quality[costs <= best_c].max()) if np.any(costs <= best_c) else float("nan"),
        "cost_at_best_quality": float(costs[reach].min()) if np.any(reach) else float("nan"),
        "max_quality": float(quality.max()),
        "min_cost": float(costs.min()),
    }


def _jobs(args, regime: str, split: str, seeds, systems, price_weights, steps,
          prior_strength: float = 1.0) -> list[dict]:
    slots, rho, latency_weight = REGIMES[regime]
    base = dict(
        latency_weight=latency_weight, pair_quality_feature=args.pair_quality_feature,
        diffusion_mixing=args.diffusion_mixing, specialization=args.specialization,
        benchmark=args.benchmark, tasks=args.tasks, per_domain=args.per_domain,
        controllers=args.controllers, embedding_dim=args.embedding_dim,
        controller_assignment=args.controller_assignment,
        table_cfg=dict(time_unit_seconds=args.time_unit_seconds, arrival_process=args.arrival,
                       slo_scale=args.slo_scale,
                       **({"latency_profiles": args.latency_profiles} if args.latency_profiles else {})),
        regime=regime, slots=slots, rho=rho, split=split,
        max_processing_units=args.max_processing_units, max_waiting_units=args.max_waiting_units,
        rls_prior_strength=prior_strength,
    )
    jobs = []
    for seed in seeds:
        for system in systems:
            sweeps = SYSTEMS[system][2]
            for weight in (price_weights if sweeps else (DEFAULT_PRICE_WEIGHT,)):
                for step in steps.get(system, (OptimizerConfig().step_size,)):
                    jobs.append({**base, "seed": seed, "system": system,
                                 "price_weight": weight, "step_size": step})
    return jobs


def _execute(jobs: list[dict], workers: int) -> pd.DataFrame:
    if workers <= 1:
        return pd.DataFrame([run_one(job) for job in jobs])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.DataFrame(list(pool.map(run_one, jobs, chunksize=1)))


ENDPOINTS = ["mean_quality", "mean_cost", "quality_gap_recovered", "cost_ratio_vs_best_single",
             "deadline_hit_rate", "goodput_per_s", "p95_latency_s", "p99_latency_s",
             "true_quality_rmse", "own_quality_rmse", "true_processing_rmse"]


def aggregate(runs: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Frontier, summary and paired tests from per-run rows."""
    runs = runs.assign(variant=runs["system"] + "@" + runs["price_weight"].map(lambda w: f"{w:g}"))

    # Frontier per (regime, system, seed), then means over seeds.
    frontier_rows = []
    for (regime, system, seed), group in runs.groupby(["regime", "system", "seed"]):
        ref = group.iloc[0]
        lo = min(ref["cheapest_single_cost_usd"], ref["oracle_cost_usd"], group["mean_cost"].min())
        hi = max(ref["best_single_cost_usd"], group["mean_cost"].max())
        frontier_rows.append({"regime": regime, "system": system, "seed": seed,
                              **frontier_metrics(group, lo, hi, ref["best_single_quality"],
                                                 ref["best_single_cost_usd"])})
    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(out / "frontier_by_seed.csv", index=False)
    frontier.groupby(["regime", "system"]).agg(["mean", "std"]).to_csv(out / "frontier_summary.csv")

    reference_cols = [c for c in runs.columns if c.startswith(("best_single", "oracle_", "cheapest_single",
                                                                "random_quality", "reference_router"))]
    summary = runs.groupby(["regime", "variant"])[ENDPOINTS + reference_cols].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")

    # Paired tests at the default operating point: every learned system
    # against every other system, within each regime.
    default = runs[np.isclose(runs["price_weight"], DEFAULT_PRICE_WEIGHT)]
    for regime, group in default.groupby("regime"):
        group = group.assign(variant=group["system"])
        for reference in [s for s in ("aspsa", "aspsa_mlp", "zo_gt", "local_rls") if s in set(group["system"])]:
            _paired_significance(group, reference, out / f"significance_{regime}_{reference}.csv",
                                 tuple(ENDPOINTS))
    return frontier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", default="llmrouterbench", choices=sorted(real_benchmarks.BUILDERS))
    parser.add_argument("--output-dir", type=Path, default=Path("real_benchmark_outputs"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--validation-seeds", type=int, default=2)
    parser.add_argument("--tasks", type=int, default=1000)
    parser.add_argument("--per-domain", type=int, default=None)
    parser.add_argument("--controllers", type=int, default=3)
    parser.add_argument("--controller-assignment", default="load", choices=("load", "domain", "skewed"))
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--arrival", default="azure_code", choices=("poisson", "azure_code", "azure_conv"))
    parser.add_argument("--slo-scale", type=float, default=3.0)
    # 30 s puts the median measured service time near 0.5 units and the
    # 40-minute maximum near 80, inside --max-processing-units.
    parser.add_argument("--time-unit-seconds", type=float, default=30.0)
    parser.add_argument("--max-processing-units", type=float, default=100.0)
    parser.add_argument("--max-waiting-units", type=float, default=200.0)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pair-feature", dest="pair_quality_feature", action="store_false",
                        help="controllers without the task--agent offline quality coordinate")
    parser.add_argument("--diffusion-mixing", type=float, default=0.0,
                        help="adapt-then-combine mixing between ring neighbours (0 = off)")
    parser.add_argument("--specialization", type=float, default=0.5)
    parser.add_argument("--common-steps", type=float, nargs="*", default=None,
                        help="equal-step comparison: run every learned method with this same step on "
                             "all three blocks, one run per step, no selection")
    parser.add_argument("--per-block-steps", action="store_true",
                        help="select h_p, h_w, h_q separately, each by its own out-of-sample functional")
    parser.add_argument("--fixed-block-step", type=float, default=None,
                        help="transfer protocol: every learned method uses this step on all three blocks, "
                             "fixed in advance; no step is selected on this benchmark")
    parser.add_argument("--price-weights", type=float, nargs="+", default=list(PRICE_WEIGHTS),
                        help="price weights of the test runs; must include the default operating point")
    parser.add_argument("--latency-variant", default="default", choices=LATENCY_VARIANTS,
                        help="latency-profile sensitivity variant")
    parser.add_argument("--aggregate-from", type=Path, nargs="+", default=None,
                        help="skip running; merge runs_*.csv of these directories (later ones "
                             "replace the (regime, system) rows they contain) and aggregate into --output-dir")
    args = parser.parse_args()
    out = args.output_dir
    if args.aggregate_from:
        aggregate_main(args.aggregate_from, out)
        return
    if not any(np.isclose(w, DEFAULT_PRICE_WEIGHT) for w in args.price_weights):
        parser.error(f"--price-weights must include the default operating point {DEFAULT_PRICE_WEIGHT}")
    out.mkdir(parents=True, exist_ok=True)

    # Warm the caches once so worker processes only read parquet files.
    real_benchmarks.load(args.benchmark)
    PromptFeatures(args.benchmark, args.embedding_dim)
    if any(SYSTEMS[s][0] == "mlp_router" or s in PRIOR_ROUTER for s in args.systems):
        mlp_quality(args.benchmark)
    args.latency_profiles = latency_profiles(args.benchmark, args.latency_variant)

    learned = [s for s in args.systems if s in LEARNED]
    uses_rls = [s for s in args.systems if s in LEARNED or s == "local_rls"]
    selected_prior: dict[str, float] = {}
    val_seeds = [1000 + i for i in range(args.validation_seeds)]
    test_seeds = list(range(1, args.seeds + 1))
    selected: dict[str, dict[str, float]] = {}
    all_frames = []
    for regime in args.regimes:
        prior = 1.0
        if uses_rls:
            prior_runs = pd.concat([
                _execute(_jobs(args, regime, "validation", val_seeds, ["local_rls"],
                               (DEFAULT_PRICE_WEIGHT,), {}, strength), args.workers)
                .assign(rls_prior_strength=strength)
                for strength in PRIOR_STRENGTHS], ignore_index=True)
            prior_runs.to_csv(out / f"prior_tuning_{regime}.csv", index=False)
            prior = float(prior_runs.groupby("rls_prior_strength")["true_quality_rmse"].mean().idxmin())
            selected_prior[regime] = prior
        if learned and args.common_steps:
            # Equal-step comparison: every method, every block, the same
            # nominal step h, no selection.  The manuscript's A-SPSA keeps h
            # fixed; the others decay it as h / (k + 10)^0.602.
            base_jobs = _jobs(args, regime, "test", test_seeds, learned, (DEFAULT_PRICE_WEIGHT,), {}, prior)
            common_jobs = [{**job, "block_steps": {b: step for b in BLOCK_CRITERION}}
                           for job in base_jobs for step in args.common_steps]
            frame = _execute(common_jobs, args.workers)
            frame["common_step"] = frame["block_steps"].map(lambda s: json.loads(s)["processing"])
            all_frames.append(frame)
            frame.to_csv(out / f"runs_{regime}.csv", index=False)
            continue
        if learned and args.fixed_block_step is not None:
            # Transfer protocol: the step is fixed before the run and shared by
            # every learned method and block; only the RLS prior, shared by all
            # methods, is chosen on this benchmark's validation split.
            steps = {b: args.fixed_block_step for b in BLOCK_CRITERION}
            test_jobs = [{**job, "block_steps": steps} if job["system"] in learned else job
                         for job in _jobs(args, regime, "test", test_seeds, args.systems,
                                          args.price_weights, {}, prior)]
            frame = _execute(test_jobs, args.workers)
            all_frames.append(frame)
            frame.to_csv(out / f"runs_{regime}.csv", index=False)
            continue
        # Step selection on validation: one criterion for every method.
        if learned and args.per_block_steps:
            # One step per prediction model, h_p, h_w, h_q, as the manuscript
            # allows.  Each block is tuned alone (the others left to RLS), the
            # procedure of the simulated protocol, and selected by its own
            # functional evaluated out of sample: the loss of the prediction
            # made before the task ran against the outcome it then had.
            base_jobs = _jobs(args, regime, "validation", val_seeds, learned,
                              (DEFAULT_PRICE_WEIGHT,), {}, prior)
            tuning_jobs = [{**job, "blocks": (block,), "block_steps": {block: step}}
                           for job in base_jobs for block in BLOCK_CRITERION for step in BLOCK_STEP_GRID]
            tuning = _execute(tuning_jobs, args.workers)
            tuning.to_csv(out / f"tuning_{regime}.csv", index=False)
            chosen: dict[str, dict[str, float]] = {}
            for system in learned:
                chosen[system] = {}
                for block, criterion in BLOCK_CRITERION.items():
                    part = tuning[(tuning["system"] == system) & (tuning["blocks"] == block)]
                    steps_of = part["block_steps"].map(lambda s: json.loads(s)[block])
                    score = part.assign(step=steps_of).groupby("step")[criterion].mean()
                    chosen[system][block] = float(score.idxmin())
            selected[regime] = chosen
            test_jobs = [{**job, "block_steps": chosen[job["system"]]} if job["system"] in chosen else job
                         for job in _jobs(args, regime, "test", test_seeds, args.systems, args.price_weights,
                                          {}, prior)]
            frame = _execute(test_jobs, args.workers)
            all_frames.append(frame)
            frame.to_csv(out / f"runs_{regime}.csv", index=False)
            continue
        if learned:
            tuning = _execute(_jobs(args, regime, "validation", val_seeds, learned,
                                    (DEFAULT_PRICE_WEIGHT,), {s: STEP_GRID for s in learned}, prior),
                              args.workers)
            tuning.to_csv(out / f"tuning_{regime}.csv", index=False)
            score = tuning.groupby(["system", "step_size"])["true_quality_rmse"].mean()
            selected[regime] = {s: float(score.loc[s].idxmin()) for s in learned}
        steps = {s: (selected.get(regime, {}).get(s, OptimizerConfig().step_size),) for s in learned}
        frame = _execute(_jobs(args, regime, "test", test_seeds, args.systems, args.price_weights, steps,
                               prior), args.workers)
        all_frames.append(frame)
        frame.to_csv(out / f"runs_{regime}.csv", index=False)

    runs = pd.concat(all_frames, ignore_index=True)
    # Several rows per (system, seed) in the equal-step mode: the frontier and
    # the paired tests of ``aggregate`` assume one, so they are skipped there.
    frontier = aggregate(runs, out) if not args.common_steps else pd.DataFrame()
    manifest = {
        "code_revision": _git_revision(),
        "benchmark": real_benchmarks.REVISIONS[args.benchmark],
        "provenance": real_benchmarks.PROVENANCE[args.benchmark],
        "latency_profiles_source": table_oracle.LATENCY_PROFILES_SOURCE,
        "latency_variant": args.latency_variant,
        "latency_profiles": ({name: [ttft, rate] for name, ttft, rate in args.latency_profiles}
                             if args.latency_profiles else
                             {m: table_oracle.DEFAULT_LATENCY_PROFILES[m]
                              for m in sorted(real_benchmarks.load(args.benchmark)[1]["model"].unique())}),
        "fixed_block_step": args.fixed_block_step,
        "prefill_tokens_per_second": table_oracle.PREFILL_TOKENS_PER_SECOND,
        "encoder": ENCODER,
        "regimes": {r: {"slots_per_model": REGIMES[r][0], "rho": REGIMES[r][1],
                        "latency_weight": REGIMES[r][2], "load_reference": LOAD_REFERENCE[r]}
                    for r in args.regimes},
        "price_weights": args.price_weights,
        "step_grid": STEP_GRID,
        "step_selection": "min mean validation true_quality_rmse, shared criterion for every method",
        "profile_prior": "processing and quality predictors start from the train-split per-domain profile",
        "selected_steps": selected,
        "per_block_steps": bool(args.per_block_steps),
        "common_steps": args.common_steps,
        "block_criterion": BLOCK_CRITERION if args.per_block_steps else None,
        "block_step_grid": BLOCK_STEP_GRID if args.per_block_steps else None,
        "prior_strengths": PRIOR_STRENGTHS,
        "selected_rls_prior_strength": selected_prior,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "test_seeds": test_seeds, "validation_seeds": val_seeds,
    }
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if not frontier.empty:
        print(frontier.groupby(["regime", "system"])[["aiq", "max_quality", "quality_at_best_cost"]].mean().round(4))


def merge_runs(directories: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    """Combine ``runs_*.csv`` of several output directories.

    A later directory replaces every (regime, system) it contains, so a rerun
    of some systems supersedes their rows from an earlier run while the rows
    of systems it did not rerun are kept.  The returned provenance says which
    directory each (regime, system) came from.
    """
    merged: dict[tuple[str, str], pd.DataFrame] = {}
    source: dict[tuple[str, str], str] = {}
    for directory in directories:
        for path in sorted(directory.glob("runs_*.csv")):
            frame = pd.read_csv(path)
            for key, group in frame.groupby(["regime", "system"]):
                merged[key] = group
                source[key] = str(path)
    provenance = [{"regime": k[0], "system": k[1], "runs_file": v} for k, v in sorted(source.items())]
    return pd.concat(merged.values(), ignore_index=True), provenance


def aggregate_main(directories: list[Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runs, provenance = merge_runs(directories)
    runs.to_csv(out / "runs_merged.csv", index=False)
    frontier = aggregate(runs, out)
    manifests = {str(d): json.loads((d / "experiment_manifest.json").read_text(encoding="utf-8"))
                 for d in directories if (d / "experiment_manifest.json").exists()}
    (out / "merge_manifest.json").write_text(json.dumps(
        {"sources": provenance, "source_manifests": manifests}, indent=2, default=str), encoding="utf-8")
    print(frontier.groupby(["regime", "system"])[["aiq", "max_quality", "quality_at_best_cost"]].mean().round(4))


if __name__ == "__main__":
    main()
