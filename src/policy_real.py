"""Distributed tuning of the scheduler policy on measured LLM outcomes.

This is the protocol of ``policy_search_drift`` without drift -- the setting in
which A-SPSA ranked first on the simulated benchmark (RAZVILKI R4.11, R5.11) --
transferred unchanged to replayed real outcomes (``table_oracle``).  Nothing in
it is tuned for the transfer:

* five sites, each owning its own domain group of the stream;
* five buffer-priority weights per site, same defaults and bounds;
* objective: deadline miss rate, 1 - DHR, on the site's own traffic;
* ring consensus, omega = 0.3, step 0.2, perturbation 0.15, 30 rounds;
* nine optimizer seeds, five held-out test seeds, paired Wilcoxon with Holm.

Stated before the run, as in R4.11: acceleration accumulates oracle noise, so
A-SPSA should gain most from averaging the oracle over several streams, and the
low-noise condition is the one in which it can win.  Both conditions are run so
the prediction can fail.

The priority weights reach the outcome only through queueing and an argmax
matching, so there is no gradient and no closed form: zeroth-order search is
the applicable tool here, unlike for the linear predictors.
"""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")  # see run_real_benchmark

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import table_oracle
from .assignment_buffer import BufferPolicy, GlobalAssignmentBuffer
from .paper_experiment import ControlledReplayExperiment, ExperimentConfig
from .paper_optimizers import DistributedZerothOrderOptimizer, OptimizerConfig
from .policy_search import DEFAULTS, LOWER, UPPER, WEIGHT_NAMES
from .prompt_embeddings import PromptFeatures
from .run_paper_experiments import _git_revision, _paired_significance

SECONDARY = ("mean_quality", "p95_wait", "throughput", "mean_cost", "goodput")
TABLE_SEED = 0  # latency jitter of the replayed table; fixed across all runs
# Classic SLO-aware scheduling rules evaluated in the same buffer on the same
# test streams: they need no tuning, so they cost no oracle calls.
RULES = ("edf", "llf", "sjf")
# Priority families searched by the optimizers: names, start, lower and upper
# bounds.  "weighted5" is the manuscript's buffer.  "weighted6" adds deadline
# urgency, so earliest-deadline-first is a member (large weight) and the
# search, which starts from the same hand-set policy with urgency off, runs in
# a family that contains the best classic rule.
FAMILIES = {
    "weighted5": (WEIGHT_NAMES, DEFAULTS, LOWER, UPPER),
    "weighted6": (WEIGHT_NAMES + ("deadline_urgency_weight",), np.append(DEFAULTS, 0.0),
                  np.append(LOWER, 0.0), np.append(UPPER, 3.0)),
}


def family(cfg: dict) -> tuple:
    return FAMILIES[cfg.get("family", "weighted5")]


@lru_cache(maxsize=2)
def _world(benchmark: str, slots: int, embedding_dim: int):
    table = table_oracle.OutcomeTable(benchmark, table_oracle.TableConfig(time_unit_seconds=30.0),
                                      TABLE_SEED)
    agents = table_oracle.make_table_agents(table, slots)
    prompts = PromptFeatures(benchmark, embedding_dim)
    return table, agents, prompts


def site_domains(site: int, sites: int, domains: int = 6) -> set[int]:
    """Same partition as ``policy_search_distributed._slice_for``."""
    return {d for d in range(domains) if (d * sites) // domains == site}


def simulate(weights: np.ndarray, cfg: dict, site: int, seed: int, split: str,
             order: str = "weighted") -> dict:
    """One run of the full system on a site's stream with these weights (or a rule)."""
    table, agents, prompts = _world(cfg["benchmark"], cfg["slots"], cfg["embedding_dim"])
    tasks = table_oracle.load_table_tasks(
        table, agents, seed, cfg["rho"], limit=cfg["tasks_per_site"], split=split,
        prompt_features=prompts, domains=site_domains(site, cfg["sites"]))
    config = ExperimentConfig(
        controllers=1, agents=len(agents), rho=cfg["rho"], sync_interval=5, seed=seed,
        architecture="global_exact", optimizer="local_rls", embedding_dim=cfg["embedding_dim"],
        max_processing_units=100.0, max_waiting_units=200.0, profile_prior=True,
        rls_prior_strength=100.0, pair_quality_feature=True)
    experiment = ControlledReplayExperiment(
        tasks, config, agents=agents, oracle=table_oracle.TableOracle(table, agents, prompts))
    experiment.buffer = GlobalAssignmentBuffer(BufferPolicy(
        matching="exact", order=order, **dict(zip(family(cfg)[0], (float(v) for v in weights)))))
    return experiment.run()


class Site:
    """One site's priority weights, with the interface the optimizers expect."""

    def __init__(self, site: int, cfg: dict, oracle_seeds: tuple[int, ...]):
        self.controller_id = site
        self.cfg = cfg
        self.seeds = oracle_seeds
        _, start, self.lower, self.upper = family(cfg)
        self.x = start.copy()
        self.calls = 0
        self.round = 0
        self.history: list[tuple[int, float]] = []   # (round, oracle value) per call

    def pack_parameters(self) -> np.ndarray:
        return self.x.copy()

    def unpack_parameters(self, vector: np.ndarray) -> None:
        self.x = self.project(vector)

    def project(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(vector, dtype=float), self.lower, self.upper)

    def objective(self, vector: np.ndarray) -> float:
        self.calls += 1
        weights = self.project(vector)
        value = float(np.mean([
            1.0 - simulate(weights, self.cfg, self.controller_id, seed, "train")["deadline_hit_rate"]
            for seed in self.seeds]))
        self.history.append((self.round, value))
        return value

    def constraint(self, vector: np.ndarray, budget: float):
        return float(np.mean(vector) - budget), np.ones_like(vector) / len(vector)


def held_out(weights_per_site: list[np.ndarray], cfg: dict, split: str = "test",
             seeds: list[int] | None = None, order: str = "weighted") -> dict:
    rows = [simulate(weights_per_site[site], cfg, site, seed, split, order)
            for seed in (seeds or cfg["test_seeds"]) for site in range(cfg["sites"])]
    out = {"deadline_hit_rate": float(np.mean([r["deadline_hit_rate"] for r in rows]))}
    out.update({k: float(np.mean([r[k] for r in rows])) for k in SECONDARY})
    return out


def run_job(job: dict) -> dict:
    cfg = job["cfg"]
    trace: list[float] = []
    if job["method"] == "hand-set defaults" or job["method"] in RULES:
        found = [family(cfg)[1].copy() for _ in range(cfg["sites"])]
        calls = 0
    else:
        sites = [Site(s, cfg, tuple(job["oracle_seeds"])) for s in range(cfg["sites"])]
        optimizer = DistributedZerothOrderOptimizer(
            job["method"],
            OptimizerConfig(step_size=job.get("step_size", 0.2), perturbation=0.15,
                            consensus_weight=cfg["omega"], gamma0=1.0, mu_minus_eta=0.001),
            seed=job["optimizer_seed"])
        for step in range(1, cfg["rounds"] + 1):
            for site in sites:
                site.round = step
            optimizer.step(sites, step)
        found = [s.pack_parameters() for s in sites]
        calls = sum(s.calls for s in sites)
        # The objective as the optimizer saw it, per round: mean over sites of
        # the oracle values queried in that round (at the probe points).
        for step in range(1, cfg["rounds"] + 1):
            values = [v for s in sites for r, v in s.history if r == step]
            trace.append(float(np.mean(values)) if values else float("nan"))
    metrics = held_out(found, cfg, job.get("split", "test"), job.get("eval_seeds"),
                       order=job["method"] if job["method"] in RULES else "weighted")
    return {"noise": job["noise"], "method": job["method"], "optimizer_seed": job["optimizer_seed"],
            "step_size": job.get("step_size", 0.2), "split": job.get("split", "test"),
            "oracle_calls": calls, "weights": json.dumps([w.round(4).tolist() for w in found]),
            "trace": json.dumps(trace), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", default="llmrouterbench")
    parser.add_argument("--output-dir", type=Path, default=Path("policy_real_outputs"))
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--slots", type=int, default=1)
    # rho = 2.0 as in the simulated protocol: the priority weights matter only
    # when a queue forms (at 0.85 the deadline-hit rate moves by 0.005 across
    # the weight settings probed, at 1.5 by 0.067).
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--tasks-per-site", type=int, default=60)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--methods", nargs="*", default=["aspsa", "spsa", "zo_gt", "zo_pgd"])
    parser.add_argument("--rules", nargs="*", default=[], choices=RULES,
                        help="also evaluate these classic scheduling rules on the test streams")
    parser.add_argument("--family", default="weighted5", choices=list(FAMILIES),
                        help="priority family searched by the optimizers")
    parser.add_argument("--optimizer-seeds", type=int, nargs="+",
                        default=[3, 11, 29, 47, 61, 83, 97, 113, 127])
    parser.add_argument("--noise-levels", nargs="+", default=["noisy", "low_noise"],
                        choices=["noisy", "low_noise"])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--tune-steps", type=float, nargs="*", default=None,
                        help="select each method's step on the validation split from this shared "
                             "grid before the test run (robustness check of the fixed-step protocol)")
    parser.add_argument("--tune-seeds", type=int, nargs="+", default=[3, 11])
    parser.add_argument("--tuning-extra", type=Path, nargs="*", default=None,
                        help="step_tuning.csv files of earlier tuning on other grid points; the step "
                             "is then selected over the union of the grids")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cfg = dict(benchmark=args.benchmark, sites=args.sites, rounds=args.rounds, slots=args.slots,
               rho=args.rho, tasks_per_site=args.tasks_per_site, embedding_dim=args.embedding_dim,
               omega=args.omega, test_seeds=[11, 23, 37, 41, 53])
    if args.family != "weighted5":
        cfg["family"] = args.family      # older manifests have no key: weighted5
    oracle_seeds = {"noisy": [131], "low_noise": [131, 149, 163, 181]}
    _world(args.benchmark, args.slots, args.embedding_dim)  # warm the parquet caches

    # Non-accelerated methods decay the step as h / (k + 10)^0.602; A-SPSA keeps
    # h fixed, as the manuscript specifies.  With one shared h the accelerated
    # method therefore takes larger steps, so --tune-steps re-runs the
    # comparison with each method's h chosen on validation from a shared grid.
    selected_steps = {(n, m): 0.2 for n in args.noise_levels for m in args.methods}
    if args.tune_steps:
        tuning = [{"cfg": cfg, "noise": noise, "method": method, "optimizer_seed": seed,
                   "oracle_seeds": oracle_seeds[noise], "step_size": step,
                   "split": "validation", "eval_seeds": [101, 103, 107]}
                  for noise in args.noise_levels for method in args.methods
                  for step in args.tune_steps for seed in args.tune_seeds]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tuned = pd.DataFrame(list(pool.map(run_job, tuning, chunksize=1)))
        tuned.to_csv(out / "step_tuning.csv", index=False)
        # Earlier tuning of the same protocol on other grid points (for
        # instance a grid extended past its upper edge) joins the selection,
        # so the step is chosen over the union of all grids evaluated.
        for extra in args.tuning_extra or []:
            tuned = pd.concat([tuned, pd.read_csv(extra)], ignore_index=True)
        tuned = tuned[tuned["method"].isin(args.methods) & tuned["noise"].isin(args.noise_levels)]
        best = tuned.groupby(["noise", "method", "step_size"])["deadline_hit_rate"].mean()
        for noise in args.noise_levels:
            for method in args.methods:
                selected_steps[(noise, method)] = float(best.loc[(noise, method)].idxmax())
    jobs = [{"cfg": cfg, "noise": "reference", "method": method, "optimizer_seed": -1, "oracle_seeds": []}
            for method in ["hand-set defaults", *args.rules]]
    for noise in args.noise_levels:
        for method in args.methods:
            for seed in args.optimizer_seeds:
                jobs.append({"cfg": cfg, "noise": noise, "method": method,
                             "optimizer_seed": seed, "oracle_seeds": oracle_seeds[noise],
                             "step_size": selected_steps[(noise, method)]})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs, chunksize=1))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "policy_real_raw.csv", index=False)

    summary = (frame.groupby(["noise", "method"], as_index=False)
               .agg(dhr_mean=("deadline_hit_rate", "mean"), dhr_std=("deadline_hit_rate", "std"),
                    dhr_min=("deadline_hit_rate", "min"), dhr_max=("deadline_hit_rate", "max"),
                    quality=("mean_quality", "mean"), p95_wait=("p95_wait", "mean"),
                    cost=("mean_cost", "mean"), oracle_calls=("oracle_calls", "mean"))
               .sort_values(["noise", "dhr_mean"], ascending=[True, False]))
    summary.to_csv(out / "policy_real_summary.csv", index=False)
    for noise in args.noise_levels:
        part = frame[frame["noise"] == noise].assign(variant=lambda f: f["method"],
                                                     seed=lambda f: f["optimizer_seed"])
        if "aspsa" in set(part["method"]):
            _paired_significance(part, "aspsa", out / f"significance_{noise}.csv",
                                 ("deadline_hit_rate",))
    manifest = {"code_revision": _git_revision(), "protocol": __doc__, "config": cfg,
                "oracle_seeds": oracle_seeds, "args": {k: str(v) for k, v in vars(args).items()},
                "selected_steps": {f"{n}/{m}": v for (n, m), v in selected_steps.items()},
                "weights": list(family(cfg)[0]), "defaults": family(cfg)[1].tolist(),
                "bounds": [family(cfg)[2].tolist(), family(cfg)[3].tolist()]}
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
