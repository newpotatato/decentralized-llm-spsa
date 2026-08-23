"""Distributed tuning of scheduler priority weights.

Each site runs its own instance of the system on its own slice of the workload
and tunes its own priority weights.  The slices differ - a site that mostly sees
ML/LLM traffic has a different service-time profile from one that sees frontend
traffic - so the local optima differ, and the question the network layer has to
answer is whether sharing across sites beats tuning each site alone.

That is the setting the manuscript's synchronisation is designed for, and unlike
the prediction blocks it is a setting where no first-order method applies at all:
the priority weights reach the outcome only through queueing, matching and
deadlines, the matching contains an argmax, and the objective is an operational
rate measured on realised outcomes.

Objective: deadline miss rate on the site's own traffic, 1 - DHR.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .assignment_buffer import BufferPolicy, GlobalAssignmentBuffer
from .paper_experiment import (
    ControlledReplayExperiment, DOMAIN_INDEX, ExperimentConfig, ReplayTask,
    load_manifest_tasks, make_agent_pool,
)
from .paper_optimizers import DistributedZerothOrderOptimizer, OptimizerConfig
from .policy_search import DEFAULTS, LOWER, UPPER, WEIGHT_NAMES, SECONDARY

METHODS = ("aspsa", "spsa", "kw", "zo_pgd", "sp_gt", "zo_gt", "pd_2pt")
ABLATIONS = ("aspsa_no_consensus", "aspsa_no_momentum", "aspsa_fixed_beta")


def _slice_for(tasks: list[ReplayTask], site: int, sites: int) -> list[ReplayTask]:
    """Domain-partitioned traffic: site s owns its own group of domains."""
    domains = len(DOMAIN_INDEX)
    chosen = [t for t in tasks if (t.domain * sites) // domains == site]
    return [
        ReplayTask(index, t.source_id, t.domain, t.skill_vector, t.hardness,
                   t.urgency, t.prompt_chars, t.has_code, t.arrival, t.deadline)
        for index, t in enumerate(chosen)
    ]


def run_site(weights: np.ndarray, tasks: list[ReplayTask], seed: int, rho: float,
             agents: int, batch: int) -> dict:
    policy = BufferPolicy(matching="exact",
                          **dict(zip(WEIGHT_NAMES, (float(v) for v in weights))))
    original = ControlledReplayExperiment.__init__

    def patched(self, task_list, config):
        original(self, task_list, config)
        self.buffer = GlobalAssignmentBuffer(policy)

    ControlledReplayExperiment.__init__ = patched
    try:
        cfg = ExperimentConfig(controllers=1, agents=agents, rho=rho, sync_interval=5,
                               seed=seed, architecture="global_exact",
                               optimizer="local_rls", arrival_batch_size=batch)
        return ControlledReplayExperiment(tasks, cfg).run()
    finally:
        ControlledReplayExperiment.__init__ = original


class Site:
    """One site's weight vector, exposed with the interface the optimizers use."""

    def __init__(self, site: int, tasks: list[ReplayTask], seeds: tuple[int, ...],
                 rho: float, agents: int, batch: int):
        self.controller_id = site
        self.tasks = tasks
        self.seeds = seeds
        self.rho = rho
        self.agents = agents
        self.batch = batch
        self.x = DEFAULTS.copy()
        self.calls = 0

    def pack_parameters(self) -> np.ndarray:
        return self.x.copy()

    def unpack_parameters(self, vector: np.ndarray) -> None:
        self.x = self.project(vector)

    def project(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(vector, dtype=float), LOWER, UPPER)

    def objective(self, vector: np.ndarray) -> float:
        self.calls += 1
        weights = self.project(vector)
        return float(np.mean([
            1.0 - run_site(weights, self.tasks, seed, self.rho,
                           self.agents, self.batch)["deadline_hit_rate"]
            for seed in self.seeds
        ]))

    def constraint(self, vector: np.ndarray, budget: float):
        return float(np.mean(vector) - budget), np.ones_like(vector) / len(vector)


def held_out(weights_per_site: list[np.ndarray], manifest: Path,
             seeds: tuple[int, ...], sites: int, rho: float, agents: int,
             batch: int) -> dict:
    rows = []
    for seed in seeds:
        tasks = load_manifest_tasks(manifest, seed, rho, make_agent_pool(agents),
                                    None, split="test", batch_size=batch)
        for site in range(sites):
            slice_tasks = _slice_for(tasks, site, sites)
            if slice_tasks:
                rows.append(run_site(weights_per_site[site], slice_tasks, seed,
                                     rho, agents, batch))
    out = {"deadline_hit_rate": float(np.mean([r["deadline_hit_rate"] for r in rows]))}
    out.update({k: float(np.mean([r[k] for r in rows])) for k in SECONDARY})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/it_routing_benchmark.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("policy_outputs"))
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--optimizer-seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHODS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds, test_seeds = (131, 149), (11, 23, 37, 41, 53)
    train_tasks = load_manifest_tasks(args.manifest, train_seeds[0], args.rho,
                                      make_agent_pool(args.agents), args.limit,
                                      split="train", batch_size=args.batch)
    slices = [_slice_for(train_tasks, s, args.sites) for s in range(args.sites)]
    print("tasks per site:", [len(s) for s in slices], flush=True)

    rows = []
    base = held_out([DEFAULTS] * args.sites, args.manifest, test_seeds, args.sites,
                    args.rho, args.agents, args.batch)
    rows.append(dict(method="hand-set defaults", optimizer_seed=-1, queries=0, **base))
    print(f"hand-set defaults: DHR {base['deadline_hit_rate']:.4f}", flush=True)

    for method in args.methods:
        for opt_seed in args.optimizer_seeds:
            sites = [Site(s, slices[s], train_seeds, args.rho, args.agents, args.batch)
                     for s in range(args.sites)]
            optimizer = DistributedZerothOrderOptimizer(
                method,
                OptimizerConfig(step_size=0.2, perturbation=0.15,
                                consensus_weight=args.omega, gamma0=1.0,
                                mu_minus_eta=0.001),
                seed=opt_seed,
            )
            for step in range(1, args.rounds + 1):
                optimizer.step(sites, step)
            found = [s.pack_parameters() for s in sites]
            metrics = held_out(found, args.manifest, test_seeds, args.sites,
                               args.rho, args.agents, args.batch)
            rows.append(dict(method=method, optimizer_seed=opt_seed,
                             queries=sum(s.calls for s in sites), **metrics))
            print(f"{method} seed={opt_seed}: DHR {metrics['deadline_hit_rate']:.4f}",
                  flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "policy_distributed_raw.csv", index=False)
    summary = (frame[frame.method != "hand-set defaults"]
               .groupby("method", as_index=False)
               .agg(dhr_mean=("deadline_hit_rate", "mean"),
                    dhr_std=("deadline_hit_rate", "std"),
                    dhr_min=("deadline_hit_rate", "min"),
                    dhr_max=("deadline_hit_rate", "max"),
                    quality=("mean_quality", "mean"),
                    p95_wait=("p95_wait", "mean"),
                    queries=("queries", "mean"))
               .sort_values("dhr_mean", ascending=False))
    summary.to_csv(args.output_dir / "policy_distributed_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print()
    print(f"hand-set defaults DHR = {base['deadline_hit_rate']:.4f}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
