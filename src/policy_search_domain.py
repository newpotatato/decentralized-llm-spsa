"""Distributed tuning of a per-domain scheduling policy: 30 parameters.

The five-weight policy of ``policy_search_distributed`` turned out to be too
small a search space to separate the optimizers: every method beat the hand-set
defaults by roughly the same 20-23% and none was significantly better than
another.  Five parameters give an accelerated method nothing to exploit.

Here each domain gets its own five weights, so a security incident and a
frontend question need not trade deadline risk against suitability in the same
proportion.  That is 6 x 5 = 30 parameters per site, a defensible model of a
real scheduler and a search space where the choice of optimizer can matter.

Objective is unchanged: deadline miss rate on the site's own traffic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .assignment_buffer import BufferPolicy, GlobalAssignmentBuffer
from .paper_experiment import (
    ControlledReplayExperiment, DOMAIN_ORDER, ExperimentConfig, ReplayTask,
    load_manifest_tasks, make_agent_pool,
)
from .paper_optimizers import DistributedZerothOrderOptimizer, OptimizerConfig
from .policy_search import DEFAULTS, LOWER, UPPER, SECONDARY
from .policy_search_distributed import _slice_for

DOMAINS = len(DOMAIN_ORDER)
FLAT_DEFAULTS = np.tile(DEFAULTS, DOMAINS)
FLAT_LOWER = np.tile(LOWER, DOMAINS)
FLAT_UPPER = np.tile(UPPER, DOMAINS)


def run_site(flat: np.ndarray, tasks: list[ReplayTask], seed: int, rho: float,
             agents: int, batch: int) -> dict:
    policy = BufferPolicy(matching="exact",
                          per_domain=np.asarray(flat, dtype=float).reshape(DOMAINS, 5))
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
    def __init__(self, site: int, tasks: list[ReplayTask], seeds: tuple[int, ...],
                 rho: float, agents: int, batch: int):
        self.controller_id = site
        self.tasks = tasks
        self.seeds = seeds
        self.rho = rho
        self.agents = agents
        self.batch = batch
        self.x = FLAT_DEFAULTS.copy()
        self.calls = 0

    def pack_parameters(self) -> np.ndarray:
        return self.x.copy()

    def unpack_parameters(self, vector: np.ndarray) -> None:
        self.x = self.project(vector)

    def project(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(vector, dtype=float), FLAT_LOWER, FLAT_UPPER)

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


def held_out(per_site: list[np.ndarray], manifest: Path, seeds: tuple[int, ...],
             sites: int, rho: float, agents: int, batch: int) -> dict:
    rows = []
    for seed in seeds:
        tasks = load_manifest_tasks(manifest, seed, rho, make_agent_pool(agents),
                                    None, split="test", batch_size=batch)
        for site in range(sites):
            slice_tasks = _slice_for(tasks, site, sites)
            if slice_tasks:
                rows.append(run_site(per_site[site], slice_tasks, seed, rho,
                                     agents, batch))
    out = {"deadline_hit_rate": float(np.mean([r["deadline_hit_rate"] for r in rows]))}
    out.update({k: float(np.mean([r[k] for r in rows])) for k in SECONDARY})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/it_routing_benchmark.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("policy_outputs_domain"))
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--optimizer-seeds", type=int, nargs="+", default=[3, 11, 29, 47, 61])
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["aspsa", "spsa", "zo_gt", "zo_pgd", "kw"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds, test_seeds = (131,), (11, 23, 37, 41, 53)
    train_tasks = load_manifest_tasks(args.manifest, train_seeds[0], args.rho,
                                      make_agent_pool(args.agents), args.limit,
                                      split="train", batch_size=args.batch)
    slices = [_slice_for(train_tasks, s, args.sites) for s in range(args.sites)]
    print("tasks per site:", [len(s) for s in slices], flush=True)
    print(f"parameters per site: {FLAT_DEFAULTS.size}", flush=True)

    rows = []
    base = held_out([FLAT_DEFAULTS] * args.sites, args.manifest, test_seeds,
                    args.sites, args.rho, args.agents, args.batch)
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
            metrics = held_out([s.pack_parameters() for s in sites], args.manifest,
                               test_seeds, args.sites, args.rho, args.agents, args.batch)
            rows.append(dict(method=method, optimizer_seed=opt_seed,
                             queries=sum(s.calls for s in sites), **metrics))
            print(f"{method} seed={opt_seed}: DHR {metrics['deadline_hit_rate']:.4f}",
                  flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "policy_domain_raw.csv", index=False)
    summary = (frame[frame.method != "hand-set defaults"]
               .groupby("method", as_index=False)
               .agg(dhr_mean=("deadline_hit_rate", "mean"),
                    dhr_std=("deadline_hit_rate", "std"),
                    dhr_min=("deadline_hit_rate", "min"),
                    dhr_max=("deadline_hit_rate", "max"),
                    quality=("mean_quality", "mean"),
                    queries=("queries", "mean"))
               .sort_values("dhr_mean", ascending=False))
    summary.to_csv(args.output_dir / "policy_domain_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print()
    print(f"hand-set defaults DHR = {base['deadline_hit_rate']:.4f}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
