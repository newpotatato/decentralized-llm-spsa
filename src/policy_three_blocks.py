"""The manuscript's three-block architecture, moved to the scheduler policy.

The paper is built on three prediction models with three losses F1/F2/F3 and
three network objectives Phi_p, Phi_w, Phi_q, synchronised across controllers.
That structure is sound; what did not work was the *thing* being optimised.
Fitting linear predictors has a closed form, so recursive least squares does it
better and the zeroth-order layer contributed nothing measurable.

Here the same three-block architecture is kept and applied to the scheduling
policy instead, where no closed form exists:

  block p   which agent runs it   objective: mean service time
  block w   which task goes first objective: p95 waiting time
  block q   which agent runs it   objective: 1 - mean quality

Service time is latency minus waiting, deliberately: an earlier version used
mean latency for block p, but latency *contains* waiting, so blocks p and w were
pulling on the same quantity in opposite directions and the split was not a
split at all.  Service time depends only on which agent was chosen, waiting only
on the order tasks were released in.

Each block owns three parameters, each has its own objective, each site owns its
own copy, and the blocks are synchronised across sites exactly as before.  So the
paper keeps its three functionals, its per-controller parameters and its
consensus layer; only the object of optimisation changes to one where a
zeroth-order method is the appropriate tool rather than a redundant one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .assignment_buffer import BufferPolicy, GlobalAssignmentBuffer
from .paper_experiment import (
    ControlledReplayExperiment, ExperimentConfig, ReplayTask,
    load_manifest_tasks, make_agent_pool,
)
from .paper_optimizers import DistributedZerothOrderOptimizer, OptimizerConfig
from .policy_search_distributed import _slice_for

#            p-block                       w-block                    q-block
# deadline gap, budget overrun, w_p | hotness, wait pressure, slope | w_q, w_r, w_c
DEFAULTS = np.array([0.30, 0.08, -0.16,
                     0.18, 0.12, 3.00,
                     1.10, 0.40, -0.08])
LOWER = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.25, 0.0, 0.0, -2.0])
UPPER = np.array([3.0, 3.0, 0.0, 3.0, 3.0, 12.0, 4.0, 3.0, 0.0])
BLOCKS = (("processing", slice(0, 3)), ("waiting", slice(3, 6)), ("quality", slice(6, 9)))
OBJECTIVE_KEY = {"processing": "mean_service", "waiting": "p95_wait",
                 "quality": "mean_quality"}


def run_site(flat: np.ndarray, tasks: list[ReplayTask], seed: int, rho: float,
             agents: int, batch: int) -> dict:
    """One simulation with this policy: three blocks act on one scheduler."""
    v = np.asarray(flat, dtype=float)
    policy = BufferPolicy(matching="exact", deadline_gap_penalty=float(v[0]),
                          budget_overrun_penalty=float(v[1]),
                          hotness_weight=float(v[3]), wait_pressure_weight=float(v[4]),
                          hotness_slope=float(v[5]))
    original = ControlledReplayExperiment.__init__

    def patched(self, task_list, config):
        original(self, task_list, config)
        self.buffer = GlobalAssignmentBuffer(policy)

    ControlledReplayExperiment.__init__ = patched
    try:
        cfg = ExperimentConfig(
            controllers=1, agents=agents, rho=rho, sync_interval=5, seed=seed,
            architecture="global_exact", optimizer="local_rls",
            arrival_batch_size=batch,
            base_score_weights=(float(v[6]), float(v[7]), float(v[2]), float(v[8])),
        )
        return ControlledReplayExperiment(tasks, cfg).run()
    finally:
        ControlledReplayExperiment.__init__ = original


class PolicySite:
    """One site, exposing the same interface the prediction controllers do."""

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

    def parameter_blocks(self):
        return BLOCKS

    def _measure(self, vector: np.ndarray) -> dict:
        self.calls += 1
        weights = self.project(vector)
        runs = [run_site(weights, self.tasks, seed, self.rho, self.agents, self.batch)
                for seed in self.seeds]
        out = {k: float(np.mean([r[k] for r in runs])) for k in
               ("mean_latency", "mean_wait", "p95_wait", "mean_quality",
                "deadline_hit_rate", "throughput", "mean_cost", "jain_fairness")}
        # Service time is what the choice of agent controls; waiting is what the
        # order of release controls.  Keeping them separate is what makes the
        # three blocks three different problems.
        out["mean_service"] = out["mean_latency"] - out["mean_wait"]
        return out

    def objective_block(self, vector: np.ndarray, block_name: str) -> float:
        m = self._measure(vector)
        if block_name == "processing":
            return m["mean_service"]
        if block_name == "waiting":
            return m["p95_wait"]
        if block_name == "quality":
            return 1.0 - m["mean_quality"]
        raise ValueError(block_name)

    def objective(self, vector: np.ndarray) -> float:
        m = self._measure(vector)
        return m["mean_service"] + m["p95_wait"] + (1.0 - m["mean_quality"])

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
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {"phi_p": mean("mean_latency") - mean("mean_wait"),
            "phi_w": mean("p95_wait"),
            "phi_q": 1.0 - mean("mean_quality"),
            "deadline_hit_rate": mean("deadline_hit_rate"),
            "throughput": mean("throughput"), "mean_cost": mean("mean_cost")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/it_routing_benchmark.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("policy_three_outputs"))
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[131, 149, 163, 181])
    parser.add_argument("--optimizer-seeds", type=int, nargs="+",
                        default=[3, 11, 29, 47, 61, 83, 97, 113, 127])
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["aspsa", "spsa", "zo_gt", "zo_pgd"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds, test_seeds = tuple(args.train_seeds), (11, 23, 37, 41, 53)
    tasks = load_manifest_tasks(args.manifest, train_seeds[0], args.rho,
                                make_agent_pool(args.agents), args.limit,
                                split="train", batch_size=args.batch)
    slices = [_slice_for(tasks, s, args.sites) for s in range(args.sites)]
    print(f"tasks per site: {[len(s) for s in slices]}", flush=True)
    print(f"blocks: {[(n, sl.stop - sl.start) for n, sl in BLOCKS]}", flush=True)

    rows = []
    base = held_out([DEFAULTS] * args.sites, args.manifest, test_seeds, args.sites,
                    args.rho, args.agents, args.batch)
    rows.append(dict(method="hand-set defaults", optimizer_seed=-1, queries=0, **base))
    print("hand-set defaults: " + " ".join(f"{k}={base[k]:.4f}"
                                           for k in ("phi_p", "phi_w", "phi_q")), flush=True)

    for method in args.methods:
        for opt_seed in args.optimizer_seeds:
            sites = [PolicySite(s, slices[s], train_seeds, args.rho, args.agents,
                                args.batch) for s in range(args.sites)]
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
            print(f"{method} seed={opt_seed}: " + " ".join(
                f"{k}={metrics[k]:.4f}" for k in ("phi_p", "phi_w", "phi_q")), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "three_blocks_raw.csv", index=False)
    summary = (frame[frame.method != "hand-set defaults"]
               .groupby("method", as_index=False)
               .agg(phi_p=("phi_p", "mean"), phi_w=("phi_w", "mean"),
                    phi_q=("phi_q", "mean"), dhr=("deadline_hit_rate", "mean"),
                    queries=("queries", "mean")))
    summary.to_csv(args.output_dir / "three_blocks_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print()
    print("hand-set: " + " ".join(f"{k}={base[k]:.4f}" for k in ("phi_p", "phi_w", "phi_q")))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
