"""Tune the scheduler's priority weights by zeroth-order search.

Why this is a different problem from fitting the predictors.  The predictors are
linear and their losses are differentiable, so recursive least squares is simply
the better tool and a zeroth-order method has nothing to add.  The priority
weights of the assignment buffer are the opposite case:

  * they act on the outcome only through the whole scheduler - queueing,
    matching, deadlines - so no closed form exists;
  * the matching contains an argmax, so the map is not differentiable;
  * the objective is an operational rate measured on realised outcomes, so the
    only way to evaluate a candidate is to run the system.

That is exactly the setting zeroth-order stochastic approximation is for.

The objective is the deadline miss rate, 1 - DHR.  It is a single prespecified
operational quantity rather than an arbitrary weighted scalarisation; quality,
waiting and throughput are reported alongside so a win cannot hide a trade.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .assignment_buffer import BufferPolicy, GlobalAssignmentBuffer
from .paper_experiment import (
    ControlledReplayExperiment, ExperimentConfig, load_manifest_tasks, make_agent_pool,
)
from .paper_optimizers import DistributedZerothOrderOptimizer, OptimizerConfig

# hotness, wait pressure, deadline gap penalty, budget overrun penalty, slope
WEIGHT_NAMES = ("hotness_weight", "wait_pressure_weight", "deadline_gap_penalty",
                "budget_overrun_penalty", "hotness_slope")
DEFAULTS = np.array([0.18, 0.12, 0.30, 0.08, 3.00])
LOWER = np.array([0.0, 0.0, 0.0, 0.0, 0.25])
UPPER = np.array([3.0, 3.0, 3.0, 3.0, 12.0])
SECONDARY = ("mean_quality", "p95_wait", "throughput", "mean_cost", "jain_fairness")


def simulate(weights: np.ndarray, manifest: Path, seed: int, rho: float,
             controllers: int, agents: int, limit: int | None,
             batch: int) -> dict:
    """Run the event-driven system once with these priority weights."""
    policy = BufferPolicy(matching="exact",
                          **dict(zip(WEIGHT_NAMES, (float(v) for v in weights))))
    original = ControlledReplayExperiment.__init__

    def patched(self, tasks, config):
        original(self, tasks, config)
        self.buffer = GlobalAssignmentBuffer(policy)

    ControlledReplayExperiment.__init__ = patched
    try:
        cfg = ExperimentConfig(
            controllers=controllers, agents=agents, rho=rho, sync_interval=5,
            seed=seed, architecture="global_exact", optimizer="local_rls",
            diffusion_mixing=0.1, task_limit=limit, arrival_batch_size=batch,
        )
        tasks = load_manifest_tasks(manifest, seed, rho, make_agent_pool(agents),
                                    limit, split="train", batch_size=batch)
        return ControlledReplayExperiment(tasks, cfg).run()
    finally:
        ControlledReplayExperiment.__init__ = original


class PolicyObjective:
    """A single controller-side view of the weight vector for the optimizers.

    The optimizers expect pack/unpack/project/objective, the same interface the
    prediction blocks expose, so the identical implementations are reused
    without a special case.
    """

    def __init__(self, manifest: Path, seeds: tuple[int, ...], rho: float,
                 controllers: int, agents: int, limit: int, batch: int,
                 controller_id: int = 0):
        self.manifest = manifest
        self.seeds = seeds
        self.rho = rho
        self.controllers = controllers
        self.agents = agents
        self.limit = limit
        self.batch = batch
        self.controller_id = controller_id
        self.x = DEFAULTS.copy()
        self.calls = 0

    def pack_parameters(self) -> np.ndarray:
        return self.x.copy()

    def unpack_parameters(self, vector: np.ndarray) -> None:
        self.x = self.project(vector)

    def project(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(vector, dtype=float), LOWER, UPPER)

    def objective(self, vector: np.ndarray) -> float:
        """Deadline miss rate over the training seeds."""
        self.calls += 1
        weights = self.project(vector)
        misses = []
        for seed in self.seeds:
            summary = simulate(weights, self.manifest, seed, self.rho,
                               self.controllers, self.agents, self.limit, self.batch)
            misses.append(1.0 - float(summary["deadline_hit_rate"]))
        return float(np.mean(misses))

    def constraint(self, vector: np.ndarray, budget: float):
        return float(np.mean(vector) - budget), np.ones_like(vector) / len(vector)


def evaluate(weights: np.ndarray, manifest: Path, seeds: tuple[int, ...], rho: float,
             controllers: int, agents: int, batch: int) -> dict:
    """Held-out evaluation of a weight vector on the test split."""
    rows = []
    original = ControlledReplayExperiment.__init__
    policy = BufferPolicy(matching="exact",
                          **dict(zip(WEIGHT_NAMES, (float(v) for v in weights))))

    def patched(self, tasks, config):
        original(self, tasks, config)
        self.buffer = GlobalAssignmentBuffer(policy)

    ControlledReplayExperiment.__init__ = patched
    try:
        for seed in seeds:
            cfg = ExperimentConfig(
                controllers=controllers, agents=agents, rho=rho, sync_interval=5,
                seed=seed, architecture="global_exact", optimizer="local_rls",
                diffusion_mixing=0.1, arrival_batch_size=batch,
            )
            tasks = load_manifest_tasks(manifest, seed, rho, make_agent_pool(agents),
                                        None, split="test", batch_size=batch)
            rows.append(ControlledReplayExperiment(tasks, cfg).run())
    finally:
        ControlledReplayExperiment.__init__ = original
    out = {"deadline_hit_rate": float(np.mean([r["deadline_hit_rate"] for r in rows]))}
    out.update({k: float(np.mean([r[k] for r in rows])) for k in SECONDARY})
    return out


def random_search(objective: PolicyObjective, budget: int, seed: int) -> np.ndarray:
    """Equal-budget baseline: the honest competitor for any black-box method."""
    rng = np.random.default_rng(seed)
    best, best_value = DEFAULTS.copy(), objective.objective(DEFAULTS)
    for _ in range(budget - 1):
        candidate = rng.uniform(LOWER, UPPER)
        value = objective.objective(candidate)
        if value < best_value:
            best, best_value = candidate, value
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/it_routing_benchmark.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("policy_outputs"))
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--controllers", type=int, default=5)
    parser.add_argument("--agents", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = (131, 149)
    test_seeds = (11, 23, 37, 41, 53)
    methods = ("aspsa", "spsa", "kw", "zo_pgd", "sp_gt", "zo_gt", "pd_2pt")

    rows = []
    baseline = evaluate(DEFAULTS, args.manifest, test_seeds, args.rho,
                        args.controllers, args.agents, args.batch)
    rows.append(dict(method="hand-set defaults", queries=0,
                     weights=np.round(DEFAULTS, 4).tolist(), **baseline))
    print(f"hand-set defaults: DHR {baseline['deadline_hit_rate']:.4f}", flush=True)

    budget = None
    for method in methods:
        objective = PolicyObjective(args.manifest, train_seeds, args.rho,
                                    args.controllers, args.agents, args.limit, args.batch)
        optimizer = DistributedZerothOrderOptimizer(
            method,
            OptimizerConfig(step_size=0.2, perturbation=0.15, consensus_weight=0.0,
                            gamma0=1.0, mu_minus_eta=0.001),
            seed=7,
        )
        for step in range(1, args.rounds + 1):
            optimizer.step([objective], step)
        found = objective.pack_parameters()
        held = evaluate(found, args.manifest, test_seeds, args.rho,
                        args.controllers, args.agents, args.batch)
        rows.append(dict(method=method, queries=objective.calls,
                         weights=np.round(found, 4).tolist(), **held))
        if method == "aspsa":
            budget = objective.calls
        print(f"{method}: {objective.calls} queries, DHR {held['deadline_hit_rate']:.4f}, "
              f"weights {np.round(found, 3).tolist()}", flush=True)

    objective = PolicyObjective(args.manifest, train_seeds, args.rho,
                                args.controllers, args.agents, args.limit, args.batch)
    found = random_search(objective, budget or 50, seed=5)
    held = evaluate(found, args.manifest, test_seeds, args.rho,
                    args.controllers, args.agents, args.batch)
    rows.append(dict(method="random search", queries=objective.calls,
                     weights=np.round(found, 4).tolist(), **held))
    print(f"random search: {objective.calls} queries, DHR {held['deadline_hit_rate']:.4f}",
          flush=True)

    frame = pd.DataFrame(rows).sort_values("deadline_hit_rate", ascending=False)
    frame.to_csv(args.output_dir / "policy_search.csv", index=False)
    pd.set_option("display.width", 220)
    print()
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
