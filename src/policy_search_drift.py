"""Distributed policy tuning under a drifting workload and unreliable links.

Two regimes the manuscript's own assumptions describe and the implementation
never exercised.

Assumption 5 is about a moving optimum: the minimiser drifts by at most
``delta_theta`` per synchronisation round.  Every experiment so far ran on a
fixed task set, where the optimum barely moves, so the tracking half of the
analysis was never tested.  Here the offered load ramps and the traffic mix
shifts across rounds, which is what a serving system actually sees, and the best
priority weights move with it.  Momentum is classically a tracking device, so
this is the regime where acceleration should pay.

Assumption 1 allows a random, time-varying communication matrix with
``E||L_tau - L_av||^2 <= sigma_B^2`` and additive link noise with variance
``sigma_zeta^2``.  The implementation used a fixed ring, so both were zero.
Here each link is present with probability ``link_probability`` and the
parameters a neighbour reports are corrupted by Gaussian noise.

Objective is unchanged: deadline miss rate on the site's own traffic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .paper_experiment import load_manifest_tasks, make_agent_pool
from .paper_optimizers import DistributedZerothOrderOptimizer, OptimizerConfig
from .policy_search import DEFAULTS, LOWER, UPPER, SECONDARY
from .policy_search_distributed import Site, _slice_for, held_out, run_site

METHODS = ("aspsa", "spsa", "zo_gt", "zo_pgd", "kw")


class DriftingSite(Site):
    """A site whose workload moves between synchronisation rounds.

    ``round_index`` is advanced by the driver.  The evaluation window slides
    through the site's stream and the offered load ramps, so the minimiser of
    the local objective is a moving target rather than a fixed point.
    """

    def __init__(self, *args, window: int = 24, ramp: float = 1.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window
        self.ramp = ramp
        self.round_index = 0
        self.total_rounds = 1

    def _current(self) -> tuple[list, float]:
        span = max(len(self.tasks) - self.window, 1)
        progress = self.round_index / max(self.total_rounds - 1, 1)
        start = int(progress * span)
        rho = self.rho * (1.0 + self.ramp * progress)
        return self.tasks[start:start + self.window], rho

    def objective(self, vector: np.ndarray) -> float:
        self.calls += 1
        weights = self.project(vector)
        tasks, rho = self._current()
        if not tasks:
            return 0.0
        return float(np.mean([
            1.0 - run_site(weights, tasks, seed, rho, self.agents,
                           self.batch)["deadline_hit_rate"]
            for seed in self.seeds
        ]))


class UnreliableNetwork(DistributedZerothOrderOptimizer):
    """Random link presence and additive link noise, as Assumption 1 allows."""

    def __init__(self, *args, link_probability: float = 1.0,
                 link_noise: float = 0.0, network_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.link_probability = float(link_probability)
        self.link_noise = float(link_noise)
        self.network_rng = np.random.default_rng(network_seed)

    def _ring_average(self, vectors: list[np.ndarray]) -> list[np.ndarray]:
        count = len(vectors)
        if count <= 1:
            return [v.copy() for v in vectors]
        out = []
        for index, vector in enumerate(vectors):
            terms, weights = [vector], [1.0]
            for neighbour in ((index - 1) % count, (index + 1) % count):
                if self.network_rng.random() > self.link_probability:
                    continue
                reported = vectors[neighbour]
                if self.link_noise > 0.0:
                    reported = reported + self.network_rng.normal(
                        0.0, self.link_noise, size=reported.shape)
                terms.append(reported)
                weights.append(1.0)
            out.append(np.average(np.asarray(terms), axis=0,
                                  weights=np.asarray(weights)))
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/it_routing_benchmark.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("policy_outputs_drift"))
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--optimizer-seeds", type=int, nargs="+",
                        default=[3, 11, 29, 47, 61])
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHODS))
    parser.add_argument("--link-probability", type=float, default=1.0)
    parser.add_argument("--link-noise", type=float, default=0.0)
    parser.add_argument("--ramp", type=float, default=1.6)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[131],
                        help="seeds averaged inside ONE oracle call; more seeds "
                             "means a less noisy oracle, which is the condition "
                             "acceleration theory requires")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = tuple(args.train_seeds)
    test_seeds = (11, 23, 37, 41, 53)
    tasks = load_manifest_tasks(args.manifest, train_seeds[0], args.rho,
                                make_agent_pool(args.agents), args.limit,
                                split="train", batch_size=args.batch)
    slices = [_slice_for(tasks, s, args.sites) for s in range(args.sites)]
    print(f"tasks per site: {[len(s) for s in slices]}", flush=True)
    print(f"drift: window {args.window}, load ramp x{1 + args.ramp:.1f} over the run",
          flush=True)
    print(f"network: link probability {args.link_probability}, "
          f"link noise {args.link_noise}", flush=True)

    rows = []
    base = held_out([DEFAULTS] * args.sites, args.manifest, test_seeds, args.sites,
                    args.rho, args.agents, args.batch)
    rows.append(dict(method="hand-set defaults", optimizer_seed=-1, queries=0, **base))
    print(f"hand-set defaults: DHR {base['deadline_hit_rate']:.4f}", flush=True)

    for method in args.methods:
        for opt_seed in args.optimizer_seeds:
            sites = [DriftingSite(s, slices[s], train_seeds, args.rho, args.agents,
                                  args.batch, window=args.window, ramp=args.ramp)
                     for s in range(args.sites)]
            for site in sites:
                site.total_rounds = args.rounds
            optimizer = UnreliableNetwork(
                method,
                OptimizerConfig(step_size=0.2, perturbation=0.15,
                                consensus_weight=args.omega, gamma0=1.0,
                                mu_minus_eta=0.001),
                seed=opt_seed,
                link_probability=args.link_probability,
                link_noise=args.link_noise,
                network_seed=opt_seed + 1000,
            )
            for step in range(1, args.rounds + 1):
                for site in sites:
                    site.round_index = step - 1
                optimizer.step(sites, step)
            metrics = held_out([s.pack_parameters() for s in sites], args.manifest,
                               test_seeds, args.sites, args.rho, args.agents, args.batch)
            rows.append(dict(method=method, optimizer_seed=opt_seed,
                             queries=sum(s.calls for s in sites), **metrics))
            print(f"{method} seed={opt_seed}: DHR {metrics['deadline_hit_rate']:.4f}",
                  flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "drift_raw.csv", index=False)
    summary = (frame[frame.method != "hand-set defaults"]
               .groupby("method", as_index=False)
               .agg(dhr_mean=("deadline_hit_rate", "mean"),
                    dhr_std=("deadline_hit_rate", "std"),
                    dhr_min=("deadline_hit_rate", "min"),
                    dhr_max=("deadline_hit_rate", "max"),
                    queries=("queries", "mean"))
               .sort_values("dhr_mean", ascending=False))
    summary.to_csv(args.output_dir / "drift_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print()
    print(f"hand-set defaults DHR = {base['deadline_hit_rate']:.4f}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
