"""The manuscript's three functionals on the scheduler policy, measured outcomes.

The protocol of the simulated ``policy_three_blocks`` (RAZVILKI R5.10),
transferred unchanged to replayed real outcomes (``table_oracle``).  The
scheduler weights form three blocks, each owning one of the manuscript's three
functionals:

  block p   (deadline gap, budget overrun, latency weight)   mean service time
  block w   (hotness, wait pressure, hotness slope)           p95 waiting time
  block q   (quality, relevance and price weights)            1 - mean quality

Every block is its own zeroth-order problem with its own two-point estimate per
round, the blocks are synchronised across five sites exactly as the prediction
blocks of the manuscript are, and every site owns its own copy.  Service time
is latency minus waiting, so blocks p and w act on different quantities.

    python -m src.policy_real_three --output-dir real_benchmark_outputs/policy_real_three
"""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import table_oracle
from .assignment_buffer import BufferPolicy, GlobalAssignmentBuffer
from .paper_experiment import ControlledReplayExperiment, ExperimentConfig
from .paper_optimizers import DistributedZerothOrderOptimizer, OptimizerConfig
from .policy_real import _world, site_domains
from .policy_three_blocks import BLOCKS, DEFAULTS, LOWER, UPPER
from .run_paper_experiments import _git_revision, _paired_significance

OBJECTIVE = {"processing": "mean_service", "waiting": "p95_wait", "quality": "one_minus_quality"}


def simulate(vector: np.ndarray, cfg: dict, site: int, seed: int, split: str) -> dict:
    """One run of the full system on a site's stream with these nine weights."""
    v = np.clip(np.asarray(vector, dtype=float), LOWER, UPPER)
    table, agents, prompts = _world(cfg["benchmark"], cfg["slots"], cfg["embedding_dim"])
    tasks = table_oracle.load_table_tasks(
        table, agents, seed, cfg["rho"], limit=cfg["tasks_per_site"], split=split,
        prompt_features=prompts, domains=site_domains(site, cfg["sites"]))
    config = ExperimentConfig(
        controllers=1, agents=len(agents), rho=cfg["rho"], sync_interval=5, seed=seed,
        architecture="global_exact", optimizer="local_rls", embedding_dim=cfg["embedding_dim"],
        max_processing_units=100.0, max_waiting_units=200.0, profile_prior=True,
        rls_prior_strength=100.0, pair_quality_feature=True,
        base_score_weights=(float(v[6]), float(v[7]), float(v[2]), float(v[8])))
    experiment = ControlledReplayExperiment(
        tasks, config, agents=agents, oracle=table_oracle.TableOracle(table, agents, prompts))
    experiment.buffer = GlobalAssignmentBuffer(BufferPolicy(
        matching="exact", deadline_gap_penalty=float(v[0]), budget_overrun_penalty=float(v[1]),
        hotness_weight=float(v[3]), wait_pressure_weight=float(v[4]), hotness_slope=float(v[5])))
    s = experiment.run()
    return {"mean_service": s["mean_latency"] - s["mean_wait"], "p95_wait": s["p95_wait"],
            "one_minus_quality": 1.0 - s["mean_quality"], "deadline_hit_rate": s["deadline_hit_rate"],
            "goodput": s["goodput"], "mean_cost": s["mean_cost"], "mean_quality": s["mean_quality"]}


class Site:
    """One site's nine weights in three blocks, with the controller interface."""

    def __init__(self, site: int, cfg: dict, oracle_seeds: tuple[int, ...]):
        self.controller_id = site
        self.cfg = cfg
        self.seeds = oracle_seeds
        self.x = DEFAULTS.copy()
        self.calls = 0
        self.round = 0
        self.history: list[tuple[int, str, float]] = []

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
        runs = [simulate(vector, self.cfg, self.controller_id, seed, "train") for seed in self.seeds]
        return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}

    def objective_block(self, vector: np.ndarray, block_name: str) -> float:
        value = self._measure(vector)[OBJECTIVE[block_name]]
        self.history.append((self.round, block_name, value))
        return value

    def objective(self, vector: np.ndarray) -> float:
        m = self._measure(vector)
        return m["mean_service"] + m["p95_wait"] + m["one_minus_quality"]

    def constraint(self, vector: np.ndarray, budget: float):
        return float(np.mean(vector) - budget), np.ones_like(vector) / len(vector)


def held_out(per_site: list[np.ndarray], cfg: dict) -> dict:
    rows = [simulate(per_site[site], cfg, site, seed, "test")
            for seed in cfg["test_seeds"] for site in range(cfg["sites"])]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def run_job(job: dict) -> dict:
    cfg = job["cfg"]
    traces: dict[str, list[float]] = {name: [] for name, _ in BLOCKS}
    if job["method"] == "hand-set defaults":
        found, calls = [DEFAULTS.copy() for _ in range(cfg["sites"])], 0
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
        for name, _ in BLOCKS:
            for step in range(1, cfg["rounds"] + 1):
                values = [v for s in sites for r, b, v in s.history if r == step and b == name]
                traces[name].append(float(np.mean(values)) if values else float("nan"))
    metrics = held_out(found, cfg)
    return {"method": job["method"], "optimizer_seed": job["optimizer_seed"], "oracle_calls": calls,
            "weights": json.dumps([w.round(4).tolist() for w in found]),
            "traces": json.dumps(traces), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", default="llmrouterbench")
    parser.add_argument("--output-dir", type=Path, default=Path("real_benchmark_outputs/policy_real_three"))
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--tasks-per-site", type=int, default=60)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--methods", nargs="+", default=["aspsa", "spsa", "zo_gt", "zo_pgd"])
    parser.add_argument("--optimizer-seeds", type=int, nargs="+",
                        default=[3, 11, 29, 47, 61, 83, 97, 113, 127])
    parser.add_argument("--oracle-seeds", type=int, nargs="+", default=[131, 149, 163, 181])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(benchmark=args.benchmark, sites=args.sites, rounds=args.rounds, slots=args.slots,
               rho=args.rho, tasks_per_site=args.tasks_per_site, embedding_dim=args.embedding_dim,
               omega=args.omega, test_seeds=[11, 23, 37, 41, 53])
    _world(args.benchmark, args.slots, args.embedding_dim)
    jobs = [{"cfg": cfg, "method": "hand-set defaults", "optimizer_seed": -1, "oracle_seeds": []}]
    jobs += [{"cfg": cfg, "method": m, "optimizer_seed": s, "oracle_seeds": args.oracle_seeds}
             for m in args.methods for s in args.optimizer_seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        frame = pd.DataFrame(list(pool.map(run_job, jobs, chunksize=1)))
    frame.to_csv(out / "three_blocks_raw.csv", index=False)
    summary = frame.groupby("method")[["mean_service", "p95_wait", "one_minus_quality",
                                       "deadline_hit_rate", "goodput", "mean_cost"]].mean()
    summary.to_csv(out / "three_blocks_summary.csv")
    part = frame[frame["method"] != "hand-set defaults"].assign(
        variant=lambda f: f["method"], seed=lambda f: f["optimizer_seed"])
    if "aspsa" in set(part["method"]):
        _paired_significance(part, "aspsa", out / "significance.csv",
                             ("mean_service", "p95_wait", "one_minus_quality", "deadline_hit_rate", "goodput"))
    manifest = {"code_revision": _git_revision(), "protocol": __doc__, "config": cfg,
                "oracle_seeds": args.oracle_seeds, "defaults": DEFAULTS.tolist(),
                "bounds": [LOWER.tolist(), UPPER.tolist()],
                "blocks": {name: [sl.start, sl.stop] for name, sl in BLOCKS},
                "optimizer": {"step_size": 0.2, "perturbation": 0.15, "omega": args.omega,
                              "gamma0": 1.0, "mu_minus_eta": 0.001}}
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
