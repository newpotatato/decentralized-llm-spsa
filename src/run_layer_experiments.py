"""Experiments for the three-layer hypothesis.

  RLS        gives the least-squares solution and tracks drift
  A-SPSA     closes the gap from least squares to the actual F1/F2/F3 minimum,
             which has no closed form because those losses are robust and
             asymmetric
  ATC        synchronises controllers by combining estimates, not by adding a
             consensus gradient

Every cell reports both the in-window objective and the held-out error against
the generative model, because the two move in opposite directions when a method
overfits.

Run as ``python -m src.run_layer_experiments --output-dir layer_outputs``.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .paper_experiment import (
    ControlledReplayExperiment, ExperimentConfig, load_manifest_tasks, make_agent_pool,
)
from .paper_optimizers import (
    ModelOptimizerConfigs, OptimizerConfig, ring_laplacian_spectrum,
)

METRICS = (
    "phi_p_auc", "phi_w_auc", "phi_q_auc",
    "data_p_final", "data_w_final", "data_q_final",
    "disagreement_p_auc", "disagreement_w_auc", "disagreement_q_auc",
    "true_processing_rmse", "true_quality_rmse",
    "mean_quality", "deadline_hit_rate", "p95_wait",
)


def one(manifest: Path, controllers: int, agents: int, epsilon: float,
        method: str, seed: int, step: float, sync: int) -> dict:
    oc = OptimizerConfig(step_size=step, perturbation=0.06, consensus_weight=0.002,
                         gamma0=1.0, mu_minus_eta=0.001)
    cfg = ExperimentConfig(
        controllers=controllers, agents=agents, rho=0.849, sync_interval=sync,
        seed=seed, architecture="fixed_replay", optimizer=method,
        use_learning=method != "none", use_rls=method != "none",
        diffusion_mixing=epsilon,
        optimizer_configs=ModelOptimizerConfigs(processing=oc, waiting=oc, quality=oc),
    )
    tasks = load_manifest_tasks(manifest, seed, cfg.rho, make_agent_pool(agents),
                                None, split="test")
    summary = ControlledReplayExperiment(tasks, cfg).run()
    row = {k: float(summary[k]) for k in METRICS}
    row.update(controllers=controllers, agents=agents, epsilon=epsilon,
               method=method, seed=seed, step=step, sync=sync)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/it_routing_benchmark.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("layer_outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spectrum = pd.DataFrame([
        dict(controllers=m, lambda_2=ring_laplacian_spectrum(m)[0],
             lambda_max=ring_laplacian_spectrum(m)[1],
             topology="complete graph" if m == 3 else "ring")
        for m in (3, 4, 5, 6, 7, 9)
    ])
    spectrum["laplacian_condition"] = spectrum.lambda_max / spectrum.lambda_2
    spectrum.to_csv(args.output_dir / "ring_spectrum.csv", index=False)
    print(spectrum.to_string(index=False), flush=True)

    rows = []
    grid = list(itertools.product(
        (3, 5, 7),                                   # controllers
        (0.0, 0.05, 0.10, 0.20, 1.0 / 3.0),          # ATC mixing
        (("local_rls", 0.0), ("spsa", 0.05), ("aspsa", 0.05)),
    ))
    for index, (controllers, epsilon, (method, step)) in enumerate(grid, 1):
        for seed in args.seeds:
            rows.append(one(args.manifest, controllers, 12, epsilon, method,
                            seed, step, 5))
        print(f"[{index}/{len(grid)}] m={controllers} eps={epsilon:.3f} {method}",
              flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "layers_raw.csv", index=False)
    summary = frame.groupby(["controllers", "epsilon", "method"], as_index=False)[
        list(METRICS)].mean()
    summary.to_csv(args.output_dir / "layers_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
