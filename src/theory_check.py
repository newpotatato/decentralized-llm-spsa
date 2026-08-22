"""Check the manuscript's assumptions against what the implementation does.

Run as ``python -m src.theory_check``.  Every line is either a derivation from
the manuscript or a measurement on a real run, so a claim in the paper can be
traced to a number here.  The corresponding regression tests live in
``tests/test_protocol_fairness.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .paper_experiment import (
    ControlledReplayExperiment, ExperimentConfig, RAW_FEATURE_NORM,
    load_manifest_tasks, make_agent_pool,
)
from .paper_losses import PaperLossConfig, strong_convexity_floor
from .paper_optimizers import (
    lipschitz_constants, ring_laplacian_spectrum, theory_parameters,
)


def _rule(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def report(manifest: Path, controllers: int = 3, agents: int = 12) -> dict:
    cfg = ExperimentConfig(
        controllers=controllers, agents=agents, rho=0.849, sync_interval=5,
        seed=11, architecture="fixed_replay", optimizer="local_rls",
    )
    pool = make_agent_pool(agents)
    tasks = load_manifest_tasks(manifest, cfg.seed, cfg.rho, pool, None, split="test")
    experiment = ControlledReplayExperiment(tasks, cfg)
    seen: list[np.ndarray] = []
    original = experiment._prediction_features

    def spy(task, now):
        vector = original(task, now)
        seen.append(vector.copy())
        return vector

    experiment._prediction_features = spy
    summary = experiment.run()
    controller = experiment.controllers[0]
    design = np.asarray(seen)
    x_max = float(np.linalg.norm(design, axis=1).max())
    loss = cfg.loss

    _rule("A1  communication graph")
    lambda_2, lambda_max = ring_laplacian_spectrum(controllers)
    print(f"  lambda_2 = {lambda_2:.4f} > 0 (connected), lambda_max = {lambda_max:.4f}")
    print("  B_tau is a fixed ring, so B_tau == B_av, sigma_B = 0, sigma_zeta = 0.")
    print("  The time-varying and noisy-communication half of A1 is not exercised.")
    if controllers == 3:
        print("  At m = 3 the ring is the complete graph, so lambda_2 == lambda_max")
        print("  and the spectral gap carries no information at this size.")

    _rule("A2  bounded successive oracle-noise differences")
    print("  Both evaluations of a round read the same frozen rolling window, so")
    print("  the sampling part of the oracle error cancels between them exactly.")

    _rule("A3  convexity and a common minimizer")
    block = dict(controller.parameter_blocks())["processing"]
    base = controller.pack_parameters()
    direction = np.zeros_like(base)
    direction[block] = 1.0
    worst = min(
        float(np.diff(np.array([
            controller.objective_block(base + t * direction, "processing")
            for t in np.linspace(-span, span, 200)
        ]), 2).min())
        for span in (5.0, 60.0)
    )
    print(f"  queried oracle, min second difference along a ray: {worst:+.3e}")
    print(f"  {'OK' if worst > -1e-9 else 'FAIL'}: the box projection is applied to the")
    print("  iterate, not inside the evaluation, so Y inherits the convexity of F.")
    print("  A common minimizer is NOT enforced anywhere and does not hold at the")
    print("  observed iterates; the controllers fit different rolling windows.")

    _rule("A4  Lipschitz constants")
    print(f"  measured X_max = {x_max:.4f} (the manuscript assumes 1)")
    declared = lipschitz_constants(loss, feature_norm=1.0)
    measured = lipschitz_constants(loss, feature_norm=x_max)
    for key in ("M_p", "M_w", "M_q"):
        flag = "OK" if measured[key] <= declared[key] + 1e-9 else "UNDERSTATED"
        print(f"    {key}: manuscript {declared[key]:8.2f}   at measured X {measured[key]:8.2f}   {flag}")

    _rule("A4b  where strong convexity actually comes from")
    print("  Hessian of Phi = blockdiag(grad^2 F_j) + omega * (L_av kron I).")
    print("  The second summand vanishes on the consensus subspace, so mu there")
    print("  is whatever the ridge supplies, not omega * lambda_2.")
    omega = cfg.network_omega[0]
    floors = strong_convexity_floor(loss)
    print(f"  omega * lambda_2 (manuscript mu) = {omega * lambda_2:.5f}")
    for name, value in floors.items():
        verdict = "OK" if value >= omega * lambda_2 else "BELOW the network modulus"
        print(f"    ridge[{name:11s}] = {value:<8g} {verdict}")
    shift = np.zeros_like(base)
    shift[block] = 1.0 / np.sqrt(block.stop - block.start)
    values = np.array([controller.objective_block(base + t * shift, "processing")
                       for t in (-1e-2, 0.0, 1e-2)])
    curvature = float((values[0] - 2 * values[1] + values[2]) / 1e-4)
    print(f"  measured curvature along a consensus shift: {curvature:+.5f}")

    _rule("A5  drift of the minimizers")
    series = np.array([r["local_p"] for r in experiment.round_records])
    if len(series) > 1:
        print(f"  rolling objective over {len(series)} rounds: "
              f"max step {np.abs(np.diff(series)).max():.4f}, "
              f"mean step {np.abs(np.diff(series)).mean():.4f}")
    print(f"  H_sync = {cfg.sync_interval} of a {cfg.history_size}-deep window is")
    print("  replaced per round, so the objective is genuinely nonstationary.")

    _rule("A6  independence of the random elements")
    print("  Delta comes from a Generator seeded independently of the replay, and")
    print("  the replay outcomes are a deterministic function of (task, agent,")
    print("  seed), so they cannot depend on Delta.")

    _rule("prescribed parameters, with the corrected modulus")
    ridges = {"processing": loss.lambda_reg_p, "waiting": loss.lambda_reg_w,
              "quality": loss.lambda_reg_q}
    keys = {"processing": "M_p", "waiting": "M_w", "quality": "M_q"}
    out = {}
    for name in ("processing", "waiting", "quality"):
        params = theory_parameters(declared[keys[name]], omega, controllers, ridges[name])
        out[name] = params
        print(f"  {name:11s} mu = {params['mu']:<8g} h_theory = {params['h_theory']:.6f}  "
              f"0 < eta < {params['eta_upper_bound']:g}")

    _rule("held-out accuracy of the fitted models")
    print(f"  true_processing_rmse = {summary['true_processing_rmse']:.4f}")
    print(f"  true_processing_mape = {100 * summary['true_processing_mape']:.2f}%")
    print(f"  true_quality_rmse    = {summary['true_quality_rmse']:.4f}")
    print(f"  worst_block_queries_per_parameter = "
          f"{summary['worst_block_queries_per_parameter']:.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/it_routing_benchmark.jsonl"))
    parser.add_argument("--controllers", type=int, default=3)
    parser.add_argument("--agents", type=int, default=12)
    args = parser.parse_args()
    report(args.manifest, args.controllers, args.agents)


if __name__ == "__main__":
    main()
