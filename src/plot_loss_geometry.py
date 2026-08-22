"""Three panels explaining why F1 is flat and F2/F3 are steep.

Panel A: F1 divides the error by the task's own duration, so the same absolute
         error costs far less on a long task.  F2 and F3 have no such division.
Panel B: the resulting curvature of each loss, which is what an optimizer feels.
Panel C: the block objectives as the optimizer actually queries them, on a real
         run, along a random direction in parameter space.

Run as ``python -m src.plot_loss_geometry``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paper_experiment import (
    ControlledReplayExperiment, ExperimentConfig, load_manifest_tasks, make_agent_pool,
)
from .paper_losses import LossObservation, PaperLossConfig, processing_loss, quality_loss, waiting_loss

C_P, C_W, C_Q = "#B0413E", "#2F6DA8", "#3E8A5B"


def _obs(processing=2.0, waiting=1.0, quality=0.6, hardness=0.6, relevance=0.5):
    return LossObservation(
        features=np.zeros(9), agent_id=0, processing_real=processing,
        waiting_real=waiting, quality_real=quality, relevance=relevance,
        hardness=hardness,
    )


def panel_a(ax, cfg: PaperLossConfig) -> None:
    """The same absolute error, on tasks of different length."""
    errors = np.linspace(0.0, 1.5, 400)
    for duration, style in ((1.0, "-"), (2.0, "--"), (4.0, ":")):
        o = _obs(processing=duration)
        target = duration / (1.0 + cfg.a_relevance_p * (1.0 - o.relevance))
        y = [processing_loss(target + e, o, cfg) for e in errors]
        ax.plot(errors, y, style, color=C_P, lw=2.2,
                label=f"task takes {duration:.0f} time units")

    marks = []
    for duration in (1.0, 2.0, 4.0):
        o = _obs(processing=duration)
        target = duration / (1.0 + cfg.a_relevance_p * (1.0 - o.relevance))
        marks.append(processing_loss(target + 1.0, o, cfg))
    ax.plot([1.0] * 3, marks, "o", color=C_P, ms=7, zorder=5)
    ax.annotate(
        f"one and the same error of 1 unit\ncosts {marks[0]/marks[2]:.0f}x less on the long task",
        xy=(1.0, marks[2]), xytext=(0.06, max(marks) * 0.80),
        arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2), fontsize=9.5, color="0.2",
    )
    ax.set_title("A.  F1 divides the error by the task's own duration",
                 fontsize=10.5, loc="left", weight="bold", pad=8)
    ax.set_xlabel("absolute prediction error (time units)")
    ax.set_ylabel("F1 loss")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)


def panel_b(ax, cfg: PaperLossConfig) -> None:
    """Curvature of each loss: how hard it pushes back on a wrong prediction."""
    durations = np.linspace(0.8, 4.5, 300)
    curvature_p = [
        1.0 / (cfg.epsilon_p + (d / (1.0 + cfg.a_relevance_p * 0.5))
               * (1.0 + cfg.a_hardness_p * 0.6)) ** 2
        for d in durations
    ]
    ax.plot(durations, curvature_p, color=C_P, lw=2.4, label="F1  processing time")
    ax.axhline(2.0 * cfg.wait_under_weight, color=C_W, lw=2.4, label="F2  waiting time")
    ax.axhline(2.0 * (1.0 + cfg.a_hardness_q * 0.6 + cfg.a_risk_q * 0.5),
               color=C_Q, lw=2.4, label="F3  response quality")

    mean_p = 0.145
    ax.axhline(mean_p, color=C_P, lw=1.1, ls="--", alpha=0.8)
    ax.annotate(f"F1 average over the real workload: {mean_p:.3f}",
                xy=(3.9, mean_p), xytext=(1.55, 0.42), fontsize=9.5, color=C_P,
                arrowprops=dict(arrowstyle="->", color=C_P, lw=1.0))
    ax.annotate(f"{2.0 * cfg.wait_under_weight / mean_p:.0f}x steeper",
                xy=(4.2, 4.0), xytext=(3.1, 1.3), fontsize=10, color="0.2", weight="bold",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))
    ax.set_yscale("log")
    ax.set_title("B.  Curvature: how hard each loss pushes back",
                 fontsize=10.5, loc="left", weight="bold", pad=8)
    ax.set_xlabel("task duration (time units)")
    ax.set_ylabel("curvature  $\\partial^2 f / \\partial \\hat{y}^2$   (log scale)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=0.25, which="both")


def panel_c(ax, manifest: Path) -> None:
    """The real block objectives along a random direction, from one run."""
    cfg = ExperimentConfig(controllers=3, agents=12, rho=0.849, sync_interval=5,
                           seed=11, architecture="fixed_replay", optimizer="local_rls")
    tasks = load_manifest_tasks(manifest, cfg.seed, cfg.rho, make_agent_pool(12),
                                None, split="test")
    experiment = ControlledReplayExperiment(tasks, cfg)
    experiment.run()
    controller = experiment.controllers[0]
    base = controller.pack_parameters()
    blocks = dict(controller.parameter_blocks())
    rng = np.random.default_rng(0)
    steps = np.linspace(-1.5, 1.5, 121)

    for name, colour, label in (("processing", C_P, "F1  processing"),
                                ("waiting", C_W, "F2  waiting"),
                                ("quality", C_Q, "F3  quality")):
        block = blocks[name]
        width = block.stop - block.start
        direction = np.zeros_like(base)
        unit = rng.normal(size=width)
        direction[block] = unit / np.linalg.norm(unit)
        centre = controller.objective_block(base, name)
        curve = [controller.objective_block(base + t * direction, name) - centre
                 for t in steps]
        ax.plot(steps, curve, color=colour, lw=2.4, label=label)

    top = max(ax.get_ylim()[1], 1e-6)
    ax.set_ylim(-0.004, top * 1.30)
    ax.annotate("F1 is nearly flat: an ordinary\nzeroth-order step barely moves it",
                xy=(0.95, 0.004), xytext=(-1.45, top * 0.82), fontsize=9.5, color=C_P,
                arrowprops=dict(arrowstyle="->", color=C_P, lw=1.1))
    ax.set_title("C.  The same, on the real objectives of one controller",
                 fontsize=10.5, loc="left", weight="bold", pad=8)
    ax.set_xlabel("shift of the block parameters along a random unit direction")
    ax.set_ylabel("change in the block objective")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    ax.axhline(0.0, color="0.6", lw=0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/it_routing_benchmark.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("loss_geometry.png"))
    args = parser.parse_args()

    cfg = PaperLossConfig()
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    panel_a(axes[0], cfg)
    panel_b(axes[1], cfg)
    panel_c(axes[2], args.manifest)
    fig.suptitle(
        "Why acceleration pays off on F1 and not on F2/F3: F1 normalises its error "
        "by the task's own scale, so it is the flattest of the three",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
