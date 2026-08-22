"""CLI runner for the global-buffer architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .buffered_orchestrator import BufferedOrchestrator
from .main import SimulationVariant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--controllers", type=int, default=3)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--tick-duration", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("buffered_outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    variant = SimulationVariant(
        name="global_buffer_aspsa",
        routing_mode="adaptive",
        controller_mode="load_balancing",
        spsa_variant="aspsa",
        spsa_interval=5,
        spsa_warmup_tasks=15,
    )
    orchestrator = BufferedOrchestrator(
        num_ctrl=args.controllers,
        num_agents=args.agents,
        variant=variant,
        simulation_tick=args.tick_duration,
        classifier_seed=args.seed,
    )
    tasks = orchestrator.generate_batched_stream(
        args.tasks,
        batch_size=args.batch_size,
        tick_duration=args.tick_duration,
    )
    summary = orchestrator.run(tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(orchestrator.task_records).to_csv(
        args.output_dir / "task_records.csv", index=False
    )
    pd.DataFrame(orchestrator.event_records).to_csv(
        args.output_dir / "event_records.csv", index=False
    )
    pd.DataFrame(orchestrator.feedback_records).to_csv(
        args.output_dir / "feedback_records.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
