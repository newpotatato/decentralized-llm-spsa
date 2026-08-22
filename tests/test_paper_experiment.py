import json
from pathlib import Path

import numpy as np
import pytest

MANIFEST = Path("data/it_routing_benchmark.jsonl")
requires_manifest = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="pinned IT routing benchmark manifest is absent; build it with src.audit_it_benchmark",
)

from src.assignment_buffer import BufferPolicy, CandidateProposal, GlobalAssignmentBuffer
from src.paper_experiment import (
    ControlledReplayExperiment, ExperimentConfig, ReplayOracle,
    ReplayTask, load_manifest_tasks, make_agent_pool,
)
from src.paper_losses import LossObservation, PaperLossConfig, processing_loss, quality_loss, waiting_loss
from src.paper_optimizers import (
    DistributedZerothOrderOptimizer, METHODS, OptimizerConfig,
)


def observation(**updates):
    values = dict(
        features=np.array([1.0, 0.0, 0.5]), agent_id=0,
        processing_real=2.0, waiting_real=3.0, quality_real=0.6,
        relevance=0.8, hardness=0.5,
    )
    values.update(updates)
    return LossObservation(**values)


def test_manuscript_losses_have_intended_asymmetry_and_risk():
    cfg = PaperLossConfig()
    obs = observation()
    assert processing_loss(2.0, obs, cfg) >= 0.0
    assert waiting_loss(2.0, obs, cfg) > waiting_loss(4.0, obs, cfg)
    assert quality_loss(0.8, obs, cfg) > quality_loss(0.4, obs, cfg)


def test_exact_matching_beats_greedy_counterexample():
    def build(matching):
        buffer = GlobalAssignmentBuffer(BufferPolicy(
            hotness_weight=0.0, wait_pressure_weight=0.0,
            deadline_gap_penalty=0.0, budget_overrun_penalty=0.0,
            matching=matching,
        ))
        buffer.submit(task_id=1, controller_id=0, arrival_time=0, deadline=10, time_budget=10,
                      proposals=[CandidateProposal(0, 10, 1), CandidateProposal(1, 9, 1)])
        buffer.submit(task_id=2, controller_id=0, arrival_time=0, deadline=10, time_budget=10,
                      proposals=[CandidateProposal(0, 8, 1), CandidateProposal(1, 0, 1)])
        return buffer
    greedy = build("greedy").dispatch(0, {0: 0, 1: 0})
    exact = build("exact").dispatch(0, {0: 0, 1: 0})
    assert sum(x.base_score for x in exact) == 17
    assert sum(x.base_score for x in greedy) == 10


def test_replay_oracle_is_order_independent():
    agents = make_agent_pool(3)
    task = ReplayTask(1, "source-1", 0, (1, 0, 0, 0, 0, 0), 0.6, 0.5, 500, True, 0, 5)
    oracle = ReplayOracle(17, agents)
    first = oracle.outcome(task, 1)
    oracle.outcome(task, 2)
    assert oracle.outcome(task, 1) == first


@requires_manifest
def test_manifest_split_is_balanced_and_disjoint():
    path = MANIFEST
    agents = make_agent_pool(3)
    splits = {
        name: load_manifest_tasks(path, 1, 0.8, agents, split=name)
        for name in ("train", "validation", "test")
    }
    assert len(splits["train"]) == 720
    assert len(splits["validation"]) == len(splits["test"]) == 240
    ids = [{task.source_id for task in tasks} for tasks in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for tasks in splits.values():
        counts = np.bincount([task.domain for task in tasks], minlength=6)
        assert len(set(counts)) == 1


def test_short_experiment_logs_task_and_round_losses():
    agents = make_agent_pool(3)
    tasks = []
    for index in range(18):
        domain = index % 6
        vector = tuple(1.0 if pos == domain else 0.0 for pos in range(6))
        tasks.append(ReplayTask(index, f"q-{index}", domain, vector, 0.5, 0.4, 300, False,
                                index * 0.4, index * 0.4 + 6.0))
    cfg = ExperimentConfig(controllers=2, agents=3, seed=5, warmup=6, sync_interval=3)
    experiment = ControlledReplayExperiment(tasks, cfg)
    summary = experiment.run()
    assert len(experiment.task_records) == 18
    assert len(experiment.round_records) == 5
    assert all(key in experiment.task_records[0] for key in ("f1", "f2", "f3"))
    assert summary["oracle_queries"] > 0
    first_round = experiment.round_records[0]
    assert first_round["oracle_queries"] == 12.0
    for short_name in ("p", "w", "q"):
        assert np.isclose(
            first_round[f"phi_{short_name}"],
            first_round[f"local_{short_name}"]
            + cfg.network_omega[("p", "w", "q").index(short_name)]
            * first_round[f"disagreement_{short_name}"],
        )
        assert np.isfinite(summary[f"phi_{short_name}_auc"])
    assert 0 <= summary["jain_fairness"] <= 1

    frozen = ControlledReplayExperiment(
        tasks,
        ExperimentConfig(
            controllers=2, agents=3, seed=5, warmup=6, sync_interval=3,
            use_learning=False, use_rls=False, optimizer="none",
        ),
    )
    frozen_summary = frozen.run()
    assert frozen_summary["oracle_queries"] == 0.0
    assert all(frozen_summary[f"phi_{name}_auc"] > 0.0 for name in ("p", "w", "q"))


def test_fixed_replay_keeps_feedback_stream_identical_across_optimizers():
    tasks = []
    for index in range(30):
        domain = index % 6
        vector = tuple(1.0 if pos == domain else 0.0 for pos in range(6))
        tasks.append(ReplayTask(
            index, f"fixed-{index}", domain, vector, 0.5, 0.4, 300, False,
            index * 0.2, index * 0.2 + 6.0,
        ))
    streams = []
    for method in ("spsa", "aspsa"):
        experiment = ControlledReplayExperiment(
            tasks,
            ExperimentConfig(
                controllers=3, agents=6, architecture="fixed_replay",
                optimizer=method, seed=17, warmup=6, sync_interval=3,
            ),
        )
        experiment.run()
        streams.append([
            (
                row["task_id"], row["controller_id"], row["agent_id"],
                row["service_real"], row["quality_real"], row["wait_real"],
            )
            for row in experiment.task_records
        ])
    assert streams[0] == streams[1]


class QuadraticController:
    def __init__(self, cid):
        self.controller_id = cid
        self.x = np.array([0.8, 0.6, 0.4])
    def pack_parameters(self):
        return self.x.copy()
    def unpack_parameters(self, vector):
        self.x = self.project(vector)
    def project(self, vector):
        return np.clip(vector, -2, 2)
    def objective(self, vector):
        return float(np.sum(np.asarray(vector) ** 2))
    def constraint(self, vector, budget):
        return float(np.mean(vector) - budget), np.ones_like(vector) / len(vector)


def test_aspsa_alpha_and_gamma_follow_manuscript_recurrence():
    cfg = OptimizerConfig(
        step_size=0.01, gamma0=1.0, mu_minus_eta=0.02,
        initial_alpha=0.0, acceleration_epsilon=0.05,
    )
    controllers = [QuadraticController(0), QuadraticController(1)]
    optimizer = DistributedZerothOrderOptimizer("aspsa", cfg, seed=9)
    optimizer.step(controllers, 1)
    alpha = optimizer.state.scalars["aspsa_alpha:all"]
    gamma = optimizer.state.scalars["aspsa_gamma:all"]
    assert np.isclose(gamma, cfg.gamma0)
    left = alpha**2 * (1.0 + cfg.acceleration_epsilon) / cfg.step_size
    right = (1.0 - alpha) * cfg.gamma0 + alpha * cfg.mu_minus_eta
    assert np.isclose(left, right)


def test_all_optimizers_update_complete_parameter_vector_finitely():
    for method in METHODS:
        controllers = [QuadraticController(0), QuadraticController(1)]
        optimizer = DistributedZerothOrderOptimizer(method, seed=9)
        trace = optimizer.step(controllers, 1)
        assert trace["oracle_queries"] > 0
        assert all(np.isfinite(controller.x).all() for controller in controllers)


@requires_manifest
def test_arrival_stream_and_external_sla_do_not_depend_on_architecture_size():
    path = MANIFEST
    small = load_manifest_tasks(path, 31, 0.85, make_agent_pool(3), limit=30, split="test")
    large = load_manifest_tasks(path, 31, 0.85, make_agent_pool(12), limit=30, split="test")
    assert [(task.source_id, task.arrival, task.deadline) for task in small] == [
        (task.source_id, task.arrival, task.deadline) for task in large
    ]


@requires_manifest
def test_burst_arrivals_preserve_batches_for_matching_experiment():
    tasks = load_manifest_tasks(
        MANIFEST, 31, 0.85, make_agent_pool(6), limit=12, split="test", batch_size=4
    )
    assert len({task.arrival for task in tasks[:4]}) == 1
    assert len({task.arrival for task in tasks[4:8]}) == 1
    assert tasks[0].arrival < tasks[4].arrival < tasks[8].arrival


def test_gradient_tracking_is_not_identical_to_plain_zo_pgd_after_two_rounds():
    plain = [QuadraticController(i) for i in range(3)]
    tracked = [QuadraticController(i) for i in range(3)]
    for index, controller in enumerate(plain + tracked):
        controller.x = controller.x + 0.05 * index
    zo = DistributedZerothOrderOptimizer("zo_pgd", seed=12)
    gt = DistributedZerothOrderOptimizer("zo_gt", seed=12)
    for step in (1, 2):
        zo.step(plain, step)
        gt.step(tracked, step)
    assert any(not np.allclose(left.x, right.x) for left, right in zip(plain, tracked))





