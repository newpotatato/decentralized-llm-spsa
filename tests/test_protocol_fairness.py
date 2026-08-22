"""Regression tests for the fairness and identifiability defects found in the
controlled paper protocol.

Each test pins one property that a previous revision violated, so a future
change cannot silently reintroduce a comparison that measures something other
than the optimizer it claims to compare.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.assignment_buffer import (
    BufferPolicy, CandidateProposal, CandidateStatus, GlobalAssignmentBuffer,
)
from src.paper_experiment import (
    CANONICAL_AGENTS, ControlledReplayExperiment, ExperimentConfig, MAX_AGENTS,
    ReplayTask, load_manifest_tasks, make_agent_pool, pool_capacity,
    semantic_relevance,
)
from src.paper_losses import (
    LossObservation, PaperLossConfig, mean_losses, quality_loss, ridge_penalty,
)
from src.paper_optimizers import (
    METHODS, DistributedZerothOrderOptimizer, ModelOptimizerConfigs,
    OptimizerConfig, lipschitz_constants, ring_laplacian_spectrum,
    theory_parameters,
)

MANIFEST = Path("data/it_routing_benchmark.jsonl")
requires_manifest = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="pinned IT routing benchmark manifest is absent; build it with src.audit_it_benchmark",
)


def synthetic_tasks(count: int, agents: int = 6) -> list[ReplayTask]:
    tasks = []
    for index in range(count):
        domain = index % 6
        vector = tuple(1.0 if pos == domain else 0.0 for pos in range(6))
        tasks.append(ReplayTask(
            index, f"synthetic-{index}", domain, vector,
            (0.25, 0.60, 1.0)[index % 3], 0.4, 300 + index, False,
            index * 0.3, index * 0.3 + 6.0,
        ))
    return tasks


class _Quadratic:
    """Separable strongly convex controller with a known unique minimiser.

    ``start`` is deliberately distinct from ``optimum`` so a step actually has
    a gradient to follow; starting at the minimiser makes every method report a
    zero displacement and hides exactly the scale differences under test.
    """

    def __init__(self, cid: int, start: float, optimum: float = 0.0):
        self.controller_id = cid
        self.offset = optimum
        self.x = np.full(6, start)

    def pack_parameters(self):
        return self.x.copy()

    def unpack_parameters(self, vector):
        self.x = self.project(vector)

    def project(self, vector):
        return np.clip(vector, -50.0, 50.0)

    def objective(self, vector):
        return float(np.sum((np.asarray(vector) - self.offset) ** 2))

    def constraint(self, vector, budget):
        return float(np.mean(vector) - budget), np.ones_like(vector) / len(vector)


# --------------------------------------------------------------------------
# Consensus is one shared mechanism with one shared strength
# --------------------------------------------------------------------------

def test_every_method_applies_the_same_consensus_gradient():
    """omega * (theta_j - ring_mean) for all methods, not 5% mixing for some.

    A previous revision mixed the baselines towards the ring average by a
    step-independent 5% while A-SPSA received ``omega * h``.  With a tuned
    h = 12 that made A-SPSA average twelve times harder than every method it
    was compared against, which is what its reported advantage measured.
    """
    config = OptimizerConfig(step_size=0.1, consensus_weight=0.25)
    point = np.array([1.0, -2.0, 3.0])
    ring_mean = np.array([0.0, 0.0, 0.0])
    for method in METHODS:
        optimizer = DistributedZerothOrderOptimizer(method, config, seed=0)
        gradient = optimizer._consensus_gradient(point, ring_mean)
        assert np.allclose(gradient, 0.25 * (point - ring_mean))
        assert np.allclose(optimizer._consensus_gradient(point, ring_mean, False), 0.0)


def test_ring_average_difference_equals_the_laplacian_term():
    """``theta_j - ring_mean`` must equal ``sum_l b_jl (theta_j - theta_l)``."""
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 2.0]), np.array([-1.0, 1.0])]
    averages = DistributedZerothOrderOptimizer._ring_average(vectors)
    for index, (vector, average) in enumerate(zip(vectors, averages)):
        neighbours = (vectors[(index - 1) % 3], vectors[(index + 1) % 3])
        expected = sum((vector - other) / 3.0 for other in neighbours)
        assert np.allclose(vector - average, expected)


# Placing each controller at its own minimiser isolates consensus as the only
# force acting, because a two-point or coordinate estimate of a quadratic is
# exactly zero there.  Three methods legitimately carry an extra term and are
# therefore excluded from the exact-equality check:
#   sp_gt   - single point, so f(x + beta*u) is non-zero at the minimiser;
#   pd_2pt  - adds dual * grad h, and the constraint is violated at the start;
#   aspsa   - probes at theta~, not at theta, so its estimate is non-zero.
PLAIN_DESCENT_METHODS = ("spsa", "kw", "zo_pgd", "zo_gt")


def test_consensus_contracts_at_the_same_rate_for_every_method():
    """Identical omega and h must produce identical contraction, whatever the
    gradient estimator, because the consensus term is now the same operator."""
    def disagreement_after(method: str, rounds: int) -> float:
        starts = (-3.0, 0.0, 3.0)
        controllers = [_Quadratic(i, start, optimum=start) for i, start in enumerate(starts)]
        optimizer = DistributedZerothOrderOptimizer(
            method,
            OptimizerConfig(step_size=0.05, perturbation=0.02, consensus_weight=0.5),
            seed=3,
        )
        for round_index in range(1, rounds + 1):
            optimizer.step(controllers, round_index)
        vectors = [c.pack_parameters() for c in controllers]
        centre = np.mean(vectors, axis=0)
        return float(np.mean([np.linalg.norm(v - centre) for v in vectors]))

    start = float(np.mean([abs(o) * np.sqrt(6) for o in (-3.0, 0.0, 3.0)]))

    # After one round the gradient estimates are still exactly zero, so the
    # displacement is pure consensus and must be bit-identical.
    first = {m: disagreement_after(m, 1) for m in PLAIN_DESCENT_METHODS}
    assert np.allclose(list(first.values()), first["spsa"], rtol=1e-12), first

    # Consensus then moves each controller off its own minimiser, so the
    # estimators start to differ; the rates must still agree closely.
    later = {m: disagreement_after(m, 5) for m in PLAIN_DESCENT_METHODS}
    for method, value in later.items():
        assert value < start, f"{method} did not contract towards consensus"
    assert np.allclose(list(later.values()), later["spsa"], rtol=1e-2), later

    # Every remaining method must still contract, only not at the same rate.
    for method in ("aspsa", "pd_2pt", "sp_gt"):
        assert disagreement_after(method, 5) < start


# --------------------------------------------------------------------------
# The perturbation convention must not hide a factor of d
# --------------------------------------------------------------------------

def test_compensated_estimator_is_unbiased_for_the_gradient():
    """Delta in {+-1/sqrt(d)} biases the raw estimate towards grad F / d."""
    config = OptimizerConfig(compensate_perturbation_dimension=True)
    optimizer = DistributedZerothOrderOptimizer("spsa", config, seed=0)
    scale, gain = optimizer._perturbation_scale(16)
    assert np.isclose(scale, 0.25)
    assert np.isclose(gain, 16.0)

    raw = DistributedZerothOrderOptimizer(
        "spsa", replace(config, compensate_perturbation_dimension=False), seed=0
    )
    assert np.isclose(raw._perturbation_scale(16)[1], 1.0)


def test_shared_step_grid_contains_a_working_configuration_for_every_method():
    """Equal budget is only equal opportunity if the grid covers each optimum.

    The methods do not share a usable step region, so the shared grid has to be
    wide rather than narrow.  A previous revision instead hand-picked a
    per-method range - up to h = 20 for A-SPSA against h = 0.05 for the
    baselines - which by itself decided which method won.
    """
    from src.run_paper_experiments import STEP_GRID

    def distance_after(method: str, step: float, pert: float, rounds: int = 40) -> float:
        controllers = [_Quadratic(i, 4.0) for i in range(3)]
        optimizer = DistributedZerothOrderOptimizer(
            method,
            OptimizerConfig(step_size=step, perturbation=pert, consensus_weight=0.0),
            seed=11,
        )
        for round_index in range(1, rounds + 1):
            optimizer.step(controllers, round_index)
        return float(np.linalg.norm(controllers[0].pack_parameters()))

    start = float(np.linalg.norm(np.full(6, 4.0)))
    best = {
        method: min(distance_after(method, step, pert) for step, pert in STEP_GRID)
        for method in METHODS
    }
    for method, residual in best.items():
        assert residual < start, (
            f"no configuration of the shared grid lets {method} make progress "
            f"(best residual {residual:.3f} from {start:.3f})"
        )
    # Every method that estimates the gradient from two points must get most of
    # the way there.  SP-GT is single point and intrinsically far slower, so it
    # is only required to improve, not to converge in the same budget.
    for method in ("spsa", "aspsa", "kw", "zo_pgd", "zo_gt", "pd_2pt"):
        assert best[method] < 0.5 * start, (method, best[method])


def test_methods_do_not_share_a_usable_step_region():
    """Documents why the shared grid spans four decades instead of one."""
    def diverged(method: str, step: float) -> bool:
        controllers = [_Quadratic(i, 4.0) for i in range(3)]
        optimizer = DistributedZerothOrderOptimizer(
            method,
            OptimizerConfig(step_size=step, perturbation=0.02, consensus_weight=0.0),
            seed=11,
        )
        for round_index in range(1, 41):
            optimizer.step(controllers, round_index)
        return float(np.linalg.norm(controllers[0].pack_parameters())) > np.linalg.norm(
            np.full(6, 4.0)
        )

    # The two-point baselines are still improving where A-SPSA has diverged.
    assert not diverged("spsa", 0.2)
    assert diverged("aspsa", 0.2)
    # A-SPSA is usable exactly where the baselines have barely moved.
    assert not diverged("aspsa", 0.001)


def test_dimension_compensation_removes_the_factor_of_d_between_estimators():
    def displacement(method: str, compensate: bool) -> float:
        controllers = [_Quadratic(0, 4.0)]
        before = controllers[0].pack_parameters().copy()
        optimizer = DistributedZerothOrderOptimizer(
            method,
            OptimizerConfig(
                step_size=0.01, perturbation=0.02, consensus_weight=0.0,
                compensate_perturbation_dimension=compensate,
                surrogate_init="theta",
            ),
            seed=11,
        )
        optimizer.step(controllers, 1)
        return float(np.linalg.norm(controllers[0].pack_parameters() - before))

    compensated = displacement("spsa", True) / displacement("aspsa", True)
    raw = displacement("spsa", False) / displacement("aspsa", False)
    # Both draw Delta in {+-1/sqrt(d)}; only the compensation differs, so the
    # uncompensated pair must sit further from parity than the compensated one.
    assert abs(np.log(compensated)) < abs(np.log(raw))


# --------------------------------------------------------------------------
# A-SPSA recurrence details
# --------------------------------------------------------------------------

def test_aspsa_consensus_uses_the_probing_points():
    """The manuscript correction is omega * sum b_jl (theta~_j - theta~_l)."""
    controllers = [_Quadratic(i, float(i) + 1.0) for i in range(3)]
    optimizer = DistributedZerothOrderOptimizer(
        "aspsa",
        OptimizerConfig(step_size=0.05, consensus_weight=0.4, gamma0=1.0, mu_minus_eta=0.001),
        seed=5,
    )
    captured = {}
    original = optimizer._consensus_gradient

    def spy(point, ring_average, allow=True):
        captured.setdefault("points", []).append(np.asarray(point).copy())
        captured.setdefault("averages", []).append(np.asarray(ring_average).copy())
        return original(point, ring_average, allow)

    optimizer._consensus_gradient = spy
    optimizer.step(controllers, 1)
    points = captured["points"]
    averages = captured["averages"]
    assert len(points) == 3
    # The ring average handed to the correction must be the average of the very
    # probing points the gradients were taken at, not of the incoming vectors.
    expected = DistributedZerothOrderOptimizer._ring_average(points)
    for produced, want in zip(averages, expected):
        assert np.allclose(produced, want)


def test_surrogate_initialisation_is_selectable_and_recorded():
    """z_{j,0}=0 is the manuscript choice but shrinks a non-centred model."""
    def first_probe(mode: str) -> np.ndarray:
        controllers = [_Quadratic(0, 5.0, optimum=1.0)]
        optimizer = DistributedZerothOrderOptimizer(
            "aspsa",
            OptimizerConfig(step_size=0.05, consensus_weight=0.0, surrogate_init=mode),
            seed=1,
        )
        vector = controllers[0].pack_parameters()
        return optimizer._surrogate_center(controllers[0], vector).copy()

    assert np.allclose(first_probe("zero"), 0.0)
    assert np.allclose(first_probe("theta"), 5.0)


def test_theory_parameters_match_the_manuscript_prescription():
    """mu = omega*lambda_2, h = 1/max{M, omega*lambda_max}, 0 < eta < mu."""
    lambda_2, lambda_max = ring_laplacian_spectrum(3)
    assert lambda_2 > 0.0 and lambda_max >= lambda_2
    constants = lipschitz_constants(PaperLossConfig())
    # M_p = X^2 / epsilon_p^2 with X = 1 and epsilon_p = 0.05.
    assert np.isclose(constants["M_p"], 400.0)
    prescription = theory_parameters(constants["M_p"], 0.002, 3)
    assert np.isclose(prescription["mu"], 0.002 * lambda_2)
    assert np.isclose(prescription["h_theory"], 1.0 / 400.0)
    assert prescription["eta_upper_bound"] == prescription["mu"]


# --------------------------------------------------------------------------
# Losses follow the manuscript
# --------------------------------------------------------------------------

def test_relevance_is_the_manuscript_cosine_and_prefers_the_matching_agent():
    pool = make_agent_pool(6)
    own = semantic_relevance((1, 0, 0, 0, 0, 0), pool[0].capability)
    other = semantic_relevance((0, 0, 1, 0, 0, 0), pool[0].capability)
    assert own > other
    phi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    psi = np.asarray(pool[0].capability)
    assert np.isclose(own, float(phi @ psi) / (np.linalg.norm(phi) * np.linalg.norm(psi)), atol=1e-6)


def test_quality_loss_penalty_scales_with_relevance_not_urgency():
    cfg = PaperLossConfig()
    base = dict(
        features=np.array([1.0, 0.0]), agent_id=0, processing_real=2.0,
        waiting_real=1.0, quality_real=0.5, hardness=0.5,
    )
    low = LossObservation(relevance=0.1, **base)
    high = LossObservation(relevance=0.9, **base)
    # Overprediction only: the extra penalty must grow with R_ik.
    assert quality_loss(0.9, high, cfg) > quality_loss(0.9, low, cfg)
    # Underprediction: the extra term is inactive, so R_ik must not matter.
    assert np.isclose(quality_loss(0.1, high, cfg), quality_loss(0.1, low, cfg))


def test_ridge_terms_are_reported_with_the_losses():
    cfg = PaperLossConfig(lambda_reg_p=0.5, lambda_reg_q=0.25)
    theta_p = np.array([1.0, 1.0])
    theta_q = np.array([2.0])
    f1, _, f3 = mean_losses(
        [], lambda o: 0.0, lambda o: 0.0, lambda o: 0.0, cfg,
        processing_parameters=theta_p, quality_parameters=theta_q,
    )
    assert np.isclose(f1, ridge_penalty(theta_p, 0.5)) and np.isclose(f1, 0.5)
    assert np.isclose(f3, ridge_penalty(theta_q, 0.25)) and np.isclose(f3, 0.5)


def test_block_objectives_sum_to_the_whole_vector_objective():
    """A block evaluation and a whole-vector evaluation must agree."""
    tasks = synthetic_tasks(24)
    experiment = ControlledReplayExperiment(
        tasks, ExperimentConfig(controllers=2, agents=3, seed=4, warmup=6, sync_interval=3)
    )
    experiment.run()
    controller = experiment.controllers[0]
    vector = controller.pack_parameters()
    blocks = sum(
        controller.objective_block(vector, name)
        for name, _ in controller.parameter_blocks()
    )
    assert np.isclose(blocks, controller.objective(vector), rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------
# Identifiability and budget instrumentation
# --------------------------------------------------------------------------

@requires_manifest
def test_queue_feature_is_not_identically_zero_under_fixed_replay():
    """fixed_replay bypasses the buffer; the queue feature must still move.

    While it read the unused global buffer it stayed at zero for every
    observation, leaving one coordinate per agent unidentifiable and perturbed
    by every zeroth-order method for nothing.
    """
    agents = make_agent_pool(6)
    tasks = load_manifest_tasks(MANIFEST, 11, 1.4, agents, 90, split="test")
    config = ExperimentConfig(
        controllers=3, agents=6, rho=1.4, seed=11, warmup=10,
        sync_interval=5, architecture="fixed_replay",
    )
    experiment = ControlledReplayExperiment(tasks, config)
    seen = []
    original = experiment._prediction_features

    def spy(task, now):
        vector = original(task, now)
        seen.append(vector.copy())
        return vector

    experiment._prediction_features = spy
    experiment.run()
    matrix = np.asarray(seen)
    assert matrix.shape[1] == 9
    assert matrix[:, -1].max() > 0.0, "queue occupancy feature never became positive"
    assert np.linalg.matrix_rank(matrix) == matrix.shape[1]


def test_summary_reports_the_query_budget_per_parameter():
    tasks = synthetic_tasks(40)
    config = ExperimentConfig(
        controllers=3, agents=6, seed=7, warmup=6, sync_interval=3,
        architecture="fixed_replay", optimizer="spsa",
    )
    summary = ControlledReplayExperiment(tasks, config).run()
    for key in ("sync_rounds", "tuned_parameters", "queries_per_parameter",
                "worst_block_queries_per_parameter", "query_budget_sufficient",
                "rho_effective"):
        assert key in summary
    expected = summary["oracle_queries"] / summary["tuned_parameters"]
    assert np.isclose(summary["queries_per_parameter"], expected)
    assert summary["query_budget_sufficient"] in (0.0, 1.0)

    # Every block receives the same number of evaluations per round whatever
    # its dimension, so the pooled ratio must not be used on its own: it can
    # look sufficient while the large blocks are not.  The verdict is the worst
    # block.
    per_block = {
        name: summary[f"queries_per_parameter_{name}"]
        for name in ("processing", "waiting", "quality")
    }
    assert np.isclose(summary["worst_block_queries_per_parameter"], min(per_block.values()))
    assert summary["query_budget_sufficient"] == float(min(per_block.values()) >= 1.0)
    # The waiting block is an order of magnitude smaller, so it is resolved
    # while the two large blocks are not; a pooled ratio would hide that.
    assert per_block["waiting"] > per_block["processing"]
    assert summary["queries_per_parameter"] > summary["worst_block_queries_per_parameter"]


def test_two_point_methods_are_under_resolved_on_the_paper_blocks():
    """The reported budget shortfall must be visible, not averaged away."""
    tasks = synthetic_tasks(120)
    config = ExperimentConfig(
        controllers=3, agents=12, seed=7, warmup=10, sync_interval=5,
        architecture="fixed_replay", optimizer="spsa",
    )
    summary = ControlledReplayExperiment(tasks, config).run()
    assert summary["query_budget_sufficient"] == 0.0
    assert summary["worst_block_queries_per_parameter"] < 1.0

    # Coordinate-wise KW spends 2d evaluations per round, so it is the one
    # method that does resolve the block, at a proportionally larger cost.
    kw = ControlledReplayExperiment(
        tasks, replace(config, optimizer="kw")
    ).run()
    assert kw["worst_block_queries_per_parameter"] >= 1.0
    assert kw["oracle_queries"] > 10.0 * summary["oracle_queries"]


def test_rho_effective_exposes_the_canonical_capacity_reference():
    """rho is normalised to a six-agent pool, so a larger system runs cooler."""
    tasks = synthetic_tasks(30)
    small = ControlledReplayExperiment(
        tasks, ExperimentConfig(controllers=3, agents=CANONICAL_AGENTS, rho=0.85,
                                seed=2, warmup=6, sync_interval=5)
    ).run()
    large = ControlledReplayExperiment(
        tasks, ExperimentConfig(controllers=3, agents=12, rho=0.85,
                                seed=2, warmup=6, sync_interval=5)
    ).run()
    assert np.isclose(small["rho_effective"], 0.85, atol=1e-9)
    assert large["rho_effective"] < small["rho_effective"]
    assert np.isclose(
        large["rho_effective"],
        0.85 * pool_capacity(make_agent_pool(CANONICAL_AGENTS)) / pool_capacity(make_agent_pool(12)),
    )


def test_agent_pool_extends_past_twelve_with_distinct_profiles():
    pool = make_agent_pool(MAX_AGENTS)
    signatures = {
        (tuple(np.round(a.capability, 6)), tuple(np.round(a.mean_service, 6)))
        for a in pool
    }
    assert len(signatures) == MAX_AGENTS
    assert all(0.0 <= value <= 1.0 for agent in pool for value in agent.capability)


# --------------------------------------------------------------------------
# Buffer priority
# --------------------------------------------------------------------------

def test_availability_cannot_discriminate_between_free_agents():
    """Documented reason the availability term was removed from the priority."""
    buffer = GlobalAssignmentBuffer()
    buffer.submit(
        task_id=1, controller_id=0, arrival_time=0.0, deadline=10.0, time_budget=10.0,
        proposals=[CandidateProposal(0, 1.0, 1.0), CandidateProposal(1, 0.9, 4.0)],
    )
    buffer.refresh(5.0, {0: 0.0, 1: 2.0}, {0, 1})
    scores = {
        agent_id: candidate.availability_score
        for agent_id, candidate in buffer.tasks[1].proposals.items()
    }
    assert set(scores.values()) == {1.0}
    assert not hasattr(BufferPolicy(), "availability_weight")


def test_waiting_prediction_changes_dispatch_order_without_defining_the_deadline():
    """Improving F2 must be able to change an operational outcome."""
    def winner(tight_budget: float, relaxed_budget: float) -> int:
        buffer = GlobalAssignmentBuffer(BufferPolicy(hotness_weight=0.0))
        for task_id, budget in ((1, tight_budget), (2, relaxed_budget)):
            buffer.submit(
                task_id=task_id, controller_id=0, arrival_time=0.0,
                deadline=100.0, time_budget=100.0,
                proposals=[CandidateProposal(
                    0, 0.5, 1.0, predicted_wait_budget=budget
                )],
            )
        return buffer.dispatch(6.0, {0: 0.0})[0].task_id

    # The task whose admissible wait is nearly exhausted must be served first,
    # and the ranking must flip when the predictions swap.
    assert winner(1.0, 50.0) == 1
    assert winner(50.0, 1.0) == 2


def test_hotness_uses_only_agents_this_round_can_reach():
    """A busy agent's slack must not make a task look cold."""
    buffer = GlobalAssignmentBuffer()
    buffer.submit(
        task_id=1, controller_id=0, arrival_time=0.0, deadline=100.0, time_budget=100.0,
        proposals=[
            CandidateProposal(0, 0.5, 95.0),  # free, but almost out of time
            CandidateProposal(1, 0.5, 0.5),   # huge slack, currently busy
        ],
    )
    availability = {0: 0.0, 1: 10.0}
    buffer.refresh(0.0, availability, {0})
    reachable_only = buffer.tasks[1].proposals[0].priority
    buffer.refresh(0.0, availability, None)
    all_agents = buffer.tasks[1].proposals[0].priority
    assert reachable_only > all_agents


def test_pending_count_is_maintained_incrementally():
    buffer = GlobalAssignmentBuffer()
    for task_id in (1, 2, 3):
        buffer.submit(
            task_id=task_id, controller_id=0, arrival_time=0.0, deadline=10.0,
            time_budget=10.0, proposals=[CandidateProposal(0, 1.0 / task_id, 1.0)],
        )
    assert buffer.pending_count() == 3
    buffer.dispatch(0.0, {0: 0.0})
    assert buffer.pending_count() == 2
    assert len(list(buffer.pending_tasks())) == 2
    assert all(
        task.status.value == "pending" for task in buffer.pending_tasks()
    )


def test_invalidated_alternatives_are_retained_for_audit():
    buffer = GlobalAssignmentBuffer()
    buffer.submit(
        task_id=1, controller_id=0, arrival_time=0.0, deadline=10.0, time_budget=10.0,
        proposals=[CandidateProposal(0, 0.9, 1.0), CandidateProposal(1, 0.1, 1.0)],
    )
    buffer.dispatch(0.0, {0: 0.0, 1: 0.0})
    statuses = {c.status for c in buffer.tasks[1].proposals.values()}
    assert statuses == {CandidateStatus.ASSIGNED, CandidateStatus.INVALIDATED}


# --------------------------------------------------------------------------
# The tuning protocol itself
# --------------------------------------------------------------------------

def test_every_method_receives_the_identical_tuning_grid():
    from src.run_paper_experiments import STEP_GRID, NETWORK_OMEGA, _tune_methods
    import inspect

    source = inspect.getsource(_tune_methods)
    assert "STEP_GRID" in source
    # One grid, no method-specific or block-specific branch.
    assert "if method" not in source.split("def tuning_grid")[1].split("return")[0]
    assert len(STEP_GRID) == 6
    # Four decades, so no method's optimum falls outside the shared range.
    steps = [step for step, _ in STEP_GRID]
    assert max(steps) / min(steps) >= 1000.0
    assert NETWORK_OMEGA == ExperimentConfig().network_omega[0]


def test_optimizer_primary_metrics_separate_accuracy_from_agreement():
    from src.run_paper_experiments import OPTIMIZER_PRIMARY_METRICS

    assert any(name.startswith("local_") for name in OPTIMIZER_PRIMARY_METRICS)
    assert any(name.startswith("disagreement_") for name in OPTIMIZER_PRIMARY_METRICS)
    assert any(name.startswith("phi_") for name in OPTIMIZER_PRIMARY_METRICS)


def test_significance_table_flags_the_exact_test_resolution_floor(tmp_path):
    import pandas as pd
    from src.run_paper_experiments import _paired_significance

    seeds = list(range(10))
    frame = pd.DataFrame(
        [{"variant": "aspsa", "seed": s, "phi_p_auc": 1.0 + 1e-9 * (s + 1)} for s in seeds]
        + [{"variant": "spsa", "seed": s, "phi_p_auc": 1.0} for s in seeds]
    )
    path = tmp_path / "significance.csv"
    _paired_significance(frame, "aspsa", path, ("phi_p_auc",))
    row = pd.read_csv(path).iloc[0]
    # A consistent sign on a 1e-9 effect still reaches the floor of the test.
    assert bool(row.reject_holm_0_05)
    assert bool(row.p_at_resolution_floor)
    assert int(row.effective_pairs) == 10
    assert np.isclose(row.p_resolution_floor, 2.0 / 2 ** 10)
    assert not bool(row.practical_effect_ge_1pct)


def test_resolution_floor_accounts_for_discarded_zero_differences(tmp_path):
    """The signed-rank test drops ties, which raises its attainable floor."""
    import pandas as pd
    from src.run_paper_experiments import _paired_significance

    seeds = list(range(10))
    frame = pd.DataFrame(
        [{"variant": "aspsa", "seed": s, "phi_p_auc": 1.0 + 1e-9 * s} for s in seeds]
        + [{"variant": "spsa", "seed": s, "phi_p_auc": 1.0} for s in seeds]
    )
    path = tmp_path / "significance.csv"
    _paired_significance(frame, "aspsa", path, ("phi_p_auc",))
    row = pd.read_csv(path).iloc[0]
    assert int(row.effective_pairs) == 9
    assert np.isclose(row.p_resolution_floor, 2.0 / 2 ** 9)
    assert bool(row.p_at_resolution_floor)


# --------------------------------------------------------------------------
# Stream heterogeneity and out-of-window accuracy
# --------------------------------------------------------------------------

def test_domain_assignment_gives_controllers_disjoint_streams():
    """The manuscript motivates synchronization with divergent local streams.

    Under load-based assignment every controller draws from one common
    distribution, so their models differ only by sampling noise and there is
    nothing for synchronization to transfer.  Domain ownership instantiates the
    situation the method is designed for.
    """
    tasks = synthetic_tasks(120)
    shared = dict(controllers=3, agents=6, seed=5, warmup=10, sync_interval=5,
                  architecture="fixed_replay", optimizer="spsa")
    balanced = ControlledReplayExperiment(
        tasks, ExperimentConfig(controller_assignment="load", **shared)
    ).run()
    partitioned = ControlledReplayExperiment(
        tasks, ExperimentConfig(controller_assignment="domain", **shared)
    ).run()
    assert np.isclose(partitioned["domain_mix_total_variation"], 1.0)
    assert balanced["domain_mix_total_variation"] < 0.5


def test_true_processing_error_is_reported_and_is_not_the_windowed_loss():
    """Phi is an in-sample loss on a rolling window; this is held-out accuracy.

    A model can drive the windowed objective down while moving away from the
    relationship that generated the data, so the two must be reported side by
    side and must not be assumed to move together.
    """
    tasks = synthetic_tasks(120)
    shared = dict(controllers=3, agents=6, seed=5, warmup=10, sync_interval=5,
                  architecture="fixed_replay")
    frozen = ControlledReplayExperiment(
        tasks,
        ExperimentConfig(optimizer="none", use_learning=False, use_rls=False, **shared),
    ).run()
    fitted = ControlledReplayExperiment(
        tasks, ExperimentConfig(optimizer="local_rls", **shared)
    ).run()

    for summary in (frozen, fitted):
        for key in ("true_processing_rmse", "true_processing_mape",
                    "domain_mix_total_variation"):
            assert key in summary and np.isfinite(summary[key])

    # Fitting lowers the windowed objective ...
    assert fitted["local_p_final"] < frozen["local_p_final"]
    # ... and the two quantities are genuinely different measurements, so a
    # drop in one must never be reported as evidence about the other.
    assert not np.isclose(fitted["true_processing_rmse"], fitted["local_p_final"])


def test_frozen_prior_recovers_the_truth_better_than_the_fitted_model():
    """Documents the overfitting regime the benchmark currently sits in.

    108 processing parameters against roughly 80 observations per controller,
    with a median of one sample per (agent, domain) cell: recursive least
    squares reduces the rolling-window loss while increasing the error against
    the generative model.  If this ever stops holding, the budget or the
    parameterisation has changed and the audit needs revisiting.
    """
    tasks = synthetic_tasks(240)
    shared = dict(controllers=3, agents=12, seed=11, warmup=10, sync_interval=5,
                  architecture="fixed_replay")
    frozen = ControlledReplayExperiment(
        tasks,
        ExperimentConfig(optimizer="none", use_learning=False, use_rls=False, **shared),
    ).run()
    fitted = ControlledReplayExperiment(
        tasks, ExperimentConfig(optimizer="local_rls", **shared)
    ).run()
    assert fitted["local_p_final"] < frozen["local_p_final"]
    assert fitted["true_processing_rmse"] > frozen["true_processing_rmse"]
