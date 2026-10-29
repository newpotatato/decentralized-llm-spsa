"""Pins for the postановка-C redesign: preprocessing, heterogeneity, omega.

Each test here corresponds to a defect that was measured, not to a style
preference.  The comment above each one names what broke.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.paper_experiment import (
    COMPLEXITY_INDICATOR_WEIGHT,
    DOMAIN_COMPLEXITY_RESPONSE,
    DOMAIN_LENGTH_RESPONSE,
    DOMAIN_ORDER,
    DOMAIN_QUALITY_RESPONSE,
    ControlledReplayExperiment,
    ExperimentConfig,
    load_manifest_tasks,
    make_agent_pool,
    task_complexity,
)

MANIFEST = Path("data/it_routing_benchmark.jsonl")


def _config(**kwargs) -> ExperimentConfig:
    base = dict(controllers=5, agents=12, rho=0.849, sync_interval=5, seed=11,
                architecture="fixed_replay", optimizer="local_rls")
    base.update(kwargs)
    return ExperimentConfig(**base)


def _tasks(seed: int = 11, agents: int = 12):
    return load_manifest_tasks(MANIFEST, seed, 0.849, make_agent_pool(agents),
                               None, split="test")


# The complexity feature took three values derived from a post's view count, so
# it graded popularity rather than difficulty and left only 18 distinct feature
# vectors across 240 tasks.
def test_complexity_follows_the_manuscript_formula() -> None:
    got = task_complexity(prompt_chars=400, has_code=True, tag_count=4,
                          task_kind="debugging", primary_share=0.5)
    expected = (2.0 + 4.0 * 0.5 + 2.0 * 0.5 + COMPLEXITY_INDICATOR_WEIGHT * 3) / 10.0
    assert got == pytest.approx(expected)


def test_complexity_is_clipped_into_the_unit_interval() -> None:
    low = task_complexity(0, False, 0, "question_answering", 0.0)
    high = task_complexity(10_000, True, 9, "design", 1.0)
    assert 0.0 < low <= high <= 1.0
    assert high == pytest.approx(1.0)


def test_complexity_is_continuous_not_three_valued() -> None:
    tasks = _tasks()
    assert len({round(t.hardness, 6) for t in tasks}) > 50


def test_feature_vectors_are_mostly_distinct() -> None:
    tasks = _tasks()
    distinct = {tuple(np.round(t.features, 9)) for t in tasks}
    assert len(distinct) > 0.75 * len(tasks)


# The Lipschitz constants quoted in the manuscript assume ||x|| <= 1.  The raw
# task vector gained two more unit-valued coordinates, so RAW_FEATURE_NORM had
# to grow with them; had it not, every bound in the paper would have been
# quoted for the wrong feature scale.  The bound applies to the vector actually
# handed to the controllers, not to the unnormalised task vector.
def test_prediction_features_stay_inside_the_unit_ball() -> None:
    experiment = ControlledReplayExperiment(_tasks(), _config())
    original = experiment._prediction_features
    norms: list[float] = []

    def recording(task, now):
        vector = original(task, now)
        norms.append(float(np.linalg.norm(vector)))
        return vector

    experiment._prediction_features = recording
    experiment.run()
    assert len(norms) > 100
    assert max(norms) <= 1.0 + 1e-9


def test_raw_feature_norm_matches_the_largest_possible_vector() -> None:
    from src.paper_experiment import RAW_FEATURE_NORM
    # one domain coordinate + complexity + length + code flag + busy + queue,
    # every one of them capable of reaching one simultaneously
    assert RAW_FEATURE_NORM == pytest.approx(np.sqrt(6.0))
    for task in _tasks():
        assert np.linalg.norm(task.features) <= RAW_FEATURE_NORM + 1e-9


# A single shared complexity slope makes the optimal predictor identical for
# every controller, so no domain mix can produce different optima.
def test_domain_responses_are_not_all_equal() -> None:
    for response in (DOMAIN_COMPLEXITY_RESPONSE, DOMAIN_LENGTH_RESPONSE,
                     DOMAIN_QUALITY_RESPONSE):
        assert len(response) == len(DOMAIN_ORDER)
        assert len(set(response)) > 1


# "skewed" must interpolate between load balancing and disjoint ownership; if it
# collapsed to one end the interior optimum of omega could not be exhibited.
@pytest.mark.parametrize("low,high", [(0.0, 0.5), (0.5, 1.0)])
def test_specialization_increases_stream_skew(low: float, high: float) -> None:
    tasks = _tasks()
    skews = []
    for share in (low, high):
        config = _config(controller_assignment="skewed", specialization=share)
        skews.append(ControlledReplayExperiment(tasks, config).run()
                     ["domain_mix_total_variation"])
    assert skews[0] < skews[1]


def test_specialization_endpoints_match_the_named_modes() -> None:
    tasks = _tasks()
    full = ControlledReplayExperiment(
        tasks, _config(controller_assignment="skewed", specialization=1.0)).run()
    domain = ControlledReplayExperiment(
        tasks, _config(controller_assignment="domain")).run()
    assert full["domain_mix_total_variation"] == pytest.approx(
        domain["domain_mix_total_variation"])


def test_specialization_is_validated() -> None:
    with pytest.raises(ValueError):
        _config(controller_assignment="skewed", specialization=1.5)
    with pytest.raises(ValueError):
        _config(controller_assignment="nonsense")


# network_omega weighted the *reported* Phi while OptimizerConfig.consensus_weight
# was what the algorithm applied; their defaults differed twentyfold, so the
# objective being reported was not the one being minimised.
def test_network_omega_is_what_the_optimizer_applies() -> None:
    omega = (0.11, 0.22, 0.33)
    experiment = ControlledReplayExperiment(_tasks(), _config(network_omega=omega))
    applied = [experiment.optimizer.block_configs[name].consensus_weight
               for name in ("processing", "waiting", "quality")]
    assert applied == pytest.approx(list(omega))


def test_consensus_strength_changes_the_parameters_it_reaches() -> None:
    tasks = _tasks()
    thetas = []
    for omega in (0.002, 0.5):
        config = _config(optimizer="aspsa", network_omega=(omega,) * 3)
        experiment = ControlledReplayExperiment(tasks, config)
        experiment.run()
        thetas.append(np.array([c.theta_time for c in experiment.controllers]))
    assert not np.allclose(thetas[0], thetas[1])
