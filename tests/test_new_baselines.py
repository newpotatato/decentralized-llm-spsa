"""Classic buffer rules, the trained router baseline and the latency sensitivity variants."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from src.assignment_buffer import BufferPolicy, CandidateProposal, GlobalAssignmentBuffer

PROCESSED = Path("data/external/processed")
requires_llmrouterbench = pytest.mark.skipif(
    not (PROCESSED / "llmrouterbench_items.parquet").exists()
    or not (PROCESSED / "llmrouterbench_emb_all-MiniLM-L6-v2.parquet").exists(),
    reason="LLMRouterBench cache absent; build with python -m src.real_benchmarks llmrouterbench --download",
)


def _submit(buffer, task_id, deadline, processing=2.0, scores=(0.5,)):
    buffer.submit(task_id=task_id, controller_id=0, arrival_time=0.0, deadline=deadline,
                  time_budget=deadline,
                  proposals=[CandidateProposal(agent, score, processing) for agent, score in enumerate(scores)])


def _served_first(order, tasks, free=(0,)):
    buffer = GlobalAssignmentBuffer(BufferPolicy(order=order))
    for task in tasks:
        _submit(buffer, *task)
    return buffer.dispatch(0.0, {agent: 0.0 for agent in free})


def test_edf_serves_the_earliest_deadline():
    served = _served_first("edf", [(1, 30.0), (2, 10.0), (3, 20.0)])
    assert [a.task_id for a in served] == [2]


def test_llf_serves_the_least_laxity_and_can_disagree_with_edf():
    # Task 1: deadline 10, 1 unit of work, laxity 9.  Task 2: deadline 12, 8 units, laxity 4.
    tasks = [(1, 10.0, 1.0), (2, 12.0, 8.0)]
    assert _served_first("llf", tasks)[0].task_id == 2
    assert _served_first("edf", tasks)[0].task_id == 1


def test_sjf_serves_the_shortest_predicted_job():
    assert _served_first("sjf", [(1, 20.0, 5.0), (2, 20.0, 1.0)])[0].task_id == 2


def test_a_rule_is_lexicographic_over_the_base_score():
    # The weighted priority prefers the far better-scoring task; EDF must still
    # serve the earlier deadline first.
    tasks = [(1, 8.0, 2.0, (0.10,)), (2, 50.0, 2.0, (0.95,))]
    assert _served_first("weighted", tasks)[0].task_id == 2
    assert _served_first("edf", tasks)[0].task_id == 1


def test_the_urgent_task_goes_to_its_best_free_agent():
    served = _served_first("edf", [(1, 5.0, 2.0, (0.2, 0.9)), (2, 40.0, 2.0, (0.3, 0.4))], free=(0, 1))
    assignment = {a.task_id: a.agent_id for a in served}
    assert assignment[1] == 1 and assignment[2] == 0


def test_unknown_order_is_rejected():
    with pytest.raises(ValueError):
        BufferPolicy(order="fifo")


def test_deadline_urgency_turns_the_weighted_priority_into_edf():
    tasks = [(1, 8.0, 2.0, (0.10,)), (2, 50.0, 2.0, (0.95,))]
    urgent = GlobalAssignmentBuffer(BufferPolicy(deadline_urgency_weight=3.0))
    for task in tasks:
        _submit(urgent, *task)
    assert urgent.dispatch(0.0, {0: 0.0})[0].task_id == 1


def test_zero_urgency_leaves_every_priority_unchanged():
    buffers = [GlobalAssignmentBuffer(BufferPolicy()), GlobalAssignmentBuffer(BufferPolicy(deadline_urgency_weight=0.0))]
    for buffer in buffers:
        for task in [(1, 8.0, 2.0, (0.10, 0.3)), (2, 50.0, 1.0, (0.95, 0.2)), (3, 4.0, 3.0, (0.5, 0.6))]:
            _submit(buffer, *task)
        buffer.refresh(1.0, {0: 0.0, 1: 2.0})
    left, right = ([c.priority for t in b.tasks.values() for c in t.proposals.values()] for b in buffers)
    assert left == right


def test_six_weight_family_starts_from_the_hand_set_policy_with_urgency_off():
    from src.policy_real import FAMILIES
    names5, start5, _, _ = FAMILIES["weighted5"]
    names6, start6, lower6, upper6 = FAMILIES["weighted6"]
    assert names6[:5] == names5 and names6[5] == "deadline_urgency_weight"
    assert np.array_equal(start6[:5], start5) and start6[5] == 0.0
    assert lower6[5] == 0.0 and upper6[5] == 3.0
    BufferPolicy(**dict(zip(names6, start6)))      # every name is a policy field


def test_mlp_router_is_fitted_on_train_prompts_only(tmp_path, monkeypatch):
    """Changing every test outcome leaves every prediction unchanged."""
    from src import prompt_embeddings, real_benchmarks
    rng = np.random.default_rng(0)
    ids = [f"x:{i}" for i in range(60)]
    split = ["train"] * 40 + ["test"] * 20
    vectors = pd.DataFrame(rng.normal(size=(60, 8)), index=ids, columns=[f"e{i}" for i in range(8)])
    train_quality = rng.uniform(size=(60, 2))

    def outcomes(test_value):
        rows = [{"item_id": item, "model": model,
                 "quality": float(train_quality[i, m]) if split[i] == "train" else test_value}
                for i, item in enumerate(ids) for m, model in enumerate(("a", "b"))]
        return pd.DataFrame(rows)

    items = pd.DataFrame({"item_id": ids, "split": split})
    monkeypatch.setattr(prompt_embeddings, "encode", lambda benchmark, root=None: vectors)
    predictions = []
    for label, test_value in (("low", 0.0), ("high", 1.0)):
        root = tmp_path / label
        (root / "processed").mkdir(parents=True)
        monkeypatch.setattr(real_benchmarks, "load", lambda name, root=None, v=test_value: (items, outcomes(v)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions.append(prompt_embeddings.mlp_quality("fake", root))
    assert predictions[0].equals(predictions[1])
    assert list(predictions[0].columns) == ["a", "b"]
    assert predictions[0].to_numpy().min() >= 0.0 and predictions[0].to_numpy().max() <= 1.0


@requires_llmrouterbench
def test_latency_variants_scale_every_model_and_leave_the_default_alone():
    from src.run_real_benchmark import latency_profiles
    from src.table_oracle import DEFAULT_LATENCY_PROFILES
    assert latency_profiles("llmrouterbench", "default") is None
    for name, ttft, rate in latency_profiles("llmrouterbench", "slow"):
        base_ttft, base_rate = DEFAULT_LATENCY_PROFILES[name]
        assert ttft == pytest.approx(2.0 * base_ttft) and rate == pytest.approx(0.5 * base_rate)
    first, again = latency_profiles("llmrouterbench", "shuffle1"), latency_profiles("llmrouterbench", "shuffle1")
    assert first == again
    speeds = [rate / DEFAULT_LATENCY_PROFILES[name][1] for name, _, rate in first]
    assert all(0.5 <= s <= 2.0 for s in speeds)
    assert first != latency_profiles("llmrouterbench", "shuffle2")


@requires_llmrouterbench
def test_prior_switch_changes_only_the_controllers_estimate():
    from src.prompt_embeddings import PromptFeatures
    from src.table_oracle import (OutcomeTable, TableConfig, TableOracle, load_table_tasks,
                                  make_table_agents)
    table = OutcomeTable("llmrouterbench", TableConfig(time_unit_seconds=30.0), seed=0)
    agents = make_table_agents(table)
    prompts = PromptFeatures("llmrouterbench", 8)
    tasks = load_table_tasks(table, agents, 1, 0.5, limit=5, split="test", prompt_features=prompts)
    knn, mlp = TableOracle(table, agents, prompts), TableOracle(table, agents, prompts, prior="mlp")
    for task in tasks:
        for agent in agents:
            assert knn.prior_quality(task, agent.agent_id) == knn.knn_quality(task, agent.agent_id)
            assert mlp.prior_quality(task, agent.agent_id) == mlp.router_quality(task, agent.agent_id)
            assert mlp.knn_quality(task, agent.agent_id) == knn.knn_quality(task, agent.agent_id)
    with pytest.raises(ValueError):
        TableOracle(table, agents, prompts, prior="oracle")
