"""Policy-layer experiment on measured outcomes and the per-block step path."""
from pathlib import Path

import numpy as np
import pytest

from src import real_benchmarks

ARCHIVE = Path("data/external/llmrouterbench/bench-release")
EMBEDDINGS = Path("data/external/processed/llmrouterbench_emb_all-MiniLM-L6-v2.parquet")
AZURE = Path("data/external/azure/AzureLLMInferenceTrace_code.csv")
requires_llmrouterbench = pytest.mark.skipif(
    not ARCHIVE.exists() or not EMBEDDINGS.exists() or not AZURE.exists(),
    reason="LLMRouterBench, its embeddings or the Azure trace are absent; see REAL_BENCHMARK_REPORT.md",
)


def test_sites_partition_the_six_domains():
    from src.policy_real import site_domains
    groups = [site_domains(site, 5) for site in range(5)]
    assert set().union(*groups) == set(range(6))
    assert sum(len(g) for g in groups) == 6
    assert groups[0] == {0, 1}


TINY = dict(benchmark="llmrouterbench", sites=5, rounds=2, slots=1, rho=2.0,
            tasks_per_site=12, embedding_dim=8, omega=0.3, test_seeds=[11])


@requires_llmrouterbench
def test_optimizer_run_records_one_trace_value_per_round():
    from src.policy_real import run_job
    row = run_job({"cfg": TINY, "noise": "noisy", "method": "aspsa", "optimizer_seed": 3,
                   "oracle_seeds": [131]})
    import json
    trace = json.loads(row["trace"])
    assert len(trace) == TINY["rounds"]
    assert all(0.0 <= v <= 1.0 for v in trace)          # deadline-miss rates
    # Two-point estimate: two oracle calls per site per round.
    assert row["oracle_calls"] == 2 * TINY["sites"] * TINY["rounds"]
    assert 0.0 <= row["deadline_hit_rate"] <= 1.0


@requires_llmrouterbench
def test_hand_set_reference_spends_no_oracle_calls_and_is_deterministic():
    from src.policy_real import run_job
    job = {"cfg": TINY, "noise": "reference", "method": "hand-set defaults",
           "optimizer_seed": -1, "oracle_seeds": []}
    first, second = run_job(job), run_job(job)
    assert first["oracle_calls"] == 0
    assert first["deadline_hit_rate"] == second["deadline_hit_rate"]


@requires_llmrouterbench
def test_site_stream_contains_only_its_own_domains():
    from src import table_oracle
    from src.policy_real import _world, site_domains
    table, agents, prompts = _world("llmrouterbench", 1, 8)
    for site in range(5):
        tasks = table_oracle.load_table_tasks(table, agents, 131, 2.0, limit=30, split="train",
                                              prompt_features=prompts, domains=site_domains(site, 5))
        assert {t.domain for t in tasks} <= site_domains(site, 5)


@requires_llmrouterbench
def test_per_block_step_activates_only_the_named_block():
    from src import run_real_benchmark as R
    base = dict(benchmark="llmrouterbench", tasks=40, per_domain=None, controllers=3, embedding_dim=8,
                controller_assignment="load",
                table_cfg=dict(time_unit_seconds=30.0, arrival_process="azure_code", slo_scale=3.0),
                split="validation", seed=1000, price_weight=-0.08, step_size=0.04,
                max_processing_units=100.0, max_waiting_units=200.0, rls_prior_strength=100.0,
                pair_quality_feature=True, diffusion_mixing=0.0, specialization=0.5,
                regime="capacity", slots=4, rho=0.85, latency_weight=-0.16)
    waiting_only = R.run_one({**base, "system": "aspsa", "blocks": ("waiting",),
                              "block_steps": {"waiting": 5e-2}})
    rls = R.run_one({**base, "system": "local_rls"})
    # The processing and quality blocks are left to RLS: their losses match.
    assert waiting_only["f1"] == pytest.approx(rls["f1"])
    assert waiting_only["f3"] == pytest.approx(rls["f3"])
    assert waiting_only["blocks"] == "waiting"
    assert R.BLOCK_CRITERION == {"processing": "f1", "waiting": "f2", "quality": "f3"}
