# Global-Buffer Adaptive Multi-LLM Orchestration with A-SPSA

This repository is an independent fork that implements an event-driven global
assignment buffer. Controllers evaluate task--agent pairs but never dispatch
tasks themselves. The global buffer owns the final one-task/one-agent matching
decision, refreshes time and availability terms, and dispatches work only to
free agents.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the state machine, equations, and
component responsibilities, and [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md)
for the experiment protocol.

*Anonymous submission*

## Abstract

Adaptive routing in multi-LLM systems is difficult to evaluate because policies can observe different feedback. This repository implements a replay protocol that fixes task streams and potential task-agent outcomes, an event-driven global assignment buffer, and distributed zeroth-order optimization of the three manuscript losses. The controlled optimizer benchmark uses exogenous balanced routing so every method receives identical feedback, applies one shared consensus mechanism and one shared step grid to every method, and reports prediction accuracy and controller disagreement as separate endpoints rather than only their weighted sum. Architecture experiments are reported separately using operational quality, deadline, waiting, throughput, cost, and fairness metrics. Real-LLM validation has not yet been run, and all agent capabilities are simulated.

## Repository Structure

```
src/                Python simulation and plotting code
  paper_experiment.py     controlled six-domain replay experiment
  paper_losses.py         manuscript losses F1/F2/F3
  paper_optimizers.py     the seven compared zeroth-order methods
  run_paper_experiments.py  paper protocol runner (tuning, search, ablation)
  it_benchmark.py         pinned Stack Overflow six-domain benchmark builder
  audit_it_benchmark.py   benchmark construction audit and quality gates
  assignment_buffer.py    global task--agent candidate buffer and matching
  buffered_orchestrator.py event-driven orchestration and delayed feedback
  run_buffered_simulation.py CLI for the global-buffer architecture
  main.py                 legacy system model, RLS, A-SPSA, judge module
  spsa_variants.py        legacy optimizer implementations (see caveat below)
  dataset_loader.py       legacy MBPP / TriviaQA / XSum / OPUS-100 loader
  llm_api.py              OpenAI-compatible API client
  llm_config.py           [NOT COMMITTED] API keys and model aliases
  llm_config_example.py   template for llm_config.py
tests/              pytest unit tests
data/               pinned routing manifest (versioned) and caches (ignored)
requirements.txt    Python dependencies
```

The manuscript LaTeX source is distributed as `Sc_Rep_26_A_SPSA_LLM.zip`. There
is no `paper/` directory in this fork.

### Two optimizer code paths

`src/paper_optimizers.py` is the implementation used by the manuscript
protocol. `src/spsa_variants.py` is the legacy ApPLIED-era implementation kept
for the older `run_spsa_comparison.py` and `run_real_llm_comparison.py`
entry points. The two differ in substance: the legacy module optimizes plain
MSE and a one-sided hinge rather than F1/F2/F3, draws perturbations from the
global NumPy RNG, and keeps its own copy of the parameters so RLS updates
between synchronization rounds are ignored. **Do not mix numbers from the two
paths.** Only `src.run_paper_experiments` produces paper evidence.

## Setup

```bash
pip install -r requirements.txt
```

### API keys (real-LLM mode only)

```bash
cp src/llm_config_example.py src/llm_config.py
# edit src/llm_config.py — add your Groq / OpenRouter key
```

Or export the Groq key as an environment variable:

```bash
export GROQ_API_KEY="gsk_..."
```

## Reproducing Results

### Controlled paper experiments

The protocol, losses, fairness constraints, and statistics are documented in
[PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md); the audited outcome is in
[PAPER_RESULTS_AUDIT.md](PAPER_RESULTS_AUDIT.md).

```bash
# fast protocol check (diagnostics only, marked NOT FOR PAPER)
python -m src.run_paper_experiments --profile pilot \
       --output-dir paper_experiment_outputs_corrected_pilot

# paper-scale simulated run
python -m src.run_paper_experiments --profile full \
       --output-dir paper_experiment_outputs_corrected_full
```

Every output directory carries `experiment_manifest.json` recording the commit
(`code_revision`), the selected optimizer parameters, the theory check, and the
protocol. Artifacts produced before the fairness corrections are not comparable
with these; see the audit for what changed and why.

### Rebuilding the benchmark

`data/it_routing_benchmark.jsonl` is versioned. To rebuild and re-audit it:

```bash
python -m src.audit_it_benchmark --per-skill 200 --seed 42 \
       --output data/it_routing_benchmark.jsonl \
       --audit-output data/it_routing_benchmark.audit.json
```

### Global-buffer architecture demo

```bash
python -m src.run_buffered_simulation --tasks 200 --controllers 3 --agents 5 \
       --seed 42 --output-dir buffered_outputs
```

The runner writes task, event, feedback, and summary tables. In simulation,
time advances to the next arrival or completion event. With real endpoints,
the same state machine uses measured API latency.

### Legacy comparison scripts

These use the legacy loss and optimizer path described above and are retained
for continuity, not as paper evidence:

```bash
python -m src.run_spsa_comparison --tasks 500 --seeds 1,2,3,4,5 \
       --output-dir spsa_comparison
python -m src.run_real_llm_comparison --tasks 100 --seeds 42,43 \
       --output-dir real_llm_outputs
```

## Tests

```bash
python -m pytest -q
```

`tests/test_protocol_fairness.py` pins the properties that make the optimizer
comparison a comparison of optimizers: one consensus operator, one perturbation
scale, one step grid, manuscript-faithful losses, an identifiable feature
matrix, and an honest report of the query budget and the statistical resolution
floor.

Tests that need an optional corpus skip with a reason when it is absent. They
never pass silently: when the corpus is present the assertions run in full.

## License

MIT
