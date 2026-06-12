# Adaptive Decentralized Multi-LLM Orchestration with Accelerated Consensus-Based SPSA

*Anonymous submission*

## Abstract

Adaptive routing in multi-LLM systems is difficult to evaluate reliably because different policies may observe different model outputs,
latencies, costs, and judge scores. We introduce a replay-based evaluation protocol that fixes both the task stream and the potential
outcome of each task–agent pair, allowing routing policies to be compared under identical conditions. Within this framework, we adapt
accelerated consensus-based simultaneous perturbation stochastic approximation (A-SPSA) to the online optimization of controller
parameters used for processing-time and wait-time prediction. The method combines two-point zeroth-order feedback, accelerated
surrogate updates, and consensus among distributed controllers. We evaluate A-SPSA against six distributed zeroth-order baselines in
a discrete-event simulation with heterogeneous agents, and further validate the policies using cached outputs from real LLM endpoints.
The experiments include aggregate comparison, component-level ablation, and hyperparameter sensitivity analysis. A-SPSA achieves
the strongest overall balance between routing objective, deadline hit rate, and wait-time prediction error, while the ablation results
show that lower intermediate prediction loss does not necessarily lead to better downstream routing decisions. The proposed protocol
provides a reproducible basis for evaluating adaptive agent-routing policies under stochastic and partially observable conditions.

## Repository Structure

```
paper/              LaTeX source (main.tex) and figures (figs/)
src/                Python simulation and plotting code
  main.py               core system model, RLS, A-SPSA, judge module
  spsa_variants.py      all 7 compared optimization algorithms
  run_spsa_comparison.py  main simulation experiment (Table 1)
  run_real_llm_comparison.py  real-LLM validation (Table 2)
  plot_real_llm.py      figures for Section 5 real-LLM results
  plot_paper_figures.py   figures for Sections 4–5 simulation results
  dataset_loader.py     MBPP / TriviaQA / XSum / OPUS-100 / Hermes-FC loader
  llm_api.py            OpenAI-compatible API client
  llm_config.py         [NOT COMMITTED] API keys and model aliases — see setup
  llm_config_example.py template for llm_config.py
tests/              pytest unit tests
requirements.txt    Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

### API keys (real-LLM mode only)

Copy the template and fill in your keys:

```bash
cp src/llm_config_example.py src/llm_config.py
# edit src/llm_config.py — add your Groq / OpenRouter key
```

Or export the Groq key as an environment variable (the config reads `GROQ_API_KEY`):

```bash
export GROQ_API_KEY="gsk_..."
```

## Reproducing Results

### Table 1 — simulation (seeds 1–5, ~5 min)

```bash
python -m src.run_spsa_comparison --tasks 500 --seeds 1,2,3,4,5 \
       --output-dir spsa_comparison
```

### Table 2 — real-LLM validation (requires Groq API key, ~30 min)

```bash
python -m src.run_real_llm_comparison --tasks 100 --seeds 42,43 \
       --output-dir real_llm_outputs
```

### Paper figures

```bash
python -m src.plot_paper_figures   # simulation figures → paper/figs/
python -m src.plot_real_llm        # real-LLM figures   → paper/figs/
```

### Compile the paper

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## License

MIT
