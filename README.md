# Adaptive Decentralized Multi-LLM Orchestration with Accelerated Consensus-Based SPSA

*Anonymous submission*

## Abstract

We study the problem of online parameter balancing in a decentralized multi-LLM
task orchestration system operating under partial observability and bounded disturbances.
Building on a prior two-layer architecture — in which adaptive controllers predict
task parameters via recursive least squares and synchronize models via SPSA-based
consensus — we replace the baseline distributed SPSA with an accelerated
consensus-based variant (A-SPSA) that incorporates a Nesterov momentum term and a
quadratic surrogate function.
We evaluate the adaptation in a high-fidelity simulation with five heterogeneous
LLM agent profiles across seven zeroth-order distributed optimization algorithms
using three metrics designed to capture routing quality, deadline compliance, and
wait-prediction accuracy; the simulation uses real task prompts drawn from five
public benchmarks (MBPP, TriviaQA, XSum, OPUS-100, Hermes-FC) and is further
validated on live LLM API calls via a shared-execution-cache protocol.
A-SPSA achieves the highest deadline hit rate and lowest wait-prediction loss
among all compared methods, and ranks first on routing objective with statistically
significant gains over four of six baselines.

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
