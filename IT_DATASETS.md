# IT benchmark: design, sources, and scientific scope

## 1. Primary controlled routing benchmark

The paper's primary IT-routing workload is built by `src.it_benchmark` from one pinned source:

- dataset: [StackPulse](https://huggingface.co/datasets/Omarrran/StackPulse_778K_QnA_Code_dataset), configuration `high_quality`;
- revision: `18c9ddf`;
- sample: 200 tasks for each of six domains (1,200 total), seed 42;
- domains: Frontend, Backend/API, Databases/Data Engineering, ML/LLM, DevOps/Cloud, Security;
- labels: explicit Stack Overflow tags only; prompt keywords are never class ground truth;
- multi-label information: `skill_vector` retains every recognized domain tag, while `primary_skill` defines the balanced evaluation stratum;
- global uniqueness: a Stack Overflow question can appear in only one primary class.

Using one source for all classes controls the strongest defect in the earlier draft: a model cannot identify the class merely from source-specific formatting. Selection is balanced by class and spread round-robin across code presence, prompt length, and a declared difficulty proxy. It does not claim random sampling from the natural Stack Overflow population; it is a controlled, stratified routing benchmark.

Build and audit it with:

```bash
python -m src.audit_it_benchmark --per-skill 200 --seed 42 \
  --output data/it_routing_benchmark.jsonl \
  --audit-output data/it_routing_benchmark.audit.json
```

Text export is disabled by default. The manifest retains source URLs and metadata. Stack Overflow posts have revision-dependent CC BY-SA licenses; the wrapper dataset's Apache metadata must not be treated as relicensing the upstream posts. If text is redistributed, attribution and the applicable per-post license must be handled explicitly.

### Quality gates

Construction fails instead of silently producing a paper artifact when any gate is violated:

| Gate | Limit | Seed-42 result |
|---|---:|---:|
| Class count delta | 0 | 0 |
| Globally unique normalized prompts | >= 99% | 100% |
| Explicit-tag label purity | 100% | 100% |
| Largest/smallest class median prompt length | <= 3.0 | 1.697 |
| Maximum class-vs-global task-kind total variation | <= 0.55 | 0.190 |

The six class counts are exactly 200. Median prompt lengths range from 599 to 1,016.5 characters. These are dataset diagnostics, not evidence that agents possess different skills.

### What this benchmark can and cannot establish

It can test routing, queueing, assignment, regret under a fixed replay oracle, and whether learned task/agent compatibility improves system objectives under controlled IT-domain inputs.

It cannot by itself measure answer correctness: StackPulse supplies question metadata/text but the selected configuration has no reference answers. `task_kind` and `difficulty_bucket` are deterministic heuristic covariates, not gold labels. Tag membership is an observable proxy for domain, not proof that a task requires only that skill. Claims that agents specialize differently require a separately measured agent-by-domain capability matrix from real LLM runs; until those runs exist, specialization must be described as simulated.

To avoid leakage, train/validation/test splitting must group by source question ID before any fitting or coefficient selection. Report both macro averages (equal domain weight) and per-domain results. Do not report only micro averages.

## 2. External-validity / correctness track

Heterogeneous datasets remain useful as separate stress tests, not as one pooled six-class routing benchmark. Pooling them would reintroduce `source == class`, formatting, length, and task-type confounding.

| Intended stress test | Dataset | License shown by source | Use |
|---|---|---|---|
| Backend/API tool calls | [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | Structured tool/API behavior |
| SQL generation | [SQL Create Context](https://huggingface.co/datasets/b-mc2/sql-create-context) | CC BY 4.0 | Schema-conditioned SQL |
| ML/data-science code | [DS-1000](https://github.com/xlang-ai/DS-1000) | CC BY-SA 4.0 | Executable correctness tests |
| DevOps QA | [DevOps QA](https://huggingface.co/datasets/Skilln/devops-qa-dataset) | CC BY-SA 4.0 | Short operational QA |
| Defensive security | [Trendyol Cybersecurity](https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset) | Apache-2.0 | Security instruction following |

`src.it_dataset_loader` is retained only for this heterogeneous external-validity track. Its keyword-derived soft overlaps are weak labels and must not be used as the paper's ground truth.

The earlier Frontend candidate `xlelords/vulcan` is rejected for scientific evaluation: inspection found thousands of rows but only 14 unique prompts, largely output-style variants. It must not be used to claim frontend coverage or class balance.

## 3. Remaining pre-run requirements

Before real paper experiments:

1. freeze source revisions and archive the audit JSON with every run;
2. define grouped train/validation/test splits and confirm zero question-ID overlap;
3. obtain real per-agent, per-domain outcomes on a disjoint calibration subset;
4. specify the correctness judge/tests for each external-validity dataset;
5. report confidence intervals across seeds and paired comparisons on identical replay tables;
6. keep the primary routing claims separate from external correctness claims.
