# Handoff: global-buffer A-SPSA multi-LLM architecture

This file captures the decisions made while deriving the new architecture from
the original repository, handwritten diagrams, and the new Scientific Reports
draft. It is intended as context for continuing the work in a new coding
session opened directly in this repository.

## Repository boundaries

- Original research repository: `C:\tarasova\KDD\decentralized-llm-spsa`
- New independent fork: `C:\tarasova\sc_rep_a_spsa_llm`
- Do not modify the original repository while developing the new architecture.
- The fork has its own `.git` directory and isolated `.venv`.
- New article archive copied into the fork: `Sc_Rep_26_A_SPSA_LLM.zip`.

## Why the architecture was changed

The old implementation immediately selected a controller and then an agent for
each arriving task. The controller effectively owned the dispatch decision,
and an agent could accumulate queued assignments.

The new architecture introduces a global scheduling buffer. Controllers no
longer assign work. They estimate how well every agent could perform the task
assigned to that controller. The buffer owns the final scheduling decision and
only dispatches a task when an agent is free.

## Final architecture agreed in the conversation

1. A stateless classifier/feature extractor processes an incoming task.
2. Load balancing assigns the task to exactly one responsible controller.
3. The responsible controller evaluates every pair `(task i, agent k)`.
4. For every pair, the controller publishes a feature vector and scalar base
   score into one global buffer.
5. The controller does not assign an agent and does not change a proposal after
   it has been buffered.
6. The buffer updates only quantities derived from current time and agent
   availability.
7. The buffer compares all pending task--agent pairs and performs the final
   matching.
8. One task is executed by exactly one agent.
9. An agent receives a new task only after finishing its current task.
10. Alternative proposals for an assigned task remain in history but become
    logically invalid and can never produce a duplicate assignment.
11. The judge evaluates each executed request--response pair independently.
12. Feedback is returned only to the controller responsible for that task.
13. An execution/API failure is a terminal unsuccessful result; there is no
    implicit retry.
14. Controller routing weights are local and adaptive.
15. RLS performs local prediction updates after completed feedback.
16. A-SPSA periodically synchronizes prediction models across controllers.

## Clarification about candidate pairs

A task belongs to one controller, so the same `(task, agent)` pair cannot be
published by multiple controllers. Before dispatch, however, one task has one
candidate pair per agent:

```text
(T_i, E_1), (T_i, E_2), ..., (T_i, E_n).
```

After the buffer chooses one pair, all other pairs of that task are marked
`INVALIDATED`. They are retained for auditing but excluded from scheduling.

## Controller feature vector and score

The controller builds a vector for every task--agent pair. The inherited vector
contains the following components:

```text
f_ik = [
    relevance,
    availability,
    predicted_quality,
    time_score,
    queue_score,
    cost_score,
    reputation,
    urgency
].
```

The controller owns local weights `w_j` and publishes the scalar suitability
score

```text
s_jik = w_j^T f_ik.
```

The score answers: "How well can agent `k` execute task `i`?" The controller's
base score is immutable after publication. Availability and queue components
are owned by the buffer and are dynamically refreshed there.

## Time and deadline quantities from the new article

For task `i` and agent `k`:

```text
finish_hat_ik(t) = max(t, available_k(t)) + p_hat_ik

d_ik(t) = max(0, finish_hat_ik(t) - D_i) / b_i

r_ik(t) = (
    max(t, available_k(t)) - arrival_i + p_hat_ik
) / b_i

K_ik(t) = (
    D_i - max(t, available_k(t))
) / (p_hat_ik + epsilon_K)
```

`K_ik` is the slack coefficient for the pair, not an agent index and not a
team:

- `K_ik > 1`: predicted completion has relative slack;
- `K_ik ~= 1`: predicted completion is close to the deadline;
- `0 < K_ik < 1`: insufficient time is predicted;
- `K_ik <= 0`: the agent is predicted to become available at or after the
  deadline.

If every pair has non-positive slack, the task is still scheduled in
best-effort mode. A hard rejection policy was not selected.

## Suitability versus scheduling priority

Agent suitability and task hotness are intentionally separated. Directly
rewarding `1 / K_ik` inside the agent score could make a poorly timed agent win
simply because that particular pair is in danger.

The buffer first computes task-level hotness using the best slack available for
the task:

```text
K_i_best(t) = max_k K_ik(t)

H_i(t) = sigmoid(alpha_H * (1 - K_i_best(t))).
```

The dynamic scheduling priority is

```text
P_jik(t) = s_jik
         + lambda_A * availability_ik(t)
         + lambda_H * H_i(t)
         - lambda_D * d_ik(t)
         - lambda_R * max(0, r_ik(t) - 1).
```

`H_i` is identical for all candidate agents of one task. It raises a hot task
in the global buffer without changing which agent is most suitable for that
task.

## Global matching

At every scheduling event, the buffer sorts feasible pending edges by dynamic
priority and greedily reserves distinct tasks and distinct free agents.

The conceptual optimization problem is

```text
maximize  sum_i sum_k x_ik P_ik(t)

subject to
    sum_k x_ik <= 1       for every task i
    sum_i x_ik <= 1       for every free agent k
    x_ik in {0, 1}.
```

The current implementation uses deterministic greedy matching. Replacing it
with exact maximum-weight bipartite matching is a possible future experiment,
not an unresolved architecture question.

## Time model

The architecture is event-driven.

Simulation events:

- one or more task arrivals;
- an agent completion;
- judge feedback after completion;
- an A-SPSA synchronization round after enough completed feedback.

Numerical experiments may use discrete ticks. For example, five tasks can
arrive at one tick and service durations can be quantized to integer ticks.
The simulator advances to the next arrival or completion rather than polling
every empty time unit.

Real endpoint mode uses the same state transitions with continuous timestamps
and measured API latency.

## Feedback and learning

Only the task's responsible controller receives feedback. Feedback contains
the executed agent, quality, success, latency, wait, cost, and deadline result.

Local updates:

- RLS processing-time predictor;
- RLS admissible-wait predictor;
- RLS response-quality predictor;
- local agent reputation;
- local routing weights `w_j`.

Routing weights remain controller-specific. They currently use a projected
online utility update with a small anchor toward the initial policy. They are
not strongly synchronized because that would remove controller specialization.

A-SPSA synchronizes the processing-time, waiting-time, and quality prediction
models after completed observations accumulate. It does not make the final
dispatch decision.

## Compatibility constraints

The current text/code LLM experiments use soft relevance rather than binary
capability restrictions: code-specialized models can process natural language,
but perform different task types with different quality.

An optional hard compatibility mask may be added later for heterogeneous agents
such as embedding-only, speech, image, tool-restricted, or context-limited
models. It is not required by the current experiments.

## Implemented files

- `src/assignment_buffer.py`
  - candidate/task states;
  - dynamic `K_ik`, deadline gap, budget ratio, hotness, and priority;
  - global greedy matching;
  - terminal completion/failure state machine.
- `src/buffered_orchestrator.py`
  - classification;
  - LVP-style controller load balancing;
  - evaluation of every task--agent pair;
  - event queue;
  - dispatch and delayed completion;
  - synthetic and real API execution;
  - judge and controller-local feedback;
  - local weight adaptation and periodic A-SPSA.
- `src/main.py`
  - controller quality observation history;
  - local routing-weight adaptation.
- `src/spsa_variants.py`
  - A-SPSA response-quality loss and synchronization.
- `src/run_buffered_simulation.py`
  - CLI numerical runner;
  - batch/tick parameters;
  - CSV exports.
- `tests/test_assignment_buffer.py`
  - buffer state and matching invariants.
- `tests/test_buffered_orchestrator.py`
  - end-to-end architecture invariants.
- `ARCHITECTURE.md`
  - concise technical specification.

## Verification already completed

Focused test command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_assignment_buffer.py `
  tests\test_buffered_orchestrator.py `
  tests\test_main.py
```

Result at handoff:

```text
27 passed
```

Smoke run command:

```powershell
.\.venv\Scripts\python.exe -m src.run_buffered_simulation `
  --tasks 20 `
  --controllers 3 `
  --agents 5 `
  --seed 42 `
  --batch-size 5 `
  --tick-duration 1 `
  --output-dir smoke_outputs
```

Observed smoke invariants:

- 20 tasks buffered;
- 20 unique assignments;
- 20 terminal results;
- no duplicated task execution;
- integer service times in tick mode;
- two A-SPSA synchronization events.

The copied legacy dataset tests expect local MBPP and TriviaQA files that are
absent from both the original repository and the fork. They fall back to one
template and therefore fail their minimum-pool-size assertions. This is a data
availability issue, not a global-buffer failure. Several legacy benchmark
tests are also long-running and exceeded the interactive test timeout without
showing failures before termination.

## Useful commands in a new session

```powershell
cd C:\tarasova\sc_rep_a_spsa_llm

# Focused architecture tests
.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_assignment_buffer.py `
  tests\test_buffered_orchestrator.py `
  tests\test_main.py

# Numerical experiment
.\.venv\Scripts\python.exe -m src.run_buffered_simulation `
  --tasks 200 --controllers 3 --agents 5 `
  --seed 42 --batch-size 5 --tick-duration 1 `
  --output-dir buffered_outputs

# Inspect uncommitted implementation
git status --short
git diff --check
```

## Current repository state

At handoff, implementation changes are intentionally uncommitted. The original
repository under `C:\tarasova\KDD` is clean and was not modified. Continue all
new work inside `C:\tarasova\sc_rep_a_spsa_llm`.
