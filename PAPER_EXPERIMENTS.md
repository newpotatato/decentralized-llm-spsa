# Controlled paper experiment protocol

This is the paper-facing protocol for the balanced six-domain IT benchmark. It is separate from the legacy five-task experiment path.

## Scope and evidence level

The current replay capability matrix and task-agent potential outcomes are simulated. The run validates the optimization, routing, queueing, and statistical protocol; it does not establish superiority on real LLM endpoints. Real LLM runs remain a future validation stage.

All paper-scale comparisons use grouped 60/20/20 train/validation/test splits by Stack Overflow question ID, deterministic potential outcomes for every `(task, agent, seed)`, external difficulty-dependent deadlines, and ten paired test seeds. Only the outcome of the selected task-agent pair is observable.

## Manuscript objectives

`src.paper_losses` implements the three losses from the manuscript:

- F1: relevance-corrected, hardness-scaled pseudo-Huber processing-time loss, including the manuscript ridge term `(lambda_reg_p / 2) ||theta||^2`;
- F2: asymmetric squared waiting-time loss with the larger penalty on underprediction;
- F3: hardness-weighted quality error with an additional relevance-weighted overprediction penalty.

`R_ik` in F1 and F3 is the manuscript's semantic relevance, the cosine of the task semantic vector and the agent specialisation profile (`src.paper_experiment.semantic_relevance`). It is observable to a controller. Earlier revisions substituted the oracle capability matrix inside F1 and task urgency inside F3; both substitutions are removed.

Optimizer evaluation uses the three independent network objectives, not routing utility:

`Phi_nu = sum_j F_j^nu(theta_j) + (omega_nu / 4) sum_j sum_l b_jl ||theta_j - theta_l||^2`, for `nu in {p,w,q}`.

Here `F_j^nu` is the controller's rolling-window empirical objective, the controllers form a ring, `b_jl = 1/3` for the two ring neighbours, and `omega_p = omega_w = omega_q = 0.002`.

### Why Phi is reported in decomposed form

`Phi` mixes prediction accuracy with parameter agreement. At the earlier `omega = 0.05` the agreement penalty reached 0.6-1.9 while the whole local objective was 0.02-0.18, so a ranking by `Phi` alone ranked methods by how hard they averaged parameters. Two changes follow:

1. `omega` is calibrated to 0.002 so the penalty is commensurate with the term it regularises;
2. `local_*` and `disagreement_*` are prespecified primary endpoints in their own right, alongside `phi_*`.

A method may therefore win `Phi` and still be reported as less accurate. Both facts are published.

## Fair optimizer comparison

SPSA, A-SPSA, Kiefer-Wolfowitz, ZO-PGD, SP-GT, ZO-GT, and PD-2PT update the same processing, waiting, and quality parameter blocks and call the same block-local objectives. Three properties make the comparison a comparison of optimizers rather than of their configurations.

**One consensus mechanism.** Every method adds the same correction `omega * sum_l b_jl (theta_j - theta_l)` to its descent direction, which is the exact gradient of the disagreement term of `Phi`. The realised consensus strength is therefore `omega * h` for all of them. Earlier revisions mixed the baselines towards the ring average by a step-independent 5% while A-SPSA received `omega * h`; with a tuned `h = 12` that made A-SPSA average twelve times harder than the methods it was compared against, and accounted for most of its reported advantage on `Phi_p`.

**One perturbation scale.** The manuscript draws `Delta` from `{-1/sqrt(d), +1/sqrt(d)}^d`, for which `E[Delta Delta^T] = I/d` and the raw two-point estimate is biased towards `grad F / d`. The perturbation points are kept exactly as in the manuscript and the estimate is compensated by the factor `d`, so a step size means the same displacement for every method. Without this, no single step grid could serve both conventions.

**One step grid.** Six configurations per method and parameter block, from one shared four-decade grid, selected on validation. The methods genuinely do not share a usable step region: on a quadratic with `M = 2` the two-point estimators improve up to the theoretical `h = 1/M`, A-SPSA is best near `1e-3` and diverges above `5e-2` because the accelerated recurrence amplifies oracle noise, and single-point SP-GT only survives at `1e-4`. The grid is therefore wide rather than narrow, so each method's optimum lies inside it. `mu - eta` is derived from `omega * lambda_2(L_av)` rather than tuned.

The optimizer benchmark uses `fixed_replay`, a deterministic balanced assignment (`task_id mod number_of_agents`) with `H_sync = 5`. Predictions cannot change the selected feedback stream: for a given seed, every optimizer sees identical task, controller, agent, service-time, quality, and waiting-time observations. The event-driven global architecture is evaluated separately.

A-SPSA follows the manuscript recurrence using `gamma_(tau-1)` in the alpha equation and a fixed descent parameter `h`. The consensus correction is evaluated at the probing points `theta~`, as the manuscript specifies, not at the incoming parameters. The fixed-beta ablation fixes perturbation beta, not acceleration alpha. Objective-query counts are explicit: KW uses `2d` evaluations and is not presented as a two-query method.

### Recorded implementation choices

The manuscript leaves two quantities unspecified. Both are recorded in every artifact rather than chosen silently.

- `alpha_0` is not specified; the neutral choice `alpha_0 = 0` is used.
- `z_{j,0} = 0` is specified. For a regression model whose parameters are not centred at the origin this is an implicit shrinkage prior, so `surrogate_init` is a selectable field and the manuscript value remains the default.

### Theory conditions are checked, not assumed

`src.paper_optimizers.theory_parameters` computes the manuscript prescription `mu = omega * lambda_2(L_av)`, `h = 1 / max{M_nu, omega * lambda_max(L_av)}`, and `0 < eta < mu` from the verified Lipschitz constants `M_p = X^2/eps_p^2`, `M_w = 2 a_w^- X^2`, `M_q = 2(1 + a_h^q + a_R^q) Z^2`. Nothing in the search enforces these bounds, so `experiment_manifest.json` records them under `theory_check` and any violation is visible. For the paper loss configuration `M_p = 400`, so the prescribed processing step is `h = 0.0025`; the previous run selected `h = 12`, roughly 4800 times the theoretical bound.

### Query budget

Every summary reports `sync_rounds`, `tuned_parameters`, `queries_per_parameter`, the per-block ratios `queries_per_parameter_{processing,waiting,quality}`, `worst_block_queries_per_parameter`, and the boolean `query_budget_sufficient`.

A two-point zeroth-order method cannot resolve a block whose dimension exceeds its query budget. The ratio must be read per block, not pooled: every block receives the same two evaluations per controller per round whatever its dimension, while the blocks differ in size by an order of magnitude. At the selected 3/12 configuration a run completes 43 synchronization rounds, so each block gets 86 evaluations per controller against 108 processing parameters, 9 waiting parameters, and 108 quality parameters. The two-point methods therefore sit at 0.80 on processing and quality and 9.56 on waiting; the pooled ratio of 1.15 looks sufficient and is not. `query_budget_sufficient` tests the worst block and is false for every two-point method. Coordinate-wise KW spends `2d` evaluations per round and is the one method that does resolve the blocks, at roughly 75 times the query cost.

A ranking obtained below one query per parameter reflects the deterministic parts of each recurrence, not gradient information, and must be reported as such. This is a budget limitation of the pinned 1,200-question benchmark at a 60/20/20 split, not a defect of any method.

## Architecture evaluation

The operational comparison includes legacy local queues, random routing, static skill routing, shortest queue, global greedy matching, global exact matching, centralized exact decisions, and an oracle upper bound. All systems instantiate at least three controllers and three agents; centralized/oracle modes use one logical authority.

The bounded search covers 3--7 controllers, 3/4/6/8/10/12/15/18 agents, and synchronization intervals 5/10/20. The agent grid extends past twelve so a selected configuration is not pinned to the boundary of its own search space. Train produces a quality/resource-diverse shortlist. Validation first requires deadline-hit rate at least 0.80, keeps candidates within 0.01 mean quality of the best, and then orders by smallest infrastructure, higher deadline-hit rate, lower p95 wait, and lower cost. Routing utility is absent from this rule.

### Buffer priority

The scheduling priority is

```text
P_jik(t) = s_jik + lambda_H * H_i(t) + lambda_W * W_i(t)
         - lambda_D * d_ik(t) - lambda_R * max(0, r_ik(t) - 1).
```

There is no availability term. Dispatch considers agents with `available_k(t) <= t` only, so `availability_ik(t)` evaluates to `exp(0) = 1` for every candidate edge of a dispatch round: the term added the same constant to every edge and could never change a matching. `availability_score` is still refreshed and retained as an audit field.

`H_i(t)` uses the best slack among the agents this dispatch round can actually reach. Taking the best slack over every agent, including occupied ones, reported a comfortable task when the only options available now were critical.

`W_i(t)` is waiting pressure, `min(elapsed / w_hat_i, cap)`. It is task-level and never redefines the external deadline `D_i` or the time budget `b_i`, so the waiting model influences dispatch order without defining the metric it is scored on. Before this term the waiting predictor was written into the proposal and read by nothing: `Phi_w` was a primary endpoint whose improvement could not change any operational outcome, and the manuscript's motivation for F2 was not true of the implemented system.

### Offered load

`rho` is normalised against a fixed canonical six-agent capacity so that offered load does not change when a candidate architecture changes its agent count. A system with more agents than the reference therefore runs cooler than `rho` suggests: at `rho = 0.85` a twelve-agent system sits near 0.43 utilisation, where queues are rare and greedy and exact matching frequently coincide. Every summary reports `rho_effective` and `mean_utilization` so a load level cannot be read as a utilisation. Architecture endpoints are mean quality, deadline-hit rate, p95 waiting time, throughput, mean cost, and Jain fairness, complemented by a burst stress test at `rho = 2.0`, batch size 6. Load evaluation uses `rho = 0.50, 0.85, 1.00, 1.15, 1.40, 2.00`.

## Statistical reporting

Comparisons use paired Wilcoxon signed-rank tests. Holm correction is applied across variants separately within each prespecified endpoint.

Every significance row also reports `effective_pairs`, `p_resolution_floor`, and `p_at_resolution_floor`. The exact signed-rank test discards tied pairs, so with ten seeds the smallest attainable two-sided p is `2/2^n` for the `n` non-zero pairs. Every significant row of a ten-seed design sits on that floor: the test reports only that the sign was consistent and cannot separate a `1e-8` effect from a large one. `practical_effect_ge_1pct` is a descriptive 1% relevance flag, not an additional hypothesis test, and it is the column that distinguishes the two cases. Means and 95% confidence intervals are exported for plots.

## Known limitations of the simulated generative model

`ReplayOracle` draws service time from `mean_service[domain] * (1 + 0.7 * hardness)` and quality from `capability[domain] - 0.18 * hardness`, both with independent noise. The controller features are the domain vector, hardness, and two load coordinates. The true model therefore lies exactly inside the linear hypothesis class of the predictors, with no misspecification and i.i.d. noise. This is a favourable setting for recursive least squares and leaves little room for a zeroth-order method to add anything. It is a deliberate property of a controlled replay, and it bounds what the optimizer comparison can claim.

`hardness` takes three values and the domain vector is near one-hot, so there are at most eighteen distinct feature patterns before the two load coordinates. On the test split each controller sees roughly one observation per `(agent, domain)` cell.

## Commands

Fast protocol check:

```bash
python -m src.run_paper_experiments --profile pilot \
  --output-dir paper_experiment_outputs_corrected_pilot
```

Paper-scale simulated run:

```bash
python -m src.run_paper_experiments --profile full \
  --output-dir paper_experiment_outputs_corrected_full
```

Every artifact directory carries `experiment_manifest.json` with `code_revision`, the selected optimizer parameters, the theory check, and the protocol description. Do not use pilot directories, or any directory produced before the fairness corrections, as paper evidence. The audited artifacts are in `paper_experiment_outputs_corrected_full`; see `PAPER_RESULTS_AUDIT.md`.
