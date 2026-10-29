"""Controlled event-driven replay experiment for the six IT domains.

This module is independent of the legacy five-task taxonomy. Potential outcomes
are fixed for every (task, agent, seed), so all routing and optimization methods
see the same counterfactual table while observing feedback only for selected
pairs.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
import hashlib
import heapq
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .assignment_buffer import BufferPolicy, CandidateProposal, GlobalAssignmentBuffer
from .it_dataset_loader import ITSkill
from .paper_losses import (
    LossObservation, PaperLossConfig, mean_losses, processing_loss,
    quality_loss, ridge_penalty, waiting_loss,
)
from .paper_optimizers import (
    DistributedZerothOrderOptimizer, ModelOptimizerConfigs, OptimizerConfig,
)


DOMAIN_ORDER = tuple(ITSkill)
DOMAIN_INDEX = {skill.value: index for index, skill in enumerate(DOMAIN_ORDER)}
DIFFICULTY = {"easy": 0.25, "medium": 0.60, "hard": 1.0}
TASK_KINDS = ("code_editing", "code_generation", "debugging", "design",
              "question_answering")
TASK_KIND_INDEX = {kind: index for index, kind in enumerate(TASK_KINDS)}
# Complexity indicators, standing in for the manuscript's set I_i.  The pinned
# manifest carries no prompt text, so the indicators are the structural signals
# that survive: whether the post contains code, whether it carries many tags,
# and whether the task kind is one of the two that require reading an existing
# artefact rather than producing a new one.
COMPLEXITY_INDICATOR_WEIGHT = 0.6
HARD_TASK_KINDS = frozenset({"debugging", "design"})


def task_complexity(prompt_chars: int, has_code: bool, tag_count: int,
                    task_kind: str, primary_share: float) -> float:
    """Manuscript appendix complexity score, normalised to [0, 1].

    ``h_i = clip(2 + 4 phi_{i,type} + 2 min(|s_i|/800, 1.5) + c_i^I, 1, 10)``

    ``phi_{i,type}`` is the share of the task's recognised tags that belong to
    its primary domain, ``|s_i|`` the prompt length, and ``c_i^I`` the weighted
    count of complexity indicators.  The three-valued ``difficulty_bucket``
    proxy it replaces took only the values 0.25, 0.60 and 1.0 and was derived
    from view counts, so it graded popularity rather than difficulty.
    """
    indicators = (int(bool(has_code)) + int(tag_count >= 4)
                  + int(task_kind in HARD_TASK_KINDS))
    raw = (2.0
           + 4.0 * float(primary_share)
           + 2.0 * min(float(prompt_chars) / 800.0, 1.5)
           + COMPLEXITY_INDICATOR_WEIGHT * indicators)
    return float(np.clip(raw, 1.0, 10.0) / 10.0)


def length_score(prompt_chars: int) -> float:
    """Saturating prompt length, the manuscript's ``min(|s_i|/800, 1.5)``."""
    return float(min(float(prompt_chars) / 800.0, 1.5) / 1.5)
EPSILON_RELEVANCE = 1e-9
# Largest attainable norm of the raw feature vector: one active domain
# indicator plus hardness, busy fraction and queue occupancy, each in [0, 1].
RAW_FEATURE_NORM = float(np.sqrt(6.0))
# Indices of the coordinates that describe system load rather than the task.
# The last two coordinates describe system load rather than the task.
LOAD_FEATURE_INDICES = (len(DOMAIN_ORDER) + 3, len(DOMAIN_ORDER) + 4)


def raw_feature_norm(embedding_dim: int = 0, pair_feature: bool = False) -> float:
    """Largest attainable norm of the raw feature vector.

    A prompt embedding is scaled to norm at most one (``prompt_embeddings``),
    so it adds one unit to the bound whatever its dimension; the pair quality
    coordinate lies in [0, 1] and adds one more.
    """
    return float(np.sqrt(6.0 + (1.0 if embedding_dim else 0.0) + (1.0 if pair_feature else 0.0)))
# Raised from 12 so the selected configuration is no longer pinned to the
# upper boundary of the architecture search grid.
MAX_AGENTS = 18
# rho is normalised against this fixed reference pool so that offered load is
# comparable across candidate architectures.  A system with more agents than
# the reference therefore runs at a lower true utilisation than rho suggests,
# which is why every summary also reports rho_effective and mean_utilization.
CANONICAL_AGENTS = 6


def semantic_relevance(skill_vector, capability, epsilon: float = EPSILON_RELEVANCE) -> float:
    """Manuscript ``R_ik = phi_i^T psi_k / (||phi_i|| ||psi_k|| + eps_R)``.

    This is the cosine similarity of the task semantic vector and the agent
    specialisation profile.  It is observable to a controller and is the only
    pair-level correction the manuscript uses inside F1 and F3.
    """
    phi = np.asarray(skill_vector, dtype=float)
    psi = np.asarray(capability, dtype=float)
    denominator = float(np.linalg.norm(phi) * np.linalg.norm(psi)) + epsilon
    return float(np.clip(float(np.dot(phi, psi)) / denominator, 0.0, 1.0))


@dataclass(frozen=True)
class ReplayTask:
    task_id: int
    source_id: str
    domain: int
    skill_vector: tuple[float, ...]
    hardness: float
    urgency: float
    prompt_chars: int
    has_code: bool
    arrival: float
    deadline: float
    length: float = 0.0
    task_kind: int = 0
    # Name of the domain slot.  Empty for the IT benchmark, whose names come
    # from ``DOMAIN_ORDER``; measured benchmarks bring their own six families.
    domain_name: str = ""
    # Low-dimensional prompt embedding, norm at most one.  Empty unless the
    # experiment sets ``embedding_dim``; it sits between the task covariates
    # and the load coordinates so the rest of the layout is unchanged.
    embedding: tuple[float, ...] = ()

    @property
    def domain_label(self) -> str:
        return self.domain_name or DOMAIN_ORDER[self.domain].value

    @property
    def features(self) -> np.ndarray:
        """One-hot primary domain plus task hardness.

        The replay oracle conditions the outcome on ``primary_skill`` alone, so
        the predictor must be indexed the same way.  Using the multi-label
        ``skill_vector`` here activated two or three domain coordinates for the
        16.7% of tasks that carry several recognised tags, while the target
        still came from one of them; least squares then split the credit and
        biased every domain coefficient towards a fraction of its true value.
        The full tag vector is retained on the task and is still what
        ``semantic_relevance`` consumes.
        """
        out = np.zeros(len(DOMAIN_ORDER) + 3, dtype=float)
        out[self.domain] = 1.0
        out[len(DOMAIN_ORDER)] = self.hardness
        out[len(DOMAIN_ORDER) + 1] = self.length
        out[len(DOMAIN_ORDER) + 2] = float(self.has_code)
        if self.embedding:
            out = np.concatenate((out, np.asarray(self.embedding, dtype=float)))
        return out


@dataclass(frozen=True)
class AgentSpec:
    agent_id: int
    name: str
    capability: np.ndarray
    mean_service: np.ndarray
    token_cost: float
    # Serving slots of one model share its predictor.  -1 means the agent is a
    # model of its own, as in the simulated pool.
    model_id: int = -1


@dataclass(frozen=True)
class PotentialOutcome:
    service_time: float
    quality: float
    success: bool
    cost: float


@dataclass(frozen=True)
class ExperimentConfig:
    controllers: int = 3
    agents: int = 6
    rho: float = 0.85
    seed: int = 42
    architecture: str = "global_exact"
    optimizer: str = "aspsa"
    use_learning: bool = True
    use_rls: bool = True
    sync_interval: int = 10
    warmup: int = 30
    history_size: int = 100
    task_limit: int | None = None
    arrival_batch_size: int = 1
    loss: PaperLossConfig = field(default_factory=PaperLossConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    optimizer_configs: ModelOptimizerConfigs | None = None
    optimizer_blocks: tuple[str, ...] = ("processing", "waiting", "quality")
    # How a task is assigned to a responsible controller.
    #   "load"   - least-loaded controller, independent of the task domain.
    #              Every controller then sees i.i.d. draws from one common
    #              distribution, so their models differ only by sampling noise
    #              and there is nothing for synchronization to transfer.
    #   "domain" - controllers own disjoint domain groups, which is the
    #              situation the manuscript motivates synchronization with: a
    #              controller never observes the domains it does not route, so
    #              those parameters can only improve through its neighbours.
    #   "skewed" - each controller owns a domain group but only receives a
    #              ``specialization`` share of that group's traffic, the rest
    #              being load balanced.  Disjoint ownership is the extreme case
    #              in which a neighbour's parameters carry no information about
    #              a controller's own traffic, only bias; a real deployment sits
    #              between the two, and that is where the coupling weight of the
    #              network objective has an interior optimum rather than being
    #              driven to full agreement or switched off.
    controller_assignment: str = "load"
    # Share of a domain group's traffic that goes to its owner under the
    # "skewed" assignment.  0 reproduces "load", 1 reproduces "domain".
    specialization: float = 0.5
    # Adapt-then-combine synchronisation.  After the local update, each
    # controller replaces its parameters by a convex combination with its ring
    # neighbours:  theta_j <- (1 - 2*eps) theta_j + eps theta_{j-1} + eps
    # theta_{j+1}.  This is the diffusion strategy of the distributed-estimation
    # literature, as opposed to the manuscript's consensus strategy, which adds
    # omega * sum_l b_jl (theta_j - theta_l) inside the descent step.
    #
    # The two differ by orders of magnitude in how much a neighbour matters: the
    # consensus term is scaled by the descent step, giving a realised strength
    # of omega*h ~ 1e-4 here, while the combination acts at the scale of the
    # estimate itself.  0 disables it; 1/(controllers) is full averaging.
    diffusion_mixing: float = 0.0
    # Weights of the controller's base suitability score, previously hard-coded.
    # Order: predicted quality, semantic relevance, predicted processing time,
    # token cost.  Exposing them makes the agent-choice half of the policy
    # tunable, which is what the quality objective acts on.
    base_score_weights: tuple[float, float, float, float] = (1.10, 0.40, -0.16, -0.08)
    # The manuscript's verification of Assumption 4 sets X_max = Z_max = 1 and
    # derives M_p = X^2/eps_p^2 = 400 from it.  The raw feature vector has
    # ||x|| up to 2, so every constant it states is understated by X_max^2 and
    # the prescribed step h = 1/M is four times too large.  Scaling the
    # features by their largest attainable norm is a change of units - it
    # rescales theta and h and leaves the model class untouched - which makes
    # the manuscript's own constants correct rather than approximately correct.
    normalize_features: bool = True
    # Strength of the prior in the local estimator, as a ridge in the
    # sum-of-squares sense: P_0 = I / rls_prior_strength.  This is a separate
    # quantity from loss.lambda_reg_*, which is a mean-form ridge inside the
    # A-SPSA objective and is sized to supply strong convexity.  Recursive
    # least squares minimises a SUM, the objective a MEAN, so one number cannot
    # play both roles - it would differ by the window length.  The previous
    # P_0 = 20 I made the prior almost vacuous and the fitted model ended up
    # further from the truth than the prior it started from.
    rls_prior_strength: float = 1.0
    # The replay oracle draws service time and quality independently of system
    # load, so the busy-fraction and queue coordinates carry no signal for
    # those two targets and only absorb noise.  Realised waiting time is a
    # queueing quantity and genuinely depends on both, so the waiting model
    # keeps them.  Set False to give every block all coordinates.
    load_features_for_waiting_only: bool = True
    # omega is calibrated so the disagreement penalty is commensurate with the
    # local losses it regularises.  At omega = 0.05 the penalty reached 0.6-1.9
    # while the whole local objective was 0.02-0.18, so Phi ranked methods by
    # how hard they averaged parameters rather than by prediction accuracy.
    # local_* and disagreement_* are additionally reported as separate primary
    # endpoints so this coefficient can no longer decide a conclusion alone.
    network_omega: tuple[float, float, float] = (0.002, 0.002, 0.002)
    # Dimension of the prompt embedding appended to the task covariates.  Zero
    # reproduces the manuscript feature vector.  Every published router
    # conditions on the prompt text; domain and hardness alone can only learn
    # per-family means and cannot tell two prompts of one family apart.
    embedding_dim: int = 0
    # Task--agent pair coordinate: the offline estimate of this agent's quality
    # on this prompt from its nearest train prompts (``TableOracle.prior_quality``).
    # The manuscript defines the controller features on the pair, f_ik; the
    # original implementation carried task features only, so the controller
    # could not tell two prompts of one family apart while the kNN baseline it
    # is compared against could.  Off by default.
    pair_quality_feature: bool = False
    # Upper clipping bounds of the processing and waiting predictors, in time
    # units.  They suit the simulated service times of order one; measured
    # reasoning-model latencies run to tens of minutes and need wider bounds,
    # or the predictor cannot represent the tail the scheduler must plan for.
    max_processing_units: float = 20.0
    max_waiting_units: float = 30.0
    # Initialise the processing and quality predictors from the agents' offline
    # profile (per-domain mean service and quality) instead of the generic
    # constants.  Measured-outcome runs set it: the profile is what an operator
    # knows from offline evaluation, the same train data the kNN and per-domain
    # baselines use.
    profile_prior: bool = False
    # Which tasks the global buffer serves first: the weighted priority of
    # ``BufferPolicy`` or one of the classic rules "edf", "llf", "sjf".
    buffer_order: str = "weighted"

    def __post_init__(self) -> None:
        if self.controllers <= 0 or self.agents <= 0:
            raise ValueError("controllers and agents must be positive")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if self.arrival_batch_size <= 0:
            raise ValueError("arrival_batch_size must be positive")
        valid_blocks = {"processing", "waiting", "quality"}
        if not self.optimizer_blocks or not set(self.optimizer_blocks) <= valid_blocks:
            raise ValueError("optimizer_blocks must select processing/waiting/quality")
        if len(self.network_omega) != 3 or any(weight < 0.0 for weight in self.network_omega):
            raise ValueError("network_omega must contain three non-negative coefficients")
        if self.controller_assignment not in {"load", "domain", "skewed"}:
            raise ValueError("controller_assignment must be load, domain or skewed")
        if not 0.0 <= self.specialization <= 1.0:
            raise ValueError("specialization must lie in [0, 1]")
        if not 0.0 <= self.diffusion_mixing <= 0.5:
            raise ValueError("diffusion_mixing must lie in [0, 0.5]")
        if self.architecture not in {
            "legacy_local_queue", "random", "static_skill", "shortest_queue",
            "global_greedy", "global_exact", "centralized_exact", "oracle",
            "fixed_replay", "static_domain", "quality_greedy", "cost_greedy",
            "knn_router", "mlp_router",
        }:
            raise ValueError(f"unknown architecture: {self.architecture}")
        if self.buffer_order not in BufferPolicy.ORDERS:
            raise ValueError(f"buffer_order must be one of {BufferPolicy.ORDERS}")


@dataclass(order=True)
class _Completion:
    finish: float
    sequence: int
    task_id: int = field(compare=False)
    controller_id: int = field(compare=False)
    agent_id: int = field(compare=False)
    predicted_processing: float = field(compare=False)
    predicted_waiting: float = field(compare=False)
    predicted_quality: float = field(compare=False)
    processing_regularizer: float = field(compare=False)
    assigned_at: float = field(compare=False)
    outcome: PotentialOutcome = field(compare=False)
    prediction_features: np.ndarray = field(compare=False)
    buffered: bool = field(compare=False, default=True)


def _stable_rng(seed: int, *parts: object) -> np.random.Generator:
    key = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def make_agent_pool(count: int) -> list[AgentSpec]:
    """Deterministic specialist/generalist pool; explicitly simulated."""
    if count <= 0 or count > MAX_AGENTS:
        raise ValueError(f"agent count must be in [1, {MAX_AGENTS}]")
    agents: list[AgentSpec] = []
    domains = len(DOMAIN_ORDER)
    for agent_id in range(count):
        specialist = agent_id % domains
        generation = agent_id // domains
        if generation == 0:
            capability = np.full(domains, 0.56)
            capability[specialist] = 0.90
            left, right = (specialist - 1) % domains, (specialist + 1) % domains
            capability[[left, right]] = 0.68
            service = np.full(domains, 2.45)
            service[specialist] = 1.05
            service[[left, right]] = 1.70
            label = f"specialist_{DOMAIN_ORDER[specialist].value}"
            token_cost = 0.22 + 0.035 * specialist
        elif generation == 1:
            capability = np.full(domains, 0.73 + 0.015 * (specialist % 3))
            capability[specialist] += 0.08
            service = np.full(domains, 1.55 + 0.08 * (specialist % 3))
            service[specialist] *= 0.82
            label = f"generalist_{specialist}"
            token_cost = 0.34 + 0.025 * specialist
        else:
            # A third tier keeps every additional agent distinct, so grid sizes
            # above twelve are real alternatives rather than duplicates of the
            # generalist tier.  It is cheaper and slower than tier one.
            capability = np.full(domains, 0.64 + 0.012 * (specialist % 4))
            capability[specialist] += 0.11
            service = np.full(domains, 1.95 + 0.06 * (specialist % 4))
            service[specialist] *= 0.75
            label = f"economy_{specialist}"
            token_cost = 0.16 + 0.02 * specialist
        capability = np.clip(capability, 0.0, 1.0)
        agents.append(AgentSpec(agent_id, label, capability, service, token_cost))
    return agents


def pool_capacity(agents: list[AgentSpec]) -> float:
    """Service capacity of an agent pool in tasks per unit time."""
    return float(sum(1.0 / float(np.mean(agent.mean_service)) for agent in agents))


# How strongly each domain's outcome responds to complexity and to prompt
# length.  A single shared slope, as the earlier oracle used, makes the optimal
# predictor identical for every controller no matter which domains it sees: the
# coefficient of a domain is a property of the agent, so a controller that only
# handles ML/LLM estimates the same number a controller handling everything
# would.  Domain-specific responses break that.  Debugging a long trace scales
# with the log; generating code from a short prompt does not, and an ML job hits
# memory limits that a frontend question never reaches.
#
# The predictors carry one complexity coefficient and one length coefficient per
# agent, not one per (agent, domain), so no single parameter vector fits every
# domain at once.  The best fit therefore depends on the domain mix a controller
# actually sees, which is what makes the sites genuinely heterogeneous - and
# what costs the common minimiser of Assumption 3.
DOMAIN_COMPLEXITY_RESPONSE = (0.35, 0.55, 0.80, 1.35, 1.00, 0.65)
DOMAIN_LENGTH_RESPONSE = (0.45, 0.30, 0.20, 0.10, 0.35, 0.55)
DOMAIN_QUALITY_RESPONSE = (0.10, 0.14, 0.20, 0.32, 0.24, 0.16)


class ReplayOracle:
    def __init__(self, seed: int, agents: list[AgentSpec]):
        self.seed = int(seed)
        self.agents = agents

    def outcome(self, task: ReplayTask, agent_id: int) -> PotentialOutcome:
        agent = self.agents[agent_id]
        rng = _stable_rng(self.seed, task.source_id, agent_id, "outcome")
        response = DOMAIN_COMPLEXITY_RESPONSE[task.domain]
        length_response = DOMAIN_LENGTH_RESPONSE[task.domain]
        mean = float(agent.mean_service[task.domain]) * (
            1.0 + response * task.hardness + length_response * task.length
        )
        service = max(0.05, mean * float(rng.lognormal(mean=-0.5 * 0.22**2, sigma=0.22)))
        quality = float(np.clip(
            agent.capability[task.domain]
            - DOMAIN_QUALITY_RESPONSE[task.domain] * task.hardness
            + rng.normal(0.0, 0.045),
            0.0, 1.0,
        ))
        success_probability = float(np.clip(0.10 + 0.90 * quality, 0.0, 1.0))
        success = bool(rng.random() < success_probability)
        tokens = 220.0 + 0.35 * task.prompt_chars + 180.0 * task.hardness
        cost = float(agent.token_cost * tokens / 1000.0)
        return PotentialOutcome(service, quality, success, cost)


class PaperController:
    def __init__(self, controller_id: int, agents: list[AgentSpec], cfg: ExperimentConfig):
        self.controller_id = controller_id
        self.agents = agents
        self.cfg = cfg
        # six domain coordinates + complexity + length + code flag
        # + busy fraction + buffer pressure
        self.feature_dim = (len(DOMAIN_ORDER) + 5 + cfg.embedding_dim
                            + (1 if cfg.pair_quality_feature else 0))
        self.load_indices = (self.feature_dim - 2, self.feature_dim - 1)
        # The pair coordinate sits just before the two load coordinates.
        self.pair_index = self.feature_dim - 3 if cfg.pair_quality_feature else None
        # Predictor row of every agent.  Slots of one model are capacity, not
        # distinct predictors: learning a separate vector per slot splits one
        # model's feedback across its replicas and leaves each near the prior.
        self.row_of = np.array([a.model_id if a.model_id >= 0 else a.agent_id for a in agents])
        self.rows = int(self.row_of.max()) + 1
        if sorted(set(self.row_of.tolist())) != list(range(self.rows)):
            raise ValueError("model ids must be contiguous from zero")
        # Scaling the features by RAW_FEATURE_NORM is a change of units, so the
        # initial parameters and the feasible box must be expressed in the same
        # units or every prediction is off by exactly that factor.
        self.parameter_scale = (raw_feature_norm(cfg.embedding_dim, cfg.pair_quality_feature)
                                if cfg.normalize_features else 1.0)
        unit = self.parameter_scale
        self.theta_time = np.zeros((self.rows, self.feature_dim))
        self.theta_time[:, :len(DOMAIN_ORDER)] = 1.6 * unit
        self.theta_time[:, len(DOMAIN_ORDER)] = 1.0 * unit       # complexity
        self.theta_time[:, len(DOMAIN_ORDER) + 1] = 0.4 * unit   # length
        self.theta_time[:, len(DOMAIN_ORDER) + 2] = 0.1 * unit   # has code
        self.theta_wait = np.zeros(self.feature_dim)
        self.theta_wait[:len(DOMAIN_ORDER)] = 0.20 * unit
        self.theta_wait[len(DOMAIN_ORDER)] = 0.20 * unit
        self.theta_wait[-2:] = (1.0 * unit, 0.5 * unit)
        self.theta_quality = np.zeros((self.rows, self.feature_dim))
        self.theta_quality[:, :len(DOMAIN_ORDER)] = 0.62 * unit
        self.theta_quality[:, len(DOMAIN_ORDER)] = -0.12 * unit      # complexity
        self.theta_quality[:, len(DOMAIN_ORDER) + 1] = -0.02 * unit  # length
        self.theta_quality[:, len(DOMAIN_ORDER) + 2] = 0.0           # has code
        if cfg.profile_prior:
            first = {}
            for agent, row in zip(agents, self.row_of):
                first.setdefault(int(row), agent)
            for row, agent in first.items():
                self.theta_time[row] = 0.0
                self.theta_time[row, :len(DOMAIN_ORDER)] = np.asarray(agent.mean_service) * unit
                self.theta_quality[row] = 0.0
                if self.pair_index is not None:
                    # Start from the offline per-prompt estimate itself.
                    self.theta_quality[row, self.pair_index] = 1.0 * unit
                else:
                    self.theta_quality[row, :len(DOMAIN_ORDER)] = np.asarray(agent.capability) * unit
        # Recursive least squares started at theta_0 with covariance P_0 =
        # (1/lambda) I is exactly ridge regression about theta_0 with
        # coefficient lambda, so the local estimator and the ridge inside the
        # A-SPSA objective are the same regularisation rather than two
        # unrelated ones.  P_0 = 20 I made the prior almost vacuous.
        if cfg.loss.ridge_reference == "origin":
            self.prior_time = np.zeros_like(self.theta_time)
            self.prior_wait = np.zeros_like(self.theta_wait)
            self.prior_quality = np.zeros_like(self.theta_quality)
        else:
            self.prior_time = self.theta_time.copy()
            self.prior_wait = self.theta_wait.copy()
            self.prior_quality = self.theta_quality.copy()
        self.ridge_p = cfg.loss.lambda_reg_p / self.parameter_scale**2
        self.ridge_w = cfg.loss.lambda_reg_w / self.parameter_scale**2
        self.ridge_q = cfg.loss.lambda_reg_q / self.parameter_scale**2
        prior_strength = max(cfg.rls_prior_strength, 1e-9) / self.parameter_scale**2
        covariance = np.eye(self.feature_dim) / prior_strength
        self.p_time = np.stack([covariance.copy() for _ in range(self.rows)])
        self.p_wait = covariance.copy()
        self.p_quality = np.stack([covariance.copy() for _ in range(self.rows)])
        self.observations: deque[LossObservation] = deque(maxlen=cfg.history_size)
        self.active_tasks = 0

    def task_features(self, features: np.ndarray) -> np.ndarray:
        """Features visible to the processing and quality models.

        The load coordinates are masked out for these two blocks: in the replay
        oracle neither service time nor quality depends on system load, so
        those coordinates are pure noise absorbers there.  Masking rather than
        removing keeps the parameter layout stable for every consumer.
        """
        if not self.cfg.load_features_for_waiting_only:
            return features
        masked = np.asarray(features, dtype=float).copy()
        masked[list(self.load_indices)] = 0.0
        return masked

    def predict_processing(self, agent_id: int, features: np.ndarray) -> float:
        x = self.task_features(features)
        return float(np.clip(self.theta_time[self.row_of[agent_id]] @ x, 0.05,
                             self.cfg.max_processing_units))

    def wait_features(self, features: np.ndarray) -> np.ndarray:
        """Waiting is a task-level quantity; the pair coordinate is masked."""
        if self.pair_index is None:
            return features
        masked = np.asarray(features, dtype=float).copy()
        masked[self.pair_index] = 0.0
        return masked

    def predict_waiting(self, features: np.ndarray) -> float:
        return float(np.clip(self.theta_wait @ self.wait_features(features), 0.0,
                             self.cfg.max_waiting_units))

    def predict_quality(self, agent_id: int, features: np.ndarray) -> float:
        x = self.task_features(features)
        return float(np.clip(self.theta_quality[self.row_of[agent_id]] @ x, 0.0, 1.0))

    @staticmethod
    def _rls(theta: np.ndarray, covariance: np.ndarray, x: np.ndarray, target: float, forgetting: float = 0.99):
        px = covariance @ x
        gain = px / max(forgetting + float(x @ px), 1e-12)
        updated = theta + gain * (target - float(theta @ x))
        new_covariance = (covariance - np.outer(gain, x) @ covariance) / forgetting
        return updated, new_covariance

    def observe(self, observation: LossObservation) -> None:
        self.observations.append(observation)
        if not self.cfg.use_rls:
            return
        aid, x = self.row_of[observation.agent_id], observation.features
        task_only = self.task_features(x)
        self.theta_time[aid], self.p_time[aid] = self._rls(
            self.theta_time[aid], self.p_time[aid], task_only, observation.processing_real
        )
        self.theta_wait, self.p_wait = self._rls(
            self.theta_wait, self.p_wait, self.wait_features(x), observation.waiting_real
        )
        self.theta_quality[aid], self.p_quality[aid] = self._rls(
            self.theta_quality[aid], self.p_quality[aid], task_only, observation.quality_real
        )
        self.unpack_parameters(self.pack_parameters())

    def pack_parameters(self) -> np.ndarray:
        return np.concatenate((self.theta_time.ravel(), self.theta_wait, self.theta_quality.ravel()))

    def _split(self, vector: np.ndarray):
        matrix_size = self.rows * self.feature_dim
        time = vector[:matrix_size].reshape(self.rows, self.feature_dim)
        wait = vector[matrix_size:matrix_size + self.feature_dim]
        quality = vector[matrix_size + self.feature_dim:].reshape(self.rows, self.feature_dim)
        return time, wait, quality

    def parameter_blocks(self) -> tuple[tuple[str, slice], ...]:
        matrix_size = self.rows * self.feature_dim
        return (
            ("processing", slice(0, matrix_size)),
            ("waiting", slice(matrix_size, matrix_size + self.feature_dim)),
            ("quality", slice(matrix_size + self.feature_dim, 2 * matrix_size + self.feature_dim)),
        )

    def block_regularizer(self, block_name: str) -> float:
        """Ridge term of one block at the current parameters."""
        if block_name == "processing":
            return ridge_penalty(self.theta_time, self.ridge_p, self.prior_time)
        if block_name == "waiting":
            return ridge_penalty(self.theta_wait, self.ridge_w, self.prior_wait)
        if block_name == "quality":
            return ridge_penalty(self.theta_quality, self.ridge_q, self.prior_quality)
        raise ValueError(f"unknown parameter block: {block_name}")

    def objective_block(self, vector: np.ndarray, block_name: str) -> float:
        """Rolling-window objective of one block, evaluated as written.

        The vector is deliberately NOT projected here.  Composing a convex loss
        with a box projection gives a non-convex function, and Assumption 3 of
        the manuscript is about the function the oracle returns.  Feasibility is
        maintained where it belongs, on the iterate, by the optimizer calling
        ``project`` after each step.
        """
        if not self.observations:
            return 0.0
        time, wait, quality = self._split(np.asarray(vector, dtype=float))
        observations = list(self.observations)
        if block_name == "processing":
            values = [processing_loss(float(time[self.row_of[o.agent_id]] @ self.task_features(o.features)), o, self.cfg.loss)
                      for o in observations]
            regularizer = ridge_penalty(time, self.ridge_p, self.prior_time)
        elif block_name == "waiting":
            values = [waiting_loss(float(wait @ self.wait_features(o.features)), o, self.cfg.loss)
                      for o in observations]
            regularizer = ridge_penalty(wait, self.ridge_w, self.prior_wait)
        elif block_name == "quality":
            values = [quality_loss(float(quality[self.row_of[o.agent_id]] @ self.task_features(o.features)), o, self.cfg.loss)
                      for o in observations]
            regularizer = ridge_penalty(quality, self.ridge_q, self.prior_quality)
        else:
            raise ValueError(f"unknown parameter block: {block_name}")
        return float(np.mean(values) + regularizer)
    def unpack_parameters(self, vector: np.ndarray) -> None:
        time, wait, quality = self._split(self.project(vector))
        self.theta_time, self.theta_wait, self.theta_quality = time, wait, quality

    def project(self, vector: np.ndarray) -> np.ndarray:
        unit = self.parameter_scale
        time, wait, quality = self._split(np.asarray(vector, dtype=float))
        return np.concatenate((
            np.clip(time, -5.0 * unit, self.cfg.max_processing_units * unit).ravel(),
            np.clip(wait, -10.0 * unit, self.cfg.max_waiting_units * unit),
            np.clip(quality, -2.0 * unit, 2.0 * unit).ravel(),
        ))

    def objective(self, vector: np.ndarray) -> float:
        """Sum of the three regularised block objectives.

        Uses exactly the same ridge convention as ``objective_block`` so that a
        whole-vector evaluation and the sum of block evaluations agree.
        """
        if not self.observations:
            return 0.0
        time, wait, quality = self._split(np.asarray(vector, dtype=float))
        f1, f2, f3 = mean_losses(
            list(self.observations),
            lambda o: float(time[self.row_of[o.agent_id]] @ self.task_features(o.features)),
            lambda o: float(wait @ self.wait_features(o.features)),
            lambda o: float(quality[self.row_of[o.agent_id]] @ self.task_features(o.features)),
            replace(self.cfg.loss, lambda_reg_p=self.ridge_p,
                    lambda_reg_w=self.ridge_w, lambda_reg_q=self.ridge_q),
            processing_parameters=time, quality_parameters=quality,
            waiting_parameters=wait,
            processing_reference=self.prior_time, waiting_reference=self.prior_wait,
            quality_reference=self.prior_quality,
        )
        return f1 + f2 + f3

    def true_processing_error(self, agents: list[AgentSpec]) -> tuple[float, float]:
        """RMSE and MAPE of the processing model against the generative truth.

        The rolling-window objective is an *in-sample* loss over the last
        observations of the pairs this controller happened to select, so a
        model can drive it down while moving away from the relationship that
        generated the data.  This measures the latter directly, over every
        (agent, domain, hardness) cell including those never observed.
        """
        errors, relative = [], []
        for agent_id, agent in enumerate(agents):
            for domain in range(len(DOMAIN_ORDER)):
                for hardness in DIFFICULTY.values():
                    features = np.zeros(self.feature_dim)
                    features[domain] = 1.0
                    features[len(DOMAIN_ORDER)] = hardness
                    # The live features are expressed in scaled units, so the
                    # probe must be too or every prediction is off by exactly
                    # the scale factor.
                    features = self.task_features(features / self.parameter_scale)
                    truth = float(agent.mean_service[domain]) * (
                        1.0 + DOMAIN_COMPLEXITY_RESPONSE[domain] * hardness
                    )
                    prediction = float(self.theta_time[self.row_of[agent_id]] @ features)
                    errors.append(prediction - truth)
                    relative.append(abs(prediction - truth) / max(truth, 1e-9))
        errors = np.asarray(errors, dtype=float)
        return float(np.sqrt(np.mean(errors**2))), float(np.mean(relative))

    def true_quality_error(self, agents: list[AgentSpec]) -> tuple[float, float]:
        """The same held-out check for the response-quality model.

        Quality carries the largest weight in the routing score, so it is the
        block whose accuracy matters most operationally.
        """
        errors, relative = [], []
        for agent_id, agent in enumerate(agents):
            for domain in range(len(DOMAIN_ORDER)):
                for hardness in DIFFICULTY.values():
                    features = np.zeros(self.feature_dim)
                    features[domain] = 1.0
                    features[len(DOMAIN_ORDER)] = hardness
                    features = self.task_features(features / self.parameter_scale)
                    truth = float(np.clip(
                        agent.capability[domain]
                        - DOMAIN_QUALITY_RESPONSE[domain] * hardness, 0.0, 1.0))
                    prediction = float(self.theta_quality[self.row_of[agent_id]] @ features)
                    errors.append(prediction - truth)
                    relative.append(abs(prediction - truth) / max(truth, 1e-9))
        errors = np.asarray(errors, dtype=float)
        return float(np.sqrt(np.mean(errors**2))), float(np.mean(relative))

    def current_losses(self) -> tuple[float, float, float]:
        return mean_losses(
            list(self.observations),
            lambda o: self.predict_processing(o.agent_id, o.features),
            lambda o: self.predict_waiting(o.features),
            lambda o: self.predict_quality(o.agent_id, o.features),
            replace(self.cfg.loss, lambda_reg_p=self.ridge_p,
                    lambda_reg_w=self.ridge_w, lambda_reg_q=self.ridge_q),
            processing_parameters=self.theta_time,
            quality_parameters=self.theta_quality,
            waiting_parameters=self.theta_wait,
            processing_reference=self.prior_time, waiting_reference=self.prior_wait,
            quality_reference=self.prior_quality,
        )

    def constraint(self, vector: np.ndarray, budget: float) -> tuple[float, np.ndarray]:
        time, _, _ = self._split(np.asarray(vector, dtype=float))
        gradient = np.zeros_like(vector)
        observations = list(self.observations)
        if observations:
            predictions = []
            for observation in observations:
                predictions.append(float(time[self.row_of[observation.agent_id]] @ self.task_features(observation.features)))
                start = self.row_of[observation.agent_id] * self.feature_dim
                gradient[start:start + self.feature_dim] += observation.features / len(observations)
            violation = float(np.mean(predictions) - budget)
        else:
            violation = float(np.mean(time) - budget)
            gradient[:time.size] = 1.0 / time.size
        return violation, gradient

def load_manifest_tasks(
    path: str | Path,
    seed: int,
    rho: float,
    agents: list[AgentSpec],
    limit: int | None = None,
    split: str = "all",
    batch_size: int = 1,
) -> list[ReplayTask]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if split not in {"all", "train", "validation", "test"}:
        raise ValueError("split must be all, train, validation, or test")
    if split != "all":
        selected = []
        for domain in DOMAIN_INDEX:
            domain_rows = [row for row in rows if row["primary_skill"] == domain]
            domain_rows.sort(key=lambda row: hashlib.sha256(f"split|{row['id']}".encode()).hexdigest())
            left = int(0.60 * len(domain_rows))
            right = int(0.80 * len(domain_rows))
            slices = {"train": domain_rows[:left], "validation": domain_rows[left:right], "test": domain_rows[right:]}
            selected.extend(slices[split])
        rows = selected
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    # Offered load is referenced to one fixed six-agent canonical system.
    # It must not change when a candidate architecture changes its agent count.
    capacity = pool_capacity(make_agent_pool(CANONICAL_AGENTS))
    arrival_rate = max(1e-6, rho * capacity)
    arrival = 0.0
    tasks: list[ReplayTask] = []
    canonical_service = 2.0
    for index, row in enumerate(rows):
        if index % batch_size == 0:
            # Keep the expected offered load per task while exposing a burst
            # workload in which multi-task assignment is identifiable.
            arrival += float(rng.exponential(batch_size / arrival_rate))
        tags = row.get("tags") or []
        skill = tuple(float(x) for x in row["skill_vector"])
        primary_share = 1.0 / max(sum(skill), 1.0)
        hardness = task_complexity(
            int(row["prompt_chars"]), bool(row["has_code"]), len(tags),
            str(row["task_kind"]), primary_share,
        )
        urgency = float(np.clip(0.15 + 0.60 * hardness + rng.uniform(0.0, 0.20), 0.0, 1.0))
        sla = canonical_service * (1.8 + 1.4 * hardness)
        tasks.append(ReplayTask(
            index, str(row["id"]), DOMAIN_INDEX[str(row["primary_skill"])],
            skill, hardness, urgency,
            int(row["prompt_chars"]), bool(row["has_code"]), arrival, arrival + sla,
            length_score(int(row["prompt_chars"])),
            TASK_KIND_INDEX.get(str(row["task_kind"]), 0),
        ))
    return tasks


class ControlledReplayExperiment:
    def __init__(self, tasks: list[ReplayTask], config: ExperimentConfig,
                 agents: list[AgentSpec] | None = None, oracle=None):
        """``agents`` and ``oracle`` default to the simulated pool.

        Passing a ``table_oracle.TableOracle`` and its agents replays measured
        LLM outcomes through the identical architecture and optimizers.
        """
        self.tasks = tasks
        self.config = config
        self.agents = agents if agents is not None else make_agent_pool(config.agents)
        if len(self.agents) != config.agents:
            raise ValueError("config.agents must equal the number of agents supplied")
        self.oracle = oracle if oracle is not None else ReplayOracle(config.seed, self.agents)
        if any(len(task.embedding) != config.embedding_dim for task in tasks):
            raise ValueError("every task embedding must have config.embedding_dim coordinates")
        self.feature_norm = raw_feature_norm(config.embedding_dim, config.pair_quality_feature)
        if config.pair_quality_feature and not hasattr(self.oracle, "prior_quality"):
            raise ValueError("pair_quality_feature needs an oracle with prior_quality")
        self.controllers = [PaperController(i, self.agents, config) for i in range(config.controllers)]
        matching = "greedy" if config.architecture == "global_greedy" else "exact"
        self.buffer = GlobalAssignmentBuffer(BufferPolicy(matching=matching, order=config.buffer_order))
        optimizer_method = "aspsa" if config.optimizer in {"none", "local_rls"} else config.optimizer
        block_configs = config.optimizer_configs.as_dict() if config.optimizer_configs else {}
        # The manuscript has one omega per block: it weights the disagreement
        # term of Phi and it is the coefficient the descent step applies.  The
        # code had two independent names for it - ``network_omega``, used to
        # report Phi, and ``OptimizerConfig.consensus_weight``, used by the
        # algorithm - whose defaults differed by a factor of twenty, so the
        # objective that was reported was not the one being minimised.  The
        # driver in run_paper_experiments set both to the same number and was
        # unaffected; a config built directly from the defaults was not.
        # ``network_omega`` is now authoritative for both.
        block_configs = {
            name: replace(block_configs.get(name, config.optimizer_config),
                          consensus_weight=omega)
            for name, omega in zip(("processing", "waiting", "quality"),
                                   config.network_omega)
        }
        self.optimizer = DistributedZerothOrderOptimizer(
            optimizer_method, config.optimizer_config, config.seed + 701,
            block_configs=block_configs, active_blocks=config.optimizer_blocks,
        )
        self.agent_available = np.zeros(config.agents, dtype=float)
        self.agent_busy = np.zeros(config.agents, dtype=float)
        self.controller_cursor = 0
        # Routing draws are kept on their own stream so that changing the
        # specialization share does not shift the service-time or quality noise.
        self.routing_rng = _stable_rng(config.seed, "routing")
        self.completions: list[_Completion] = []
        self.task_records: list[dict] = []
        self.round_records: list[dict] = []
        self.load_records: list[dict] = []
        self._sequence = 0
        self._completed = 0
        self._task_map = {task.task_id: task for task in tasks}

    def _network_objectives(self) -> dict[str, float]:
        """Rolling-window objectives from the manuscript network problem."""
        blocks = {"processing": "p", "waiting": "w", "quality": "q"}
        omega = dict(zip(blocks, self.config.network_omega))
        slices = dict(self.controllers[0].parameter_blocks())
        metrics: dict[str, float] = {}
        count = len(self.controllers)
        for block_name, short_name in blocks.items():
            local_sum = sum(
                controller.objective_block(controller.pack_parameters(), block_name)
                for controller in self.controllers
            )
            vectors = [
                controller.pack_parameters()[slices[block_name]]
                for controller in self.controllers
            ]
            edge_sum = 0.0
            if count > 1:
                for index, vector in enumerate(vectors):
                    for neighbor in ((index - 1) % count, (index + 1) % count):
                        edge_sum += (1.0 / 3.0) * float(np.sum((vector - vectors[neighbor]) ** 2))
            disagreement = 0.25 * edge_sum
            # The ridge is part of F_j as the manuscript writes it, so it
            # belongs in Phi.  It is also reported on its own, because a
            # regulariser large enough to supply strong convexity is large
            # enough to dominate the number a reader takes for the data fit.
            ridge_sum = sum(
                controller.block_regularizer(block_name) for controller in self.controllers
            )
            metrics[f"local_{short_name}"] = float(local_sum)
            metrics[f"data_{short_name}"] = float(local_sum - ridge_sum)
            metrics[f"ridge_{short_name}"] = float(ridge_sum)
            metrics[f"disagreement_{short_name}"] = float(disagreement)
            metrics[f"phi_{short_name}"] = float(local_sum + omega[block_name] * disagreement)
        return metrics

    def _generalization_metrics(self) -> dict[str, float]:
        """Out-of-window accuracy of the processing model, and stream skew.

        ``domain_mix_total_variation`` is the largest pairwise total variation
        between the controllers' realised domain distributions.  Under
        load-based assignment it reflects finite-sample noise only; under
        domain ownership it is the designed heterogeneity.
        """
        if hasattr(self.oracle, "potential"):
            rmse, mape, quality_rmse, quality_mape = self._counterfactual_errors()
        else:
            scores = [c.true_processing_error(self.agents) for c in self.controllers]
            rmse = float(np.mean([value[0] for value in scores]))
            mape = float(np.mean([value[1] for value in scores]))
            quality = [c.true_quality_error(self.agents) for c in self.controllers]
            quality_rmse = float(np.mean([value[0] for value in quality]))
            quality_mape = float(np.mean([value[1] for value in quality]))
        counts = np.zeros((len(self.controllers), len(DOMAIN_ORDER)), dtype=float)
        for row in self.task_records:
            counts[row["controller_id"], row["domain_index"]] += 1.0
        totals = counts.sum(axis=1, keepdims=True)
        mixes = counts / np.maximum(totals, 1.0)
        skew = 0.0
        for left in range(len(self.controllers)):
            for right in range(left + 1, len(self.controllers)):
                skew = max(skew, 0.5 * float(np.abs(mixes[left] - mixes[right]).sum()))
        return {
            "true_processing_rmse": rmse,
            "true_processing_mape": mape,
            "true_quality_rmse": quality_rmse,
            "true_quality_mape": quality_mape,
            "domain_mix_total_variation": float(skew),
        }

    def _counterfactual_predictions(self, tasks: list[ReplayTask]):
        """Every controller's predictions for every task--agent pair of ``tasks``.

        Load coordinates are zero, as in the synthetic probe.  With the pair
        feature the agent's offline estimate enters its own coordinate.
        """
        pair = self.config.pair_quality_feature
        features = np.stack([
            np.concatenate((task.features, [0.0] if pair else [], [0.0, 0.0])) for task in tasks
        ])
        scale = self.feature_norm if self.config.normalize_features else 1.0
        features = features / scale
        if pair:
            prior = np.array([[self.oracle.prior_quality(task, agent.agent_id) for agent in self.agents]
                              for task in tasks]) / scale
        times, qualities = [], []
        for controller in self.controllers:
            x = np.stack([controller.task_features(row) for row in features])
            time = (x @ controller.theta_time.T)[:, controller.row_of]
            quality = (x @ controller.theta_quality.T)[:, controller.row_of]
            if pair:
                k = controller.pair_index
                time = time + prior * controller.theta_time[controller.row_of, k][None, :]
                quality = quality + prior * controller.theta_quality[controller.row_of, k][None, :]
            times.append(np.clip(time, 0.05, self.config.max_processing_units))
            qualities.append(np.clip(quality, 0.0, 1.0))
        return times, qualities

    def _own_traffic_quality_rmse(self) -> float:
        """Quality error of each controller on the tasks it actually routed.

        A controller only routes its own traffic, so under domain-skewed
        assignment scoring it on every domain rewards averaging with neighbours
        automatically (RAZBOR 8.6).  Weighted by the number of tasks routed.
        """
        errors, weights = [], []
        by_id = {task.source_id: task for task in self.tasks}
        for controller in self.controllers:
            own = [by_id[row["task_id"]] for row in self.task_records
                   if row["controller_id"] == controller.controller_id]
            if not own:
                continue
            index = self.controllers.index(controller)
            _, quality = self._counterfactual_predictions(own)
            truth = self.oracle.potential(own)["quality"]
            errors.append(float(np.sqrt(np.mean((quality[index] - truth) ** 2))))
            weights.append(len(own))
        return float(np.average(errors, weights=weights)) if errors else float("nan")

    def _counterfactual_errors(self) -> tuple[float, float, float, float]:
        """Prediction error against measured outcomes of every task--agent pair.

        With a measured table there is no generative formula to compare with,
        but there is something better: the outcome every agent actually
        produced on every task of the stream, including the pairs that were
        never dispatched.  Load coordinates are zero, as in the synthetic probe.
        The quality "MAPE" slot carries the mean absolute error, because
        measured quality is often exactly zero.
        """
        potential = self.oracle.potential(self.tasks)
        time_pred, quality_pred = self._counterfactual_predictions(self.tasks)
        stats = []
        for time, quality in zip(time_pred, quality_pred):
            time_error = time - potential["service"]
            quality_error = quality - potential["quality"]
            stats.append((
                float(np.sqrt(np.mean(time_error**2))),
                float(np.mean(np.abs(time_error) / np.maximum(potential["service"], 1e-9))),
                float(np.sqrt(np.mean(quality_error**2))),
                float(np.mean(np.abs(quality_error))),
            ))
        return tuple(float(v) for v in np.mean(stats, axis=0))

    def _routing_references(self) -> dict[str, float]:
        """Where the realised operating point sits among the pool's references.

        Reference points are computed on exactly the tasks of this stream from
        their full counterfactual rows, so they share the stream's composition:
        the per-task oracle, the best single model and the cheapest single model
        (both chosen on the train profile, as a deployer would), and the
        uniform-random expectation.  ``quality_gap_recovered`` is the share of
        the gap between the best single model and the oracle that the system
        closed, the quantity RouterBench and RouteLLM report.  The single-model
        references ignore queueing: they are what one model would score with
        unlimited capacity.
        """
        potential = self.oracle.potential(self.tasks)
        quality, cost = potential["quality"], potential["cost"]
        profile = np.array([np.mean(agent.capability) for agent in self.agents])
        price = np.array([agent.token_cost for agent in self.agents])
        best, cheapest = int(np.argmax(profile)), int(np.argmin(price))
        realised_q = float(np.mean([row["quality_real"] for row in self.task_records]))
        realised_c = float(np.mean([row["cost"] for row in self.task_records]))
        oracle_q = float(quality.max(axis=1).mean())
        # Among the agents reaching the best quality, the oracle takes the cheapest.
        top = quality >= quality.max(axis=1, keepdims=True) - 1e-12
        oracle_c = float(np.where(top, cost, np.inf).min(axis=1).mean())
        best_q, best_c = float(quality[:, best].mean()), float(cost[:, best].mean())
        gap = oracle_q - best_q
        reference = (self.oracle.reference_summary(self.tasks)
                     if hasattr(self.oracle, "reference_summary") else {})
        return {
            **reference,
            "oracle_quality": oracle_q, "oracle_cost_usd": oracle_c,
            "best_single_quality": best_q, "best_single_cost_usd": best_c,
            "cheapest_single_quality": float(quality[:, cheapest].mean()),
            "cheapest_single_cost_usd": float(cost[:, cheapest].mean()),
            "random_quality": float(quality.mean()), "random_cost_usd": float(cost.mean()),
            "quality_gap_recovered": float((realised_q - best_q) / gap) if gap > 1e-12 else float("nan"),
            "quality_regret": float(oracle_q - realised_q),
            "cost_ratio_vs_best_single": float(realised_c / best_c) if best_c > 0 else float("nan"),
        }

    def _controller_for(self, task: ReplayTask) -> PaperController:
        if self.config.architecture in {"oracle", "centralized_exact"} or self.config.controllers == 1:
            return self.controllers[0]
        owner = self.controllers[
            (task.domain * self.config.controllers) // len(DOMAIN_ORDER)
        ]
        if self.config.controller_assignment == "domain":
            # Disjoint domain ownership: the six domains are split as evenly as
            # possible across the controllers, so each one observes only its
            # own slice of the workload.
            controller = owner
        elif self.config.controller_assignment == "skewed":
            # Partial ownership.  The draw uses the experiment's routing stream
            # so that the realised mix is reproducible for a given seed.
            if self.routing_rng.random() < self.config.specialization:
                controller = owner
            else:
                controller = min(self.controllers,
                                 key=lambda c: (c.active_tasks, c.controller_id))
        else:
            controller = min(self.controllers, key=lambda c: (c.active_tasks, c.controller_id))
        controller.active_tasks += 1
        return controller

    def _pending_count(self, now: float) -> int:
        """Tasks accepted but not yet started, for every architecture.

        ``legacy_local_queue`` and ``fixed_replay`` bypass the global buffer, so
        reading the buffer returned a constant zero and left the corresponding
        feature coordinate identically zero.  That made the feature matrix rank
        deficient and left one parameter per agent unidentifiable while the
        zeroth-order methods kept perturbing it.
        """
        if self.config.architecture in {"legacy_local_queue", "fixed_replay"}:
            return sum(1 for event in self.completions if event.assigned_at > now)
        return self.buffer.pending_count()

    def _prediction_features(self, task: ReplayTask, now: float) -> np.ndarray:
        base = task.features
        busy_fraction = float(np.mean(self.agent_available > now))
        pending = self._pending_count(now)
        # Queue occupancy is saturated at one agent-equivalent.  Without the
        # cap this coordinate is unbounded under load - it reached 3.8 in a
        # congested run - which breaks the ||x|| <= 1 bound that the
        # manuscript's constants M_p, M_w, M_q are derived from.
        queue_pressure = min(pending / max(1, self.config.agents), 1.0)
        pair = [0.0] if self.config.pair_quality_feature else []
        vector = np.concatenate((base, pair, [busy_fraction, queue_pressure]))
        if self.config.normalize_features:
            vector = vector / self.feature_norm
        return vector

    def _proposal(self, controller: PaperController, task: ReplayTask, agent: AgentSpec, now: float,
                  shared: dict | None = None) -> CandidateProposal:
        """One candidate edge.

        ``shared`` caches, for one task at one instant, what does not depend on
        the individual agent: the feature vector and waiting prediction, and
        per predictor row the processing and quality predictions and the
        relevance.  Slots of one model share a row, so a sixteen-slot pool
        computes each prediction once instead of sixteen times.  The values are
        identical to computing them per agent.
        """
        if shared is None:
            shared = {}
        if "x" not in shared:
            x = self._prediction_features(task, now)
            shared["x"], shared["x_tuple"] = x, tuple(x)
            shared["w_hat"] = controller.predict_waiting(x)
        x, w_hat = shared["x"], shared["w_hat"]
        row = int(controller.row_of[agent.agent_id])
        if row not in shared:
            x_pair, x_pair_tuple = x, shared["x_tuple"]
            if controller.pair_index is not None:
                x_pair = x.copy()
                scale = self.feature_norm if self.config.normalize_features else 1.0
                x_pair[controller.pair_index] = self.oracle.prior_quality(task, agent.agent_id) / scale
                x_pair_tuple = tuple(x_pair)
            shared[row] = (
                controller.predict_processing(agent.agent_id, x_pair),
                controller.predict_quality(agent.agent_id, x_pair),
                semantic_relevance(task.skill_vector, agent.capability),
                x_pair_tuple,
            )
        p_hat, q_hat, relevance, pair_tuple = shared[row]
        if self.config.architecture == "oracle":
            outcome = self.oracle.outcome(task, agent.agent_id)
            base = 1.35 * outcome.quality + 0.45 * float(outcome.success) - 0.12 * outcome.service_time - 0.10 * outcome.cost
            p_hat, q_hat = outcome.service_time, outcome.quality
        elif self.config.architecture == "random":
            base = float(_stable_rng(self.config.seed, task.source_id, agent.agent_id, "route").uniform())
        elif self.config.architecture == "static_skill":
            base = relevance
        elif self.config.architecture == "shortest_queue":
            base = -float(self.agent_available[agent.agent_id])
        elif self.config.architecture == "static_domain":
            # Best model per task family from offline evaluation: the
            # per-category lookup router every routing benchmark reports.
            base = float(agent.capability[task.domain])
        elif self.config.architecture == "quality_greedy":
            # Task-agnostic: the best model overall whenever it is free.
            base = float(np.mean(agent.capability))
        elif self.config.architecture == "cost_greedy":
            base = -float(agent.token_cost)
        elif self.config.architecture == "knn_router":
            # RouterBench's kNN router: mean quality of the agent's model over
            # the nearest train prompts, traded against price with the same
            # weights the controllers use, so a cost sweep moves both alike.
            w_q, _, _, w_c = self.config.base_score_weights
            base = w_q * self.oracle.knn_quality(task, agent.agent_id) + w_c * agent.token_cost
        elif self.config.architecture == "mlp_router":
            # A trained router: an MLP on the prompt embedding predicts every
            # model's quality (RouterBench's MLP router), traded against price
            # with the same weights as the kNN router.
            w_q, _, _, w_c = self.config.base_score_weights
            base = w_q * self.oracle.router_quality(task, agent.agent_id) + w_c * agent.token_cost
        else:
            w_q, w_r, w_p, w_c = self.config.base_score_weights
            base = (w_q * q_hat + w_r * relevance
                    + w_p * p_hat + w_c * agent.token_cost)
        return CandidateProposal(
            agent.agent_id, float(base), max(p_hat, 0.05), q_hat, relevance,
            agent.token_cost, pair_tuple, w_hat,
        )

    def _submit(self, task: ReplayTask, now: float) -> None:
        controller = self._controller_for(task)
        if self.config.architecture in {"legacy_local_queue", "fixed_replay"}:
            shared: dict = {}
            proposals = [self._proposal(controller, task, agent, now, shared) for agent in self.agents]
            if self.config.architecture == "fixed_replay":
                # Exogenous balanced assignment: optimizer predictions cannot
                # change the feedback stream used by the optimizer benchmark.
                chosen = proposals[task.task_id % len(self.agents)]
            else:
                chosen = max(proposals, key=lambda p: (p.base_score, -p.agent_id))
            assigned_at = max(now, float(self.agent_available[chosen.agent_id]))
            self._schedule(task, controller, chosen, assigned_at, buffered=False)
            return
        shared: dict = {}
        proposals = [self._proposal(controller, task, agent, now, shared) for agent in self.agents]
        self.buffer.submit(
            task_id=task.task_id, controller_id=controller.controller_id,
            arrival_time=task.arrival, deadline=task.deadline,
            time_budget=task.deadline - task.arrival, proposals=proposals,
            payload=task,
        )

    def _schedule(self, task: ReplayTask, controller: PaperController, proposal: CandidateProposal, assigned_at: float, buffered: bool) -> None:
        outcome = self.oracle.outcome(task, proposal.agent_id)
        finish = assigned_at + outcome.service_time
        self.agent_available[proposal.agent_id] = finish
        self.agent_busy[proposal.agent_id] += outcome.service_time
        self._sequence += 1
        heapq.heappush(self.completions, _Completion(
            finish, self._sequence, task.task_id, controller.controller_id,
            proposal.agent_id, proposal.predicted_processing,
            proposal.predicted_wait_budget, proposal.predicted_quality,
            ridge_penalty(controller.theta_time, controller.ridge_p,
                          controller.prior_time),
            assigned_at, outcome, np.asarray(proposal.feature_vector, dtype=float), buffered,
        ))

    def _dispatch(self, now: float) -> None:
        if self.config.architecture == "legacy_local_queue":
            return
        assignments = self.buffer.dispatch(now, {i: float(v) for i, v in enumerate(self.agent_available)})
        for assignment in assignments:
            task = self._task_map[assignment.task_id]
            proposal = self.buffer.tasks[task.task_id].proposals[assignment.agent_id].proposal
            controller = self.controllers[assignment.controller_id]
            self._schedule(task, controller, proposal, now, buffered=True)

    def _finish(self, event: _Completion) -> None:
        task = self._task_map[event.task_id]
        controller = self.controllers[event.controller_id]
        wait_real = event.assigned_at - task.arrival
        # F1 and F3 use the manuscript semantic relevance R_ik.  Earlier
        # revisions passed the oracle capability matrix (privileged
        # information the controller cannot observe) and task urgency in place
        # of R_ik in F3.
        observation = LossObservation(
            event.prediction_features, event.agent_id, event.outcome.service_time, wait_real,
            event.outcome.quality,
            semantic_relevance(task.skill_vector, self.agents[event.agent_id].capability),
            task.hardness,
        )
        f1 = processing_loss(event.predicted_processing, observation, self.config.loss) + event.processing_regularizer
        f2 = waiting_loss(event.predicted_waiting, observation, self.config.loss)
        f3 = quality_loss(event.predicted_quality, observation, self.config.loss)
        # Store every selected outcome in the evaluation window. RLS is
        # applied inside observe only when use_rls=True.
        controller.observe(observation)
        if event.buffered:
            self.buffer.complete(task.task_id, event.finish, event.outcome.success)
        controller.active_tasks = max(0, controller.active_tasks - 1)
        self._completed += 1
        deadline_hit = event.finish <= task.deadline
        latency = event.finish - task.arrival
        utility = (
            0.45 * event.outcome.quality + 0.20 * float(event.outcome.success)
            + 0.20 * float(deadline_hit) - 0.10 * min(latency / 10.0, 1.0)
            - 0.05 * min(event.outcome.cost, 1.0)
        )
        self.task_records.append({
            "completion_index": self._completed, "task_id": task.source_id,
            "domain": task.domain_label, "domain_index": task.domain,
            "controller_id": event.controller_id,
            "agent_id": event.agent_id, "arrival": task.arrival,
            "assigned_at": event.assigned_at, "finish": event.finish,
            "deadline": task.deadline, "wait_real": wait_real,
            "service_real": event.outcome.service_time, "latency": latency,
            "quality_real": event.outcome.quality, "success": int(event.outcome.success),
            "deadline_hit": int(deadline_hit), "cost": event.outcome.cost,
            "p_hat": event.predicted_processing, "w_hat": event.predicted_waiting,
            "q_hat": event.predicted_quality, "f1": f1, "f2": f2, "f3": f3,
            "utility": utility, "sync_round": len(self.round_records),
        })
        if (
            self._completed >= self.config.warmup
            and self._completed % self.config.sync_interval == 0
        ):
            if self.config.use_learning and self.config.optimizer not in {"none", "local_rls"}:
                trace = self.optimizer.step(self.controllers, len(self.round_records) + 1)
            else:
                trace = {
                    "oracle_queries": 0.0,
                    "oracle_queries_cumulative": float(self.optimizer.state.query_count),
                }
            self._combine_controllers()
            losses = np.asarray([c.current_losses() for c in self.controllers], dtype=float)
            network = self._network_objectives()
            self.round_records.append({
                "round": len(self.round_records) + 1, "completion_index": self._completed,
                "event_time": event.finish, "f1": float(losses[:, 0].mean()),
                "f2": float(losses[:, 1].mean()), "f3": float(losses[:, 2].mean()),
                **network, **trace,
            })
    def _combine_controllers(self) -> None:
        """Adapt-then-combine: average each controller with its ring neighbours.

        Runs after the local update, so it is the combine half of ATC.  It costs
        no objective evaluations - only one parameter exchange per neighbour,
        the same message the consensus term already needs.
        """
        epsilon = float(self.config.diffusion_mixing)
        count = len(self.controllers)
        if epsilon <= 0.0 or count < 2:
            return
        vectors = [controller.pack_parameters() for controller in self.controllers]
        for index, controller in enumerate(self.controllers):
            left = vectors[(index - 1) % count]
            right = vectors[(index + 1) % count]
            mixed = (1.0 - 2.0 * epsilon) * vectors[index] + epsilon * (left + right)
            controller.unpack_parameters(mixed)

    def _snapshot_load(self, now: float) -> None:
        pending = self._pending_count(now)
        row = {"event_time": now, "pending_tasks": pending}
        for agent_id, available in enumerate(self.agent_available):
            row[f"agent_{agent_id}_busy"] = int(available > now)
        self.load_records.append(row)

    def run(self) -> dict[str, float]:
        next_task = 0
        while next_task < len(self.tasks) or self.completions or (
            self.config.architecture != "legacy_local_queue" and any(True for _ in self.buffer.pending_tasks())
        ):
            next_arrival = self.tasks[next_task].arrival if next_task < len(self.tasks) else float("inf")
            next_finish = self.completions[0].finish if self.completions else float("inf")
            now = min(next_arrival, next_finish)
            if not np.isfinite(now):
                raise RuntimeError("experiment cannot make progress")
            while self.completions and self.completions[0].finish <= now + 1e-12:
                self._finish(heapq.heappop(self.completions))
            while next_task < len(self.tasks) and self.tasks[next_task].arrival <= now + 1e-12:
                self._submit(self.tasks[next_task], now)
                next_task += 1
            self._dispatch(now)
            self._snapshot_load(now)
        return self.summary()

    def summary(self) -> dict[str, float]:
        if not self.task_records:
            return {}
        values = lambda key: np.asarray([row[key] for row in self.task_records], dtype=float)
        makespan = max(values("finish")) - min(values("arrival"))
        utilization = self.agent_busy / max(makespan, 1e-12)
        assignment_counts = np.bincount([row["agent_id"] for row in self.task_records], minlength=len(self.agents)).astype(float)
        jain = float(assignment_counts.sum() ** 2 / (len(assignment_counts) * np.sum(assignment_counts**2))) if np.sum(assignment_counts**2) else 0.0
        network_summary = {}
        for short_name in ("p", "w", "q"):
            series = np.asarray([row[f"phi_{short_name}"] for row in self.round_records], dtype=float)
            local = np.asarray([row[f"local_{short_name}"] for row in self.round_records], dtype=float)
            disagreement = np.asarray(
                [row[f"disagreement_{short_name}"] for row in self.round_records], dtype=float
            )
            data = np.asarray(
                [row[f"data_{short_name}"] for row in self.round_records], dtype=float
            )
            network_summary.update({
                f"phi_{short_name}_auc": float(series.mean()) if len(series) else np.nan,
                f"phi_{short_name}_final": float(series[-1]) if len(series) else np.nan,
                f"local_{short_name}_final": float(local[-1]) if len(local) else np.nan,
                f"data_{short_name}_final": float(data[-1]) if len(data) else np.nan,
                f"data_{short_name}_auc": float(data.mean()) if len(data) else np.nan,
                f"disagreement_{short_name}_auc": float(disagreement.mean()) if len(disagreement) else np.nan,
                f"disagreement_{short_name}_final": float(disagreement[-1]) if len(disagreement) else np.nan,
            })
        blocks = dict(self.controllers[0].parameter_blocks())
        active = [name for name in self.config.optimizer_blocks]
        dimensions = {
            name: blocks[name].stop - blocks[name].start for name in active
        }
        active_dimension = sum(dimensions.values())
        tuned_parameters = float(active_dimension * self.config.controllers)
        queries = float(self.optimizer.state.query_count)
        # A two-point zeroth-order method cannot resolve a block whose
        # dimension exceeds its query budget.  These ratios are reported so a
        # comparison run below one cannot be read as an optimizer ranking.
        #
        # The ratio is reported per block as well.  Every block receives the
        # same number of evaluations per round irrespective of its dimension,
        # while the blocks differ in size by an order of magnitude, so the
        # pooled ratio can look sufficient while the two large blocks are not.
        # The binding constraint is the worst block, which is what
        # query_budget_sufficient tests.
        queries_per_parameter = queries / max(tuned_parameters, 1.0)
        block_queries: dict[str, float] = {name: 0.0 for name in active}
        for record in self.round_records:
            for name in active:
                block_queries[name] += float(record.get(f"oracle_queries_{name}", 0.0))
        block_ratios = {
            name: block_queries[name]
            / max(float(dimensions[name] * self.config.controllers), 1.0)
            for name in active
        }
        worst_block_ratio = min(block_ratios.values()) if block_ratios else 0.0
        actual_capacity = pool_capacity(self.agents)
        # A measured table loads its own pool directly (table_oracle), so the
        # synthetic canonical reference does not apply there.
        canonical_capacity = (actual_capacity if hasattr(self.oracle, "potential")
                              else pool_capacity(make_agent_pool(CANONICAL_AGENTS)))
        rho_effective = float(self.config.rho) * canonical_capacity / max(actual_capacity, 1e-12)
        return {
            "tasks": float(len(self.task_records)), "controllers": float(self.config.controllers),
            "agents": float(self.config.agents), "rho": float(self.config.rho),
            "rho_effective": float(rho_effective),
            "sync_rounds": float(len(self.round_records)),
            "tuned_parameters": tuned_parameters,
            "queries_per_parameter": float(queries_per_parameter),
            "worst_block_queries_per_parameter": float(worst_block_ratio),
            "query_budget_sufficient": float(worst_block_ratio >= 1.0),
            **{
                f"queries_per_parameter_{name}": float(value)
                for name, value in block_ratios.items()
            },
            "success_rate": float(values("success").mean()),
            "mean_quality": float(values("quality_real").mean()),
            "deadline_hit_rate": float(values("deadline_hit").mean()),
            "mean_latency": float(values("latency").mean()),
            "p50_latency": float(np.quantile(values("latency"), 0.50)),
            "p95_latency": float(np.quantile(values("latency"), 0.95)),
            "p99_latency": float(np.quantile(values("latency"), 0.99)),
            # SLO attainment is deadline_hit_rate; goodput counts only work that
            # was both correct and on time, per unit of time.
            "goodput": float(np.sum(values("success") * values("deadline_hit")) / max(makespan, 1e-12)),
            "mean_wait": float(values("wait_real").mean()),
            "p95_wait": float(np.quantile(values("wait_real"), 0.95)),
            "throughput": float(len(self.task_records) / max(makespan, 1e-12)),
            "mean_cost": float(values("cost").mean()),
            "mean_utility": float(values("utility").mean()),
            "f1": float(values("f1").mean()), "f2": float(values("f2").mean()),
            "f3": float(values("f3").mean()), "mean_utilization": float(utilization.mean()),
            "max_utilization": float(utilization.max()), "jain_fairness": jain,
            "oracle_queries": float(self.optimizer.state.query_count),
            "infrastructure_units": float(self.config.agents + 0.25 * self.config.controllers),
            **self._generalization_metrics(),
            **(self._routing_references() if hasattr(self.oracle, "potential") else {}),
            **({"own_quality_rmse": self._own_traffic_quality_rmse()}
               if hasattr(self.oracle, "potential") else {}),
            **network_summary,
        }
















