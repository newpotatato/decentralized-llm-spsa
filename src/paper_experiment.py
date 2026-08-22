"""Controlled event-driven replay experiment for the six IT domains.

This module is independent of the legacy five-task taxonomy. Potential outcomes
are fixed for every (task, agent, seed), so all routing and optimization methods
see the same counterfactual table while observing feedback only for selected
pairs.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
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
EPSILON_RELEVANCE = 1e-9
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

    @property
    def features(self) -> np.ndarray:
        out = np.zeros(len(DOMAIN_ORDER) + 1, dtype=float)
        out[:len(DOMAIN_ORDER)] = np.asarray(self.skill_vector, dtype=float)
        out[-1] = self.hardness
        return out


@dataclass(frozen=True)
class AgentSpec:
    agent_id: int
    name: str
    capability: np.ndarray
    mean_service: np.ndarray
    token_cost: float


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
    # omega is calibrated so the disagreement penalty is commensurate with the
    # local losses it regularises.  At omega = 0.05 the penalty reached 0.6-1.9
    # while the whole local objective was 0.02-0.18, so Phi ranked methods by
    # how hard they averaged parameters rather than by prediction accuracy.
    # local_* and disagreement_* are additionally reported as separate primary
    # endpoints so this coefficient can no longer decide a conclusion alone.
    network_omega: tuple[float, float, float] = (0.002, 0.002, 0.002)

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
        if self.architecture not in {
            "legacy_local_queue", "random", "static_skill", "shortest_queue",
            "global_greedy", "global_exact", "centralized_exact", "oracle",
            "fixed_replay",
        }:
            raise ValueError(f"unknown architecture: {self.architecture}")


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


class ReplayOracle:
    def __init__(self, seed: int, agents: list[AgentSpec]):
        self.seed = int(seed)
        self.agents = agents

    def outcome(self, task: ReplayTask, agent_id: int) -> PotentialOutcome:
        agent = self.agents[agent_id]
        rng = _stable_rng(self.seed, task.source_id, agent_id, "outcome")
        mean = float(agent.mean_service[task.domain]) * (1.0 + 0.70 * task.hardness)
        service = max(0.05, mean * float(rng.lognormal(mean=-0.5 * 0.22**2, sigma=0.22)))
        quality = float(np.clip(
            agent.capability[task.domain] - 0.18 * task.hardness + rng.normal(0.0, 0.045),
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
        # six domain coordinates + hardness + busy fraction + buffer pressure
        self.feature_dim = len(DOMAIN_ORDER) + 3
        self.theta_time = np.zeros((len(agents), self.feature_dim))
        self.theta_time[:, :len(DOMAIN_ORDER)] = 1.6
        self.theta_time[:, len(DOMAIN_ORDER)] = 1.0
        self.theta_wait = np.zeros(self.feature_dim)
        self.theta_wait[:len(DOMAIN_ORDER)] = 0.20
        self.theta_wait[len(DOMAIN_ORDER)] = 0.20
        self.theta_wait[-2:] = (1.0, 0.5)
        self.theta_quality = np.zeros((len(agents), self.feature_dim))
        self.theta_quality[:, :len(DOMAIN_ORDER)] = 0.62
        self.theta_quality[:, len(DOMAIN_ORDER)] = -0.12
        self.p_time = np.stack([np.eye(self.feature_dim) * 20.0 for _ in agents])
        self.p_wait = np.eye(self.feature_dim) * 20.0
        self.p_quality = np.stack([np.eye(self.feature_dim) * 20.0 for _ in agents])
        self.observations: deque[LossObservation] = deque(maxlen=cfg.history_size)
        self.active_tasks = 0

    def predict_processing(self, agent_id: int, features: np.ndarray) -> float:
        return float(np.clip(self.theta_time[agent_id] @ features, 0.05, 20.0))

    def predict_waiting(self, features: np.ndarray) -> float:
        return float(np.clip(self.theta_wait @ features, 0.0, 30.0))

    def predict_quality(self, agent_id: int, features: np.ndarray) -> float:
        return float(np.clip(self.theta_quality[agent_id] @ features, 0.0, 1.0))

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
        aid, x = observation.agent_id, observation.features
        self.theta_time[aid], self.p_time[aid] = self._rls(
            self.theta_time[aid], self.p_time[aid], x, observation.processing_real
        )
        self.theta_wait, self.p_wait = self._rls(
            self.theta_wait, self.p_wait, x, observation.waiting_real
        )
        self.theta_quality[aid], self.p_quality[aid] = self._rls(
            self.theta_quality[aid], self.p_quality[aid], x, observation.quality_real
        )
        self.unpack_parameters(self.pack_parameters())

    def pack_parameters(self) -> np.ndarray:
        return np.concatenate((self.theta_time.ravel(), self.theta_wait, self.theta_quality.ravel()))

    def _split(self, vector: np.ndarray):
        matrix_size = len(self.agents) * self.feature_dim
        time = vector[:matrix_size].reshape(len(self.agents), self.feature_dim)
        wait = vector[matrix_size:matrix_size + self.feature_dim]
        quality = vector[matrix_size + self.feature_dim:].reshape(len(self.agents), self.feature_dim)
        return time, wait, quality

    def parameter_blocks(self) -> tuple[tuple[str, slice], ...]:
        matrix_size = len(self.agents) * self.feature_dim
        return (
            ("processing", slice(0, matrix_size)),
            ("waiting", slice(matrix_size, matrix_size + self.feature_dim)),
            ("quality", slice(matrix_size + self.feature_dim, 2 * matrix_size + self.feature_dim)),
        )

    def objective_block(self, vector: np.ndarray, block_name: str) -> float:
        if not self.observations:
            return 0.0
        time, wait, quality = self._split(self.project(vector))
        observations = list(self.observations)
        if block_name == "processing":
            values = [processing_loss(float(time[o.agent_id] @ o.features), o, self.cfg.loss) for o in observations]
            regularizer = ridge_penalty(time, self.cfg.loss.lambda_reg_p)
        elif block_name == "waiting":
            values = [waiting_loss(float(wait @ o.features), o, self.cfg.loss) for o in observations]
            regularizer = 0.0
        elif block_name == "quality":
            values = [quality_loss(float(quality[o.agent_id] @ o.features), o, self.cfg.loss) for o in observations]
            regularizer = ridge_penalty(quality, self.cfg.loss.lambda_reg_q)
        else:
            raise ValueError(f"unknown parameter block: {block_name}")
        return float(np.mean(values) + regularizer)
    def unpack_parameters(self, vector: np.ndarray) -> None:
        time, wait, quality = self._split(self.project(vector))
        self.theta_time, self.theta_wait, self.theta_quality = time, wait, quality

    def project(self, vector: np.ndarray) -> np.ndarray:
        time, wait, quality = self._split(np.asarray(vector, dtype=float))
        return np.concatenate((
            np.clip(time, -5.0, 20.0).ravel(),
            np.clip(wait, -10.0, 30.0),
            np.clip(quality, -2.0, 2.0).ravel(),
        ))

    def objective(self, vector: np.ndarray) -> float:
        """Sum of the three regularised block objectives.

        Uses exactly the same ridge convention as ``objective_block`` so that a
        whole-vector evaluation and the sum of block evaluations agree.
        """
        if not self.observations:
            return 0.0
        time, wait, quality = self._split(self.project(vector))
        f1, f2, f3 = mean_losses(
            list(self.observations),
            lambda o: float(time[o.agent_id] @ o.features),
            lambda o: float(wait @ o.features),
            lambda o: float(quality[o.agent_id] @ o.features),
            self.cfg.loss,
            processing_parameters=time,
            quality_parameters=quality,
        )
        return f1 + f2 + f3

    def current_losses(self) -> tuple[float, float, float]:
        return mean_losses(
            list(self.observations),
            lambda o: self.predict_processing(o.agent_id, o.features),
            lambda o: self.predict_waiting(o.features),
            lambda o: self.predict_quality(o.agent_id, o.features),
            self.cfg.loss,
            processing_parameters=self.theta_time,
            quality_parameters=self.theta_quality,
        )

    def constraint(self, vector: np.ndarray, budget: float) -> tuple[float, np.ndarray]:
        time, _, _ = self._split(self.project(vector))
        gradient = np.zeros_like(vector)
        observations = list(self.observations)
        if observations:
            predictions = []
            for observation in observations:
                predictions.append(float(time[observation.agent_id] @ observation.features))
                start = observation.agent_id * self.feature_dim
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
        hardness = DIFFICULTY[str(row["difficulty_bucket"])]
        urgency = float(np.clip(0.15 + 0.60 * hardness + rng.uniform(0.0, 0.20), 0.0, 1.0))
        sla = canonical_service * (1.8 + 1.4 * hardness)
        tasks.append(ReplayTask(
            index, str(row["id"]), DOMAIN_INDEX[str(row["primary_skill"])],
            tuple(float(x) for x in row["skill_vector"]), hardness, urgency,
            int(row["prompt_chars"]), bool(row["has_code"]), arrival, arrival + sla,
        ))
    return tasks


class ControlledReplayExperiment:
    def __init__(self, tasks: list[ReplayTask], config: ExperimentConfig):
        self.tasks = tasks
        self.config = config
        self.agents = make_agent_pool(config.agents)
        self.oracle = ReplayOracle(config.seed, self.agents)
        self.controllers = [PaperController(i, self.agents, config) for i in range(config.controllers)]
        matching = "greedy" if config.architecture == "global_greedy" else "exact"
        self.buffer = GlobalAssignmentBuffer(BufferPolicy(matching=matching))
        optimizer_method = "aspsa" if config.optimizer in {"none", "local_rls"} else config.optimizer
        block_configs = config.optimizer_configs.as_dict() if config.optimizer_configs else {}
        self.optimizer = DistributedZerothOrderOptimizer(
            optimizer_method, config.optimizer_config, config.seed + 701,
            block_configs=block_configs, active_blocks=config.optimizer_blocks,
        )
        self.agent_available = np.zeros(config.agents, dtype=float)
        self.agent_busy = np.zeros(config.agents, dtype=float)
        self.controller_cursor = 0
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
            metrics[f"local_{short_name}"] = float(local_sum)
            metrics[f"disagreement_{short_name}"] = float(disagreement)
            metrics[f"phi_{short_name}"] = float(local_sum + omega[block_name] * disagreement)
        return metrics

    def _controller_for(self) -> PaperController:
        if self.config.architecture in {"oracle", "centralized_exact"} or self.config.controllers == 1:
            return self.controllers[0]
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
        return np.concatenate((base, [busy_fraction, pending / max(1, self.config.agents)]))

    def _proposal(self, controller: PaperController, task: ReplayTask, agent: AgentSpec, now: float) -> CandidateProposal:
        x = self._prediction_features(task, now)
        p_hat = controller.predict_processing(agent.agent_id, x)
        w_hat = controller.predict_waiting(x)
        q_hat = controller.predict_quality(agent.agent_id, x)
        relevance = semantic_relevance(task.skill_vector, agent.capability)
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
        else:
            base = 1.10 * q_hat + 0.40 * relevance - 0.16 * p_hat - 0.08 * agent.token_cost
        return CandidateProposal(
            agent.agent_id, float(base), max(p_hat, 0.05), q_hat, relevance,
            agent.token_cost, tuple(x), w_hat,
        )

    def _submit(self, task: ReplayTask, now: float) -> None:
        controller = self._controller_for()
        if self.config.architecture in {"legacy_local_queue", "fixed_replay"}:
            proposals = [self._proposal(controller, task, agent, now) for agent in self.agents]
            if self.config.architecture == "fixed_replay":
                # Exogenous balanced assignment: optimizer predictions cannot
                # change the feedback stream used by the optimizer benchmark.
                chosen = proposals[task.task_id % len(self.agents)]
            else:
                chosen = max(proposals, key=lambda p: (p.base_score, -p.agent_id))
            assigned_at = max(now, float(self.agent_available[chosen.agent_id]))
            self._schedule(task, controller, chosen, assigned_at, buffered=False)
            return
        proposals = [self._proposal(controller, task, agent, now) for agent in self.agents]
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
            ridge_penalty(controller.theta_time, self.config.loss.lambda_reg_p),
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
            "domain": DOMAIN_ORDER[task.domain].value, "controller_id": event.controller_id,
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
            losses = np.asarray([c.current_losses() for c in self.controllers], dtype=float)
            network = self._network_objectives()
            self.round_records.append({
                "round": len(self.round_records) + 1, "completion_index": self._completed,
                "event_time": event.finish, "f1": float(losses[:, 0].mean()),
                "f2": float(losses[:, 1].mean()), "f3": float(losses[:, 2].mean()),
                **network, **trace,
            })
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
            network_summary.update({
                f"phi_{short_name}_auc": float(series.mean()) if len(series) else np.nan,
                f"phi_{short_name}_final": float(series[-1]) if len(series) else np.nan,
                f"local_{short_name}_final": float(local[-1]) if len(local) else np.nan,
                f"disagreement_{short_name}_auc": float(disagreement.mean()) if len(disagreement) else np.nan,
                f"disagreement_{short_name}_final": float(disagreement[-1]) if len(disagreement) else np.nan,
            })
        blocks = dict(self.controllers[0].parameter_blocks())
        active_dimension = sum(
            blocks[name].stop - blocks[name].start
            for name in self.config.optimizer_blocks
        )
        tuned_parameters = float(active_dimension * self.config.controllers)
        queries = float(self.optimizer.state.query_count)
        # A two-point zeroth-order method cannot resolve a block whose
        # dimension exceeds its query budget.  This ratio is reported so a
        # comparison run below one cannot be read as an optimizer ranking.
        queries_per_parameter = queries / max(tuned_parameters, 1.0)
        actual_capacity = pool_capacity(self.agents)
        canonical_capacity = pool_capacity(make_agent_pool(CANONICAL_AGENTS))
        rho_effective = float(self.config.rho) * canonical_capacity / max(actual_capacity, 1e-12)
        return {
            "tasks": float(len(self.task_records)), "controllers": float(self.config.controllers),
            "agents": float(self.config.agents), "rho": float(self.config.rho),
            "rho_effective": float(rho_effective),
            "sync_rounds": float(len(self.round_records)),
            "tuned_parameters": tuned_parameters,
            "queries_per_parameter": float(queries_per_parameter),
            "query_budget_sufficient": float(queries_per_parameter >= 1.0),
            "success_rate": float(values("success").mean()),
            "mean_quality": float(values("quality_real").mean()),
            "deadline_hit_rate": float(values("deadline_hit").mean()),
            "mean_latency": float(values("latency").mean()),
            "p95_latency": float(np.quantile(values("latency"), 0.95)),
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
            **network_summary,
        }
















