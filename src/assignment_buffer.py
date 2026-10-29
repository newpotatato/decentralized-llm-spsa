"""Global event-driven task-to-agent assignment buffer.

Controllers publish one immutable candidate evaluation for every task--agent
pair.  The buffer owns the final dispatch decision.  It only refreshes values
that depend on time and agent availability; controller-produced base scores
remain unchanged while a task waits in the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"


class CandidateStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class CandidateProposal:
    """Immutable evaluation published by one task's responsible controller."""

    agent_id: int
    base_score: float
    predicted_processing: float
    predicted_quality: float = 0.0
    relevance: float = 0.0
    cost: float = 0.0
    feature_vector: tuple[float, ...] = ()
    predicted_wait_budget: float = 0.0

    def __post_init__(self) -> None:
        if self.predicted_processing <= 0.0:
            raise ValueError("predicted_processing must be positive")


@dataclass
class CandidateState:
    """A proposal plus the dynamic values maintained by the buffer."""

    proposal: CandidateProposal
    status: CandidateStatus = CandidateStatus.PENDING
    agent_available_at: float = 0.0
    predicted_start: float = 0.0
    predicted_finish: float = 0.0
    availability_score: float = 0.0
    deadline_gap: float = 0.0
    required_time_ratio: float = 0.0
    slack_coefficient: float = 0.0
    wait_pressure: float = 0.0
    priority: float = float("-inf")


@dataclass
class BufferedTask:
    task_id: int
    controller_id: int
    arrival_time: float
    deadline: float
    time_budget: float
    proposals: Dict[int, CandidateState]
    payload: object = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[int] = None
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass(frozen=True)
class Assignment:
    task_id: int
    controller_id: int
    agent_id: int
    dispatch_time: float
    predicted_processing: float
    predicted_finish: float
    base_score: float
    priority: float
    slack_coefficient: float
    feature_vector: tuple[float, ...] = ()
    payload: object = None


@dataclass(frozen=True, eq=False)
class BufferPolicy:
    """Weights of the dynamic scheduling layer.

    ``base_score`` already expresses task--agent suitability.  These weights
    add only task hotness, waiting pressure, and deadline-risk effects.

    There is deliberately no availability weight.  Dispatch considers agents
    with ``available_k(t) <= t`` only, so ``availability_ik(t)`` evaluates to
    ``exp(0) = 1`` for every candidate edge of a dispatch round: the term used
    to add the same constant to every edge and could never change a matching.
    ``availability_score`` is still refreshed and retained as an audit field.
    """

    hotness_weight: float = 0.18
    wait_pressure_weight: float = 0.12
    deadline_gap_penalty: float = 0.30
    budget_overrun_penalty: float = 0.08
    hotness_slope: float = 3.0
    wait_pressure_cap: float = 3.0
    matching: str = "exact"
    epsilon: float = 1e-8
    # Optional per-domain weights.  When present, a task is scheduled with the
    # weights of its own domain rather than one global set: a security incident
    # and a frontend question need not trade deadline risk against suitability
    # in the same proportion.  Shape (domains, 5), columns in the order
    # hotness, wait pressure, deadline gap, budget overrun, hotness slope.
    per_domain: Optional[np.ndarray] = None
    # Which tasks are served first.  "weighted" is the priority above.  The
    # classic SLO-aware rules of the serving literature serve tasks strictly in
    # the order of one urgency key and send each to its best free agent by
    # base score: "edf" earliest deadline, "llf" least laxity (deadline minus
    # now minus the fastest reachable predicted processing time), "sjf"
    # shortest predicted processing time.  The weights are then unused.
    order: str = "weighted"
    # Deadline urgency: -weight * (deadline - now) / scale is added to every
    # edge of a task.  It is task-level, so it decides which tasks are served
    # first and never which agent serves one.  A large weight makes the
    # weighted priority earliest-deadline-first; with the budget-overrun
    # penalty it can serve early deadlines first while passing over tasks that
    # can no longer make theirs.  Zero, the default, leaves the priority as it was.
    deadline_urgency_weight: float = 0.0
    urgency_scale: float = 1.0

    ORDERS = ("weighted", "edf", "llf", "sjf")
    # Rank gap that makes a rule lexicographic: it exceeds any sum of base-score
    # differences a matching over the agent pool could trade against one rank.
    RANK_GAP = 1e4

    def weights_for(self, domain: Optional[int]) -> tuple[float, float, float, float, float]:
        """The five priority weights that apply to a task of this domain."""
        if self.per_domain is None or domain is None:
            return (self.hotness_weight, self.wait_pressure_weight,
                    self.deadline_gap_penalty, self.budget_overrun_penalty,
                    self.hotness_slope)
        row = np.asarray(self.per_domain)[int(domain)]
        return (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))

    def __post_init__(self) -> None:
        if self.matching not in {"exact", "greedy"}:
            raise ValueError("matching must be exact or greedy")
        if self.order not in self.ORDERS:
            raise ValueError(f"order must be one of {self.ORDERS}")


class GlobalAssignmentBuffer:
    """Global buffer that matches pending tasks to free agents.

    The greedy dispatcher sorts all currently feasible task--agent edges by
    priority.  Once an edge is selected, both its task and its agent are
    reserved for the current dispatch round.  Alternative proposals are kept
    for auditability but marked ``INVALIDATED``.
    """

    def __init__(self, policy: Optional[BufferPolicy] = None) -> None:
        self.policy = policy or BufferPolicy()
        self.tasks: Dict[int, BufferedTask] = {}
        self.assignment_history: List[Assignment] = []
        # Live set of pending ids.  Scanning ``tasks`` on every event made the
        # simulation quadratic in the number of submitted tasks.
        self._pending: Dict[int, BufferedTask] = {}

    def submit(
        self,
        *,
        task_id: int,
        controller_id: int,
        arrival_time: float,
        deadline: float,
        time_budget: float,
        proposals: Sequence[CandidateProposal],
        payload: object = None,
    ) -> BufferedTask:
        if task_id in self.tasks:
            raise ValueError(f"task {task_id} is already present in the buffer")
        if deadline < arrival_time:
            raise ValueError("deadline cannot precede arrival_time")
        if time_budget <= 0.0:
            raise ValueError("time_budget must be positive")
        if not proposals:
            raise ValueError("at least one agent proposal is required")

        proposal_map: Dict[int, CandidateState] = {}
        for proposal in proposals:
            if proposal.agent_id in proposal_map:
                raise ValueError(
                    f"controller {controller_id} published duplicate pair "
                    f"({task_id}, {proposal.agent_id})"
                )
            proposal_map[proposal.agent_id] = CandidateState(proposal=proposal)

        task = BufferedTask(
            task_id=task_id,
            controller_id=controller_id,
            arrival_time=float(arrival_time),
            deadline=float(deadline),
            time_budget=float(time_budget),
            proposals=proposal_map,
            payload=payload,
        )
        self.tasks[task_id] = task
        self._pending[task_id] = task
        return task

    def refresh(
        self,
        now: float,
        agent_available_at: Mapping[int, float],
        dispatchable_agents: Optional[set[int]] = None,
    ) -> None:
        """Refresh only time- and availability-dependent candidate values.

        ``dispatchable_agents`` restricts the hotness reduction to the agents
        that this dispatch round can actually use.  Taking the best slack over
        every agent, including busy ones, made a task look cold because some
        occupied agent had slack, even when all of its reachable options were
        already past their deadline.
        """

        eps = self.policy.epsilon
        rule_rows: List[tuple[BufferedTask, List[CandidateState], List[CandidateState]]] = []
        for task in self.pending_tasks():
            dynamic_rows: List[CandidateState] = []
            reachable: List[CandidateState] = []
            for agent_id, candidate in task.proposals.items():
                if candidate.status is not CandidateStatus.PENDING:
                    continue
                available_at = max(float(now), float(agent_available_at[agent_id]))
                processing = candidate.proposal.predicted_processing
                wait = max(0.0, available_at - float(now))
                finish = available_at + processing

                candidate.agent_available_at = available_at
                candidate.predicted_start = available_at
                candidate.predicted_finish = finish
                candidate.availability_score = exp(
                    -wait / max(task.time_budget, eps)
                )
                candidate.deadline_gap = max(0.0, finish - task.deadline) / max(
                    task.time_budget, eps
                )
                candidate.required_time_ratio = (
                    available_at - task.arrival_time + processing
                ) / max(task.time_budget, eps)
                candidate.slack_coefficient = (
                    task.deadline - available_at
                ) / (processing + eps)
                dynamic_rows.append(candidate)
                if dispatchable_agents is None or agent_id in dispatchable_agents:
                    reachable.append(candidate)

            if not dynamic_rows:
                continue

            # A task is hot when even its best *reachable* execution option has
            # little relative slack.  This term is task-level, so urgency does
            # not distort the ranking of agents for the same task.
            domain = getattr(task.payload, "domain", None)
            (w_hot, w_wait, w_gap, w_over, slope) = self.policy.weights_for(domain)

            slack_source = reachable or dynamic_rows
            best_slack = max(row.slack_coefficient for row in slack_source)
            hotness = self._sigmoid(slope * (1.0 - best_slack))

            # Waiting pressure uses the controller's published admissible-wait
            # prediction.  It is task-level and never redefines the external
            # deadline or the time budget, so the waiting model influences
            # dispatch order without defining the metric it is scored on.
            elapsed = max(0.0, float(now) - task.arrival_time)
            budget = max(
                max(row.proposal.predicted_wait_budget for row in dynamic_rows), eps
            )
            wait_pressure = min(elapsed / budget, self.policy.wait_pressure_cap)

            for candidate in dynamic_rows:
                candidate.wait_pressure = wait_pressure
                candidate.priority = (
                    candidate.proposal.base_score
                    + w_hot * hotness
                    + w_wait * wait_pressure
                    - w_gap * candidate.deadline_gap
                    - w_over * max(0.0, candidate.required_time_ratio - 1.0)
                )
            if self.policy.deadline_urgency_weight:
                urgency = (self.policy.deadline_urgency_weight * (task.deadline - float(now))
                           / self.policy.urgency_scale)
                for candidate in dynamic_rows:
                    candidate.priority -= urgency
            if self.policy.order != "weighted":
                rule_rows.append((task, dynamic_rows, slack_source))

        if rule_rows:
            self._rank_by_rule(now, rule_rows)

    def _rank_by_rule(
        self,
        now: float,
        rule_rows: List[tuple[BufferedTask, List[CandidateState], List[CandidateState]]],
    ) -> None:
        """Priorities of a classic rule: urgency rank first, base score second."""
        order = self.policy.order

        def urgency(entry) -> tuple[float, int]:
            task, _, source = entry
            fastest = min(row.proposal.predicted_processing for row in source)
            if order == "edf":
                key = task.deadline
            elif order == "llf":
                key = task.deadline - float(now) - fastest
            else:
                key = fastest
            return (key, task.task_id)

        for rank, (_, rows, _) in enumerate(sorted(rule_rows, key=urgency)):
            for candidate in rows:
                candidate.priority = candidate.proposal.base_score - self.policy.RANK_GAP * rank

    def dispatch(
        self,
        now: float,
        agent_available_at: Mapping[int, float],
    ) -> List[Assignment]:
        """Assign distinct pending tasks to distinct agents free at ``now``."""

        free_agents = {
            agent_id
            for agent_id, available_at in agent_available_at.items()
            if float(available_at) <= float(now)
        }
        if not free_agents:
            return []
        self.refresh(now, agent_available_at, free_agents)

        if self.policy.matching == "exact":
            return self._dispatch_exact(now, free_agents)

        ranked_edges = []
        for task in self.pending_tasks():
            for agent_id, candidate in task.proposals.items():
                if (
                    candidate.status is CandidateStatus.PENDING
                    and agent_id in free_agents
                ):
                    ranked_edges.append((candidate.priority, task.task_id, agent_id))

        # Stable tie-breaking keeps the greedy baseline reproducible.
        ranked_edges.sort(key=lambda row: (-row[0], row[1], row[2]))
        reserved_tasks = set()
        reserved_agents = set()
        assignments: List[Assignment] = []

        for _, task_id, agent_id in ranked_edges:
            if task_id in reserved_tasks or agent_id in reserved_agents:
                continue
            task = self.tasks[task_id]
            if task.status is not TaskStatus.PENDING:
                continue
            candidate = task.proposals[agent_id]
            assignment = self._assign(task, candidate, agent_id, now)
            assignments.append(assignment)
            reserved_tasks.add(task_id)
            reserved_agents.add(agent_id)

        return assignments

    def _dispatch_exact(self, now: float, free_agents: set[int]) -> List[Assignment]:
        """Maximum-weight bipartite matching with deterministic tie-breaking."""
        from scipy.optimize import linear_sum_assignment

        tasks = sorted(self.pending_tasks(), key=lambda task: task.task_id)
        agents = sorted(free_agents)
        if not tasks or not agents:
            return []
        weights = np.full((len(tasks), len(agents)), -1e12, dtype=float)
        for row, task in enumerate(tasks):
            for col, agent_id in enumerate(agents):
                candidate = task.proposals.get(agent_id)
                if candidate is None or candidate.status is not CandidateStatus.PENDING:
                    continue
                tie = -1e-12 * (row * max(1, len(agents)) + col)
                weights[row, col] = candidate.priority + tie
        row_ids, col_ids = linear_sum_assignment(weights, maximize=True)
        assignments: List[Assignment] = []
        for row, col in sorted(zip(row_ids, col_ids)):
            if weights[row, col] <= -1e11:
                continue
            task = tasks[int(row)]
            agent_id = agents[int(col)]
            assignments.append(
                self._assign(task, task.proposals[agent_id], agent_id, now)
            )
        return assignments
    def complete(self, task_id: int, completed_at: float, success: bool) -> None:
        task = self.tasks[task_id]
        if task.status is not TaskStatus.ASSIGNED:
            raise ValueError(f"task {task_id} has not been assigned")
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.completed_at = float(completed_at)

    def pending_tasks(self) -> Iterable[BufferedTask]:
        return iter(list(self._pending.values()))

    def pending_count(self) -> int:
        """Number of tasks waiting in the buffer, in constant time."""
        return len(self._pending)

    def next_agent_release(
        self,
        now: float,
        agent_available_at: Mapping[int, float],
    ) -> Optional[float]:
        future = [
            float(value) for value in agent_available_at.values() if float(value) > now
        ]
        return min(future) if future else None

    def _assign(
        self,
        task: BufferedTask,
        candidate: CandidateState,
        agent_id: int,
        now: float,
    ) -> Assignment:
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent_id = agent_id
        task.assigned_at = float(now)
        self._pending.pop(task.task_id, None)
        candidate.status = CandidateStatus.ASSIGNED

        for other_agent_id, other in task.proposals.items():
            if other_agent_id != agent_id:
                other.status = CandidateStatus.INVALIDATED

        assignment = Assignment(
            task_id=task.task_id,
            controller_id=task.controller_id,
            agent_id=agent_id,
            dispatch_time=float(now),
            predicted_processing=candidate.proposal.predicted_processing,
            predicted_finish=candidate.predicted_finish,
            base_score=candidate.proposal.base_score,
            priority=candidate.priority,
            slack_coefficient=candidate.slack_coefficient,
            feature_vector=candidate.proposal.feature_vector,
            payload=task.payload,
        )
        self.assignment_history.append(assignment)
        return assignment

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            z = exp(-value)
            return 1.0 / (1.0 + z)
        z = exp(value)
        return z / (1.0 + z)

