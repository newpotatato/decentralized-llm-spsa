"""Common zeroth-order optimizers for the controlled paper experiment.

All methods update the same processing, waiting, and quality blocks and call
the same block-local objectives. Model-specific algorithm parameters are allowed,
while the network-objective coefficient is fixed across compared methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


METHODS = ("spsa", "aspsa", "kw", "zo_pgd", "sp_gt", "zo_gt", "pd_2pt")
ABLATIONS = (
    "no_learning", "local_rls", "spsa", "aspsa_no_momentum",
    "aspsa_fixed_beta", "aspsa_no_consensus", "aspsa",
)


@dataclass(frozen=True)
class OptimizerConfig:
    """Parameters of one method on one parameter block.

    ``consensus_weight`` is the network coefficient ``omega`` of the shared
    objective ``Phi``.  It is fixed across compared methods and is never a
    method-specific tuning dimension: every method now descends the *same*
    consensus gradient ``omega * sum_l b_jl (theta_j - theta_l)`` scaled by its
    own step size, so the realised consensus strength is ``omega * h`` for all
    of them.  Earlier revisions applied a step-independent 5% mixing to the
    baselines while A-SPSA received ``omega * h``; with a tuned ``h = 12`` that
    made A-SPSA average 12x harder than every method it was compared against.
    """

    step_size: float = 0.04
    perturbation: float = 0.10
    consensus_weight: float = 0.04
    alpha_decay: float = 0.602
    perturbation_decay: float = 0.101
    acceleration_epsilon: float = 0.05
    constraint_budget: float = 3.0
    dual_step: float = 0.02
    gamma0: float = 1.0
    mu_minus_eta: float = 0.01
    initial_alpha: float = 0.0
    # The manuscript draws Delta in {-1/sqrt(d), +1/sqrt(d)}^d, for which
    # E[Delta Delta^T] = I/d and the two-point estimate is biased towards
    # grad F / d.  With compensation enabled every estimator in METHODS is
    # unbiased for grad F, so one shared step grid is genuinely comparable and
    # the manuscript prescription h = 1/max{M, omega*lambda_max} applies
    # directly.  Disable it only to reproduce the raw manuscript recurrence.
    compensate_perturbation_dimension: bool = True
    # Manuscript initialisation is z_{j,0} = 0.  Because theta_0 is not the
    # origin for a regression model, that is an implicit shrinkage prior; the
    # alternative anchors the surrogate at the incoming parameters.
    surrogate_init: str = "zero"


@dataclass(frozen=True)
class ModelOptimizerConfigs:
    processing: OptimizerConfig = field(default_factory=OptimizerConfig)
    waiting: OptimizerConfig = field(default_factory=OptimizerConfig)
    quality: OptimizerConfig = field(default_factory=OptimizerConfig)

    def as_dict(self) -> dict[str, OptimizerConfig]:
        return {
            "processing": self.processing,
            "waiting": self.waiting,
            "quality": self.quality,
        }


@dataclass
class OptimizerState:
    vectors: dict[Any, dict[str, np.ndarray]] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    query_count: int = 0


class _BlockProxy:
    def __init__(self, controller: Any, block_name: str, block_slice: slice):
        self.controller = controller
        self.block_name = block_name
        self.block_slice = block_slice
        self.controller_id = (block_name, controller.controller_id)

    def pack_parameters(self) -> np.ndarray:
        return self.controller.pack_parameters()[self.block_slice].copy()

    def unpack_parameters(self, vector: np.ndarray) -> None:
        complete = self.controller.pack_parameters()
        complete[self.block_slice] = vector
        self.controller.unpack_parameters(complete)

    def project(self, vector: np.ndarray) -> np.ndarray:
        complete = self.controller.pack_parameters()
        complete[self.block_slice] = vector
        return self.controller.project(complete)[self.block_slice]

    def objective(self, vector: np.ndarray) -> float:
        complete = self.controller.pack_parameters()
        complete[self.block_slice] = vector
        return float(self.controller.objective_block(complete, self.block_name))

    def constraint(self, vector: np.ndarray, budget: float):
        if self.block_name != "processing":
            return -float(budget), np.zeros_like(vector)
        complete = self.controller.pack_parameters()
        complete[self.block_slice] = vector
        violation, gradient = self.controller.constraint(complete, budget)
        return violation, gradient[self.block_slice]

class DistributedZerothOrderOptimizer:
    def __init__(
        self,
        method: str,
        config: OptimizerConfig | None = None,
        seed: int = 0,
        block_configs: dict[str, OptimizerConfig] | None = None,
        active_blocks: tuple[str, ...] = ("processing", "waiting", "quality"),
    ):
        if method not in METHODS and method not in {
            "aspsa_no_momentum", "aspsa_fixed_beta", "aspsa_no_consensus"
        }:
            raise ValueError(f"unknown optimizer: {method}")
        valid_blocks = {"processing", "waiting", "quality"}
        if not active_blocks or not set(active_blocks) <= valid_blocks:
            raise ValueError("active_blocks must be a non-empty subset of processing/waiting/quality")
        self.method = method
        self.config = config or OptimizerConfig()
        self.block_configs = dict(block_configs or {})
        self.active_blocks = tuple(active_blocks)
        self.rng = np.random.default_rng(seed)
        self.state = OptimizerState()

    def _evaluate(self, controller: Any, vector: np.ndarray) -> float:
        self.state.query_count += 1
        return float(controller.objective(vector))

    def _schedule(self, round_index: int) -> tuple[float, float]:
        k = max(1, int(round_index))
        step = self.config.step_size / (k + 10.0) ** self.config.alpha_decay
        pert = self.config.perturbation / k ** self.config.perturbation_decay
        return float(step), float(max(pert, 1e-6))

    @staticmethod
    def _ring_average(vectors: list[np.ndarray]) -> list[np.ndarray]:
        n = len(vectors)
        if n <= 1:
            return [v.copy() for v in vectors]
        return [
            (vectors[(i - 1) % n] + vectors[i] + vectors[(i + 1) % n]) / 3.0
            for i in range(n)
        ]

    def step(self, controllers: list[Any], round_index: int) -> dict[str, float]:
        if not controllers:
            return {"oracle_queries": 0.0}
        if not hasattr(controllers[0], "parameter_blocks"):
            return self._step_core(controllers, round_index)
        before = self.state.query_count
        original_config = self.config
        trace: dict[str, float] = {}
        for block_name, block_slice in controllers[0].parameter_blocks():
            if block_name not in self.active_blocks:
                continue
            self.config = self.block_configs.get(block_name, original_config)
            proxies = [_BlockProxy(controller, block_name, block_slice) for controller in controllers]
            block_trace = self._step_core(proxies, round_index)
            trace[f"oracle_queries_{block_name}"] = block_trace["oracle_queries"]
            trace[f"step_size_{block_name}"] = block_trace["step_size"]
            trace[f"perturbation_{block_name}"] = block_trace["perturbation"]
        self.config = original_config
        trace.update({
            "oracle_queries": float(self.state.query_count - before),
            "oracle_queries_cumulative": float(self.state.query_count),
        })
        return trace

    def _step_core(self, controllers: list[Any], round_index: int) -> dict[str, float]:
        if not controllers:
            return {"oracle_queries": 0.0}
        before = self.state.query_count
        vectors = [c.pack_parameters() for c in controllers]
        step_size, pert = self._schedule(round_index)
        consensus = self._ring_average(vectors)

        if self.method.startswith("aspsa"):
            # The manuscript uses a fixed descent parameter h. The fixed-beta
            # ablation removes only the perturbation decay.
            step_size = self.config.step_size
            if self.method == "aspsa_fixed_beta":
                pert = self.config.perturbation
            updated = self._aspsa(controllers, vectors, consensus, round_index, step_size, pert)
        elif self.method == "kw":
            updated = self._kw(controllers, vectors, consensus, step_size, pert)
        elif self.method == "zo_pgd":
            updated = self._two_point_sphere(controllers, vectors, consensus, step_size, pert, tracking=False)
        elif self.method == "sp_gt":
            updated = self._single_point_tracking(controllers, vectors, consensus, step_size, pert)
        elif self.method == "zo_gt":
            updated = self._two_point_sphere(controllers, vectors, consensus, step_size, pert, tracking=True)
        elif self.method == "pd_2pt":
            updated = self._primal_dual(controllers, vectors, consensus, step_size, pert)
        else:
            updated = self._spsa(controllers, vectors, consensus, step_size, pert)

        for controller, vector in zip(controllers, updated):
            controller.unpack_parameters(vector)
        return {
            "oracle_queries": float(self.state.query_count - before),
            "oracle_queries_cumulative": float(self.state.query_count),
            "step_size": step_size,
            "perturbation": pert,
        }

    def _consensus_gradient(
        self, point: np.ndarray, ring_average: np.ndarray, allow: bool = True
    ) -> np.ndarray:
        """``omega * sum_l b_jl (theta_j - theta_l)`` for the ring with b = 1/3.

        ``ring_average`` is ``(theta_{j-1} + theta_j + theta_{j+1}) / 3``, so
        ``point - ring_average`` equals ``sum_l b_jl (point - theta_l)``
        exactly.  This is the gradient of the disagreement term of ``Phi``:
        every method therefore descends the same network objective with the
        same realised consensus strength ``omega * h``.
        """
        if not allow:
            return np.zeros_like(point)
        return self.config.consensus_weight * (point - ring_average)

    def _perturbation_scale(self, dimension: int) -> tuple[float, float]:
        """Return ``(delta_scale, estimator_gain)`` for the SPSA family."""
        scale = 1.0 / np.sqrt(dimension)
        gain = float(dimension) if self.config.compensate_perturbation_dimension else 1.0
        return float(scale), gain

    def _spsa(self, controllers, vectors, averages, step_size, pert):
        out = []
        for c, x, avg in zip(controllers, vectors, averages):
            scale, gain = self._perturbation_scale(x.size)
            delta = self.rng.choice((-1.0, 1.0), size=x.size) * scale
            grad = gain * (
                self._evaluate(c, x + pert * delta) - self._evaluate(c, x - pert * delta)
            ) / (2.0 * pert) * delta
            grad = grad + self._consensus_gradient(x, avg)
            out.append(c.project(x - step_size * grad))
        return out

    @staticmethod
    def _solve_alpha(h: float, kappa: float, mu_eta: float, epsilon: float) -> float:
        leading = 1.0 + max(0.0, epsilon)
        coefficient = h * (kappa - mu_eta)
        discriminant = max(0.0, coefficient**2 + 4.0 * leading * h * kappa)
        return float(np.clip((-coefficient + np.sqrt(discriminant)) / (2.0 * leading), 0.01, 0.99))

    def _surrogate_center(self, controller: Any, vector: np.ndarray) -> np.ndarray:
        """Initial ``z_{j,0}``.

        The manuscript prescribes ``z_{j,0} = 0``.  For a regression model whose
        parameters are not centred at the origin that is an implicit shrinkage
        prior, so the alternative anchoring at ``theta_{j,0}`` is available and
        recorded as an explicit implementation choice.
        """
        slot = self.state.vectors.setdefault(controller.controller_id, {})
        if "surrogate_center" not in slot:
            slot["surrogate_center"] = (
                vector.copy() if self.config.surrogate_init == "theta"
                else np.zeros_like(vector)
            )
        return slot["surrogate_center"]

    def _aspsa(self, controllers, vectors, averages, round_index, step_size, pert):
        """Accelerated three-sequence update from the manuscript equations."""
        h = float(step_size)
        if self.method == "aspsa_no_momentum":
            out = []
            for controller, vector, average in zip(controllers, vectors, averages):
                scale, gain = self._perturbation_scale(vector.size)
                delta = self.rng.choice((-1.0, 1.0), size=vector.size) * scale
                grad = gain * (
                    self._evaluate(controller, vector + pert * delta)
                    - self._evaluate(controller, vector - pert * delta)
                ) / (2.0 * pert) * delta
                grad = grad + self._consensus_gradient(vector, average)
                out.append(controller.project(vector - h * grad))
            return out
        mu_eta = float(self.config.mu_minus_eta)
        block_key = getattr(controllers[0], "block_name", "all")
        gamma_key = f"aspsa_gamma:{block_key}"
        alpha_key = f"aspsa_alpha:{block_key}"
        previous_gamma = self.state.scalars.get(gamma_key, self.config.gamma0)
        previous_alpha = self.state.scalars.get(alpha_key, self.config.initial_alpha)
        gamma = max(
            (1.0 - previous_alpha) * previous_gamma + previous_alpha * mu_eta,
            1e-12,
        )
        alpha = self._solve_alpha(
            h, previous_gamma, mu_eta, self.config.acceleration_epsilon
        )
        allow_consensus = self.method != "aspsa_no_consensus"
        denominator = max(previous_gamma + alpha * mu_eta, 1e-12)

        # The manuscript consensus correction is evaluated at the probing
        # points theta~, not at the incoming parameters, so the probes of every
        # controller must be formed before any gradient is taken.
        centers, probes = [], []
        for controller, vector in zip(controllers, vectors):
            center = self._surrogate_center(controller, vector)
            centers.append(center)
            probes.append((alpha * previous_gamma * center + gamma * vector) / denominator)
        probe_averages = self._ring_average(probes)

        out = []
        for c, center, probe, probe_avg in zip(controllers, centers, probes, probe_averages):
            scale, gain = self._perturbation_scale(probe.size)
            delta = self.rng.choice((-1.0, 1.0), size=probe.size) * scale
            grad = gain * (
                self._evaluate(c, probe + pert * delta)
                - self._evaluate(c, probe - pert * delta)
            ) / (2.0 * pert) * delta
            grad = grad + self._consensus_gradient(probe, probe_avg, allow_consensus)
            theta_bar = probe - h * grad
            new_center = (
                (1.0 - alpha) * previous_gamma * center
                + alpha * mu_eta * probe
                - alpha * grad
            ) / gamma
            self.state.vectors[c.controller_id]["surrogate_center"] = new_center
            out.append(c.project(theta_bar))
        self.state.scalars[gamma_key] = gamma
        self.state.scalars[alpha_key] = alpha
        return out
    def _kw(self, controllers, vectors, averages, step_size, pert):
        out = []
        for c, x, avg in zip(controllers, vectors, averages):
            grad = np.zeros_like(x)
            for index in range(x.size):
                unit = np.zeros_like(x)
                unit[index] = 1.0
                grad[index] = (self._evaluate(c, x + pert * unit) - self._evaluate(c, x - pert * unit)) / (2.0 * pert)
            grad = grad + self._consensus_gradient(x, avg)
            out.append(c.project(x - step_size * grad))
        return out

    def _sphere_gradient(self, controller, x, pert):
        direction = self.rng.normal(size=x.size)
        direction /= np.linalg.norm(direction) + 1e-12
        delta = self._evaluate(controller, x + pert * direction) - self._evaluate(controller, x - pert * direction)
        return x.size * delta / (2.0 * pert) * direction

    def _two_point_sphere(self, controllers, vectors, averages, step_size, pert, tracking):
        raw = [self._sphere_gradient(c, x, pert) for c, x in zip(controllers, vectors)]
        gradients = raw
        if tracking:
            previous_gradients = []
            previous_trackers = []
            for controller, grad in zip(controllers, raw):
                slot = self.state.vectors.setdefault(controller.controller_id, {})
                previous_gradients.append(slot.get("gradient", np.zeros_like(grad)))
                previous_trackers.append(slot.get("tracker", np.zeros_like(grad)))
            mixed_trackers = self._ring_average(previous_trackers)
            gradients = []
            for controller, grad, previous_grad, mixed in zip(controllers, raw, previous_gradients, mixed_trackers):
                tracker = mixed + grad - previous_grad
                slot = self.state.vectors[controller.controller_id]
                slot["gradient"] = grad.copy()
                slot["tracker"] = tracker.copy()
                gradients.append(tracker)
        return [
            c.project(x - step_size * (grad + self._consensus_gradient(x, avg)))
            for c, x, avg, grad in zip(controllers, vectors, averages, gradients)
        ]
    def _single_point_tracking(self, controllers, vectors, averages, step_size, pert):
        raw = []
        for controller, vector in zip(controllers, vectors):
            direction = self.rng.normal(size=vector.size)
            direction /= np.linalg.norm(direction) + 1e-12
            raw.append(vector.size * self._evaluate(controller, vector + pert * direction) / pert * direction)
        previous_gradients, previous_trackers = [], []
        for controller, grad in zip(controllers, raw):
            slot = self.state.vectors.setdefault(controller.controller_id, {})
            previous_gradients.append(slot.get("gradient", np.zeros_like(grad)))
            previous_trackers.append(slot.get("tracker", np.zeros_like(grad)))
        mixed_trackers = self._ring_average(previous_trackers)
        out = []
        for controller, vector, average, grad, previous_grad, mixed in zip(
            controllers, vectors, averages, raw, previous_gradients, mixed_trackers
        ):
            tracker = mixed + grad - previous_grad
            slot = self.state.vectors[controller.controller_id]
            slot["gradient"] = grad.copy()
            slot["tracker"] = tracker.copy()
            direction = tracker + self._consensus_gradient(vector, average)
            out.append(controller.project(vector - step_size * direction))
        return out
    def _primal_dual(self, controllers, vectors, averages, step_size, pert):
        out = []
        for c, x, avg in zip(controllers, vectors, averages):
            grad = self._sphere_gradient(c, x, pert)
            key = f"dual:{c.controller_id}"
            dual = self.state.scalars.get(key, 0.0)
            violation, constraint_grad = c.constraint(x, self.config.constraint_budget)
            dual = max(0.0, dual + self.config.dual_step * violation)
            self.state.scalars[key] = dual
            direction = grad + dual * constraint_grad + self._consensus_gradient(x, avg)
            out.append(c.project(x - step_size * direction))
        return out


def ring_laplacian_spectrum(controllers: int) -> tuple[float, float]:
    """``(lambda_2, lambda_max)`` of the averaged ring Laplacian with b = 1/3.

    The experiment fixes the communication graph to a ring in which every
    controller has exactly two neighbours of weight ``1/3``.
    """
    if controllers <= 1:
        return 0.0, 0.0
    weights = np.zeros((controllers, controllers), dtype=float)
    for index in range(controllers):
        for neighbour in ((index - 1) % controllers, (index + 1) % controllers):
            weights[index, neighbour] = 1.0 / 3.0
    laplacian = np.diag(weights.sum(axis=1)) - weights
    eigenvalues = np.sort(np.linalg.eigvalsh((laplacian + laplacian.T) / 2.0))
    return float(eigenvalues[1]), float(eigenvalues[-1])


def theory_parameters(
    smoothness: float, omega: float, controllers: int, ridge: float = 0.0
) -> dict[str, float]:
    """Strong-convexity modulus and step size of the network problem.

    The manuscript states ``mu = omega * lambda_2(L_av)``.  That is the
    curvature of the disagreement term, and the Hessian of the network
    objective is

        blockdiag(grad^2 F_j)  +  omega * (L_av kron I),

    whose second summand is positive semidefinite with a null space spanned by
    the consensus subspace ``theta_1 = ... = theta_m``.  On that subspace the
    disagreement term contributes nothing, so

        mu  =  lambda_min  >=  ridge + 0  =  ridge,

    and ``omega * lambda_2`` is only the modulus *restricted to the orthogonal
    complement*.  With an unregularised block the problem is convex but not
    strongly convex, and the acceleration is calibrated to a modulus the
    objective does not have.  ``mu_manuscript`` is reported alongside so the
    discrepancy stays visible.
    """
    lambda_2, lambda_max = ring_laplacian_spectrum(controllers)
    mu_manuscript = omega * lambda_2
    mu = float(ridge)
    step = 1.0 / max(smoothness, omega * lambda_max, 1e-12)
    return {
        "lambda_2": lambda_2,
        "lambda_max": lambda_max,
        "mu": mu,
        "mu_manuscript": mu_manuscript,
        "mu_on_disagreement_subspace": mu_manuscript,
        "h_theory": step,
        "eta_upper_bound": mu,
        "strongly_convex": bool(mu > 0.0),
    }


def lipschitz_constants(
    loss_config: Any, feature_norm: float = 1.0
) -> dict[str, float]:
    """``M_p``, ``M_w``, ``M_q`` from the manuscript verification of Assumption 4."""
    squared = float(feature_norm) ** 2
    return {
        "M_p": squared / float(loss_config.epsilon_p) ** 2,
        "M_w": 2.0 * float(loss_config.wait_under_weight) * squared,
        "M_q": 2.0 * (
            1.0 + float(loss_config.a_hardness_q) + float(loss_config.a_risk_q)
        ) * squared,
    }
