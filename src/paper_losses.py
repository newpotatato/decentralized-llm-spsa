"""Losses used in the manuscript for the IT routing experiments.

The module is deliberately independent from a particular optimizer.  Every
method is evaluated on exactly the same F1/F2/F3 definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PaperLossConfig:
    delta_p: float = 1.0
    epsilon_p: float = 0.05
    a_relevance_p: float = 0.25
    a_hardness_p: float = 0.50
    lambda_reg_p: float = 0.002
    wait_under_weight: float = 2.0
    wait_over_weight: float = 1.0
    a_hardness_q: float = 0.50
    a_risk_q: float = 0.50
    # The manuscript writes a ridge only in F1, as (lambda/2)||theta||^2, and
    # leaves F2 and F3 unregularised.  Two corrections are made here.
    #
    # 1. The penalty is taken about the initial parameters, not about the
    #    origin.  Processing times live in [1, 4] and qualities in [0.4, 0.9],
    #    so pulling theta towards zero biases the predictor towards predicting
    #    no time and no quality: measured held-out RMSE gets monotonically
    #    worse with lambda under the manuscript form and never beats the frozen
    #    prior, while the same lambda about the prior beats it by a quarter.
    #
    # 2. All three blocks carry one.  The manuscript sets mu = omega*lambda_2,
    #    but that is the curvature of the disagreement term alone and it
    #    vanishes on the consensus subspace theta_1 = ... = theta_m.  The only
    #    curvature Phi has there is the ridge, so acceleration is calibrated to
    #    a strong convexity the problem does not possess unless every block is
    #    regularised with lambda >= mu.
    lambda_reg_q: float = 0.002
    lambda_reg_w: float = 0.002
    # "prior" penalises ||theta - theta_0||^2; "origin" is the manuscript's
    # literal ||theta||^2.  Same curvature, so the same strong-convexity
    # modulus; they differ only in what the shrinkage target is.
    ridge_reference: str = "prior"


@dataclass(frozen=True)
class LossObservation:
    """One completed task--agent observation.

    ``relevance`` is the manuscript's semantic relevance ``R_{i,k_i}``.  It is
    the *only* pair-level correction used by F1 and F3: the manuscript writes
    ``p^corr = p / (1 + a_R^{(p)}(1 - R))`` and weights the F3 overprediction
    penalty by ``a_R^{(q)} R``.  Earlier revisions substituted the oracle
    capability matrix here (privileged information) and task urgency in F3;
    both substitutions are removed.
    """

    features: np.ndarray
    agent_id: int
    processing_real: float
    waiting_real: float
    quality_real: float
    relevance: float
    hardness: float


def corrected_processing_target(obs: LossObservation, cfg: PaperLossConfig) -> float:
    denominator = 1.0 + cfg.a_relevance_p * (1.0 - float(obs.relevance))
    return float(obs.processing_real) / max(denominator, 1e-12)


def processing_loss(prediction: float, obs: LossObservation, cfg: PaperLossConfig) -> float:
    """F1: relevance-corrected, hardness-scaled pseudo-Huber loss."""
    target = corrected_processing_target(obs, cfg)
    scale = cfg.delta_p * (
        cfg.epsilon_p + target * (1.0 + cfg.a_hardness_p * float(obs.hardness))
    )
    residual = (float(prediction) - target) / max(scale, 1e-12)
    return float(cfg.delta_p**2 * (np.sqrt(1.0 + residual**2) - 1.0))


def waiting_loss(prediction: float, obs: LossObservation, cfg: PaperLossConfig) -> float:
    """F2: asymmetric squared loss; underprediction is more expensive."""
    under = max(float(obs.waiting_real) - float(prediction), 0.0)
    over = max(float(prediction) - float(obs.waiting_real), 0.0)
    return float(cfg.wait_under_weight * under**2 + cfg.wait_over_weight * over**2)


def quality_loss(prediction: float, obs: LossObservation, cfg: PaperLossConfig) -> float:
    """F3: hardness-weighted MSE plus relevance-weighted overprediction penalty."""
    residual = float(prediction) - float(obs.quality_real)
    base = (1.0 + cfg.a_hardness_q * float(obs.hardness)) * residual**2
    risk = cfg.a_risk_q * float(obs.relevance) * max(residual, 0.0) ** 2
    return float(base + risk)


def ridge_penalty(
    parameters: np.ndarray | None,
    coefficient: float,
    reference: np.ndarray | None = None,
) -> float:
    """``(lambda_reg / 2) * ||theta - reference||^2``.

    The manuscript states the penalty per sample; because it does not depend on
    the observation, the sample mean equals the same term, so it is added once
    per objective evaluation rather than once per observation.

    ``reference`` defaults to the origin, which reproduces the manuscript form
    exactly.  Passing the initial parameters instead keeps the same curvature -
    and therefore the same strong-convexity modulus - while removing the bias
    towards predicting zero.
    """
    if parameters is None or coefficient <= 0.0:
        return 0.0
    theta = np.asarray(parameters, dtype=float)
    if reference is not None:
        theta = theta - np.asarray(reference, dtype=float)
    return float(0.5 * coefficient * float(np.sum(theta ** 2)))


def strong_convexity_floor(cfg: PaperLossConfig) -> dict[str, float]:
    """Curvature each block contributes on the consensus subspace."""
    return {
        "processing": float(cfg.lambda_reg_p),
        "waiting": float(cfg.lambda_reg_w),
        "quality": float(cfg.lambda_reg_q),
    }


def mean_losses(
    observations: list[LossObservation],
    predict_processing,
    predict_waiting,
    predict_quality,
    cfg: PaperLossConfig,
    processing_parameters: np.ndarray | None = None,
    quality_parameters: np.ndarray | None = None,
    waiting_parameters: np.ndarray | None = None,
    processing_reference: np.ndarray | None = None,
    waiting_reference: np.ndarray | None = None,
    quality_reference: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Mean F1/F2/F3 including the manuscript ridge terms of F1 and F3.

    ``processing_parameters`` / ``quality_parameters`` must be supplied for the
    reported values to match the objective actually optimised; omitting them
    reproduces the unregularised data terms only.
    """
    reg_p = ridge_penalty(processing_parameters, cfg.lambda_reg_p, processing_reference)
    reg_w = ridge_penalty(waiting_parameters, cfg.lambda_reg_w, waiting_reference)
    reg_q = ridge_penalty(quality_parameters, cfg.lambda_reg_q, quality_reference)
    if not observations:
        return reg_p, reg_w, reg_q
    f1 = [processing_loss(predict_processing(o), o, cfg) for o in observations]
    f2 = [waiting_loss(predict_waiting(o), o, cfg) for o in observations]
    f3 = [quality_loss(predict_quality(o), o, cfg) for o in observations]
    return (
        float(np.mean(f1)) + reg_p,
        float(np.mean(f2)) + reg_w,
        float(np.mean(f3)) + reg_q,
    )
