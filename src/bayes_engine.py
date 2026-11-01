"""
Baymax - Probabilistic Reasoning Engine (Bayesian)
==================================================

This module implements the probabilistic inference layer of the
Baymax expert system using a Naive Bayes formulation of
Bayes' Theorem.

Given a set of user-reported symptoms, this engine computes the
posterior probability of each disease in the knowledge base:

    P(D | S) ∝ P(D) * Π P(S_i | D)

Implementation notes
--------------------
- Conditional symptom probabilities are parameterized using three
  profiles aligned with the synthetic dataset generator:

      Core symptom of disease     -> P(symptom=1 | D) = 0.85
      Optional symptom of disease -> P(symptom=1 | D) = 0.40
      Unrelated symptom           -> P(symptom=1 | D) = 0.05

- Log-probabilities are used internally to avoid floating-point
  underflow when multiplying many small values.

- Posteriors are normalized so they sum to 1 across all diseases,
  giving an interpretable probability distribution suitable for
  ranking, display, and downstream use by the search engine.

This engine satisfies the assignment requirements for:
    - Probabilistic reasoning
    - Bayes' Theorem
    - Reasoning under uncertainty
"""

import math
from typing import Dict, Iterable, List, Tuple

from .knowledge_base import get_symptom_ids, load_diseases
from .utils import normalize_symptoms, sort_diagnoses


# ---------------------------------------------------------------------------
# Likelihood parameters
# ---------------------------------------------------------------------------
#
# These values define how likely each symptom is to be present, given
# the disease, depending on whether the symptom is core, optional, or
# unrelated. They are aligned with the synthetic data generation model.

P_SYMPTOM_GIVEN_CORE: float = 0.85
P_SYMPTOM_GIVEN_OPTIONAL: float = 0.40
P_SYMPTOM_GIVEN_UNRELATED: float = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_symptom(symptom_id: str, disease_info: dict) -> str:
    """
    Classify a symptom relative to a disease as one of:
        - "core"
        - "optional"
        - "unrelated"
    """
    core = set(disease_info.get("core_symptoms", []) or [])
    optional = set(disease_info.get("optional_symptoms", []) or [])

    if symptom_id in core:
        return "core"
    if symptom_id in optional:
        return "optional"
    return "unrelated"


def _p_present_given_disease(symptom_id: str, disease_info: dict) -> float:
    """
    Return P(symptom = 1 | disease) using the parameterized profiles.
    """
    category = _classify_symptom(symptom_id, disease_info)
    if category == "core":
        return P_SYMPTOM_GIVEN_CORE
    if category == "optional":
        return P_SYMPTOM_GIVEN_OPTIONAL
    return P_SYMPTOM_GIVEN_UNRELATED


def _safe_log(value: float) -> float:
    """
    Compute log(value) with a small floor to avoid log(0).
    """
    return math.log(max(value, 1e-12))


# ---------------------------------------------------------------------------
# Single-disease posterior (log-space, unnormalized)
# ---------------------------------------------------------------------------

def compute_posterior(
    symptoms: Iterable[str],
    disease_info: dict,
) -> float:
    """
    Compute the unnormalized log-posterior log( P(D) * Π P(S_i | D) )
    for a single disease, using a Naive Bayes assumption.

    All symptoms in the master symptom list are considered:
        - If the user reports a symptom, P(symptom=1 | D) is used.
        - If the user does NOT report a symptom, P(symptom=0 | D) is used.

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user.
    disease_info : dict
        A disease entry from the knowledge base.

    Returns
    -------
    float
        log( P(D) * Π P(S_i | D) ), unnormalized.
    """
    user_symptoms = set(normalize_symptoms(symptoms))
    all_symptom_ids = get_symptom_ids()

    prior = float(disease_info.get("prior", 1e-6))
    log_posterior = _safe_log(prior)

    for sid in all_symptom_ids:
        p_present = _p_present_given_disease(sid, disease_info)
        if sid in user_symptoms:
            log_posterior += _safe_log(p_present)
        else:
            log_posterior += _safe_log(1.0 - p_present)

    return log_posterior


# ---------------------------------------------------------------------------
# Multi-disease diagnosis (normalized posteriors)
# ---------------------------------------------------------------------------

def bayesian_diagnosis(
    symptoms: Iterable[str],
) -> List[Tuple[str, float]]:
    """
    Compute posterior probabilities P(D | S) for every disease in the
    knowledge base and return them as a ranked list.

    Posteriors are normalized so they sum to 1.

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user.

    Returns
    -------
    list of (str, float)
        (disease_name, posterior_probability), sorted descending.
    """
    diseases: Dict[str, dict] = load_diseases()

    if not diseases:
        return []

    log_scores: List[Tuple[str, float]] = []
    for name, info in diseases.items():
        log_scores.append((name, compute_posterior(symptoms, info)))

    # Normalize using the log-sum-exp trick for numerical stability.
    max_log = max(score for _, score in log_scores)
    exp_scores = [(name, math.exp(score - max_log)) for name, score in log_scores]
    total = sum(value for _, value in exp_scores)

    if total <= 0:
        # Fallback: equal distribution if everything underflowed.
        n = len(exp_scores)
        return sort_diagnoses([(name, 1.0 / n) for name, _ in exp_scores])

    posteriors = [(name, value / total) for name, value in exp_scores]
    return sort_diagnoses(posteriors)


# ---------------------------------------------------------------------------
# Explanation for transparency
# ---------------------------------------------------------------------------

def explain_bayesian_inference(
    symptoms: Iterable[str],
    disease_info: dict,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Produce a structured explanation showing how each symptom contributed
    to the posterior for a single disease.

    For each symptom in the master list, the explanation reports:
        - the symptom ID
        - whether the user reported it
        - how it was classified (core / optional / unrelated)
        - the conditional probability used
        - the resulting log-likelihood contribution

    This supports the assignment's transparency / explainability
    requirement and is used on the result page.

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user.
    disease_info : dict
        A disease entry from the knowledge base.

    Returns
    -------
    dict
        {
            "prior_log":     float,
            "contributions": [
                {
                    "symptom":    str,
                    "reported":   bool,
                    "category":   "core" | "optional" | "unrelated",
                    "p_used":     float,
                    "log_factor": float
                },
                ...
            ],
            "log_posterior": float
        }
    """
    user_symptoms = set(normalize_symptoms(symptoms))
    all_symptom_ids = get_symptom_ids()

    prior = float(disease_info.get("prior", 1e-6))
    prior_log = _safe_log(prior)

    contributions: List[Dict[str, object]] = []
    log_posterior = prior_log

    for sid in all_symptom_ids:
        category = _classify_symptom(sid, disease_info)
        p_present = _p_present_given_disease(sid, disease_info)
        reported = sid in user_symptoms
        p_used = p_present if reported else (1.0 - p_present)
        log_factor = _safe_log(p_used)
        log_posterior += log_factor

        contributions.append(
            {
                "symptom": sid,
                "reported": reported,
                "category": category,
                "p_used": p_used,
                "log_factor": log_factor,
            }
        )

    return {
        "prior_log": prior_log,
        "contributions": contributions,
        "log_posterior": log_posterior,
    }