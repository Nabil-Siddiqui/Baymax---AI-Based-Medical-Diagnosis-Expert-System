"""
Baymax - Rule-Based Inference Engine
====================================

This module implements the classical, knowledge-based inference layer
of the Baymax expert system. It compares a user's reported symptoms
against the symptom profile of each disease defined in the knowledge
base and computes a normalized match score in the range [0, 1].

The scoring function is intentionally more nuanced than simple
"matched / total" counting. It distinguishes between:

    - Core symptoms      : strongly associated with the disease
    - Optional symptoms  : sometimes associated with the disease
    - Missing core       : reduces the score slightly
    - Unrelated symptoms : reduces the score slightly

This produces a clinically reasonable, explainable score that supports
both ranking and transparent justification on the result page.

This engine is the "expert system" layer of Baymax, separate from the
probabilistic (Bayesian) and machine learning layers.
"""

from typing import Dict, Iterable, List, Tuple

from .knowledge_base import load_diseases
from .utils import normalize_symptoms, sort_diagnoses


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
#
# These weights control how much each factor contributes to the final
# match score. They are intentionally tunable so the system's behavior
# can be adjusted without changing the algorithm.

CORE_MATCH_WEIGHT: float = 1.0
"""Reward for each matched core symptom."""

OPTIONAL_MATCH_WEIGHT: float = 0.5
"""Reward for each matched optional symptom."""

MISSING_CORE_PENALTY: float = 0.5
"""Penalty for each core symptom the user does NOT have."""

UNRELATED_SYMPTOM_PENALTY: float = 0.25
"""Penalty for each user symptom that is not part of the disease profile."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_symptoms(disease_info: dict) -> Tuple[set, set]:
    """
    Extract the core and optional symptom sets from a disease entry.

    Parameters
    ----------
    disease_info : dict
        A single disease entry from the knowledge base.

    Returns
    -------
    (set, set)
        (core_symptoms, optional_symptoms)
    """
    core = set(disease_info.get("core_symptoms", []) or [])
    optional = set(disease_info.get("optional_symptoms", []) or [])
    return core, optional


def _max_possible_reward(core: set, optional: set) -> float:
    """
    Compute the maximum reward a disease could receive if the user
    matched all of its core and optional symptoms.

    This is used as a normalization factor so the final score lies
    in [0, 1].
    """
    return (
        len(core) * CORE_MATCH_WEIGHT
        + len(optional) * OPTIONAL_MATCH_WEIGHT
    )


# ---------------------------------------------------------------------------
# Single-disease scoring
# ---------------------------------------------------------------------------

def score_disease(
    symptoms: Iterable[str],
    disease_info: dict,
) -> float:
    """
    Compute the rule-based match score for a single disease.

    The score is normalized to the range [0, 1]:
        - 1.0 means a strong match (all core symptoms present, no
          unrelated symptoms reported).
        - 0.0 means no meaningful match.

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user (will be normalized internally).
    disease_info : dict
        A disease entry from the knowledge base.

    Returns
    -------
    float
        Match score in [0, 1].
    """
    user_symptoms = set(normalize_symptoms(symptoms))
    core, optional = _split_symptoms(disease_info)

    if not core and not optional:
        # Disease has no symptoms defined; cannot score.
        return 0.0

    matched_core = user_symptoms & core
    matched_optional = user_symptoms & optional
    missing_core = core - user_symptoms
    unrelated = user_symptoms - (core | optional)

    raw_score = (
        len(matched_core) * CORE_MATCH_WEIGHT
        + len(matched_optional) * OPTIONAL_MATCH_WEIGHT
        - len(missing_core) * MISSING_CORE_PENALTY
        - len(unrelated) * UNRELATED_SYMPTOM_PENALTY
    )

    max_reward = _max_possible_reward(core, optional)
    if max_reward <= 0:
        return 0.0

    # Normalize: divide by the maximum possible reward.
    normalized = raw_score / max_reward

    # Clamp to [0, 1] for clean display and downstream use.
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


# ---------------------------------------------------------------------------
# Multi-disease diagnosis
# ---------------------------------------------------------------------------

def rule_based_diagnosis(
    symptoms: Iterable[str],
) -> List[Tuple[str, float]]:
    """
    Run the rule-based inference engine across all diseases in the
    knowledge base and return a ranked list of candidate diagnoses.

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user.

    Returns
    -------
    list of (str, float)
        A list of (disease_name, score) tuples, sorted from highest
        to lowest score.
    """
    diseases: Dict[str, dict] = load_diseases()
    results: List[Tuple[str, float]] = []

    for disease_name, info in diseases.items():
        score = score_disease(symptoms, info)
        results.append((disease_name, score))

    return sort_diagnoses(results)


# ---------------------------------------------------------------------------
# Explanation for transparency
# ---------------------------------------------------------------------------

def explain_match(
    symptoms: Iterable[str],
    disease_info: dict,
) -> Dict[str, List[str]]:
    """
    Produce a structured, human-readable explanation of why a disease
    received its score.

    This supports the assignment's transparency requirement and is used
    on the result page to show the user how Baymax reached its conclusion.

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
            "matched_core":     [...],
            "matched_optional": [...],
            "missing_core":     [...],
            "unrelated":        [...],
        }
    """
    user_symptoms = set(normalize_symptoms(symptoms))
    core, optional = _split_symptoms(disease_info)

    matched_core = sorted(user_symptoms & core)
    matched_optional = sorted(user_symptoms & optional)
    missing_core = sorted(core - user_symptoms)
    unrelated = sorted(user_symptoms - (core | optional))

    return {
        "matched_core": matched_core,
        "matched_optional": matched_optional,
        "missing_core": missing_core,
        "unrelated": unrelated,
    }