"""
Baymax - Diagnosis Pipeline
===========================

This module is the orchestration layer of the Baymax expert system.
It exposes a single high-level function, `run_diagnosis`, which the
Flask web app calls to produce a complete diagnostic response from a
list of user-reported symptoms.

The pipeline integrates four complementary reasoning components:

    1. Rule-based inference engine        (src/rule_engine.py)
    2. Probabilistic Bayesian engine      (src/bayes_engine.py)
    3. Best-First Search ranking          (src/search_engine.py)
    4. Supervised machine learning model  (src/ml_model.py)

It returns a structured dictionary suitable for direct rendering in
the result page. A final unified ranking is also produced as a
weighted ensemble across the available components, providing a single
clear top diagnosis along with the supporting evidence.

This module satisfies the assignment requirement to combine
knowledge-based reasoning, probabilistic reasoning, search, and
machine learning into one coherent diagnostic system.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .bayes_engine import bayesian_diagnosis, explain_bayesian_inference
from .knowledge_base import get_disease, load_diseases
from .ml_model import SUPPORTED_MODELS, load_trained_model, predict_disease
from .rule_engine import explain_match, rule_based_diagnosis, score_disease
from .search_engine import best_first_search
from .utils import (
    format_probability,
    labels_for_symptoms,
    normalize_symptoms,
    sort_diagnoses,
    top_n_diagnoses,
)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_ML_MODEL: str = "decision_tree"
DEFAULT_SEARCH_HEURISTIC: str = "bayes"
DEFAULT_HYBRID_WEIGHT: float = 0.5
DEFAULT_TOP_K: int = 3

# Final-ranking ensemble weights. Must be non-negative.
# They will be renormalized at runtime if a component is unavailable.
DEFAULT_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "bayes": 0.4,
    "rule": 0.2,
    "ml": 0.4,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_load_ml_bundle(model_type: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to load a trained ML model bundle. Return None if the
    model has not been trained yet (e.g. during early development).
    """
    try:
        return load_trained_model(model_type)
    except FileNotFoundError:
        return None


def _ml_probabilities(
    bundle: Dict[str, Any],
    symptoms: Iterable[str],
    top_k: int,
) -> Dict[str, Any]:
    """
    Run the ML predictor and return its full result dictionary.
    """
    return predict_disease(
        symptoms=symptoms,
        model_type=bundle.get("model_type", DEFAULT_ML_MODEL),
        top_k=top_k,
        bundle=bundle,
    )


def _normalize_weights(
    weights: Dict[str, float],
    available: Dict[str, bool],
) -> Dict[str, float]:
    """
    Filter the requested ensemble weights to those whose components
    are available, then renormalize so they sum to 1.

    If no components are available, return zeroed weights.
    """
    filtered = {
        name: max(0.0, float(w))
        for name, w in weights.items()
        if available.get(name, False)
    }
    total = sum(filtered.values())
    if total <= 0:
        return {name: 0.0 for name in weights}
    return {name: w / total for name, w in filtered.items()}


def _scores_to_dict(scored: List[Tuple[str, float]]) -> Dict[str, float]:
    """Convert a list of (disease, score) tuples to a dict."""
    return {name: float(score) for name, score in scored}


def _ml_scores_to_dict(ml_result: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Convert the ML predictor's `ranked` list into a {disease: prob} dict.
    Diseases not in the ranked list are treated as 0.
    """
    if not ml_result:
        return {}
    ranked = ml_result.get("ranked", [])
    return {str(name): float(prob) for name, prob in ranked}


def compute_final_ranking(
    rule_scores: Dict[str, float],
    bayes_scores: Dict[str, float],
    ml_scores: Dict[str, float],
    weights: Dict[str, float],
) -> List[Tuple[str, float]]:
    """
    Combine component scores into a single ranked list of diseases.

    The score for each disease is:

        final(D) =
              w_bayes * bayes_scores[D]
            + w_rule  * rule_scores[D]
            + w_ml    * ml_scores[D]

    Components missing a score for a disease contribute 0 for that
    disease.

    Parameters
    ----------
    rule_scores  : dict, disease -> rule score in [0, 1]
    bayes_scores : dict, disease -> posterior probability in [0, 1]
    ml_scores    : dict, disease -> ML class probability in [0, 1]
    weights      : dict, component_name -> weight (already normalized)

    Returns
    -------
    list of (str, float)
        Diseases ranked by final score, descending.
    """
    all_diseases = set(rule_scores) | set(bayes_scores) | set(ml_scores)
    final: List[Tuple[str, float]] = []

    w_bayes = float(weights.get("bayes", 0.0))
    w_rule = float(weights.get("rule", 0.0))
    w_ml = float(weights.get("ml", 0.0))

    for disease in all_diseases:
        score = (
            w_bayes * bayes_scores.get(disease, 0.0)
            + w_rule * rule_scores.get(disease, 0.0)
            + w_ml * ml_scores.get(disease, 0.0)
        )
        final.append((disease, float(score)))

    return sort_diagnoses(final)


def _format_top_diagnosis(
    final_ranking: List[Tuple[str, float]],
    user_symptoms: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Build the user-facing top diagnosis block.

    Includes the disease name, final score, description, recommendation,
    and the symptoms matched/missing relative to the disease's profile.
    """
    if not final_ranking:
        return None

    top_disease, top_score = final_ranking[0]

    try:
        disease_info = get_disease(top_disease)
    except KeyError:
        disease_info = {}

    explanation = explain_match(user_symptoms, disease_info) if disease_info else {
        "matched_core": [],
        "matched_optional": [],
        "missing_core": [],
        "unrelated": [],
    }

    matched_ids = list(explanation["matched_core"]) + list(
        explanation["matched_optional"]
    )
    missing_ids = list(explanation["missing_core"])

    return {
        "disease": top_disease,
        "final_score": float(top_score),
        "final_score_pct": format_probability(top_score),
        "description": disease_info.get("description", ""),
        "recommendation": disease_info.get("recommendation", ""),
        "matched_symptoms": matched_ids,
        "matched_symptom_labels": labels_for_symptoms(matched_ids),
        "missing_symptoms": missing_ids,
        "missing_symptom_labels": labels_for_symptoms(missing_ids),
        "unrelated_symptoms": list(explanation["unrelated"]),
        "unrelated_symptom_labels": labels_for_symptoms(
            list(explanation["unrelated"])
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnosis(
    symptoms: Iterable[str],
    *,
    ml_model: str = DEFAULT_ML_MODEL,
    search_heuristic: str = DEFAULT_SEARCH_HEURISTIC,
    hybrid_weight: float = DEFAULT_HYBRID_WEIGHT,
    top_k: int = DEFAULT_TOP_K,
    ensemble_weights: Optional[Dict[str, float]] = None,
    include_explanations: bool = True,
    require_ml: bool = False,
) -> Dict[str, Any]:
    """
    Run the full Baymax diagnostic pipeline on a set of user symptoms.

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user.
    ml_model : str
        Which trained ML model to use. Must be in SUPPORTED_MODELS.
    search_heuristic : str
        Heuristic for the search engine: "bayes", "rule", or "hybrid".
    hybrid_weight : float
        Weight on the Bayesian component when search_heuristic="hybrid".
    top_k : int
        Number of top candidates to surface in the final ranking and
        from the ML model.
    ensemble_weights : dict, optional
        Override the default final-ranking ensemble weights, e.g.
        {"bayes": 0.5, "rule": 0.2, "ml": 0.3}. Renormalized internally.
    include_explanations : bool
        If True, include explain_match and explain_bayesian_inference
        outputs in the response (used by the result page for
        transparency).
    require_ml : bool
        If True, raise FileNotFoundError when the ML model is missing.
        If False (default), gracefully skip the ML component.

    Returns
    -------
    dict
        A structured diagnostic response. See module docstring for keys.
    """
    if ml_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported ml_model '{ml_model}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    # 1. Normalize and validate input.
    user_symptoms = normalize_symptoms(symptoms)

    warnings: List[str] = []
    components_used: List[str] = []

    # 2. Knowledge-based and probabilistic reasoning are always available.
    rule_results = rule_based_diagnosis(user_symptoms)
    components_used.append("rule")

    bayes_results = bayesian_diagnosis(user_symptoms)
    components_used.append("bayes")

    # 3. Search ranking.
    search_results = best_first_search(
        symptoms=user_symptoms,
        heuristic=search_heuristic,
        weight=hybrid_weight,
    )
    components_used.append("search")

    # 4. Optional ML component.
    ml_bundle = _try_load_ml_bundle(ml_model)
    ml_results: Optional[Dict[str, Any]] = None

    if ml_bundle is not None:
        ml_results = _ml_probabilities(ml_bundle, user_symptoms, top_k=len(load_diseases()))
        components_used.append("ml")
    else:
        if require_ml:
            raise FileNotFoundError(
                f"ML model '{ml_model}' is not trained yet. "
                "Train it first via src.ml_model.train_model()."
            )
        warnings.append(
            f"ML model '{ml_model}' is not available. The final ranking "
            "uses only the rule-based and Bayesian components."
        )

    # 5. Build final ranking via weighted ensemble.
    rule_dict = _scores_to_dict(rule_results)
    bayes_dict = _scores_to_dict(bayes_results)
    ml_dict = _ml_scores_to_dict(ml_results)

    requested_weights = (
        dict(ensemble_weights) if ensemble_weights else dict(DEFAULT_ENSEMBLE_WEIGHTS)
    )

    available = {
        "rule": True,
        "bayes": True,
        "ml": ml_results is not None,
    }
    final_weights = _normalize_weights(requested_weights, available)

    final_ranking = compute_final_ranking(
        rule_scores=rule_dict,
        bayes_scores=bayes_dict,
        ml_scores=ml_dict,
        weights=final_weights,
    )

    final_top_k = top_n_diagnoses(final_ranking, n=top_k)

    # 6. Top diagnosis block for the UI.
    top_diagnosis = _format_top_diagnosis(final_ranking, user_symptoms)

    # 7. Optional per-disease explanations for transparency.
    explanations: Dict[str, Any] = {}
    if include_explanations and top_diagnosis is not None:
        try:
            top_info = get_disease(top_diagnosis["disease"])
            explanations["rule_explanation"] = explain_match(
                user_symptoms, top_info
            )
            explanations["bayes_explanation"] = explain_bayesian_inference(
                user_symptoms, top_info
            )
        except KeyError:
            # Top disease was not found in the knowledge base; skip
            # explanations rather than failing the whole request.
            pass

    return {
        "input_symptoms": user_symptoms,
        "input_symptom_labels": labels_for_symptoms(user_symptoms),

        "rule_results": rule_results,
        "bayes_results": bayes_results,
        "search_results": search_results,
        "ml_results": ml_results,

        "final_ranking": final_ranking,
        "final_top_k": final_top_k,
        "top_diagnosis": top_diagnosis,

        "components_used": components_used,
        "weights": final_weights,
        "warnings": warnings,

        "explanations": explanations if include_explanations else {},
    }