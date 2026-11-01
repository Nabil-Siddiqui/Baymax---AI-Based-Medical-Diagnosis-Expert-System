"""
Baymax - Search Engine (Best-First Search)
==========================================

This module formulates the medical diagnosis problem as a search
problem over the space of candidate diseases and applies the
Best-First Search algorithm to identify the most probable diagnoses.

Search formulation
------------------
- State space   : the set of all diseases in the knowledge base.
- Heuristic h(d): a numeric score representing how likely disease d
                  is, given the user's reported symptoms.
- Goal          : return diseases ranked by heuristic score, with the
                  most promising candidates expanded first.

Supported heuristics
--------------------
- "bayes"  : posterior probability from the Bayesian engine.
- "rule"   : normalized rule-based match score.
- "hybrid" : weighted combination of the two:
                 score = w * bayes + (1 - w) * rule
             where w is configurable (default 0.5).

Why Best-First Search (and not A*)
----------------------------------
A* requires a meaningful path-cost function g(n) between states.
The diagnostic state space lacks such semantics: there are no
"paths" between diseases. Best-First Search is therefore the more
principled choice, as it ranks candidate states purely by a
heuristic h(n).

This module satisfies the assignment requirement to:
"Implement search algorithms to find the most probable diagnoses
from a set of candidate conditions."
"""

import heapq
from typing import Dict, Iterable, List, Tuple

from .bayes_engine import bayesian_diagnosis
from .knowledge_base import load_diseases
from .rule_engine import rule_based_diagnosis
from .utils import sort_diagnoses


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALID_HEURISTICS = ("bayes", "rule", "hybrid")
DEFAULT_HEURISTIC = "bayes"
DEFAULT_HYBRID_WEIGHT = 0.5  # Weight on the Bayesian component in "hybrid".


# ---------------------------------------------------------------------------
# Heuristic computation
# ---------------------------------------------------------------------------

def _as_dict(scored_list: List[Tuple[str, float]]) -> Dict[str, float]:
    """Convert a list of (disease, score) tuples to a dict."""
    return {name: score for name, score in scored_list}


def compute_heuristic_scores(
    symptoms: Iterable[str],
    heuristic: str = DEFAULT_HEURISTIC,
    weight: float = DEFAULT_HYBRID_WEIGHT,
) -> Dict[str, float]:
    """
    Compute the heuristic score for every disease in the knowledge base.

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user.
    heuristic : str
        One of "bayes", "rule", or "hybrid".
    weight : float
        Weight on the Bayesian component when using "hybrid".
        Must be in [0, 1]. Ignored otherwise.

    Returns
    -------
    dict
        Mapping disease_name -> heuristic score in [0, 1].
    """
    if heuristic not in VALID_HEURISTICS:
        raise ValueError(
            f"Unknown heuristic '{heuristic}'. "
            f"Valid options: {VALID_HEURISTICS}"
        )

    if heuristic == "bayes":
        return _as_dict(bayesian_diagnosis(symptoms))

    if heuristic == "rule":
        return _as_dict(rule_based_diagnosis(symptoms))

    # heuristic == "hybrid"
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Hybrid weight must be in [0, 1].")

    bayes_scores = _as_dict(bayesian_diagnosis(symptoms))
    rule_scores = _as_dict(rule_based_diagnosis(symptoms))

    diseases = load_diseases().keys()
    combined: Dict[str, float] = {}
    for name in diseases:
        b = float(bayes_scores.get(name, 0.0))
        r = float(rule_scores.get(name, 0.0))
        combined[name] = weight * b + (1.0 - weight) * r

    return combined


# ---------------------------------------------------------------------------
# Best-First Search
# ---------------------------------------------------------------------------

def best_first_search(
    symptoms: Iterable[str],
    heuristic: str = DEFAULT_HEURISTIC,
    weight: float = DEFAULT_HYBRID_WEIGHT,
) -> Dict[str, object]:
    """
    Run Best-First Search over the space of candidate diseases.

    The search uses a priority queue (implemented as a max-heap on the
    heuristic score) to always expand the most promising disease first.
    The order of expansion and the number of nodes expanded are
    recorded for evaluation purposes.

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user.
    heuristic : str
        One of "bayes", "rule", or "hybrid".
    weight : float
        Weight for the Bayesian component when using "hybrid".

    Returns
    -------
    dict
        {
            "heuristic":      str,
            "weight":         float | None,
            "ranked":         [(disease_name, score), ...]   # sorted desc
            "expansion_order":[disease_name, ...]            # order popped
            "nodes_expanded": int,
            "scores":         {disease_name: score, ...}     # raw scores
        }
    """
    scores = compute_heuristic_scores(symptoms, heuristic=heuristic, weight=weight)

    if not scores:
        return {
            "heuristic": heuristic,
            "weight": weight if heuristic == "hybrid" else None,
            "ranked": [],
            "expansion_order": [],
            "nodes_expanded": 0,
            "scores": {},
        }

    # Python's heapq is a min-heap, so we push negative scores to
    # simulate a max-heap. We also push an index as a tie-breaker to
    # avoid comparing strings or dicts when scores are equal.
    priority_queue: List[Tuple[float, int, str]] = []
    for index, (name, score) in enumerate(scores.items()):
        heapq.heappush(priority_queue, (-float(score), index, name))

    expansion_order: List[str] = []
    ranked: List[Tuple[str, float]] = []

    while priority_queue:
        neg_score, _, name = heapq.heappop(priority_queue)
        score = -neg_score
        expansion_order.append(name)
        ranked.append((name, score))

    ranked = sort_diagnoses(ranked)

    return {
        "heuristic": heuristic,
        "weight": weight if heuristic == "hybrid" else None,
        "ranked": ranked,
        "expansion_order": expansion_order,
        "nodes_expanded": len(expansion_order),
        "scores": scores,
    }