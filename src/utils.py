"""
Baymax - Utility Functions
==========================

This module provides shared helper functions used across the Baymax
project, including:

- Symptom input normalization and validation
- Conversion between symptom lists and binary feature vectors
- Probability formatting for the UI
- Sorting and ranking helpers for diagnosis results

These utilities ensure consistent behavior across the rule engine,
Bayesian engine, search engine, machine learning module, and the
Flask web app.
"""

from typing import Dict, Iterable, List, Sequence, Tuple

from .knowledge_base import (
    get_symptom_ids,
    get_symptom_label_map,
)


# ---------------------------------------------------------------------------
# Symptom input handling
# ---------------------------------------------------------------------------

def normalize_symptoms(symptoms: Iterable[str]) -> List[str]:
    """
    Clean and validate a list of symptoms submitted by the user.

    Steps performed:
        1. Strip whitespace
        2. Convert to lowercase
        3. Remove empty strings
        4. Remove duplicates while preserving order
        5. Drop any symptom IDs that are not in the knowledge base

    Parameters
    ----------
    symptoms : iterable of str
        Raw symptom IDs from the frontend or another source.

    Returns
    -------
    list of str
        A clean, validated, deduplicated list of valid symptom IDs.
    """
    if symptoms is None:
        return []

    valid_ids = set(get_symptom_ids())
    seen = set()
    cleaned: List[str] = []

    for raw in symptoms:
        if not isinstance(raw, str):
            continue
        symptom = raw.strip().lower()
        if not symptom:
            continue
        if symptom in seen:
            continue
        if symptom not in valid_ids:
            # Silently skip unknown symptoms to keep the system robust.
            # If you prefer strict mode, raise ValueError here instead.
            continue
        seen.add(symptom)
        cleaned.append(symptom)

    return cleaned


# ---------------------------------------------------------------------------
# Feature vector conversions
# ---------------------------------------------------------------------------

def symptoms_to_feature_vector(symptoms: Iterable[str]) -> List[int]:
    """
    Convert a list of symptom IDs into a fixed-length binary feature vector.

    The vector is aligned to the order returned by `get_symptom_ids()`,
    which is the same order used by:
        - the synthetic dataset columns
        - the trained ML model
        - the Bayesian engine

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user (will be normalized internally).

    Returns
    -------
    list of int
        A list of 0s and 1s with the same length as the master symptom list.
    """
    normalized = set(normalize_symptoms(symptoms))
    all_ids = get_symptom_ids()
    return [1 if sid in normalized else 0 for sid in all_ids]


def feature_vector_to_symptoms(vector: Sequence[int]) -> List[str]:
    """
    Convert a binary feature vector back into a list of symptom IDs.

    Useful for debugging, logging, and displaying ML model inputs.

    Parameters
    ----------
    vector : sequence of int
        A 0/1 vector aligned to the master symptom list.

    Returns
    -------
    list of str
        Symptom IDs corresponding to positions where the vector is 1.

    Raises
    ------
    ValueError
        If the vector length does not match the master symptom list length.
    """
    all_ids = get_symptom_ids()
    if len(vector) != len(all_ids):
        raise ValueError(
            f"Feature vector length {len(vector)} does not match "
            f"symptom list length {len(all_ids)}."
        )
    return [sid for sid, value in zip(all_ids, vector) if value == 1]


# ---------------------------------------------------------------------------
# Probability formatting
# ---------------------------------------------------------------------------

def format_probability(value: float, decimals: int = 1) -> str:
    """
    Format a probability value as a percentage string.

    Examples
    --------
    >>> format_probability(0.8423)
    '84.2%'

    >>> format_probability(0.05, decimals=0)
    '5%'

    Parameters
    ----------
    value : float
        A probability between 0 and 1.
    decimals : int
        Number of decimal places to display.

    Returns
    -------
    str
        A formatted percentage string.
    """
    if value is None:
        return "0%"

    try:
        percentage = float(value) * 100.0
    except (TypeError, ValueError):
        return "0%"

    # Clamp to a sensible range for display.
    if percentage < 0:
        percentage = 0.0
    if percentage > 100:
        percentage = 100.0

    return f"{percentage:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Diagnosis ranking helpers
# ---------------------------------------------------------------------------

def sort_diagnoses(
    diagnoses: Iterable[Tuple[str, float]],
    descending: bool = True,
) -> List[Tuple[str, float]]:
    """
    Sort a list of (disease, score) tuples by score.

    Parameters
    ----------
    diagnoses : iterable of (str, float)
        Disease names paired with a numeric score (probability or match score).
    descending : bool
        If True (default), sort highest score first.

    Returns
    -------
    list of (str, float)
        Sorted diagnosis list.
    """
    return sorted(diagnoses, key=lambda item: item[1], reverse=descending)


def top_n_diagnoses(
    diagnoses: Iterable[Tuple[str, float]],
    n: int = 3,
) -> List[Tuple[str, float]]:
    """
    Return the top N most likely diagnoses.

    Parameters
    ----------
    diagnoses : iterable of (str, float)
        Disease names paired with a numeric score.
    n : int
        Number of top diagnoses to return.

    Returns
    -------
    list of (str, float)
        The top N diagnoses, sorted by score descending.
    """
    if n <= 0:
        return []
    return sort_diagnoses(diagnoses)[:n]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def safe_label(symptom_id: str) -> str:
    """
    Return the user-friendly label for a given symptom ID.

    If the symptom ID is not found in the knowledge base, a best-effort
    fallback label is returned (the ID with underscores replaced and
    title-cased).

    Parameters
    ----------
    symptom_id : str
        The internal symptom identifier (e.g., "loss_of_smell").

    Returns
    -------
    str
        The human-readable label (e.g., "Loss of Smell").
    """
    if not isinstance(symptom_id, str) or not symptom_id:
        return ""

    label_map: Dict[str, str] = get_symptom_label_map()
    if symptom_id in label_map:
        return label_map[symptom_id]

    return symptom_id.replace("_", " ").title()


def labels_for_symptoms(symptom_ids: Iterable[str]) -> List[str]:
    """
    Convert a list of symptom IDs into their human-readable labels.

    Parameters
    ----------
    symptom_ids : iterable of str

    Returns
    -------
    list of str
    """
    return [safe_label(sid) for sid in symptom_ids]