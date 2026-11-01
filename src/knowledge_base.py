"""
Knowledge Base Loader
=====================

This module is responsible for loading and providing access to the
Baymax expert system's knowledge base, which consists of:

- diseases.json     -> Diseases, their core/optional symptoms, priors,
                       descriptions, and recommendations.
- symptom_list.json -> The master list of symptoms with internal IDs
                       and human-friendly display labels.

The functions in this module are used across the entire project,
including the rule engine, Bayesian engine, machine learning module,
synthetic data generator, and the Flask web app.

Path handling is done relative to this file, so the loader works
correctly regardless of the current working directory.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

# This file is located at: Baymax/src/knowledge_base.py
# We want to reach:        Baymax/knowledge_base/
#
# Path(__file__) -> .../Baymax/src/knowledge_base.py
# .parent        -> .../Baymax/src
# .parent.parent -> .../Baymax
#
# So the project root is two levels up from this file.

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / "knowledge_base"

DISEASES_FILE: Path = KNOWLEDGE_BASE_DIR / "diseases.json"
SYMPTOMS_FILE: Path = KNOWLEDGE_BASE_DIR / "symptom_list.json"


# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------
#
# The knowledge base does not change while the app is running, so we
# load each file only once and cache the result for fast access.

_diseases_cache: Optional[Dict[str, dict]] = None
_symptoms_cache: Optional[List[dict]] = None


# ---------------------------------------------------------------------------
# Core loading functions
# ---------------------------------------------------------------------------

def load_diseases(force_reload: bool = False) -> Dict[str, dict]:
    """
    Load the diseases knowledge base from diseases.json.

    Parameters
    ----------
    force_reload : bool
        If True, reload from disk even if a cached version exists.

    Returns
    -------
    dict
        A dictionary mapping disease name -> disease data.
        Each disease entry contains:
            - name
            - core_symptoms
            - optional_symptoms
            - prior
            - description
            - recommendation
    """
    global _diseases_cache

    if _diseases_cache is not None and not force_reload:
        return _diseases_cache

    if not DISEASES_FILE.exists():
        raise FileNotFoundError(
            f"Diseases knowledge base file not found at: {DISEASES_FILE}"
        )

    with DISEASES_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Invalid format in diseases.json: expected a top-level JSON object."
        )

    _diseases_cache = data
    return _diseases_cache


def load_symptoms(force_reload: bool = False) -> List[dict]:
    """
    Load the symptom list from symptom_list.json.

    Parameters
    ----------
    force_reload : bool
        If True, reload from disk even if a cached version exists.

    Returns
    -------
    list of dict
        A list of symptom entries. Each entry has:
            - id    : internal identifier used in code and dataset columns
            - label : human-friendly display name for the UI
    """
    global _symptoms_cache

    if _symptoms_cache is not None and not force_reload:
        return _symptoms_cache

    if not SYMPTOMS_FILE.exists():
        raise FileNotFoundError(
            f"Symptom list file not found at: {SYMPTOMS_FILE}"
        )

    with SYMPTOMS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Invalid format in symptom_list.json: expected a top-level JSON array."
        )

    _symptoms_cache = data
    return _symptoms_cache


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_disease(name: str) -> dict:
    """
    Get a single disease entry by name.

    Parameters
    ----------
    name : str
        The disease name as it appears in diseases.json (e.g., "Flu").

    Returns
    -------
    dict
        The disease data.

    Raises
    ------
    KeyError
        If the disease is not found in the knowledge base.
    """
    diseases = load_diseases()
    if name not in diseases:
        raise KeyError(f"Disease '{name}' not found in knowledge base.")
    return diseases[name]


def get_disease_names() -> List[str]:
    """
    Return the list of all disease names in the knowledge base.

    Returns
    -------
    list of str
    """
    return list(load_diseases().keys())


def get_symptom_ids() -> List[str]:
    """
    Return the list of all symptom IDs (internal names) in order.

    These IDs are the canonical names used:
        - in dataset column headers
        - in JSON keys
        - in Python code

    Returns
    -------
    list of str
    """
    return [s["id"] for s in load_symptoms()]


def get_symptom_labels() -> List[str]:
    """
    Return the list of all symptom display labels in order.

    These labels are shown to the user in the website UI.

    Returns
    -------
    list of str
    """
    return [s["label"] for s in load_symptoms()]


def get_symptom_label_map() -> Dict[str, str]:
    """
    Return a mapping of symptom_id -> symptom_label.

    Useful when displaying user-selected symptoms back to the user
    in a friendly way.

    Returns
    -------
    dict
        Example: {"loss_of_smell": "Loss of Smell", ...}
    """
    return {s["id"]: s["label"] for s in load_symptoms()}


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_knowledge_base() -> None:
    """
    Validate that all symptoms used in diseases.json are present
    in symptom_list.json.

    This helps catch typos or mismatches early during development.

    Raises
    ------
    ValueError
        If any symptom referenced by a disease is not in the symptom list.
    """
    diseases = load_diseases()
    valid_symptom_ids = set(get_symptom_ids())

    invalid_references = {}

    for disease_name, info in diseases.items():
        used_symptoms = set(info.get("core_symptoms", [])) | set(
            info.get("optional_symptoms", [])
        )
        unknown = used_symptoms - valid_symptom_ids
        if unknown:
            invalid_references[disease_name] = sorted(unknown)

    if invalid_references:
        raise ValueError(
            "Knowledge base validation failed. The following diseases "
            "reference unknown symptom IDs:\n"
            + "\n".join(
                f"  - {disease}: {symptoms}"
                for disease, symptoms in invalid_references.items()
            )
        )