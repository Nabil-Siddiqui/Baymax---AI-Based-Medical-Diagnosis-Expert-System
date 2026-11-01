"""
Baymax - AI-Based Medical Diagnosis Expert System
==================================================

This package contains the core logic of the Baymax expert system:

- Knowledge base loading and validation
- Rule-based diagnosis (knowledge-based reasoning)
- Bayesian probabilistic reasoning
- Best-First Search ranking
- Machine learning model training, loading, and prediction
- Performance evaluation utilities
- A unified diagnosis pipeline that orchestrates all components

The Flask web application (`app.py`) and the dataset generator
(`scripts/generate_dataset.py`) interact with this package through the
public API re-exported below.
"""

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__version__ = "1.0.0"
__project__ = "Baymax"
__description__ = "AI-Based Expert System for Medical Diagnosis"


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

from .knowledge_base import (
    load_diseases,
    load_symptoms,
    get_disease,
    get_disease_names,
    get_symptom_ids,
    get_symptom_labels,
    get_symptom_label_map,
    validate_knowledge_base,
)


# ---------------------------------------------------------------------------
# Diagnosis engines
# ---------------------------------------------------------------------------

from .rule_engine import (
    rule_based_diagnosis,
    score_disease,
    explain_match,
)

from .bayes_engine import (
    bayesian_diagnosis,
    compute_posterior,
    explain_bayesian_inference,
)

from .search_engine import (
    best_first_search,
    compute_heuristic_scores,
)


# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------

from .ml_model import (
    SUPPORTED_MODELS,
    train_model,
    train_all_models,
    load_trained_model,
    predict_disease,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

from .evaluation import (
    evaluate_model,
    evaluate_all_models,
    save_metrics_report,
    save_classification_report,
    save_confusion_matrix_plot,
    save_model_comparison_table,
    save_experiment_notes,
    run_full_evaluation,
)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

from .diagnosis_pipeline import (
    run_diagnosis,
    compute_final_ranking,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

from .utils import (
    normalize_symptoms,
    symptoms_to_feature_vector,
    feature_vector_to_symptoms,
    format_probability,
    sort_diagnoses,
    top_n_diagnoses,
    safe_label,
    labels_for_symptoms,
)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Metadata
    "__version__",
    "__project__",
    "__description__",

    # Knowledge base
    "load_diseases",
    "load_symptoms",
    "get_disease",
    "get_disease_names",
    "get_symptom_ids",
    "get_symptom_labels",
    "get_symptom_label_map",
    "validate_knowledge_base",

    # Rule engine
    "rule_based_diagnosis",
    "score_disease",
    "explain_match",

    # Bayesian engine
    "bayesian_diagnosis",
    "compute_posterior",
    "explain_bayesian_inference",

    # Search engine
    "best_first_search",
    "compute_heuristic_scores",

    # Machine learning
    "SUPPORTED_MODELS",
    "train_model",
    "train_all_models",
    "load_trained_model",
    "predict_disease",

    # Evaluation
    "evaluate_model",
    "evaluate_all_models",
    "save_metrics_report",
    "save_classification_report",
    "save_confusion_matrix_plot",
    "save_model_comparison_table",
    "save_experiment_notes",
    "run_full_evaluation",

    # Diagnosis pipeline
    "run_diagnosis",
    "compute_final_ranking",

    # Utilities
    "normalize_symptoms",
    "symptoms_to_feature_vector",
    "feature_vector_to_symptoms",
    "format_probability",
    "sort_diagnoses",
    "top_n_diagnoses",
    "safe_label",
    "labels_for_symptoms",
]