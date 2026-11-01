"""
Baymax - Machine Learning Module
================================

This module implements the supervised learning layer of the Baymax
expert system. It trains, saves, loads, and applies machine learning
models that predict the most likely disease from a binary symptom
feature vector.

Supported models
----------------
- "decision_tree" : a single, interpretable decision tree classifier.
- "random_forest" : an ensemble of decision trees for higher accuracy.
- "naive_bayes"   : a multinomial Naive Bayes classifier, conceptually
                    aligned with the hand-built Bayesian engine.

Pipeline
--------
1. Load the synthetic dataset from data/medical_dataset.csv.
2. Align features to the master symptom list from the knowledge base
   so the model's input is consistent with the Flask app and the
   Bayesian engine.
3. Encode disease labels using a LabelEncoder.
4. Split the data, train the model, and evaluate it.
5. Persist the model, the label encoder, and the feature column order
   to disk under models/ for use by the Flask app.

This module satisfies the assignment requirements for using machine
learning to improve diagnostic accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from .knowledge_base import get_symptom_ids
from .utils import symptoms_to_feature_vector


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"

DATASET_FILE: Path = DATA_DIR / "medical_dataset.csv"
TARGET_COLUMN: str = "disease"

# Each saved artifact ends in .pkl. We store one bundle per model_type,
# containing the model, the label encoder, and the feature column order.
SUPPORTED_MODELS: Tuple[str, ...] = (
    "decision_tree",
    "random_forest",
    "naive_bayes",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_artifact_path(model_type: str) -> Path:
    """Return the path where the model bundle is saved/loaded."""
    return MODELS_DIR / f"{model_type}_bundle.pkl"


def _build_estimator(model_type: str, random_state: int = 42):
    """Construct a fresh estimator for the given model type."""
    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        )
    if model_type == "naive_bayes":
        # MultinomialNB is appropriate for binary/count features.
        return MultinomialNB()
    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        f"Supported: {SUPPORTED_MODELS}"
    )


def _ensure_dataset_exists() -> None:
    """Raise a helpful error if the dataset file is missing."""
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_FILE}\n"
            "Generate it first by running:\n"
            "    python scripts/generate_dataset.py"
        )


def _load_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load the dataset and align the feature columns to the master
    symptom list from the knowledge base.

    Returns
    -------
    (X, y, feature_columns)
        X               : DataFrame of binary features
        y               : Series of disease labels (strings)
        feature_columns : list of feature column names in canonical order
    """
    _ensure_dataset_exists()

    df = pd.read_csv(DATASET_FILE)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset must contain a '{TARGET_COLUMN}' column. "
            f"Found columns: {list(df.columns)}"
        )

    symptom_ids = get_symptom_ids()

    # Ensure every symptom from the knowledge base exists as a column.
    missing = [s for s in symptom_ids if s not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing the following expected symptom columns: "
            f"{missing}\nRegenerate the dataset to fix this."
        )

    feature_columns = list(symptom_ids)
    X = df[feature_columns].astype(int)
    y = df[TARGET_COLUMN].astype(str)
    return X, y, feature_columns


def _ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model_type: str = "decision_tree",
    test_size: float = 0.2,
    random_state: int = 42,
    save: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a single ML model on the synthetic dataset.

    Parameters
    ----------
    model_type : str
        One of "decision_tree", "random_forest", "naive_bayes".
    test_size : float
        Fraction of the data used for testing.
    random_state : int
        Random seed for reproducibility.
    save : bool
        If True, persist the model bundle to disk.

    Returns
    -------
    (model, metrics)
        model   : the fitted estimator
        metrics : dict of evaluation metrics on the test set
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    X, y, feature_columns = _load_dataset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    model = _build_estimator(model_type, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "model_type": model_type,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": list(label_encoder.classes_),
        "feature_columns": list(feature_columns),
    }

    if save:
        _ensure_models_dir()
        bundle = {
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
            "model_type": model_type,
        }
        joblib.dump(bundle, _model_artifact_path(model_type))

    return model, metrics


def train_all_models(
    test_size: float = 0.2,
    random_state: int = 42,
    save: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all supported models and return their metrics for comparison.

    Returns
    -------
    dict
        Mapping model_type -> metrics dict.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for model_type in SUPPORTED_MODELS:
        _, metrics = train_model(
            model_type=model_type,
            test_size=test_size,
            random_state=random_state,
            save=save,
        )
        results[model_type] = metrics
    return results


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trained_model(model_type: str = "decision_tree") -> Dict[str, Any]:
    """
    Load a previously trained model bundle from disk.

    Parameters
    ----------
    model_type : str

    Returns
    -------
    dict
        {
            "model":           fitted estimator,
            "label_encoder":   fitted LabelEncoder,
            "feature_columns": list of feature names in canonical order,
            "model_type":      str
        }
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    artifact_path = _model_artifact_path(model_type)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No trained model found at: {artifact_path}\n"
            f"Train it first, e.g.:\n"
            f"    from src.ml_model import train_model\n"
            f"    train_model('{model_type}')"
        )

    return joblib.load(artifact_path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_disease(
    symptoms: Iterable[str],
    model_type: str = "decision_tree",
    top_k: int = 3,
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Predict the most likely disease(s) from a list of user symptoms.

    Parameters
    ----------
    symptoms : iterable of str
        Symptom IDs reported by the user.
    model_type : str
        Which trained model to use. Ignored if `bundle` is provided.
    top_k : int
        Number of top predictions to return with their probabilities.
    bundle : dict, optional
        A preloaded model bundle (as returned by `load_trained_model`).
        If provided, this is used directly; otherwise the bundle is
        loaded from disk. This is useful for the Flask app, which can
        load the bundle once at startup.

    Returns
    -------
    dict
        {
            "model_type":      str,
            "predicted":       str,             # top-1 disease
            "confidence":      float,           # in [0, 1]
            "ranked":          [(disease, prob), ...]  # length = top_k
            "feature_vector":  list of int,
        }
    """
    if bundle is None:
        bundle = load_trained_model(model_type)

    model = bundle["model"]
    label_encoder: LabelEncoder = bundle["label_encoder"]
    feature_columns: List[str] = bundle["feature_columns"]
    used_model_type: str = bundle.get("model_type", model_type)

    # Build the feature vector aligned to the canonical symptom order,
    # then reorder/select to match the model's training feature order.
    canonical_vector = symptoms_to_feature_vector(symptoms)
    canonical_ids = get_symptom_ids()
    canonical_map = dict(zip(canonical_ids, canonical_vector))

    # The training feature order should match canonical_ids, but we
    # reindex defensively to avoid silent misalignment if the dataset
    # was regenerated with a different column order.
    feature_row = [int(canonical_map.get(col, 0)) for col in feature_columns]
    X = np.array([feature_row], dtype=int)

    # Get class probabilities if supported.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        # Fallback: produce a one-hot pseudo-probability for the predicted class.
        pred_index = int(model.predict(X)[0])
        proba = np.zeros(len(label_encoder.classes_), dtype=float)
        proba[pred_index] = 1.0

    classes = list(label_encoder.classes_)
    ranked_pairs = sorted(
        zip(classes, proba.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    if top_k <= 0:
        top_ranked = []
    else:
        top_ranked = ranked_pairs[:top_k]

    predicted_disease = ranked_pairs[0][0]
    confidence = float(ranked_pairs[0][1])

    return {
        "model_type": used_model_type,
        "predicted": predicted_disease,
        "confidence": confidence,
        "ranked": top_ranked,
        "feature_vector": feature_row,
    }