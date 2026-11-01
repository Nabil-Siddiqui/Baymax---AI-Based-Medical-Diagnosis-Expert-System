"""
Baymax - Evaluation Module
==========================

This module evaluates the trained machine learning models from
src/ml_model.py on the synthetic Baymax dataset and produces a
comprehensive set of performance artifacts:

    results/metrics.txt              - human-readable metrics summary
    results/classification_report.txt- per-class precision/recall/F1
    results/confusion_matrix.png     - confusion matrix heatmap
    results/model_comparison.csv     - side-by-side model comparison
    results/experiment_notes.txt     - experimental setup description

Design notes
------------
- This module does NOT train models. Train them first with
  src/ml_model.py. If a model bundle is missing, evaluation will
  raise a clear error.

- The same random seed, test_size, and stratified split are used
  here as in training, so the held-out test set is consistent.

- All artifacts are written to the project's results/ directory
  so they can be included directly in the assignment report.

This module satisfies the assignment requirements for performance
evaluation, metric reporting, and experimental documentation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

# Use a non-interactive backend so we can save figures even when
# running headless (e.g. in a remote terminal or CI environment).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import seaborn as sns            # noqa: E402
from sklearn.metrics import (    # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from .ml_model import (  # noqa: E402
    DATASET_FILE,
    SUPPORTED_MODELS,
    _load_dataset,
    load_trained_model,
)


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"

DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42

# Generation parameters used in scripts/generate_dataset.py.
# Documented here so the experiment notes can reference them.
DATASET_PROBS = {
    "P_core": 0.85,
    "P_optional": 0.40,
    "P_unrelated": 0.05,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_test_set(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Reproduce the same stratified train/test split used during training
    and return the test set. We need both the encoded test labels and
    the canonical class label strings.
    """
    X, y, feature_columns = _load_dataset()

    # We re-import LabelEncoder here rather than depending on the one
    # baked into each model bundle, because we need a consistent mapping
    # across all models. We then translate predictions through each
    # model's own encoder during evaluation.
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    _, X_test, _, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    return X_test, y_test, list(encoder.classes_), feature_columns


def _predict_with_bundle(bundle: Dict[str, Any], X_test: np.ndarray) -> np.ndarray:
    """
    Predict labels using a saved model bundle, returning predictions
    in the same label space as the canonical encoder used in
    `_build_test_set`. We do this by predicting integer classes from
    the bundle's encoder and then mapping back through class names.
    """
    model = bundle["model"]
    bundle_encoder = bundle["label_encoder"]

    pred_indices = model.predict(X_test)
    pred_labels = bundle_encoder.inverse_transform(pred_indices)

    # Re-encode predicted label strings into the canonical encoding
    # produced by `_build_test_set`. This avoids any mismatch when a
    # bundle's class order differs from the canonical order.
    from sklearn.preprocessing import LabelEncoder
    canonical_encoder = LabelEncoder()
    canonical_encoder.classes_ = np.array(sorted(set(bundle_encoder.classes_)))
    return canonical_encoder.transform(pred_labels)


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_type: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Evaluate a single previously trained model on the held-out test set.

    Parameters
    ----------
    model_type : str
        One of "decision_tree", "random_forest", "naive_bayes".
    test_size : float
        Fraction of the data used for testing (must match training).
    random_state : int
        Random seed (must match training).

    Returns
    -------
    dict
        A dictionary of metrics and metadata.
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_FILE}\n"
            "Generate it first by running:\n"
            "    python scripts/generate_dataset.py"
        )

    bundle = load_trained_model(model_type)

    X_test, y_test_canonical, canonical_classes, feature_columns = _build_test_set(
        test_size=test_size,
        random_state=random_state,
    )

    y_pred_canonical = _predict_with_bundle(bundle, X_test)

    metrics: Dict[str, Any] = {
        "model_type": model_type,
        "n_test": int(len(y_test_canonical)),
        "accuracy": float(accuracy_score(y_test_canonical, y_pred_canonical)),
        "precision_weighted": float(
            precision_score(
                y_test_canonical, y_pred_canonical,
                average="weighted", zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_test_canonical, y_pred_canonical,
                average="weighted", zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test_canonical, y_pred_canonical,
                average="weighted", zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_test_canonical,
            y_pred_canonical,
            target_names=canonical_classes,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test_canonical, y_pred_canonical,
        ).tolist(),
        "labels": list(canonical_classes),
        "feature_columns": list(feature_columns),
        "test_size": test_size,
        "random_state": random_state,
    }

    return metrics


def evaluate_all_models(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all supported, previously trained models.

    Returns
    -------
    dict
        Mapping model_type -> metrics dict.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for model_type in SUPPORTED_MODELS:
        results[model_type] = evaluate_model(
            model_type=model_type,
            test_size=test_size,
            random_state=random_state,
        )
    return results


# ---------------------------------------------------------------------------
# Reporting / file outputs
# ---------------------------------------------------------------------------

def save_metrics_report(
    metrics: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save a human-readable metrics summary to a text file.

    Returns the path of the file written.
    """
    _ensure_results_dir()
    if output_path is None:
        output_path = RESULTS_DIR / "metrics.txt"

    lines = [
        "Baymax - Model Evaluation Metrics",
        "=" * 40,
        f"Model:               {metrics['model_type']}",
        f"Test set size:       {metrics['n_test']}",
        f"Test fraction:       {metrics['test_size']}",
        f"Random state:        {metrics['random_state']}",
        "",
        f"Accuracy:            {metrics['accuracy']:.4f}",
        f"Precision (weighted):{metrics['precision_weighted']:.4f}",
        f"Recall (weighted):   {metrics['recall_weighted']:.4f}",
        f"F1-score (weighted): {metrics['f1_weighted']:.4f}",
        "",
        "Classification report",
        "-" * 40,
        metrics["classification_report"].rstrip(),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_classification_report(
    metrics: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save the per-class classification report (precision, recall, F1)
    as a separate text file.
    """
    _ensure_results_dir()
    if output_path is None:
        output_path = RESULTS_DIR / "classification_report.txt"

    header = (
        f"Baymax - Classification Report ({metrics['model_type']})\n"
        f"{'=' * 60}\n"
    )
    output_path.write_text(header + metrics["classification_report"], encoding="utf-8")
    return output_path


def save_confusion_matrix_plot(
    metrics: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save a confusion matrix heatmap as a PNG image.

    Returns the path of the file written.
    """
    _ensure_results_dir()
    if output_path is None:
        output_path = RESULTS_DIR / "confusion_matrix.png"

    cm = np.array(metrics["confusion_matrix"], dtype=int)
    labels = metrics["labels"]
    title = f"Confusion Matrix - {metrics['model_type']}"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_model_comparison_table(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save a CSV comparing all evaluated models on the main metrics.

    Columns:
        model, accuracy, precision_weighted, recall_weighted, f1_weighted, n_test
    """
    _ensure_results_dir()
    if output_path is None:
        output_path = RESULTS_DIR / "model_comparison.csv"

    header = [
        "model",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "n_test",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model_type, metrics in all_metrics.items():
            writer.writerow([
                model_type,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision_weighted']:.4f}",
                f"{metrics['recall_weighted']:.4f}",
                f"{metrics['f1_weighted']:.4f}",
                metrics["n_test"],
            ])

    return output_path


def save_experiment_notes(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save a textual description of the experimental setup, including
    dataset details, split configuration, and per-model results.

    This is designed to be pasted into or summarized in the
    'Experiment Details' section of the assignment report.
    """
    _ensure_results_dir()
    if output_path is None:
        output_path = RESULTS_DIR / "experiment_notes.txt"

    if not all_metrics:
        output_path.write_text(
            "No models were evaluated. Run training and evaluation first.\n",
            encoding="utf-8",
        )
        return output_path

    # Pick any metrics dict to describe the shared experimental setup.
    sample = next(iter(all_metrics.values()))

    lines: List[str] = []
    lines.append("Baymax - Experiment Details")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Dataset")
    lines.append("-" * 40)
    lines.append(f"Source file:        {DATASET_FILE}")
    lines.append(f"Number of features: {len(sample['feature_columns'])}")
    lines.append(f"Feature columns:    {sample['feature_columns']}")
    lines.append(f"Disease labels:     {sample['labels']}")
    lines.append("")
    lines.append("Generation model probabilities (Naive Bayes assumption):")
    for k, v in DATASET_PROBS.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Train/Test Split")
    lines.append("-" * 40)
    lines.append(f"Test fraction:      {sample['test_size']}")
    lines.append(f"Random state:       {sample['random_state']}")
    lines.append(f"Test set size:      {sample['n_test']}")
    lines.append("Stratification:     stratified by disease label")
    lines.append("")
    lines.append("Models and Results")
    lines.append("-" * 40)
    for model_type, metrics in all_metrics.items():
        lines.append(f"Model: {model_type}")
        lines.append(f"  Accuracy:            {metrics['accuracy']:.4f}")
        lines.append(f"  Precision (weighted):{metrics['precision_weighted']:.4f}")
        lines.append(f"  Recall (weighted):   {metrics['recall_weighted']:.4f}")
        lines.append(f"  F1-score (weighted): {metrics['f1_weighted']:.4f}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# One-shot full evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    primary_model: str = "decision_tree",
) -> Dict[str, Any]:
    """
    Convenience function: evaluate every trained model and write all
    standard report artifacts to the results/ directory.

    Parameters
    ----------
    test_size : float
    random_state : int
    primary_model : str
        The model whose detailed reports (metrics.txt,
        classification_report.txt, confusion_matrix.png) are written
        alongside the cross-model comparison.

    Returns
    -------
    dict
        {
            "all_metrics":    {model_type: metrics, ...},
            "primary_model":  str,
            "files":          {label: path, ...}
        }
    """
    if primary_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported primary_model '{primary_model}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    all_metrics = evaluate_all_models(
        test_size=test_size,
        random_state=random_state,
    )

    primary_metrics = all_metrics[primary_model]

    files: Dict[str, str] = {}
    files["metrics_txt"] = str(save_metrics_report(primary_metrics))
    files["classification_report_txt"] = str(
        save_classification_report(primary_metrics)
    )
    files["confusion_matrix_png"] = str(
        save_confusion_matrix_plot(primary_metrics)
    )
    files["model_comparison_csv"] = str(
        save_model_comparison_table(all_metrics)
    )
    files["experiment_notes_txt"] = str(
        save_experiment_notes(all_metrics)
    )

    return {
        "all_metrics": all_metrics,
        "primary_model": primary_model,
        "files": files,
    }