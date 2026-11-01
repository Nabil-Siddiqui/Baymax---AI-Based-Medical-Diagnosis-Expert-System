"""
Baymax - Synthetic Dataset Generator
====================================

This script generates a balanced, reproducible synthetic medical
dataset for the Baymax expert system.

Each row represents one simulated patient case:

    - 18 binary symptom columns (0 / 1)
    - 1 'disease' label column

Generation model
----------------
For a target disease D, each symptom S is sampled independently:

    P(S = 1 | D) =
        0.85 if S is a core symptom of D
        0.40 if S is an optional symptom of D
        0.05 otherwise (unrelated)

These probabilities are aligned with the Bayesian engine in
src/bayes_engine.py. This consistency ensures that:

    - the data follows a Naive Bayes generative process,
    - the Bayesian engine performs principled inference on it,
    - evaluation results are interpretable and academically defensible.

Usage
-----
From the project root (Baymax/):

    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --cases-per-disease 100
    python scripts/generate_dataset.py --seed 123 --output data/medical_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Allow running this script directly with `python scripts/generate_dataset.py`
# by ensuring the project root is on sys.path so we can import `src.*`.
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_base import (  # noqa: E402  (import after sys.path tweak)
    get_symptom_ids,
    load_diseases,
    validate_knowledge_base,
)


# ---------------------------------------------------------------------------
# Generation parameters (must match src/bayes_engine.py)
# ---------------------------------------------------------------------------

P_CORE: float = 0.85
P_OPTIONAL: float = 0.40
P_UNRELATED: float = 0.05

DEFAULT_CASES_PER_DISEASE: int = 75
DEFAULT_SEED: int = 42
DEFAULT_OUTPUT: Path = PROJECT_ROOT / "data" / "medical_dataset.csv"

TARGET_COLUMN: str = "disease"


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def _classify_symptom(symptom_id: str, disease_info: dict) -> str:
    """Return 'core', 'optional', or 'unrelated' for the given symptom."""
    core = set(disease_info.get("core_symptoms", []) or [])
    optional = set(disease_info.get("optional_symptoms", []) or [])
    if symptom_id in core:
        return "core"
    if symptom_id in optional:
        return "optional"
    return "unrelated"


def _p_present(symptom_id: str, disease_info: dict) -> float:
    """Return P(symptom = 1 | disease) using the parameterized profiles."""
    category = _classify_symptom(symptom_id, disease_info)
    if category == "core":
        return P_CORE
    if category == "optional":
        return P_OPTIONAL
    return P_UNRELATED


def _generate_case(
    disease_info: dict,
    symptom_ids: List[str],
    rng: random.Random,
) -> List[int]:
    """
    Generate a single patient row (binary symptom vector) for the given
    disease. The vector is aligned to `symptom_ids`.
    """
    row: List[int] = []
    for sid in symptom_ids:
        p = _p_present(sid, disease_info)
        row.append(1 if rng.random() < p else 0)
    return row


# ---------------------------------------------------------------------------
# Dataset generation entry point
# ---------------------------------------------------------------------------

def generate_dataset(
    output_path: Path = DEFAULT_OUTPUT,
    cases_per_disease: int = DEFAULT_CASES_PER_DISEASE,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    """
    Generate the synthetic dataset and write it to disk as CSV.

    Parameters
    ----------
    output_path : Path
        Destination CSV file path.
    cases_per_disease : int
        Number of patient cases generated per disease (must be > 0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Summary information about the generated dataset.
    """
    if cases_per_disease <= 0:
        raise ValueError("cases_per_disease must be a positive integer.")

    # Make sure the knowledge base is internally consistent before we
    # generate any data. This catches typos in disease symptom IDs.
    validate_knowledge_base()

    diseases = load_diseases()
    symptom_ids = get_symptom_ids()

    if not diseases:
        raise RuntimeError("Knowledge base contains no diseases.")
    if not symptom_ids:
        raise RuntimeError("Knowledge base contains no symptoms.")

    rng = random.Random(seed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate rows in a deterministic order: iterate diseases, then
    # generate `cases_per_disease` rows for each.
    header = list(symptom_ids) + [TARGET_COLUMN]
    total_rows = 0

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for disease_name, info in diseases.items():
            for _ in range(cases_per_disease):
                row = _generate_case(info, symptom_ids, rng)
                writer.writerow(row + [disease_name])
                total_rows += 1

    return {
        "output_path": str(output_path),
        "n_rows": total_rows,
        "n_features": len(symptom_ids),
        "n_diseases": len(diseases),
        "cases_per_disease": cases_per_disease,
        "seed": seed,
        "diseases": list(diseases.keys()),
        "symptom_ids": list(symptom_ids),
        "probabilities": {
            "P_core": P_CORE,
            "P_optional": P_OPTIONAL,
            "P_unrelated": P_UNRELATED,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Baymax synthetic medical dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--cases-per-disease",
        type=int,
        default=DEFAULT_CASES_PER_DISEASE,
        help=(
            "Number of patient cases per disease "
            f"(default: {DEFAULT_CASES_PER_DISEASE})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    summary = generate_dataset(
        output_path=args.output,
        cases_per_disease=args.cases_per_disease,
        seed=args.seed,
    )

    print("Baymax synthetic dataset generated successfully.")
    print(f"  File:            {summary['output_path']}")
    print(f"  Rows:            {summary['n_rows']}")
    print(f"  Features:        {summary['n_features']}")
    print(f"  Diseases:        {summary['n_diseases']}")
    print(f"  Cases / disease: {summary['cases_per_disease']}")
    print(f"  Seed:            {summary['seed']}")
    print(
        "  Symptom probs:   "
        f"core={summary['probabilities']['P_core']}, "
        f"optional={summary['probabilities']['P_optional']}, "
        f"unrelated={summary['probabilities']['P_unrelated']}"
    )


if __name__ == "__main__":
    main()