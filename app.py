"""
Baymax - Flask Web Application
==============================

This module is the HTTP entry point of the Baymax expert system.
It exposes a small, focused web interface and a JSON API around the
diagnosis pipeline defined in `src/diagnosis_pipeline.py`.

Routes
------
GET  /              - Render the homepage with the symptom selector.
POST /diagnose      - Accept selected symptoms (form submission) and
                      render the result page.
GET  /api/symptoms  - Return the canonical symptom list as JSON.
POST /api/diagnose  - JSON endpoint that runs the diagnosis pipeline.
GET  /healthz       - Lightweight health check.

Design notes
------------
- This module is intentionally thin. All AI/diagnosis logic lives in
  `src/diagnosis_pipeline.py` and the modules it orchestrates.
- The trained ML model bundle is loaded once at startup for speed.
  If no model has been trained yet, the website still works using
  only the rule-based and Bayesian components.
- A custom Jinja filter `pct` is registered for clean percentage
  formatting on the result page.

Environment variables
---------------------
- BAYMAX_DEBUG     : "1" to enable Flask debug mode (default off).
- BAYMAX_HOST      : Host to bind to (default "127.0.0.1").
- BAYMAX_PORT      : Port to bind to (default 5000).
- BAYMAX_ML_MODEL  : Which trained model to use (default "decision_tree").
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from src.diagnosis_pipeline import run_diagnosis
from src.knowledge_base import (
    get_symptom_label_map,
    load_diseases,
    load_symptoms,
    validate_knowledge_base,
)
from src.ml_model import SUPPORTED_MODELS, load_trained_model
from src.utils import format_probability


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ML_MODEL = os.environ.get("BAYMAX_ML_MODEL", "random_forest")
if DEFAULT_ML_MODEL not in SUPPORTED_MODELS:
    DEFAULT_ML_MODEL = "decision_tree"

DEBUG_MODE = os.environ.get("BAYMAX_DEBUG", "0") == "1"
HOST = os.environ.get("BAYMAX_HOST", "127.0.0.1")
PORT = int(os.environ.get("BAYMAX_PORT", "5000"))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """
    Construct and configure the Flask application.

    Using an app factory makes the app easier to test and easier to
    embed in larger systems.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # Validate the knowledge base early so any typos are caught at
    # startup rather than at request time.
    try:
        validate_knowledge_base()
    except Exception as exc:  # noqa: BLE001
        # Don't crash the import if validation fails, but log a warning
        # the developer will see in the terminal.
        app.logger.warning("Knowledge base validation failed: %s", exc)

    # Try to preload the ML model bundle once at startup. If it is not
    # available yet, we keep going - the diagnosis pipeline gracefully
    # falls back to rule-based and Bayesian reasoning.
    try:
        app.config["ML_BUNDLE"] = load_trained_model(DEFAULT_ML_MODEL)
        app.logger.info(
            "Loaded ML model bundle '%s' at startup.", DEFAULT_ML_MODEL
        )
    except FileNotFoundError:
        app.config["ML_BUNDLE"] = None
        app.logger.warning(
            "No trained ML model found for '%s'. The site will run "
            "without the ML component. Train it with "
            "src.ml_model.train_model().",
            DEFAULT_ML_MODEL,
        )

    app.config["DEFAULT_ML_MODEL"] = DEFAULT_ML_MODEL

    _register_jinja_filters(app)
    _register_routes(app)

    return app


def _register_jinja_filters(app: Flask) -> None:
    """Register small custom Jinja filters used by the templates."""

    @app.template_filter("pct")
    def _pct_filter(value: Any, decimals: int = 1) -> str:
        """Format a probability in [0, 1] as a percentage string."""
        return format_probability(value, decimals=decimals)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_symptoms_from_form(form) -> List[str]:
    """
    Extract submitted symptom IDs from a Flask form.

    Supports two input styles:

    1. Multiple <select> dropdowns each named "symptoms":
           form.getlist("symptoms")  ->  ["fever", "cough", ...]

    2. A single hidden field named "symptoms_csv" containing
       comma-separated symptom IDs (used as a fallback for the JS UI).
    """
    selected = form.getlist("symptoms")
    if selected:
        return [s for s in selected if s]

    csv_value = form.get("symptoms_csv", "").strip()
    if not csv_value:
        return []

    return [s.strip() for s in csv_value.split(",") if s.strip()]


def _extract_symptoms_from_json(payload: Any) -> List[str]:
    """
    Extract a list of symptom IDs from a JSON payload.

    Accepts:
        {"symptoms": ["fever", "cough"]}
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("symptoms", [])
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, (str, int))]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _register_routes(app: Flask) -> None:
    """Attach all HTTP routes to the given Flask app."""

    @app.route("/", methods=["GET"])
    def index():
        """Render the homepage with the symptom selector."""
        symptoms = load_symptoms()
        return render_template(
            "index.html",
            symptoms=symptoms,
            ml_available=app.config.get("ML_BUNDLE") is not None,
            ml_model=app.config.get("DEFAULT_ML_MODEL"),
        )

    @app.route("/diagnose", methods=["POST"])
    def diagnose():
        """
        Form submission handler. Reads selected symptoms, runs the
        diagnosis pipeline, and renders the result page.
        """
        selected = _extract_symptoms_from_form(request.form)

        if not selected:
            # No symptoms selected: send the user back to the form.
            return redirect(url_for("index"))

        try:
            result = run_diagnosis(
                symptoms=selected,
                ml_model=app.config["DEFAULT_ML_MODEL"],
            )
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("Diagnosis failed: %s", exc)
            return render_template(
                "result.html",
                error="An unexpected error occurred while running the "
                      "diagnosis. Please try again.",
                result=None,
                symptom_labels=get_symptom_label_map(),
            ), 500

        return render_template(
            "result.html",
            error=None,
            result=result,
            symptom_labels=get_symptom_label_map(),
        )

    @app.route("/api/symptoms", methods=["GET"])
    def api_symptoms():
        """Return the canonical symptom list as JSON."""
        return jsonify(load_symptoms())

    @app.route("/api/diagnose", methods=["POST"])
    def api_diagnose():
        """
        JSON endpoint for programmatic diagnosis.

        Request body:
            {"symptoms": ["fever", "cough", "fatigue"]}

        Response body:
            The structured dict returned by run_diagnosis(...).
        """
        if not request.is_json:
            abort(415, description="Expected application/json request body.")

        payload = request.get_json(silent=True) or {}
        selected = _extract_symptoms_from_json(payload)

        if not selected:
            return (
                jsonify({"error": "No symptoms provided."}),
                400,
            )

        try:
            result = run_diagnosis(
                symptoms=selected,
                ml_model=app.config["DEFAULT_ML_MODEL"],
            )
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("API diagnosis failed: %s", exc)
            return (
                jsonify({"error": "Diagnosis failed.", "detail": str(exc)}),
                500,
            )

        return jsonify(result)

    @app.route("/healthz", methods=["GET"])
    def healthz():
        """Lightweight health check used during development and demos."""
        diseases = load_diseases()
        return jsonify(
            {
                "status": "ok",
                "diseases": len(diseases),
                "symptoms": len(load_symptoms()),
                "ml_available": app.config.get("ML_BUNDLE") is not None,
                "ml_model": app.config.get("DEFAULT_ML_MODEL"),
            }
        )

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(_err):
        return (
            render_template(
                "result.html",
                error="The page you requested was not found.",
                result=None,
                symptom_labels={},
            ),
            404,
        )

    @app.errorhandler(415)
    def unsupported_media(err):
        return (
            jsonify({"error": getattr(err, "description", "Unsupported Media Type")}),
            415,
        )


# ---------------------------------------------------------------------------
# WSGI entry point
# ---------------------------------------------------------------------------

# The `app` object is what WSGI servers (and `flask run`) look for.
app = create_app()


if __name__ == "__main__":
    # Direct execution: `python app.py`
    app.run(host=HOST, port=PORT, debug=DEBUG_MODE)