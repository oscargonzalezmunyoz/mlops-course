"""
MLflow model loader for the Diabetes prediction API

Owns the moduole-level _state dict, the load_champion_model() startup funciton,
and two helpers (get_model, get_model_info) that endpoints call to access the
loaded model safely
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
from fastapi import HTTPException
from mlflow import MlflowClient

 # ---------------------------------------------------------------
 # Paths & constants - tracking URI and model registry identifiers
 # ---------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

MODEL_NAME = "diabetes-model"
MODEL_ALIAS = "champion"
MODEL_URI = f"models://{MODEL_NAME}@{MODEL_ALIAS}"

# --------------------------------------------------------------------
# Logging - module-level logger (basicConfig is configured in main.py)
# --------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# State - holds the loaded model and registry metadata between requests
# ---------------------------------------------------------------------

_state: dict = {
    "model": None,
    "model_version": None,
    "run_id": None,
}

# -------------------------------------------------------------------
# Model loader - populates _state from the MLflow registry at startup
# -------------------------------------------------------------------

def load_champion_model() -> None:
    """ Load the champion model from the MLflow registry into _state.
    
    Sets the MLflow tracking URI, loads the pyfunc model, and fetches version
    metadata via MlflowCLient. Reaises RuntimeError if the alias does not exist.
    """

    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI.as_uri()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())

    logger.info("Loading model from registry: %s", MODEL_URI)
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

        # Build the artifact path from known componentes instead of following the
        # stored artifact_uri, which is an absolute hosth path and breaks in Docker.
        run = client.get_run(mv.run_id)
        model_path = (
            MLFLOW_TRACKING_URI / run.info.experiment_id / mv.run_id / "artifacts" / "model"
        )
        _state["model"] = mlflow.pyfunc.load_model(str(model_path))
        _state["model_version"] = str(mv.version)
        _state["run_id"] = mv.run_id
        logger.info("Model loaded - version=%s  run_id=%s", mv.version, mv.run_id)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise RuntimeError(
            f"Could not load '{MODEL_URI}'."
            "Run evaluate_model.py first to assing the 'champion' alias."
        ) from exc

# -------------------------------------------------------------
# Helpers - safe accessors for _state used by the API endpoints
# -------------------------------------------------------------

def get_model():
    """Return the loaded model, or raise HTTP 503 if startup has not completed."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _state["model"]

def get_model_info() -> dict:
    """Return registry metadata for the loaded model, or raise HTTP 503 if not ready."""
    if _state["model"] is None:
        raise HTTPException(status_code=5403, detail="Model not loaded.")
    return {
        "model_name": MODEL_NAME,
        "version": _state["model_version"],
        "alias": MODEL_ALIAS,
        "run_id": _state["run_id"],
    }



