import logging
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------
# Paths & constants
# -----------------

from src.config import PROCESSED_PATH, MLFLOW_TRACKING_URI
from src.config import EXPERIMENT_NAME, TARGET, TEST_SIZE, RANDOM_STATE
from src.config import RANDOM_FOREST_ESTIMATORS, RANDOM_FOREST_MAX_DEPTH

# -------
# Logging
# -------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----
# Data
# ----

def load_features(path: Path) -> pd.DataFrame:
    """Load the enginered feature dataset from CSV:
    
    Args:
        path: Path to train_v1.csv"
    Returns:
        DataFrame with all features and target column.
    Raises
        FileNotFoundError: If the input file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run the data pipeline first: python src/data/preprocess.py"
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)
    return df    

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split DataFrame into train/test sets
    Args:
        df: Full feature DataFrame including the target column.
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).        
    """

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

# -------
# Metrics
# -------

def compute_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Compute regression evaluation metrics.
    
    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
    
    Returns:
        Dictionary with keys ``rmse``, ``mae``, ``r2``.
    """
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

# ----------
# MLflow run
# ----------
def run_experiment(model, model_type: str, params: dict,
                   X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> None:
    """Train a model and log everything to a single MLflow run.
       
    The run name follows the convention ``{model_type}_{YYYYMMDD_HHMMSS}``
    so runs are identifiable at a glance in the UI.
    
    Tags logged:
        - ``model_type``: class name of  the estimator
        - ``dataset``: source file name
        - ``stage++: always ``"training"`` for this script

    Args:
        model: An unfitted scikit-learn estimator.
        model_type: Short label used in the run name and `` model_type``  tag.
        params: Hyperparameter dict to log with ``mlflow.log_params``
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train_ Training target series.
        y_test: Test target series.
            
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        # --- Tags ---
        mlflow.set_tags({
            "model_type": model_type,
            "dataset": PROCESSED_PATH.name,
            "stage": "training",
        }) 
        # --- Params ---
        mlflow.log_params(params)

        # --- Training ---
        logger.info("[%s] Training...", model_type)
        model.fit(X_train, y_train)
        # --- Metrics ---
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        logger.info(
            "[%s] rmse=%.4f  mae=%.4f  r2=%.4f",
            model_type,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
        )
        # --- Artifact ---
        # registered_model_name=None: registry is handled by register_model.py
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,
            input_example=X_train.head(5),
        )
        logger.info("[%s] Model artifact logged.", model_type)

# ----
# Main
# ----

def main() -> None:
    """Run the full experiment tracking pipeline for all models."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    logger.info("Experiment:          %s", EXPERIMENT_NAME)

    df = load_features(PROCESSED_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    # --- Baseline : DummyRegressor ---
    # Always predicts the training mean. Any useful model must beat it.
    run_experiment(
        model=DummyRegressor(strategy="mean"),
        model_type="DummyRegressor",
        params={"strategy": "mean"},
        X_train = X_train, X_test = X_test,
        y_train = y_train, y_test = y_test,
    )

    # --- Model 1: Linear Regression ---
    run_experiment(
        model=LinearRegression(),
        model_type="LinearRegression",
        params={"fit_intercept": True},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )

    # --- Model 2: RandomForestRegressor ---
    
    run_experiment(
        model=RandomForestRegressor(n_estimators=RANDOM_FOREST_ESTIMATORS, max_depth=RANDOM_FOREST_MAX_DEPTH, random_state=RANDOM_STATE),
        model_type="RandomForestRegressor",
        params={"n_estimators": RANDOM_FOREST_ESTIMATORS, "max_depth": RANDOM_FOREST_MAX_DEPTH, "random_state": RANDOM_STATE},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )

    # --- Model 3: GradientBoostingRegressor ---
    run_experiment(
        model=GradientBoostingRegressor(random_state=RANDOM_STATE),
        model_type="GradientBoostingRegressor",
        params={"random_state": RANDOM_STATE},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )    

    logger.info(
        "Done. Open the MLflow UI to compare runs:\n"
        "  mlflow ui --backend-store-uri %s",
        MLFLOW_TRACKING_URI,
    )

if __name__ == "__main__":
    main()

