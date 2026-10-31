from pathlib import Path
# --- Paths ---

BASE_DIR = Path(__file__).resolve().parents[1]      # project root
RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

# --- Dataset ---
FEATURE_COLUMNS = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
TARGET = "target"
FEATURE_MIN, FEATURE_MAX = -1.0, 1.0 # features are scaled

# --- Training ---
EXPERIMENT_NAME = "diabetes"
REGISTERED_MODEL_NAME = "diabetes-model"
TARGET = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42
RANDOM_FOREST_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 15


# --- Registry & quality gate ---
REGISTERED_MODEL_NAME = "diabetes-model"
MODEL_ALIAS = "champion"
RMSE_THRESHOLD = 60.0
R2_THRESHOLD = 0.40

