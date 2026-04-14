from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "employee_data.csv"
MODEL_PATH = MODELS_DIR / "employee_performance_model.pkl"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
CLASS_PLOT_PATH = OUTPUTS_DIR / "class_distribution.png"
FEATURE_PLOT_PATH = OUTPUTS_DIR / "feature_importance.png"
PREDICTION_LOG_PATH = OUTPUTS_DIR / "sample_predictions.json"
