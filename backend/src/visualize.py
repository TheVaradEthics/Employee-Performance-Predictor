import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CLASS_PLOT_PATH, FEATURE_PLOT_PATH, METRICS_PATH
from src.modeling import PerformancePredictor


def create_class_distribution_plot(df: pd.DataFrame) -> None:
    counts = df["performance_band"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Employee Performance Band Distribution")
    plt.xlabel("Performance Band")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CLASS_PLOT_PATH)
    plt.close()


def create_feature_importance_plot(model: PerformancePredictor | object) -> None:
    pipeline = model.pipeline if isinstance(model, PerformancePredictor) else model
    pre = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["model"]

    feature_names = pre.get_feature_names_out()
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 7))
    importances.sort_values().plot(kind="barh")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_PLOT_PATH)
    plt.close()


def read_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
