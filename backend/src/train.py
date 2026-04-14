import json

from src.data_generator import save_dataset
from src.modeling import PerformancePredictor
from src.visualize import create_class_distribution_plot, create_feature_importance_plot


def main() -> None:
    df = save_dataset(num_rows=1200, random_state=42)
    predictor = PerformancePredictor()
    metrics, _, _, _ = predictor.train(df)
    create_class_distribution_plot(df)
    create_feature_importance_plot(predictor)

    print("Training complete.")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"classification_report", "confusion_matrix"}}, indent=2))
    print("Artifacts saved in models/ and outputs/ directories.")


if __name__ == "__main__":
    main()
