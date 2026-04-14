from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.modeling import PerformancePredictor


def predict_single(record: Dict[str, object]) -> Dict[str, object]:
    model = PerformancePredictor.load_model()
    frame = pd.DataFrame([record])
    predicted_band = model.predict(frame)[0]
    probabilities = model.predict_proba(frame)[0]
    classes: List[str] = model.classes_.tolist()
    probability_map = {label: round(float(prob), 4) for label, prob in zip(classes, probabilities)}
    return {
        "predicted_performance_band": predicted_band,
        "class_probabilities": probability_map,
    }
