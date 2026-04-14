from __future__ import annotations

from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import METRICS_PATH, MODEL_PATH

TARGET_COL = "performance_band"
DROP_COLS = ["employee_id", "performance_score", TARGET_COL]


class PerformancePredictor:
    def __init__(self) -> None:
        self.pipeline = None
        self.best_params_: Dict[str, object] | None = None

    @staticmethod
    def build_pipeline(df: pd.DataFrame) -> Pipeline:
        feature_df = df.drop(columns=DROP_COLS, errors="ignore")
        numeric_cols = feature_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_cols),
                ("cat", categorical_pipe, categorical_cols),
            ]
        )

        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_estimators=300,
        )

        return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    def train(self, df: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=DROP_COLS, errors="ignore")
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        base_pipeline = self.build_pipeline(df)
        param_grid = {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 16],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }

        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        self.pipeline = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        predictions = self.pipeline.predict(X_test)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
            "classification_report": classification_report(y_test, predictions, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "labels": sorted(y.unique().tolist()),
            "best_params": self.best_params_,
        }

        joblib.dump(self.pipeline, MODEL_PATH)
        pd.Series(metrics).to_json(METRICS_PATH, indent=2)
        return metrics, X_test, y_test, pd.Series(predictions, index=y_test.index)

    @staticmethod
    def load_model() -> Pipeline:
        return joblib.load(MODEL_PATH)
