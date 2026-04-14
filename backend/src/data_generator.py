import numpy as np
import pandas as pd

from src.config import DATASET_PATH


DEPARTMENTS = ["IT", "HR", "Finance", "Sales", "Marketing", "Operations"]
EDUCATION = ["Bachelor", "Master", "MBA", "Diploma"]
WORK_MODES = ["Onsite", "Hybrid", "Remote"]


def _band_from_score(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def generate_dataset(num_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(22, 58, num_rows)
    experience_years = np.clip(age - rng.integers(20, 28, num_rows), 0, 35)
    department = rng.choice(DEPARTMENTS, num_rows, p=[0.28, 0.10, 0.12, 0.20, 0.15, 0.15])
    education_level = rng.choice(EDUCATION, num_rows, p=[0.45, 0.30, 0.15, 0.10])
    work_mode = rng.choice(WORK_MODES, num_rows, p=[0.40, 0.40, 0.20])
    job_level = np.clip((experience_years // 4) + 1, 1, 6)
    salary = (250000 + experience_years * 45000 + job_level * 70000 + rng.normal(0, 50000, num_rows)).astype(int)
    training_hours = np.clip(rng.normal(35, 15, num_rows), 0, 100).round()
    projects_completed = np.clip(rng.poisson(4, num_rows) + 1, 1, 12)
    average_monthly_hours = np.clip(rng.normal(170, 18, num_rows), 120, 260).round()
    on_time_delivery_rate = np.clip(rng.normal(0.78, 0.12, num_rows), 0.30, 1.00).round(2)
    absenteeism_days = np.clip(rng.poisson(4, num_rows), 0, 20)
    satisfaction_score = np.clip(rng.normal(6.8, 1.6, num_rows), 1, 10).round(1)
    manager_feedback_score = np.clip(rng.normal(6.9, 1.5, num_rows), 1, 10).round(1)
    peer_feedback_score = np.clip(rng.normal(7.0, 1.4, num_rows), 1, 10).round(1)
    overtime_hours = np.clip(rng.normal(12, 8, num_rows), 0, 45).round()
    promotions_last_5_years = np.clip(rng.poisson(0.8, num_rows), 0, 4)

    raw_score = (
        0.18 * training_hours
        + 2.4 * projects_completed
        + 24 * on_time_delivery_rate
        + 2.8 * satisfaction_score
        + 3.4 * manager_feedback_score
        + 2.2 * peer_feedback_score
        + 1.6 * promotions_last_5_years
        + 0.20 * experience_years
        - 1.8 * absenteeism_days
        - 0.22 * np.maximum(overtime_hours - 18, 0)
        + rng.normal(0, 6, num_rows)
    )

    min_score, max_score = raw_score.min(), raw_score.max()
    normalized_score = ((raw_score - min_score) / (max_score - min_score) * 100).round(1)
    performance_band = [_band_from_score(score) for score in normalized_score]

    df = pd.DataFrame(
        {
            "employee_id": [f"EMP{1000+i}" for i in range(num_rows)],
            "age": age,
            "experience_years": experience_years,
            "department": department,
            "education_level": education_level,
            "work_mode": work_mode,
            "job_level": job_level,
            "salary": salary,
            "training_hours": training_hours.astype(int),
            "projects_completed": projects_completed.astype(int),
            "average_monthly_hours": average_monthly_hours.astype(int),
            "on_time_delivery_rate": on_time_delivery_rate,
            "absenteeism_days": absenteeism_days.astype(int),
            "satisfaction_score": satisfaction_score,
            "manager_feedback_score": manager_feedback_score,
            "peer_feedback_score": peer_feedback_score,
            "overtime_hours": overtime_hours.astype(int),
            "promotions_last_5_years": promotions_last_5_years.astype(int),
            "performance_score": normalized_score,
            "performance_band": performance_band,
        }
    )
    return df


def save_dataset(num_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    df = generate_dataset(num_rows=num_rows, random_state=random_state)
    df.to_csv(DATASET_PATH, index=False)
    return df


if __name__ == "__main__":
    df = save_dataset()
    print(f"Dataset saved to: {DATASET_PATH}")
    print(df.head())
