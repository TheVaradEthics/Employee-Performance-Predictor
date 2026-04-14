from pydantic import BaseModel, Field


class EmployeeFeatures(BaseModel):
    age: int = Field(..., ge=18, le=65)
    experience_years: int = Field(..., ge=0, le=40)
    department: str
    education_level: str
    work_mode: str
    job_level: int = Field(..., ge=1, le=8)
    salary: int = Field(..., ge=100000, le=5000000)
    training_hours: int = Field(..., ge=0, le=150)
    projects_completed: int = Field(..., ge=0, le=20)
    average_monthly_hours: int = Field(..., ge=80, le=320)
    on_time_delivery_rate: float = Field(..., ge=0, le=1)
    absenteeism_days: int = Field(..., ge=0, le=30)
    satisfaction_score: float = Field(..., ge=1, le=10)
    manager_feedback_score: float = Field(..., ge=1, le=10)
    peer_feedback_score: float = Field(..., ge=1, le=10)
    overtime_hours: int = Field(..., ge=0, le=80)
    promotions_last_5_years: int = Field(..., ge=0, le=10)


class TrainingResponse(BaseModel):
    message: str
    dataset_rows: int
    accuracy: float
    f1_macro: float


class PredictionResponse(BaseModel):
    predicted_performance_band: str
    class_probabilities: dict[str, float]
