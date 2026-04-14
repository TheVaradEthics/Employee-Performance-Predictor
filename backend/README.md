# Employee Performance Predictor Backend

A complete backend project for predicting employee performance bands (`High`, `Medium`, `Low`) using a synthetic HR dataset, scikit-learn pipeline, and FastAPI.

## Features
- Synthetic employee dataset generation
- Clean machine learning training pipeline
- Random Forest classifier with hyperparameter tuning
- Accuracy, macro F1, confusion matrix, and classification report
- Feature importance plot and class distribution plot
- FastAPI endpoints for train, predict, metrics, health, and sample data
- GitHub-ready structure

## Project Structure
```text
backend/
├── app/
│   ├── main.py
│   └── schemas.py
├── data/
├── models/
├── outputs/
├── src/
│   ├── config.py
│   ├── data_generator.py
│   ├── modeling.py
│   ├── predict.py
│   ├── train.py
│   └── visualize.py
├── tests/
│   └── sample_payload.json
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Setup
### Windows
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the model
```bash
python -m src.train
```

This will generate:
- `data/employee_data.csv`
- `models/employee_performance_model.pkl`
- `outputs/metrics.json`
- `outputs/class_distribution.png`
- `outputs/feature_importance.png`

## Run the API
```bash
uvicorn main:app --reload
```

Open:
- API root: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

## API Endpoints
- `GET /` → Basic API info
- `POST /train` → Generate dataset and train model
- `POST /predict` → Predict performance for one employee
- `GET /metrics` → View training metrics
- `GET /health` → Check artifacts
- `GET /sample-data` → Preview generated dataset rows

## Example Prediction Request
Use `tests/sample_payload.json` in Swagger or Postman.

## Suggested Workflow
1. Install dependencies
2. Run `/train`
3. Open `/metrics`
4. Use `/predict`
5. Capture screenshots for GitHub
