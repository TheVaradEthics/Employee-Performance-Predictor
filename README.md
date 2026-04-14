# 🚀 Employee Performance Predictor (ML + Dashboard)

## 📌 Overview

This project is an industry-oriented Machine Learning system that predicts employee performance (High / Medium / Low) using HR and productivity data.

It helps HR teams and managers make data-driven decisions for:

* Promotions
* Training & Development
* Performance Monitoring
* Employee Retention

---

## 🎯 Problem Statement

Organizations struggle to:

* Identify high-performing employees early
* Detect underperformers before appraisal cycles
* Allocate training budgets efficiently

This project solves these problems using Machine Learning.

---

## 🧠 Features

* Synthetic HR dataset generation
* Data preprocessing & feature engineering
* Machine Learning model (Random Forest)
* Performance prediction API (FastAPI)
* Interactive frontend dashboard (React/Vite)
* Visualization of prediction probabilities
* Training trigger from UI

---

## ⚙️ Tech Stack

### Backend

* Python
* Pandas, NumPy
* Scikit-learn
* FastAPI
* Joblib

### Frontend

* React (Vite)
* Chart.js
* Axios

---

## 🏗️ Project Structure

```
Employee-Performance-Predictor/
│
├── backend/
│   ├── data/
│   ├── src/
│   ├── models/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Train model
python -m src.train

# Run API
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### 2️⃣ Frontend Setup

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Set API URL in `.env`:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Open:

```
http://localhost:5173
```

---

## 📊 Workflow

```
Data → Preprocessing → Model Training → Prediction → Dashboard → HR Insights
```

---

## 📸 Screenshots (Add these)

* Dataset preview
* Performance distribution chart
* Model accuracy output
* Prediction result UI
* Dashboard view

---

## 🎯 Results

* Predicts employee performance category
* Provides probability distribution
* Helps HR make better decisions

---

## 🧪 Example Prediction Input

```json
{
  "age": 30,
  "experience": 5,
  "salary": 50000,
  "department": 1,
  "training_hours": 40,
  "projects": 5,
  "performance_score": 8
}
```

---

## 🚀 Future Improvements

* Add SHAP explainability
* Use XGBoost model
* Deploy using Docker
* Add real HR dataset
* Build advanced analytics dashboard

---

## 🤝 Contribution

Feel free to fork this repo and improve it!

---

## 📌 Author

**Varad Payghan**

Built as a placement-ready ML project 🚀
