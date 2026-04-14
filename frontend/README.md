# Employee Performance Predictor Frontend

A React + Vite frontend that matches the Employee Performance Predictor FastAPI backend.

## Features
- Train model from the UI
- View backend health status
- See training metrics
- Preview sample dataset rows
- Submit employee details and get performance prediction probabilities
- Clean dashboard-style interface

## Setup
### Windows
```powershell
npm install
copy .env.example .env
npm run dev
```

### Mac/Linux
```bash
npm install
cp .env.example .env
npm run dev
```

## Environment
Set the backend base URL in `.env`:
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Run
```bash
npm run dev
```

Open:
- Frontend: `http://127.0.0.1:5173`
- Backend docs: `http://127.0.0.1:8000/docs`

## Expected backend endpoints
- `GET /health`
- `POST /train`
- `GET /metrics`
- `GET /sample-data`
- `POST /predict`
