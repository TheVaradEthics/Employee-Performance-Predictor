import { useEffect, useMemo, useState } from 'react'
import { api } from './lib/api'
import StatCard from './components/StatCard'
import SectionCard from './components/SectionCard'
import ProbabilityChart from './components/ProbabilityChart'

const initialForm = {
  age: 30,
  experience_years: 5,
  department: 'Engineering',
  education_level: 'Bachelor',
  work_mode: 'Hybrid',
  job_level: 2,
  salary: 600000,
  training_hours: 24,
  projects_completed: 4,
  average_monthly_hours: 170,
  on_time_delivery_rate: 0.84,
  absenteeism_days: 2,
  satisfaction_score: 7.8,
  manager_feedback_score: 8.1,
  peer_feedback_score: 7.9,
  overtime_hours: 10,
  promotions_last_5_years: 1,
}

const departments = ['Engineering', 'HR', 'Sales', 'Finance', 'Marketing', 'Operations']
const educationLevels = ['Diploma', 'Bachelor', 'Master', 'MBA', 'PhD']
const workModes = ['Onsite', 'Hybrid', 'Remote']

export default function App() {
  const [health, setHealth] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [rows, setRows] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [form, setForm] = useState(initialForm)
  const [loading, setLoading] = useState({ health: false, metrics: false, rows: false, train: false, predict: false })
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  async function loadHealth() {
    setLoading((s) => ({ ...s, health: true }))
    try {
      const data = await api.getHealth()
      setHealth(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading((s) => ({ ...s, health: false }))
    }
  }

  async function loadMetrics() {
    setLoading((s) => ({ ...s, metrics: true }))
    try {
      const data = await api.getMetrics()
      setMetrics(data)
    } catch (err) {
      setMetrics(null)
      setError(err.message)
    } finally {
      setLoading((s) => ({ ...s, metrics: false }))
    }
  }

  async function loadRows() {
    setLoading((s) => ({ ...s, rows: true }))
    try {
      const data = await api.getSampleData(8)
      setRows(data)
    } catch (err) {
      setRows([])
      setError(err.message)
    } finally {
      setLoading((s) => ({ ...s, rows: false }))
    }
  }

  useEffect(() => {
    loadHealth()
    loadMetrics()
    loadRows()
  }, [])

  async function handleTrain() {
    setError('')
    setMessage('')
    setLoading((s) => ({ ...s, train: true }))
    try {
      const data = await api.trainModel()
      setMessage(`Training complete. Accuracy: ${Number(data.accuracy).toFixed(3)} | Macro F1: ${Number(data.f1_macro).toFixed(3)}`)
      await Promise.all([loadHealth(), loadMetrics(), loadRows()])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading((s) => ({ ...s, train: false }))
    }
  }

  async function handlePredict(event) {
    event.preventDefault()
    setError('')
    setMessage('')
    setLoading((s) => ({ ...s, predict: true }))
    try {
      const payload = {
        ...form,
        age: Number(form.age),
        experience_years: Number(form.experience_years),
        job_level: Number(form.job_level),
        salary: Number(form.salary),
        training_hours: Number(form.training_hours),
        projects_completed: Number(form.projects_completed),
        average_monthly_hours: Number(form.average_monthly_hours),
        on_time_delivery_rate: Number(form.on_time_delivery_rate),
        absenteeism_days: Number(form.absenteeism_days),
        satisfaction_score: Number(form.satisfaction_score),
        manager_feedback_score: Number(form.manager_feedback_score),
        peer_feedback_score: Number(form.peer_feedback_score),
        overtime_hours: Number(form.overtime_hours),
        promotions_last_5_years: Number(form.promotions_last_5_years),
      }
      const data = await api.predict(payload)
      setPrediction(data)
      setMessage('Prediction generated successfully.')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading((s) => ({ ...s, predict: false }))
    }
  }

  const predictionBand = prediction?.predicted_performance_band || 'Not predicted yet'
  const topProbability = useMemo(() => {
    if (!prediction?.class_probabilities) return '—'
    const [label, prob] = Object.entries(prediction.class_probabilities).sort((a, b) => b[1] - a[1])[0]
    return `${label} (${(prob * 100).toFixed(1)}%)`
  }, [prediction])

  const metricCards = [
    {
      title: 'Dataset Ready',
      value: health?.dataset_exists ? 'Yes' : 'No',
      hint: 'Generated after training',
    },
    {
      title: 'Model Ready',
      value: health?.model_exists ? 'Yes' : 'No',
      hint: 'Needed for prediction',
    },
    {
      title: 'Metrics Ready',
      value: health?.metrics_exists ? 'Yes' : 'No',
      hint: 'Training evaluation output',
    },
    {
      title: 'Predicted Band',
      value: predictionBand,
      hint: `Top confidence: ${topProbability}`,
    },
  ]

  function updateField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">HR Analytics • Machine Learning • React Dashboard</p>
          <h1>Employee Performance Predictor</h1>
          <p className="hero-copy">
            Train the model, inspect backend artifacts, preview sample employee data, and generate live performance predictions from one frontend.
          </p>
        </div>
        <div className="hero-actions">
          <button className="primary-btn" onClick={handleTrain} disabled={loading.train}>
            {loading.train ? 'Training...' : 'Train Model'}
          </button>
          <button className="secondary-btn" onClick={() => { loadHealth(); loadMetrics(); loadRows(); }}>
            Refresh Dashboard
          </button>
        </div>
      </header>

      {error ? <div className="alert error">{error}</div> : null}
      {message ? <div className="alert success">{message}</div> : null}

      <div className="stats-grid">
        {metricCards.map((item) => (
          <StatCard key={item.title} {...item} />
        ))}
      </div>

      <div className="content-grid">
        <div className="left-column">
          <SectionCard title="Backend Status" subtitle="Health and model readiness from FastAPI.">
            <div className="status-grid">
              <div className={`status-pill ${health?.dataset_exists ? 'ok' : 'bad'}`}>Dataset</div>
              <div className={`status-pill ${health?.model_exists ? 'ok' : 'bad'}`}>Model</div>
              <div className={`status-pill ${health?.metrics_exists ? 'ok' : 'bad'}`}>Metrics</div>
            </div>
            {loading.health ? <p className="muted">Loading status...</p> : null}
          </SectionCard>

          <SectionCard title="Training Metrics" subtitle="Reads the backend metrics output after training.">
            {loading.metrics ? (
              <p className="muted">Loading metrics...</p>
            ) : metrics ? (
              <div className="metrics-grid">
                {Object.entries(metrics).map(([key, value]) => (
                  <div key={key} className="metric-item">
                    <span>{key.replaceAll('_', ' ')}</span>
                    <strong>{typeof value === 'number' ? value.toFixed(3) : String(value)}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">Train the model first to view metrics.</p>
            )}
          </SectionCard>

          <SectionCard title="Sample Dataset Preview" subtitle="Pulled from the generated synthetic CSV.">
            {loading.rows ? (
              <p className="muted">Loading rows...</p>
            ) : rows.length ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      {Object.keys(rows[0]).slice(0, 8).map((key) => <th key={key}>{key}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, index) => (
                      <tr key={index}>
                        {Object.keys(rows[0]).slice(0, 8).map((key) => <td key={key}>{String(row[key])}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted">No sample data available yet.</p>
            )}
          </SectionCard>
        </div>

        <div className="right-column">
          <SectionCard title="Prediction Form" subtitle="Submit employee features to the backend /predict endpoint.">
            <form className="form-grid" onSubmit={handlePredict}>
              <label><span>Age</span><input type="number" value={form.age} onChange={(e) => updateField('age', e.target.value)} /></label>
              <label><span>Experience Years</span><input type="number" value={form.experience_years} onChange={(e) => updateField('experience_years', e.target.value)} /></label>
              <label><span>Department</span><select value={form.department} onChange={(e) => updateField('department', e.target.value)}>{departments.map((d) => <option key={d}>{d}</option>)}</select></label>
              <label><span>Education Level</span><select value={form.education_level} onChange={(e) => updateField('education_level', e.target.value)}>{educationLevels.map((d) => <option key={d}>{d}</option>)}</select></label>
              <label><span>Work Mode</span><select value={form.work_mode} onChange={(e) => updateField('work_mode', e.target.value)}>{workModes.map((d) => <option key={d}>{d}</option>)}</select></label>
              <label><span>Job Level</span><input type="number" value={form.job_level} onChange={(e) => updateField('job_level', e.target.value)} /></label>
              <label><span>Salary</span><input type="number" value={form.salary} onChange={(e) => updateField('salary', e.target.value)} /></label>
              <label><span>Training Hours</span><input type="number" value={form.training_hours} onChange={(e) => updateField('training_hours', e.target.value)} /></label>
              <label><span>Projects Completed</span><input type="number" value={form.projects_completed} onChange={(e) => updateField('projects_completed', e.target.value)} /></label>
              <label><span>Average Monthly Hours</span><input type="number" value={form.average_monthly_hours} onChange={(e) => updateField('average_monthly_hours', e.target.value)} /></label>
              <label><span>On-time Delivery Rate</span><input type="number" step="0.01" min="0" max="1" value={form.on_time_delivery_rate} onChange={(e) => updateField('on_time_delivery_rate', e.target.value)} /></label>
              <label><span>Absenteeism Days</span><input type="number" value={form.absenteeism_days} onChange={(e) => updateField('absenteeism_days', e.target.value)} /></label>
              <label><span>Satisfaction Score</span><input type="number" step="0.1" value={form.satisfaction_score} onChange={(e) => updateField('satisfaction_score', e.target.value)} /></label>
              <label><span>Manager Feedback</span><input type="number" step="0.1" value={form.manager_feedback_score} onChange={(e) => updateField('manager_feedback_score', e.target.value)} /></label>
              <label><span>Peer Feedback</span><input type="number" step="0.1" value={form.peer_feedback_score} onChange={(e) => updateField('peer_feedback_score', e.target.value)} /></label>
              <label><span>Overtime Hours</span><input type="number" value={form.overtime_hours} onChange={(e) => updateField('overtime_hours', e.target.value)} /></label>
              <label><span>Promotions Last 5 Years</span><input type="number" value={form.promotions_last_5_years} onChange={(e) => updateField('promotions_last_5_years', e.target.value)} /></label>
              <div className="form-actions">
                <button className="primary-btn" type="submit" disabled={loading.predict}>
                  {loading.predict ? 'Predicting...' : 'Predict Performance'}
                </button>
                <button className="secondary-btn" type="button" onClick={() => setForm(initialForm)}>
                  Reset Form
                </button>
              </div>
            </form>
          </SectionCard>

          <SectionCard title="Prediction Output" subtitle="Model class and probability breakdown.">
            <div className="result-box">
              <div>
                <span className="muted">Predicted Performance Band</span>
                <h3>{predictionBand}</h3>
              </div>
              <ProbabilityChart probabilities={prediction?.class_probabilities} />
            </div>
          </SectionCard>
        </div>
      </div>
    </div>
  )
}
