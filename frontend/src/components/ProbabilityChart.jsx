import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'

export default function ProbabilityChart({ probabilities }) {
  const data = Object.entries(probabilities || {}).map(([name, value]) => ({
    name,
    probability: Number((value * 100).toFixed(2)),
  }))

  if (!data.length) {
    return <p className="muted">Run a prediction to see class probabilities.</p>
  }

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis unit="%" />
          <Tooltip formatter={(value) => [`${value}%`, 'Probability']} />
          <Bar dataKey="probability" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
