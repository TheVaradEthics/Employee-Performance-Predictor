const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })

  let data = null
  const text = await response.text()
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }

  if (!response.ok) {
    const message = data?.detail || data?.message || 'Request failed'
    throw new Error(message)
  }

  return data
}

export const api = {
  getHealth: () => request('/health'),
  trainModel: () => request('/train', { method: 'POST' }),
  getMetrics: () => request('/metrics'),
  getSampleData: (limit = 8) => request(`/sample-data?limit=${limit}`),
  predict: (payload) => request('/predict', { method: 'POST', body: JSON.stringify(payload) }),
}
