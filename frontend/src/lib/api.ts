import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' },
});

// Attach JWT from localStorage
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('sb-access-token');
  if (token) {
    (config.headers as any).Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

// Auto-clear stale token on 401
api.interceptors.response.use(
  (res) => res,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('sb-access-token');
      localStorage.removeItem('sb-user-id');
    }
    return Promise.reject(error);
  }
);

export default api;

// ── Typed API helpers ──────────────────────────────────────────────────────────

export const authApi = {
  login: (email: string, password: string) =>
    api.post('/api/auth/login', { email, password }),
  register: (email: string, password: string, display_name: string) =>
    api.post('/api/auth/register', { email, password, display_name }),
};

export const transactionsApi = {
  upload: (file: File) => {
    const fd = new FormData();
    fd.append('file', file);
    return api.post('/api/transactions/upload', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  list: (params?: Record<string, unknown>) =>
    api.get('/api/transactions', { params }),
  getReviewQueue: () => api.get('/api/transactions/review'),
  updateCategory: (id: string, category_id: string) =>
    api.patch(`/api/transactions/${id}/category`, { category_id }),
};

export const analyticsApi = {
  summary: (months = 1) => api.get('/api/analytics/summary', { params: { months } }),
  byCategory: (months = 1) => api.get('/api/analytics/by-category', { params: { months } }),
  trends: (months = 6) => api.get('/api/analytics/trends', { params: { months } }),
  recurring: () => api.get('/api/analytics/recurring'),
  coldStartStatus: () => api.get('/api/analytics/cold-start-status'),
};

export const categoriesApi = {
  list: () => api.get('/api/categories'),
  create: (body: { name: string; icon?: string; color?: string }) =>
    api.post('/api/categories', body),
  delete: (id: string) => api.delete(`/api/categories/${id}`),
};

export const budgetsApi = {
  list: () => api.get('/api/budgets'),
  status: () => api.get('/api/budgets/status'),
  create: (body: Record<string, unknown>) => api.post('/api/budgets', body),
  delete: (id: string) => api.delete(`/api/budgets/${id}`),
};

export const mlApi = {
  pipelineStats: () => api.get('/api/ml/pipeline-stats'),
  performanceMetrics: () => api.get('/api/ml/performance-metrics'),
  clusteringMetrics: () => api.get('/api/ml/clustering-metrics'),
  coldstartMetrics: () => api.get('/api/ml/coldstart-metrics'),
  gatingAnalysis: () => api.get('/api/ml/gating-analysis'),
  clusterMap: () => api.get('/api/ml/cluster-map'),
  retryFailed: () => api.post('/api/ml/retry-failed'),
  retrainGating: () => api.post('/api/ml/retrain-gating'),
};

export const getMLPerformanceMetrics = async () => {
  const resp = await mlApi.performanceMetrics();
  return resp.data;
};
