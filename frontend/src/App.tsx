import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, NavLink } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuthStore } from './stores/auth';
import { 
  LayoutDashboard, Upload, List, BarChart3, 
  PiggyBank, AlertCircle, LogOut,
  Zap, Activity
} from 'lucide-react';

// Pages
import Dashboard from './pages/Dashboard';
import UploadPage from './pages/UploadPage';
import Transactions from './pages/Transactions';
import Analytics from './pages/Analytics';
import Budgets from './pages/Budgets';
import ReviewQueue from './pages/ReviewQueue';
import Login from './pages/Login';
import Register from './pages/Register';
import SystemInsights from './pages/SystemInsights';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
});

const NAV_LINKS = [
  { to: '/',             icon: LayoutDashboard, label: 'Dashboard'     },
  { to: '/upload',       icon: Upload,          label: 'Upload'        },
  { to: '/transactions', icon: List,            label: 'Transactions'  },
  { to: '/analytics',    icon: BarChart3,       label: 'Analytics'     },
  { to: '/budgets',      icon: PiggyBank,       label: 'Budgets'       },
  { to: '/review',       icon: AlertCircle,     label: 'Review Queue'  },
  { to: '/insights',     icon: Activity,        label: 'System Insights' },
];

function Sidebar() {
  const { user, logout } = useAuthStore();
  return (
    <aside className="w-64 flex-shrink-0 bg-slate-900/80 border-r border-white/[0.06] flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="p-6 border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-primary-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-primary-500/30">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-white text-lg leading-tight">SpendWise</h1>
            <p className="text-xs text-slate-500">AI Finance Tracker</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        {NAV_LINKS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `sidebar-link ${isActive ? 'active' : ''}`
            }
          >
            <Icon className="w-4.5 h-4.5 flex-shrink-0" size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* User */}
      <div className="p-4 border-t border-white/[0.06]">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-500 to-purple-600 flex items-center justify-center text-xs font-bold text-white flex-shrink-0">
            {user?.email?.[0]?.toUpperCase() ?? 'U'}
          </div>
          <div className="min-w-0">
            <p className="text-xs font-medium text-slate-300 truncate">{user?.email}</p>
            <p className="text-xs text-slate-500">Active account</p>
          </div>
        </div>
        <button onClick={logout} className="btn-ghost w-full flex items-center gap-2 text-sm text-slate-400 hover:text-red-400">
          <LogOut size={14} />
          Sign Out
        </button>
      </div>
    </aside>
  );
}

function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-slate-950">
        {children}
      </main>
    </div>
  );
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuthStore();
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <AppLayout>{children}</AppLayout>;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/login"    element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/upload"       element={<ProtectedRoute><UploadPage /></ProtectedRoute>} />
          <Route path="/transactions" element={<ProtectedRoute><Transactions /></ProtectedRoute>} />
          <Route path="/analytics"    element={<ProtectedRoute><Analytics /></ProtectedRoute>} />
          <Route path="/budgets"      element={<ProtectedRoute><Budgets /></ProtectedRoute>} />
          <Route path="/review"       element={<ProtectedRoute><ReviewQueue /></ProtectedRoute>} />
          <Route path="/insights"     element={<ProtectedRoute><SystemInsights /></ProtectedRoute>} />
          <Route path="*"             element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
