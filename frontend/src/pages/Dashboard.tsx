import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { analyticsApi, budgetsApi, mlApi } from '../lib/api';
import { Link } from 'react-router-dom';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts';
import { 
  ArrowUpRight, ArrowDownRight, Wallet, Activity, BrainCircuit, AlertCircle, ArrowRight
} from 'lucide-react';

export default function Dashboard() {
  const { data: summary } = useQuery({ queryKey: ['summary'], queryFn: async () => (await analyticsApi.summary(1)).data });
  const { data: categoryData } = useQuery({ queryKey: ['byCategory'], queryFn: async () => (await analyticsApi.byCategory(1)).data });
  const { data: stats } = useQuery({ queryKey: ['pipelineStats'], queryFn: async () => (await mlApi.pipelineStats()).data });
  const { data: coldStart } = useQuery({ queryKey: ['coldStart'], queryFn: async () => (await analyticsApi.coldStartStatus()).data });
  const { data: budgets } = useQuery({ queryKey: ['budgetStatus'], queryFn: async () => (await budgetsApi.status()).data });

  const formatMoney = (val: number) => `₹${val.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

  const COLORS = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e', '#f97316', '#eab308', '#22c55e', '#0ea5e9'];

  return (
    <div className="p-8 max-w-7xl mx-auto h-full overflow-y-auto space-y-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-2">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Dashboard Overview</h1>
          <p className="text-slate-400 mt-1">Your AI-categorised financial summary for the last 30 days.</p>
        </div>
        <Link to="/upload" className="btn-primary flex items-center gap-2">
          Upload Statement
        </Link>
      </div>

      {/* Top Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="stat-card group">
          <div className="flex justify-between items-start mb-4">
            <div className="p-3 bg-red-500/10 text-red-400 rounded-xl">
              <ArrowDownRight className="w-6 h-6" />
            </div>
            <span className="badge bg-slate-800 text-slate-300">Last 30 Days</span>
          </div>
          <p className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-1">Total Spent</p>
          <h3 className="text-3xl font-bold text-white">{formatMoney(summary?.total_spent || 0)}</h3>
        </div>
        
        <div className="stat-card group">
          <div className="flex justify-between items-start mb-4">
            <div className="p-3 bg-emerald-500/10 text-emerald-400 rounded-xl">
              <ArrowUpRight className="w-6 h-6" />
            </div>
            <span className="badge bg-slate-800 text-slate-300">Last 30 Days</span>
          </div>
          <p className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-1">Total Received</p>
          <h3 className="text-3xl font-bold text-white">{formatMoney(summary?.total_received || 0)}</h3>
        </div>

        <div className="stat-card group relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-primary-500/10 rounded-full blur-2xl transform translate-x-8 -translate-y-8" />
          <div className="flex justify-between items-start mb-4 relative z-10">
            <div className="p-3 bg-primary-500/10 text-primary-400 rounded-xl">
              <Wallet className="w-6 h-6" />
            </div>
            <span className="badge bg-slate-800 text-slate-300">Net Balance</span>
          </div>
          <p className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-1 relative z-10">Net Flow</p>
          <h3 className={`text-3xl font-bold relative z-10 ${(summary?.net || 0) >= 0 ? 'text-emerald-400' : 'text-white'}`}>
            {(summary?.net || 0) >= 0 ? '+' : ''}{formatMoney(summary?.net || 0)}
          </h3>
        </div>
      </div>

      {/* Middle Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Category Donut */}
        <div className="glass-card p-6 lg:col-span-2 flex flex-col">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-bold text-white">Expenses by Category</h3>
            <Link to="/analytics" className="text-sm text-primary-400 hover:text-primary-300 font-medium">Detailed Analysis &rarr;</Link>
          </div>
          
          <div className="flex-1 min-h-[300px] flex items-center justify-center">
            {(!categoryData || categoryData.length === 0) ? (
              <div className="text-center text-slate-500">No data available. Upload a statement.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={categoryData.slice(0, 7)} // Top 7
                    cx="50%"
                    cy="50%"
                    innerRadius={80}
                    outerRadius={120}
                    paddingAngle={5}
                    dataKey="total"
                    nameKey="category_name"
                    stroke="rgba(255,255,255,0.05)"
                    strokeWidth={2}
                  >
                    {categoryData.slice(0, 7).map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={entry.color || COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip 
                    formatter={(value: number) => formatMoney(value)}
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '12px', color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* AI Engine Status */}
        <div className="flex flex-col gap-6">
          <div className="glass-card p-6 relative overflow-hidden flex-1">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-primary-500/5 z-0" />
            <h3 className="text-lg font-bold text-white mb-6 relative z-10 flex items-center gap-2">
              <BrainCircuit className="w-5 h-5 text-purple-400" /> AI Engine Status
            </h3>
            
            <div className="space-y-5 relative z-10">
              <div>
                <div className="flex justify-between text-sm mb-1.5">
                  <span className="text-slate-400">Data Stage</span>
                  <span className="font-bold text-primary-400 uppercase tracking-wider">{coldStart?.stage || '...'}</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2">
                  <div className="bg-gradient-to-r from-primary-500 to-purple-500 h-2 rounded-full" 
                       style={{ width: `${Math.min(100, (coldStart?.transaction_count || 0) / 50 * 100)}%` }} />
                </div>
                {coldStart?.stage !== 'established' && (
                  <p className="text-xs text-slate-500 mt-2">{coldStart?.next_milestone} more transactions to reach established accuracy.</p>
                )}
              </div>

              <div className="pt-4 border-t border-white/5">
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Model Routing Distribution</p>
                <div className="space-y-2">
                  {stats?.distribution?.filter((d: any) => d.count > 0).map((d: any, idx: number) => (
                    <div key={`${d.source}-${idx}`} className="flex items-center justify-between text-sm">
                      <span className="text-slate-300 capitalize flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${
                          d.source === 'semantic' ? 'bg-blue-400' :
                          d.source === 'behavioural' ? 'bg-purple-400' :
                          d.source === 'fused' ? 'bg-emerald-400' :
                          d.source === 'manual' ? 'bg-amber-400' : 'bg-slate-500'
                        }`} />
                        {d.source?.replace('_', ' ') || 'Other/Pending'}
                      </span>
                      <span className="text-white font-mono">{d.percentage}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Review Alert */}
          {stats?.distribution?.find((d: any) => d.source === 'manual') && (
            <Link to="/review" className="glass-card p-5 border-amber-500/30 bg-amber-500/5 hover:bg-amber-500/10 transition-colors flex items-start gap-4 cursor-pointer group">
              <AlertCircle className="w-6 h-6 text-amber-500 mt-0.5 group-hover:scale-110 transition-transform" />
              <div>
                <h4 className="text-amber-500 font-bold mb-1 flex items-center gap-2">
                  Manual Review Needed <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity transform -translate-x-2 group-hover:translate-x-0" />
                </h4>
                <p className="text-slate-300 text-sm">Some transactions had low confidence and require your input to train the model.</p>
              </div>
            </Link>
          )}
        </div>
      </div>

      {/* Budgets Preview */}
      {budgets && budgets.length > 0 && (
        <div className="glass-card p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-bold text-white flex items-center gap-2"><Activity className="w-5 h-5 text-emerald-400" /> Active Budgets</h3>
            <Link to="/budgets" className="text-sm text-primary-400 hover:text-primary-300 font-medium">Manage Budgets &rarr;</Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {budgets.slice(0, 3).map((b: any) => (
              <div key={b.budget_id} className="bg-slate-900/50 rounded-xl p-4 border border-white/5">
                <div className="flex justify-between items-end mb-2">
                  <span className="font-semibold text-slate-200">{b.category_name}</span>
                  <span className="text-sm text-slate-400">{formatMoney(b.spent)} / {formatMoney(b.budgeted)}</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${b.percentage_used > 100 ? 'bg-red-500' : b.percentage_used > 85 ? 'bg-amber-500' : 'bg-emerald-500'}`} 
                    style={{ width: `${Math.min(100, b.percentage_used)}%` }} 
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
