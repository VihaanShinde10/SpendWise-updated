import { useQuery } from '@tanstack/react-query';
import { analyticsApi, categoriesApi } from '../lib/api';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area
} from 'recharts';
import { format } from 'date-fns';
import { Repeat } from 'lucide-react';

export default function Analytics() {
  const { data: summary } = useQuery({ queryKey: ['summary'], queryFn: async () => (await analyticsApi.summary(1)).data });
  const { data: trends } = useQuery({ queryKey: ['trends'], queryFn: async () => (await analyticsApi.trends(6)).data });
  const { data: recurring } = useQuery({ queryKey: ['recurring'], queryFn: async () => (await analyticsApi.recurring()).data });
  const { data: categories } = useQuery({ queryKey: ['categories'], queryFn: async () => (await categoriesApi.list()).data });
  const { data: status } = useQuery({ queryKey: ['cold-start'], queryFn: async () => (await analyticsApi.coldStartStatus()).data });

  const formatMoney = (val: number) => `₹${val.toLocaleString('en-IN')}`;

  return (
    <div className="p-8 max-w-7xl mx-auto h-full overflow-y-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Spending Analytics</h1>
          <p className="text-slate-400">Deep dive into your financial habits and trends over time.</p>
        </div>
        
        {status && (
          <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-2xl px-4 py-2 flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-mono text-emerald-400 uppercase tracking-wider">
              System Stage: {status.stage} ({status.transaction_count} Txns)
            </span>
          </div>
        )}
      </div>

      {/* Summary Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-card p-6 border-l-4 border-l-red-500">
          <p className="text-slate-500 text-sm font-medium mb-1">Total Expenses (30d)</p>
          <p className="text-2xl font-bold text-white">{formatMoney(summary?.total_spent || 0)}</p>
        </div>
        <div className="glass-card p-6 border-l-4 border-l-emerald-500">
          <p className="text-slate-500 text-sm font-medium mb-1">Total Income (30d)</p>
          <p className="text-2xl font-bold text-white">{formatMoney(summary?.total_received || 0)}</p>
        </div>
        <div className="glass-card p-6 border-l-4 border-l-purple-500">
          <p className="text-slate-500 text-sm font-medium mb-1">Pattern Recognition</p>
          <p className="text-2xl font-bold text-white">{status?.expected_coverage_pct || 0}%</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Trend Chart */}
        <div className="glass-card p-6">
          <h3 className="text-lg font-bold text-white mb-6">Income vs Expenses (6 Months)</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trends || []} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorSpent" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorReceived" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="month" stroke="#94a3b8" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" tick={{fontSize: 12}} tickLine={false} axisLine={false} tickFormatter={(val) => `₹${val/1000}k`} />
                <Tooltip 
                  formatter={(value: any) => formatMoney(Number(value))}
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '12px', color: '#fff' }}
                />
                <Area type="monotone" dataKey="received" stroke="#22c55e" strokeWidth={2} fillOpacity={1} fill="url(#colorReceived)" name="Income" />
                <Area type="monotone" dataKey="spent" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill="url(#colorSpent)" name="Expenses" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recurring Payments */}
        <div className="glass-card p-6 flex flex-col">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Repeat className="w-5 h-5 text-purple-400" />
              Detected Recurring Payments
            </h3>
          </div>
          
          <div className="flex-1 overflow-y-auto pr-2">
            {!recurring || recurring.length === 0 ? (
              <div className="text-slate-500 text-center py-10">No recurring payments detected yet.</div>
            ) : (
              <div className="space-y-4">
                {recurring.map((rpt: any, idx: number) => {
                  const cat = categories?.find((c: any) => c.id === rpt.category_id);
                  return (
                    <div key={idx} className="bg-slate-900/50 rounded-xl p-4 border border-white/5 flex justify-between items-center group hover:bg-slate-800 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-purple-500/10 flex items-center justify-center text-purple-400 font-bold">
                          {cat?.icon || '🔄'}
                        </div>
                        <div>
                          <p className="font-bold text-slate-200">{rpt.merchant_name}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-xs text-slate-500">{cat?.name || 'Uncategorised'}</span>
                            <span className="w-1 h-1 rounded-full bg-slate-700" />
                            <span className="text-xs text-purple-400">AI Confidence: {(rpt.recurrence_strength * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-bold text-white">{formatMoney(rpt.amount)}</p>
                        <p className="text-xs text-slate-500 mt-1">Last: {format(new Date(rpt.transaction_date), 'dd MMM')}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
