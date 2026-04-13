import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { transactionsApi, categoriesApi } from '../lib/api';
import { format } from 'date-fns';
import { Tag, Check, X, AlertCircle } from 'lucide-react';

export default function ReviewQueue() {
  const queryClient = useQueryClient();

  const { data: queueData, isLoading } = useQuery({
    queryKey: ['review-queue'],
    queryFn: async () => {
      const res = await transactionsApi.getReviewQueue();
      return res.data.data;
    },
    refetchInterval: 10000, // Poll every 10s
  });

  const { data: categoriesData } = useQuery({
    queryKey: ['categories'],
    queryFn: async () => {
      const res = await categoriesApi.list();
      return res.data;
    }
  });

  const updateCategory = useMutation({
    mutationFn: ({ id, categoryId }: { id: string, categoryId: string }) => 
      transactionsApi.updateCategory(id, categoryId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['review-queue'] });
      queryClient.invalidateQueries({ queryKey: ['transactions'] });
      // Suggest retargeting the gating network since we have new corrections
      queryClient.invalidateQueries({ queryKey: ['pipeline-stats'] });
    }
  });

  if (isLoading) {
    return <div className="p-8 flex justify-center"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" /></div>;
  }

  const queue = queueData || [];
  const categories = categoriesData || [];

  return (
    <div className="p-8 max-w-6xl mx-auto h-full overflow-y-auto">
      <div className="mb-8 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Review Queue</h1>
          <p className="text-slate-400">Transactions with low ML confidence ({String((queue.length > 0 ? queue[0].gating_alpha : 0) || 0).substring(0,4)}). Review to train your personal AI.</p>
        </div>
        <div className="bg-slate-800 px-4 py-2 rounded-xl border border-white/10">
          <span className="text-2xl font-bold text-amber-400">{queue.length}</span>
          <span className="text-slate-400 ml-2 text-sm font-medium uppercase tracking-wider">Pending</span>
        </div>
      </div>

      {queue.length === 0 ? (
        <div className="glass-card p-12 text-center flex flex-col items-center justify-center">
          <div className="w-16 h-16 bg-emerald-500/10 text-emerald-400 rounded-full flex items-center justify-center mb-4">
            <Check className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-bold text-slate-200 mb-2">All Caught Up!</h3>
          <p className="text-slate-400 max-w-md">The ML pipeline is confident about all recent transactions. Your active feedback helps improve accuracy over time.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {queue.map((txn: any) => (
            <div key={txn.id} className="glass-card p-5 border-l-4 border-l-amber-500 animate-fade-in flex flex-col">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-bold text-white mb-1 leading-tight">{txn.raw_description}</h3>
                  <p className="text-xs text-slate-500 font-medium tracking-wide font-mono">
                    {format(new Date(txn.transaction_date), 'dd MMM yyyy')} • {txn.payment_method || 'UPI'}
                  </p>
                </div>
                <div className={`px-3 py-1 rounded-lg font-bold ${txn.direction === 'credit' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-200'}`}>
                  {txn.direction === 'credit' ? '+' : '-'}₹{Math.abs(txn.amount).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                </div>
              </div>

              <div className="bg-slate-900/50 rounded-xl p-4 mb-4 border border-white/[0.04] grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mb-1">AI Suggestion</p>
                  <p className="text-sm font-medium text-slate-300 flex items-center gap-2">
                    {categories.find((c: any) => c.id === txn.category_id)?.name || 'Others'}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mb-1">Confidence</p>
                  <div className="flex items-center gap-2">
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div 
                        className="h-full bg-amber-500" 
                        style={{ width: `${Math.max(5, (txn.confidence_score || 0) * 100)}%` }} 
                      />
                    </div>
                    <span className="text-xs font-bold text-amber-500">{((txn.confidence_score || 0) * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>

              <div className="mt-auto pt-2 flex flex-col sm:flex-row gap-3">
                <select 
                  className="input-field py-2 text-sm flex-1 appearance-none bg-slate-800/80 cursor-pointer"
                  onChange={(e) => {
                    if (e.target.value) {
                      updateCategory.mutate({ id: txn.id, categoryId: e.target.value });
                    }
                  }}
                  defaultValue=""
                >
                  <option value="" disabled>Select correct category...</option>
                  {categories.map((c: any) => (
                    <option key={c.id} value={c.id}>{c.icon} {c.name}</option>
                  ))}
                </select>
                
                {txn.category_id && (
                  <button 
                    onClick={() => updateCategory.mutate({ id: txn.id, categoryId: txn.category_id })}
                    className="btn-primary py-2 px-4 flex items-center justify-center gap-2 whitespace-nowrap bg-emerald-600 hover:bg-emerald-500 shadow-emerald-600/20"
                  >
                    <Check className="w-4 h-4" />
                    Confirm Suggestion
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
