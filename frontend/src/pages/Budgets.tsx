import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { budgetsApi, categoriesApi } from '../lib/api';
import { PiggyBank, Plus, Trash2, Target, AlertTriangle } from 'lucide-react';
import { useForm } from 'react-hook-form';

export default function Budgets() {
  const [isAdding, setIsAdding] = useState(false);
  const queryClient = useQueryClient();
  const { register, handleSubmit, reset } = useForm();

  const { data: statusData, isLoading } = useQuery({ queryKey: ['budgetStatus'], queryFn: async () => (await budgetsApi.status()).data });
  const { data: categories } = useQuery({ queryKey: ['categories'], queryFn: async () => (await categoriesApi.list()).data });

  const createMutation = useMutation({
    mutationFn: (data: any) => budgetsApi.create({ ...data, start_date: new Date().toISOString().split('T')[0] }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['budgetStatus'] });
      setIsAdding(false);
      reset();
    }
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => budgetsApi.delete(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['budgetStatus'] })
  });

  const onSubmit = (data: any) => {
    createMutation.mutate(data);
  };

  const budgets = statusData || [];

  return (
    <div className="p-8 max-w-6xl mx-auto h-full overflow-y-auto space-y-8">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Budgets</h1>
          <p className="text-slate-400">Set limits and track your spending automatically.</p>
        </div>
        <button onClick={() => setIsAdding(true)} className="btn-primary flex items-center gap-2">
          <Plus className="w-4 h-4" /> New Budget
        </button>
      </div>

      {isAdding && (
         <div className="glass-card p-6 border-primary-500/30 animate-slide-up">
           <div className="flex justify-between items-center mb-4">
             <h3 className="text-lg font-bold text-white">Create New Budget</h3>
             <button onClick={() => setIsAdding(false)} className="text-slate-400 hover:text-white">Cancel</button>
           </div>
           <form onSubmit={handleSubmit(onSubmit)} className="flex gap-4 items-end">
              <div className="flex-1">
                <label className="block text-sm font-medium text-slate-400 mb-1">Category</label>
                <select {...register('category_id', { required: true })} className="input-field py-2">
                  <option value="">Select a category...</option>
                  {categories?.map((c: any) => <option key={c.id} value={c.id}>{c.icon} {c.name}</option>)}
                </select>
              </div>
              <div className="flex-1">
                <label className="block text-sm font-medium text-slate-400 mb-1">Monthly Limit (₹)</label>
                <input type="number" {...register('amount', { required: true, min: 1 })} className="input-field py-2" placeholder="e.g. 5000" />
              </div>
              <button type="submit" disabled={createMutation.isPending} className="btn-primary py-2.5">
                Save Budget
              </button>
           </form>
         </div>
      )}

      {isLoading ? (
        <div className="flex justify-center p-12"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" /></div>
      ) : budgets.length === 0 ? (
        <div className="glass-card p-12 text-center flex flex-col items-center justify-center">
          <div className="w-16 h-16 bg-primary-500/10 text-primary-400 rounded-full flex items-center justify-center mb-4">
            <Target className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-bold text-slate-200 mb-2">No Budgets Set</h3>
          <p className="text-slate-400">Create your first budget to start tracking your financial goals.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {budgets.map((b: any) => {
            const isOver = b.percentage_used >= 100;
            const isWarning = !isOver && b.percentage_used >= 85;
            
            return (
              <div key={b.budget_id} className="glass-card p-6 relative group">
                <button 
                  onClick={() => deleteMutation.mutate(b.budget_id)}
                  className="absolute top-4 right-4 text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Remove Budget"
                >
                  <Trash2 className="w-4 h-4" />
                </button>

                <div className="flex items-center gap-3 mb-4">
                  <div className={`p-2 rounded-lg ${isOver ? 'bg-red-500/20 text-red-400' : isWarning ? 'bg-amber-500/20 text-amber-400' : 'bg-primary-500/20 text-primary-400'}`}>
                    {isOver ? <AlertTriangle className="w-5 h-5" /> : <PiggyBank className="w-5 h-5" />}
                  </div>
                  <div>
                    <h3 className="font-bold text-lg text-white leading-tight">{b.category_name}</h3>
                    <p className="text-xs text-slate-500 uppercase tracking-wider">{b.period}</p>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between items-end mb-2">
                    <span className="text-2xl font-bold text-white">₹{b.spent.toLocaleString('en-IN')}</span>
                    <span className="text-sm text-slate-400">of ₹{b.budgeted.toLocaleString('en-IN')}</span>
                  </div>
                  
                  <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-500 ${isOver ? 'bg-red-500' : isWarning ? 'bg-amber-500' : 'bg-emerald-500'}`} 
                      style={{ width: `${Math.min(100, b.percentage_used)}%` }} 
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 pt-4 border-t border-white/5">
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Remaining</p>
                    <p className={`font-bold ${isOver ? 'text-red-400' : 'text-emerald-400'}`}>
                      {isOver ? 'Exceeded by ' : ''}₹{Math.abs(b.remaining).toLocaleString('en-IN')}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Used</p>
                    <p className={`font-bold ${isOver ? 'text-red-400' : isWarning ? 'text-amber-400' : 'text-slate-200'}`}>
                      {b.percentage_used.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
