import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { transactionsApi, categoriesApi } from '../lib/api';
import { format } from 'date-fns';
import { Search, Filter, AlertCircle, RefreshCw, Box } from 'lucide-react';

export default function Transactions() {
  const [page, setPage] = useState(1);
  const [filterStatus, setFilterStatus] = useState<string>('');
  const [filterDirection, setFilterDirection] = useState<string>('');
  
  const { data: categoriesData } = useQuery({
    queryKey: ['categories'],
    queryFn: async () => {
      const res = await categoriesApi.list();
      return res.data;
    }
  });

  const { data: txnsData, isLoading, isFetching } = useQuery({
    queryKey: ['transactions', page, filterStatus, filterDirection],
    queryFn: async () => {
      const params: any = { page, page_size: 50 };
      if (filterStatus) params.status = filterStatus;
      if (filterDirection) params.direction = filterDirection;
      const res = await transactionsApi.list(params);
      return res.data;
    },
    refetchInterval: (query) => {
      // Poll if any transactions are still processing
      const hasProcessing = query.state.data?.data?.some((t: any) => t.processing_status === 'processing');
      return hasProcessing ? 3000 : false;
    }
  });

  const categories = categoriesData || [];
  const transactions = txnsData?.data || [];

  return (
    <div className="p-8 max-w-7xl mx-auto h-full flex flex-col">
      <div className="mb-8 flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Transactions</h1>
          <p className="text-slate-400">View and manage all your categorised bank transactions.</p>
        </div>
        
        <div className="flex items-center gap-3 w-full md:w-auto">
          <div className="relative flex-1 md:w-64">
             <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
             <input type="text" placeholder="Search merchants..." className="input-field pl-9 h-10 py-1 text-sm rounded-lg" />
          </div>
          
          <select 
            value={filterDirection} 
            onChange={(e) => { setFilterDirection(e.target.value); setPage(1); }}
            className="input-field h-10 py-1 px-3 text-sm rounded-lg w-auto appearance-none bg-slate-800"
          >
            <option value="">All Types</option>
            <option value="debit">Debits (-)</option>
            <option value="credit">Credits (+)</option>
          </select>

          <select 
            value={filterStatus} 
            onChange={(e) => { setFilterStatus(e.target.value); setPage(1); }}
            className="input-field h-10 py-1 px-3 text-sm rounded-lg w-auto appearance-none bg-slate-800"
          >
            <option value="">All Statuses</option>
            <option value="completed">Completed</option>
            <option value="processing">Processing (AI)</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      <div className="glass-card flex-1 overflow-hidden flex flex-col">
        <div className="overflow-x-auto flex-1">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-white/10 bg-slate-900/50">
                <th className="p-4 table-header">Date</th>
                <th className="p-4 table-header">Description</th>
                <th className="p-4 table-header">Amount</th>
                <th className="p-4 table-header">Category</th>
                <th className="p-4 table-header">AI Confidence</th>
                <th className="p-4 table-header">Status</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan={6} className="p-8 text-center text-slate-400"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto" /></td></tr>
              ) : transactions.length === 0 ? (
                <tr>
                  <td colSpan={6} className="p-12 text-center text-slate-400">
                    <Box className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                    <p className="text-lg font-medium text-slate-300">No transactions found</p>
                    <p className="text-sm mt-1">Try adjusting your filters or upload a bank statement.</p>
                  </td>
                </tr>
              ) : (
                transactions.map((txn: any) => {
                  const cat = categories.find((c: any) => c.id === txn.category_id);
                  const isProcessing = txn.processing_status === 'processing' || txn.processing_status === 'pending';
                  
                  return (
                    <tr key={txn.id} className="border-b border-white/[0.04] hover:bg-white/[0.02] transition-colors">
                      <td className="p-4 align-top">
                        <div className="text-sm text-slate-300 whitespace-nowrap">
                          {format(new Date(txn.transaction_date), 'dd MMM yyyy')}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">{txn.payment_method || 'UPI'}</div>
                      </td>
                      <td className="p-4 align-top max-w-xs">
                        <div className="text-sm font-medium text-slate-200 truncate" title={txn.merchant_name || txn.raw_description}>
                          {txn.merchant_name || txn.raw_description}
                        </div>
                        <div className="text-xs text-slate-500 truncate mt-0.5" title={txn.raw_description}>
                          {txn.raw_description}
                        </div>
                        {txn.is_recurring && (
                          <span className="badge bg-purple-500/10 text-purple-400 text-[10px] mt-1.5 inline-flex">Recurring</span>
                        )}
                      </td>
                      <td className="p-4 align-top whitespace-nowrap">
                        <span className={`font-bold ${txn.direction === 'credit' ? 'text-emerald-400' : 'text-slate-200'}`}>
                          {txn.direction === 'credit' ? '+' : '-'}₹{Math.abs(txn.amount).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                        </span>
                      </td>
                      <td className="p-4 align-top">
                        {isProcessing ? (
                          <span className="flex items-center gap-2 text-primary-400 text-sm italic">
                            <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Processing
                          </span>
                        ) : cat ? (
                          <span className="flex items-center gap-2 text-sm text-slate-300 bg-slate-800/50 px-2.5 py-1 rounded-md border border-white/5 w-fit">
                            <span>{cat.icon}</span> {cat.name}
                          </span>
                        ) : (
                          <span className="text-sm text-slate-500">-</span>
                        )}
                      </td>
                      <td className="p-4 align-top">
                        {!isProcessing && txn.category_id && (
                          <div className="flex flex-col gap-1 w-32">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-slate-400 uppercase tracking-wider font-semibold text-[9px]">{txn.category_source?.replace('_',' ')}</span>
                              <span className={`font-bold ${txn.needs_review ? 'text-amber-400' : 'text-emerald-400'}`}>
                                {((txn.confidence_score || 0) * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                              <div 
                                className={`h-full ${txn.needs_review ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                style={{ width: `${Math.max(5, (txn.confidence_score || 0) * 100)}%` }} 
                              />
                            </div>
                          </div>
                        )}
                      </td>
                      <td className="p-4 align-top">
                        {txn.needs_review && !txn.user_corrected ? (
                          <span className="badge-warning inline-flex items-center gap-1.5">
                            <AlertCircle className="w-3 h-3" /> Review
                          </span>
                        ) : txn.processing_status === 'failed' ? (
                          <span className="badge-danger">Failed</span>
                        ) : txn.user_corrected ? (
                          <span className="badge-info">Corrected</span>
                        ) : txn.processing_status === 'completed' ? (
                          <span className="badge-success border border-emerald-500/20">Done</span>
                        ) : (
                          <span className="badge bg-slate-700 text-slate-300 border border-slate-600">Pending</span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
        
        {/* Pagination Footer */}
        <div className="p-4 border-t border-white/10 bg-slate-900/30 flex items-center justify-between">
          <div className="text-sm text-slate-500">
            Showing <span className="text-white font-medium">{transactions.length}</span> results
            {isFetching && <RefreshCw className="w-3 h-3 inline-block ml-3 animate-spin text-primary-500" />}
          </div>
          <div className="flex gap-2">
            <button 
              disabled={page === 1} 
              onClick={() => setPage(p => p - 1)}
              className="btn-secondary py-1.5 px-4 text-sm"
            >
              Previous
            </button>
            <button 
              disabled={transactions.length < 50} 
              onClick={() => setPage(p => p + 1)}
              className="btn-secondary py-1.5 px-4 text-sm"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
