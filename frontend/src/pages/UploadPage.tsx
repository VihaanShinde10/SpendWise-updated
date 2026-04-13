import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { transactionsApi } from '../lib/api';
import { useNavigate } from 'react-router-dom';

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const [stats, setStats] = useState<{ count: number } | null>(null);
  const navigate = useNavigate();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setStatus('idle');
      setMessage('');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    maxFiles: 1,
  });

  const handleUpload = async () => {
    if (!file) return;
    setStatus('uploading');
    setMessage('Parsing bank statement and uploading...');

    try {
      const res = await transactionsApi.upload(file);
      setStatus('success');
      setStats({ count: res.data.transaction_count });
      setMessage(res.data.message);
      
      // Auto redirect to transactions after 3 seconds
      setTimeout(() => navigate('/transactions'), 3000);
    } catch (err: any) {
      setStatus('error');
      setMessage(err?.response?.data?.detail || 'Failed to upload statement');
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto h-full overflow-y-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Upload Statement</h1>
        <p className="text-slate-400">Import your bank statement to categorise transactions automatically using the AI pipeline.</p>
      </div>

      <div className="grid gap-6">
        <div className="glass-card p-8 text-center" {...getRootProps()}>
          <input {...getInputProps()} />
          <div className={`
            border-2 border-dashed rounded-xl p-12 transition-all duration-200 cursor-pointer
            ${isDragActive ? 'border-primary-500 bg-primary-500/10' : 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/50'}
            ${file ? 'border-success bg-success/5' : ''}
          `}>
            {file ? (
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 rounded-full bg-success/20 flex items-center justify-center mb-4 text-success">
                  <FileText className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-bold text-slate-200 mb-1">{file.name}</h3>
                <p className="text-slate-400 text-sm">{(file.size / 1024).toFixed(1)} KB • CSV or Excel</p>
                
                <div className="mt-8">
                  {status === 'idle' && (
                    <button 
                      onClick={(e) => { e.stopPropagation(); handleUpload(); }}
                      className="btn-primary flex items-center gap-2"
                    >
                      <Upload className="w-4 h-4" />
                      Upload & Process
                    </button>
                  )}
                  {status === 'uploading' && (
                    <button disabled className="btn-primary flex items-center gap-2 opacity-80">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Uploading...
                    </button>
                  )}
                  {status === 'success' && (
                    <button onClick={(e) => { e.stopPropagation(); navigate('/transactions'); }} className="btn-secondary flex items-center gap-2 text-success">
                      <CheckCircle2 className="w-4 h-4" />
                      View Transactions
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center text-slate-400">
                <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-4">
                  <Upload className="w-8 h-8 text-slate-300" />
                </div>
                <h3 className="text-xl font-bold text-slate-200 mb-2">
                  {isDragActive ? 'Drop your file here...' : 'Drag & drop bank statement'}
                </h3>
                <p className="mb-4">or click to browse from your computer</p>
                <div className="text-sm bg-slate-800 px-4 py-2 rounded-lg">
                  Supports .csv, .xlsx, .xls
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Status Messages */}
        {status === 'error' && (
          <div className="glass-card p-6 border-red-500/30 bg-red-500/5 flex items-start gap-4 animate-fade-in">
            <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-red-400 font-bold mb-1">Upload Failed</h4>
              <p className="text-slate-300 text-sm">{message}</p>
            </div>
          </div>
        )}

        {status === 'success' && (
          <div className="glass-card p-6 border-emerald-500/30 bg-emerald-500/5 flex items-start gap-4 animate-fade-in">
            <CheckCircle2 className="w-6 h-6 text-emerald-400 flex-shrink-0 mt-0.5" />
            <div className="w-full">
              <h4 className="text-emerald-400 font-bold mb-1">Upload Successful</h4>
              <p className="text-slate-300 text-sm mb-4">{message}</p>
              
              <div className="bg-slate-900 rounded-lg p-4 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">Transactions Found</p>
                  <p className="text-2xl font-bold text-white">{stats?.count}</p>
                </div>
                <div>
                  <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">Status</p>
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />
                    <span className="text-primary-400 font-medium">Processing in background</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
