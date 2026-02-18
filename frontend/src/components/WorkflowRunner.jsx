import { useState, useRef } from 'react'
import { TrendingUp, X, Plus, Play, Loader2, Wifi } from 'lucide-react'

const PRESET_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']

export default function WorkflowRunner({ onRun, onStream, loading }) {
  const [symbols, setSymbols] = useState(['AAPL', 'MSFT'])
  const [input, setInput] = useState('')
  const [mode, setMode] = useState('stream') // 'stream' | 'batch'

  const addSymbol = (sym) => {
    const s = sym.trim().toUpperCase()
    if (s && !symbols.includes(s) && symbols.length < 6) {
      setSymbols([...symbols, s])
    }
    setInput('')
  }

  const removeSymbol = (sym) => setSymbols(symbols.filter(s => s !== sym))

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault()
      addSymbol(input)
    }
  }

  const handleRun = () => {
    if (!symbols.length || loading) return
    if (mode === 'stream') onStream(symbols)
    else onRun(symbols)
  }

  return (
    <div className="glass-card p-6 flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-brand-500/20 flex items-center justify-center">
          <TrendingUp size={18} className="text-brand-400" />
        </div>
        <div>
          <h2 className="font-semibold text-white text-sm">Run Analysis</h2>
          <p className="text-xs text-gray-500">Select stocks to analyze</p>
        </div>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-2 p-1 bg-gray-800/60 rounded-xl">
        {[
          { id: 'stream', label: 'Live Stream', icon: Wifi },
          { id: 'batch',  label: 'Batch',       icon: Play  },
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setMode(id)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium transition-all duration-200 ${
              mode === id
                ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/20'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Icon size={13} />
            {label}
          </button>
        ))}
      </div>

      {/* Symbol chips */}
      <div>
        <label className="text-xs text-gray-500 font-medium mb-2 block">Symbols</label>
        <div className="flex flex-wrap gap-2 mb-3">
          {symbols.map(sym => (
            <span
              key={sym}
              className="flex items-center gap-1.5 bg-brand-500/15 border border-brand-500/30 text-brand-300 text-xs font-mono font-semibold px-3 py-1.5 rounded-lg"
            >
              {sym}
              <button
                onClick={() => removeSymbol(sym)}
                className="text-brand-400/60 hover:text-red-400 transition-colors"
              >
                <X size={11} />
              </button>
            </span>
          ))}
        </div>

        {/* Input */}
        <div className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value.toUpperCase())}
            onKeyDown={handleKeyDown}
            placeholder="Add ticker…"
            maxLength={6}
            className="flex-1 bg-gray-800/60 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-brand-500/60 font-mono"
          />
          <button
            onClick={() => addSymbol(input)}
            disabled={!input.trim() || symbols.length >= 6}
            className="btn-secondary px-3"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>

      {/* Presets */}
      <div>
        <p className="text-xs text-gray-600 mb-2">Quick add</p>
        <div className="flex flex-wrap gap-1.5">
          {PRESET_SYMBOLS.filter(s => !symbols.includes(s)).map(sym => (
            <button
              key={sym}
              onClick={() => addSymbol(sym)}
              disabled={symbols.length >= 6}
              className="text-xs font-mono text-gray-500 hover:text-brand-400 border border-gray-800 hover:border-brand-500/40 px-2 py-1 rounded-md transition-all duration-150 disabled:opacity-30"
            >
              {sym}
            </button>
          ))}
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={handleRun}
        disabled={!symbols.length || loading}
        className="btn-primary flex items-center justify-center gap-2 w-full"
      >
        {loading ? (
          <>
            <Loader2 size={16} className="animate-spin" />
            Analyzing…
          </>
        ) : (
          <>
            {mode === 'stream' ? <Wifi size={16} /> : <Play size={16} />}
            {mode === 'stream' ? 'Stream Analysis' : 'Run Analysis'}
          </>
        )}
      </button>

      {/* Info */}
      <p className="text-xs text-gray-600 text-center leading-relaxed">
        Agent runs: <span className="text-gray-500">Research → Backtest → Reflect → Decide</span>
      </p>
    </div>
  )
}
