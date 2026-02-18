import { useState, useEffect, useRef, useCallback } from 'react'
import WorkflowRunner from '../components/WorkflowRunner'
import ResultsDisplay from '../components/ResultsDisplay'
import { Brain, Activity, Github, ExternalLink, ChevronDown, ChevronUp } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_BASE  = API_BASE.replace(/^http/, 'ws')

// ─── Live log panel ──────────────────────────────────────────────────────────

function LiveLog({ logs, loading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  if (!logs.length && !loading) return null

  return (
    <div className="glass-card p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="relative flex h-2 w-2">
          {loading && (
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75" />
          )}
          <span className={`relative inline-flex rounded-full h-2 w-2 ${loading ? 'bg-brand-500' : 'bg-gray-600'}`} />
        </span>
        <span className="text-xs font-medium text-gray-400">
          {loading ? 'Agent running…' : 'Completed'}
        </span>
      </div>
      <div className="max-h-52 overflow-y-auto space-y-0.5 pr-1">
        {logs.map((log, i) => (
          <div key={i} className={`log-line ${log.type || 'status'}`}>
            {log.type === 'step' && (
              <span className="text-gray-600 mr-2">[{log.node}]</span>
            )}
            {log.message}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

// ─── Architecture diagram ────────────────────────────────────────────────────

function ArchDiagram() {
  const [open, setOpen] = useState(false)
  const nodes = [
    { label: 'React UI',         color: 'bg-blue-500/20 border-blue-500/40 text-blue-300' },
    { label: 'FastAPI',          color: 'bg-purple-500/20 border-purple-500/40 text-purple-300' },
    { label: 'LangGraph Agent',  color: 'bg-brand-500/20 border-brand-500/40 text-brand-300' },
    { label: 'MCP Servers',      color: 'bg-yellow-500/20 border-yellow-500/40 text-yellow-300' },
    { label: 'Qdrant VectorDB',  color: 'bg-pink-500/20 border-pink-500/40 text-pink-300' },
    { label: 'C++ Backtester',   color: 'bg-orange-500/20 border-orange-500/40 text-orange-300' },
  ]

  return (
    <div className="glass-card overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between p-4 text-left hover:bg-white/5 transition-colors"
      >
        <span className="text-sm font-semibold text-white flex items-center gap-2">
          <Brain size={15} className="text-brand-400" />
          System Architecture
        </span>
        {open ? <ChevronUp size={15} className="text-gray-500" /> : <ChevronDown size={15} className="text-gray-500" />}
      </button>

      {open && (
        <div className="px-4 pb-4 animate-fade-in">
          <div className="flex flex-wrap gap-2 mb-3">
            {nodes.map(n => (
              <span key={n.label} className={`text-xs border px-3 py-1.5 rounded-lg font-medium ${n.color}`}>
                {n.label}
              </span>
            ))}
          </div>
          <div className="text-xs text-gray-600 font-mono bg-gray-900/60 rounded-xl p-3 leading-relaxed">
            UI → FastAPI → LangGraph<br />
            &nbsp;&nbsp;├─ research  → MCP News Server → NewsAPI<br />
            &nbsp;&nbsp;├─ backtest  → MCP Strategy Server → C++ Engine<br />
            &nbsp;&nbsp;├─ reflect   → Claude 3.5 Sonnet<br />
            &nbsp;&nbsp;└─ decide    → Claude 3.5 Sonnet<br />
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br />
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Qdrant (10-K RAG)
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Dashboard ───────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [analysis, setAnalysis]   = useState(null)
  const [loading,  setLoading]    = useState(false)
  const [logs,     setLogs]       = useState([])
  const wsRef = useRef(null)

  const addLog = useCallback((msg, type = 'status') => {
    setLogs(prev => [...prev, { message: msg, type }])
  }, [])

  // Batch REST call
  const handleRun = useCallback(async (symbols) => {
    setLoading(true)
    setAnalysis(null)
    setLogs([])
    addLog(`Starting batch analysis for ${symbols.join(', ')}…`)

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setAnalysis(data)
      addLog('Analysis complete.', 'decide')
    } catch (err) {
      addLog(`Error: ${err.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [addLog])

  // WebSocket streaming
  const handleStream = useCallback((symbols) => {
    if (wsRef.current) wsRef.current.close()

    setLoading(true)
    setAnalysis(null)
    setLogs([])

    const ws = new WebSocket(`${WS_BASE}/ws/analysis`)
    wsRef.current = ws

    ws.onopen = () => {
      addLog(`Connecting to agent for ${symbols.join(', ')}…`)
      ws.send(JSON.stringify({ symbols }))
    }

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)

      if (msg.type === 'status') {
        addLog(msg.message, 'status')
      } else if (msg.type === 'step') {
        const node = msg.node
        const typeMap = { research: 'research', backtest: 'backtest', reflect: 'reflect', decide: 'decide' }
        addLog(`${node} node completed`, typeMap[node] || 'status', node)

        // Accumulate state from streaming steps
        setAnalysis(prev => {
          const base = prev || {
            status: 'streaming',
            symbols,
            decision: '',
            confidence: 0,
            sentiment: {},
            backtest_results: {},
            reflection: '',
            steps: 0,
          }
          const d = msg.data || {}
          return {
            ...base,
            sentiment:       d.sentiment_scores  ?? base.sentiment,
            backtest_results:d.backtest_results  ?? base.backtest_results,
            reflection:      d.reflection        ?? base.reflection,
            decision:        d.decision          ?? base.decision,
            confidence:      d.confidence        ?? base.confidence,
            steps:           (d.step ?? base.steps),
          }
        })
      } else if (msg.type === 'complete') {
        addLog('Analysis complete.', 'decide')
        setLoading(false)
        setAnalysis(prev => prev ? { ...prev, status: 'success' } : prev)
      } else if (msg.type === 'error') {
        addLog(`Error: ${msg.message}`, 'error')
        setLoading(false)
      }
    }

    ws.onerror = () => {
      addLog('WebSocket connection failed. Is the backend running?', 'error')
      setLoading(false)
    }

    ws.onclose = () => {
      if (loading) setLoading(false)
    }
  }, [addLog, loading])

  // Cleanup on unmount
  useEffect(() => () => wsRef.current?.close(), [])

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-gray-800/60 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center shadow-lg shadow-brand-500/30">
              <Activity size={16} className="text-white" />
            </div>
            <div>
              <h1 className="font-bold text-white text-base leading-none">FinAgent</h1>
              <p className="text-xs text-gray-500 leading-none mt-0.5">Autonomous Financial Research</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              <ExternalLink size={13} />
              API Docs
            </a>
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              <Github size={13} />
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Hero */}
      <div className="max-w-7xl mx-auto px-6 pt-10 pb-6">
        <div className="mb-8">
          <div className="inline-flex items-center gap-2 bg-brand-500/10 border border-brand-500/20 text-brand-400 text-xs font-medium px-3 py-1.5 rounded-full mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse-slow" />
            LangGraph · MCP · Qdrant · Claude 3.5 Sonnet
          </div>
          <h2 className="text-3xl font-bold text-white mb-2">
            Autonomous Market Analysis
          </h2>
          <p className="text-gray-400 max-w-xl">
            AI agent that researches news, backtests strategies, reflects on decisions,
            and delivers confident trading signals — in seconds.
          </p>
        </div>

        {/* Main grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column: controls */}
          <div className="lg:col-span-1 flex flex-col gap-4">
            <WorkflowRunner
              onRun={handleRun}
              onStream={handleStream}
              loading={loading}
            />
            <ArchDiagram />
          </div>

          {/* Right column: results */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            <LiveLog logs={logs} loading={loading} />

            {!analysis && !loading && (
              <div className="glass-card p-12 flex flex-col items-center justify-center text-center gap-4">
                <div className="w-16 h-16 rounded-2xl bg-gray-800/60 flex items-center justify-center">
                  <Brain size={28} className="text-gray-600" />
                </div>
                <div>
                  <p className="text-gray-400 font-medium mb-1">Ready to analyze</p>
                  <p className="text-sm text-gray-600">
                    Select symbols and run the agent to see results
                  </p>
                </div>
              </div>
            )}

            {analysis && <ResultsDisplay data={analysis} />}
          </div>
        </div>
      </div>
    </div>
  )
}
