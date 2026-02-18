import {
  TrendingUp, TrendingDown, Minus,
  BarChart2, Activity, ShieldCheck, Zap
} from 'lucide-react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  Tooltip, Cell
} from 'recharts'

// ─── helpers ────────────────────────────────────────────────────────────────

function parseDecision(text = '') {
  const lines = text.split('\n').filter(Boolean)
  return lines.map(line => {
    const m = line.match(/^([A-Z]+):\s*(BUY|SELL|HOLD)/i)
    if (!m) return null
    const confMatch = line.match(/confidence:\s*(\d+)%/i)
    return {
      symbol: m[1],
      action: m[2].toUpperCase(),
      confidence: confMatch ? parseInt(confMatch[1]) : 70,
      rationale: line.replace(/^[^-]+-\s*/, '').trim(),
    }
  }).filter(Boolean)
}

function actionColor(action) {
  if (action === 'BUY')  return { text: 'text-emerald-400', bg: 'bg-emerald-500/15 border-emerald-500/30' }
  if (action === 'SELL') return { text: 'text-red-400',     bg: 'bg-red-500/15 border-red-500/30' }
  return                        { text: 'text-yellow-400',  bg: 'bg-yellow-500/15 border-yellow-500/30' }
}

function ActionIcon({ action }) {
  if (action === 'BUY')  return <TrendingUp  size={16} className="text-emerald-400" />
  if (action === 'SELL') return <TrendingDown size={16} className="text-red-400" />
  return                        <Minus        size={16} className="text-yellow-400" />
}

const CHART_COLORS = ['#25a36e', '#818cf8', '#f59e0b', '#f87171', '#34d399']

// ─── sub-components ─────────────────────────────────────────────────────────

function MetricCard({ label, value, sub, icon: Icon, color = 'text-brand-400' }) {
  return (
    <div className="metric-card">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-500">{label}</span>
        {Icon && <Icon size={14} className={color} />}
      </div>
      <span className={`text-xl font-bold ${color}`}>{value}</span>
      {sub && <span className="text-xs text-gray-600">{sub}</span>}
    </div>
  )
}

function DecisionCard({ item }) {
  const { text, bg } = actionColor(item.action)
  return (
    <div className="glass-card p-4 flex flex-col gap-2 animate-slide-up">
      <div className="flex items-center justify-between">
        <span className="font-mono font-bold text-white text-lg">{item.symbol}</span>
        <span className={`flex items-center gap-1.5 border text-xs font-bold px-3 py-1 rounded-full ${bg} ${text}`}>
          <ActionIcon action={item.action} />
          {item.action}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-brand-600 to-brand-400 transition-all duration-700"
            style={{ width: `${item.confidence}%` }}
          />
        </div>
        <span className="text-xs text-gray-400 font-mono">{item.confidence}%</span>
      </div>
      {item.rationale && (
        <p className="text-xs text-gray-500 leading-relaxed">{item.rationale}</p>
      )}
    </div>
  )
}

function SentimentSection({ sentiment }) {
  const entries = Object.entries(sentiment)
  if (!entries.length) return null

  const radarData = entries.map(([sym, s]) => ({
    symbol: sym,
    positive: Math.round((s.positive_ratio ?? 0.5) * 100),
    articles: s.total_articles ?? 0,
  }))

  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
        <Activity size={15} className="text-brand-400" />
        Sentiment Analysis
      </h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        {entries.map(([sym, s]) => (
          <div key={sym} className="bg-gray-800/50 rounded-xl p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-xs font-bold text-white">{sym}</span>
              <span className={`text-xs font-medium ${
                (s.positive_ratio ?? 0.5) > 0.5 ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {Math.round((s.positive_ratio ?? 0.5) * 100)}% pos
              </span>
            </div>
            <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-emerald-500 rounded-full"
                style={{ width: `${(s.positive_ratio ?? 0.5) * 100}%` }}
              />
            </div>
            <p className="text-xs text-gray-600 mt-1">{s.total_articles ?? 0} articles</p>
          </div>
        ))}
      </div>
      {radarData.length > 1 && (
        <ResponsiveContainer width="100%" height={160}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#374151" />
            <PolarAngleAxis dataKey="symbol" tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <Radar dataKey="positive" stroke="#25a36e" fill="#25a36e" fillOpacity={0.2} />
          </RadarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

function BacktestSection({ backtest }) {
  const entries = Object.entries(backtest)
  if (!entries.length) return null

  const chartData = entries.map(([sym, b]) => ({
    symbol: sym,
    return: Math.round((b.total_return ?? 0) * 100),
    sharpe: parseFloat((b.sharpe_ratio ?? 0).toFixed(2)),
    winRate: Math.round((b.win_rate ?? 0) * 100),
  }))

  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
        <BarChart2 size={15} className="text-purple-400" />
        Backtest Results
      </h3>

      {/* Table */}
      <div className="overflow-x-auto mb-4">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-800">
              <th className="text-left pb-2 font-medium">Symbol</th>
              <th className="text-right pb-2 font-medium">Return</th>
              <th className="text-right pb-2 font-medium">Sharpe</th>
              <th className="text-right pb-2 font-medium">Win%</th>
              <th className="text-right pb-2 font-medium">Trades</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([sym, b]) => (
              <tr key={sym} className="border-b border-gray-800/50">
                <td className="py-2 font-mono font-bold text-white">{sym}</td>
                <td className={`py-2 text-right font-mono ${
                  (b.total_return ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {((b.total_return ?? 0) * 100).toFixed(1)}%
                </td>
                <td className="py-2 text-right text-gray-300 font-mono">
                  {(b.sharpe_ratio ?? 0).toFixed(2)}
                </td>
                <td className="py-2 text-right text-gray-300 font-mono">
                  {Math.round((b.win_rate ?? 0) * 100)}%
                </td>
                <td className="py-2 text-right text-gray-500 font-mono">
                  {b.total_trades ?? 0}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Bar chart */}
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={chartData} barSize={20}>
          <XAxis dataKey="symbol" tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} unit="%" />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#f9fafb' }}
          />
          <Bar dataKey="return" name="Return %" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={entry.return >= 0 ? '#25a36e' : '#f87171'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── main export ─────────────────────────────────────────────────────────────

export default function ResultsDisplay({ data }) {
  if (!data) return null

  const decisions = parseDecision(data.decision ?? '')
  const confidence = Math.round((data.confidence ?? 0) * 100)

  return (
    <div className="flex flex-col gap-5 animate-fade-in">

      {/* Top metrics row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="Confidence"
          value={`${confidence}%`}
          icon={ShieldCheck}
          color={confidence >= 70 ? 'text-brand-400' : 'text-yellow-400'}
        />
        <MetricCard
          label="Agent Steps"
          value={data.steps ?? '—'}
          sub="nodes executed"
          icon={Zap}
          color="text-purple-400"
        />
        <MetricCard
          label="Symbols"
          value={data.symbols?.length ?? '—'}
          sub="analyzed"
          icon={BarChart2}
          color="text-blue-400"
        />
        <MetricCard
          label="Status"
          value={data.status === 'success' ? 'Done' : 'Error'}
          icon={Activity}
          color={data.status === 'success' ? 'text-emerald-400' : 'text-red-400'}
        />
      </div>

      {/* Decisions */}
      {decisions.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <TrendingUp size={15} className="text-brand-400" />
            Trading Decisions
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {decisions.map((d, i) => <DecisionCard key={i} item={d} />)}
          </div>
        </div>
      )}

      {/* Raw decision text if no parsed decisions */}
      {decisions.length === 0 && data.decision && (
        <div className="glass-card p-5">
          <h3 className="text-sm font-semibold text-white mb-3">Decision</h3>
          <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">{data.decision}</p>
        </div>
      )}

      {/* Sentiment */}
      {data.sentiment && Object.keys(data.sentiment).length > 0 && (
        <SentimentSection sentiment={data.sentiment} />
      )}

      {/* Backtest */}
      {data.backtest_results && Object.keys(data.backtest_results).length > 0 && (
        <BacktestSection backtest={data.backtest_results} />
      )}

      {/* Reflection */}
      {data.reflection && (
        <div className="glass-card p-5">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <ShieldCheck size={15} className="text-yellow-400" />
            Agent Reflection
          </h3>
          <p className="text-sm text-gray-400 leading-relaxed whitespace-pre-wrap">{data.reflection}</p>
        </div>
      )}
    </div>
  )
}
