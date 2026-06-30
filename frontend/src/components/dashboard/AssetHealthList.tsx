import { useState } from 'react'
import { Search, SlidersHorizontal, ChevronLeft, ChevronRight } from 'lucide-react'
import type { AssetHealthItem, RiskLevel, SortOption } from '../../types'
import { useAssetHealth } from '../../hooks/useAssetHealth'

const RISK_COLORS: Record<RiskLevel, string> = {
  Critical: 'bg-red-100 text-red-700 border-red-200',
  High: 'bg-orange-100 text-orange-700 border-orange-200',
  Medium: 'bg-amber-100 text-amber-700 border-amber-200',
  Low: 'bg-green-100 text-green-700 border-green-200',
}

const SCORE_COLOR = (s: number) =>
  s >= 75 ? 'text-green-600'
  : s >= 50 ? 'text-amber-600'
  : s >= 25 ? 'text-orange-600'
  : 'text-red-600'

interface Props {
  selectedId: string | null
  onSelect: (id: string) => void
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null
  const w = 52
  const h = 22
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const pts = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w
      const y = h - ((v - min) / range) * (h - 4) - 2
      return `${x},${y}`
    })
    .join(' ')

  const last = values[values.length - 1]
  const color = last >= 75 ? '#22c55e' : last >= 50 ? '#f59e0b' : last >= 25 ? '#f97316' : '#ef4444'

  return (
    <svg width={w} height={h} className="inline-block">
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
    </svg>
  )
}

export function AssetHealthList({ selectedId, onSelect }: Props) {
  const [sort, setSort] = useState<SortOption>('risk_level')
  const [riskFilter, setRiskFilter] = useState<RiskLevel | null>(null)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  const { data, loading } = useAssetHealth(sort, riskFilter, page)

  const filtered = data?.assets.filter(a =>
    search ? a.display_name.toLowerCase().includes(search.toLowerCase()) : true,
  ) ?? []

  const total = data?.total ?? 0
  const totalPages = Math.ceil(total / 5)
  const start = (page - 1) * 5 + 1
  const end = Math.min(page * 5, total)

  return (
    <div className="bg-white rounded-xl border border-slate-100 shadow-sm flex flex-col h-full">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b border-slate-100">
        <h2 className="font-semibold text-slate-800 mb-3">Asset Health</h2>
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(1) }}
              placeholder="Search assets..."
              className="w-full pl-8 pr-3 py-1.5 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            onClick={() => setShowFilters(v => !v)}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-sm border rounded-lg transition-colors ${
              showFilters ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 text-slate-600 hover:bg-slate-50'
            }`}
          >
            <SlidersHorizontal size={13} />
            Filters
          </button>
        </div>
        {showFilters && (
          <div className="mt-2 flex gap-2 flex-wrap">
            {/* Sort */}
            <select
              value={sort}
              onChange={e => { setSort(e.target.value as SortOption); setPage(1) }}
              className="text-xs border border-slate-200 rounded px-2 py-1 text-slate-600 focus:outline-none"
            >
              <option value="risk_level">Sort: Risk Level</option>
              <option value="health_score">Sort: Health Score</option>
              <option value="eta">Sort: ETA</option>
            </select>
            {/* Risk filter */}
            {(['Critical', 'High', 'Medium', 'Low'] as RiskLevel[]).map(r => (
              <button
                key={r}
                onClick={() => { setRiskFilter(riskFilter === r ? null : r); setPage(1) }}
                className={`text-xs px-2 py-1 rounded border font-medium transition-colors ${
                  riskFilter === r
                    ? RISK_COLORS[r]
                    : 'border-slate-200 text-slate-500 hover:bg-slate-50'
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-100">
              <th className="text-left px-4 py-2 font-semibold text-slate-500">Asset</th>
              <th className="text-left px-3 py-2 font-semibold text-slate-500">Health Score</th>
              <th className="text-left px-3 py-2 font-semibold text-slate-500">Risk Level</th>
              <th className="text-left px-3 py-2 font-semibold text-slate-500">Top Risk</th>
              <th className="text-left px-3 py-2 font-semibold text-slate-500">Predicted Issue</th>
              <th className="text-left px-3 py-2 font-semibold text-slate-500">ETA</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={6} className="text-center py-8 text-slate-400">Loading...</td>
              </tr>
            )}
            {!loading && filtered.map(asset => (
              <AssetRow
                key={asset.asset_id}
                asset={asset}
                selected={selectedId === asset.asset_id}
                onClick={() => onSelect(asset.asset_id)}
              />
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="px-4 py-3 border-t border-slate-100 flex items-center justify-between">
        <span className="text-xs text-slate-500">
          {total > 0 ? `${start}–${end} of ${total.toLocaleString()} assets` : '—'}
        </span>
        <div className="flex items-center gap-1">
          {[...Array(Math.min(5, totalPages))].map((_, i) => {
            const p = i + 1
            return (
              <button
                key={p}
                onClick={() => setPage(p)}
                className={`w-6 h-6 rounded text-xs font-medium transition-colors ${
                  page === p ? 'bg-blue-600 text-white' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {p}
              </button>
            )
          })}
          {totalPages > 5 && <span className="text-slate-400 text-xs">…</span>}
          <button
            onClick={() => setPage(p => Math.min(p + 1, totalPages))}
            disabled={page >= totalPages}
            className="ml-1 text-slate-400 hover:text-slate-700 disabled:opacity-30"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  )
}

function AssetRow({
  asset,
  selected,
  onClick,
}: {
  asset: AssetHealthItem
  selected: boolean
  onClick: () => void
}) {
  return (
    <tr
      onClick={onClick}
      className={`border-b border-slate-50 cursor-pointer transition-colors ${
        selected ? 'bg-blue-50 border-blue-100' : 'hover:bg-slate-50'
      }`}
    >
      <td className="px-4 py-2.5">
        <div className="flex items-center gap-2">
          <GeneratorIcon />
          <div>
            <div className="font-semibold text-slate-800">{asset.display_name}</div>
            <div className="text-slate-400">{asset.capacity_kw} kW</div>
          </div>
        </div>
      </td>
      <td className="px-3 py-2.5">
        <div className={`font-bold text-sm ${SCORE_COLOR(asset.health_score)}`}>
          {asset.health_score}
        </div>
        <Sparkline values={asset.sparkline} />
      </td>
      <td className="px-3 py-2.5">
        <span className={`px-2 py-0.5 rounded border text-[11px] font-semibold ${RISK_COLORS[asset.risk_level]}`}>
          {asset.risk_level}
        </span>
      </td>
      <td className="px-3 py-2.5 text-slate-600">{asset.top_risk_parameter ?? '—'}</td>
      <td className="px-3 py-2.5 text-slate-600">{asset.predicted_issue ?? '—'}</td>
      <td className="px-3 py-2.5 text-slate-600 whitespace-nowrap">
        {asset.predicted_eta_label ?? '—'}
      </td>
    </tr>
  )
}

function GeneratorIcon() {
  return (
    <div className="w-8 h-8 rounded bg-slate-100 flex items-center justify-center flex-shrink-0">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="1.5">
        <rect x="2" y="7" width="20" height="14" rx="2" />
        <path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2" />
        <line x1="12" y1="12" x2="12" y2="16" />
        <line x1="10" y1="14" x2="14" y2="14" />
      </svg>
    </div>
  )
}
