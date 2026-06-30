import { useState } from 'react'
import { Wrench, Calendar, Eye, CheckCircle, ChevronRight } from 'lucide-react'
import type { AssetRecommendations } from '../../types'
import { api } from '../../api/client'

const TYPE_ICON: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  CREATE_WO: Wrench,
  ADJUST_PM: Calendar,
  ADD_TASK: Eye,
  MONITOR: Eye,
}

const BTN_COLOR: Record<string, string> = {
  CREATE_WO: 'bg-red-600 hover:bg-red-700 text-white',
  ADJUST_PM: 'bg-blue-600 hover:bg-blue-700 text-white',
  ADD_TASK: 'bg-slate-600 hover:bg-slate-700 text-white',
  MONITOR: 'bg-slate-500 hover:bg-slate-600 text-white',
}

interface Props {
  recs: AssetRecommendations | null
  loading: boolean
}

export function AIRecommendations({ recs, loading }: Props) {
  const [accepted, setAccepted] = useState<Set<string>>(new Set())
  const [toast, setToast] = useState<string | null>(null)

  const handleAccept = async (recId: string, title: string) => {
    await api.acceptRecommendation(recId)
    setAccepted(prev => new Set(prev).add(recId))
    setToast(`"${title}" submitted`)
    setTimeout(() => setToast(null), 3000)
  }

  return (
    <div className="bg-white rounded-xl border border-slate-100 shadow-sm flex flex-col">
      <div className="px-4 pt-4 pb-3 border-b border-slate-100">
        <h2 className="font-semibold text-slate-800">AI Recommended Actions</h2>
      </div>

      {/* Toast */}
      {toast && (
        <div className="mx-4 mt-3 flex items-center gap-2 bg-green-50 border border-green-200 text-green-700 text-xs px-3 py-2 rounded-lg">
          <CheckCircle size={14} />
          {toast}
        </div>
      )}

      <div className="p-4 space-y-3 flex-1">
        {loading && (
          <p className="text-xs text-slate-400 text-center py-4">Loading recommendations...</p>
        )}
        {!loading && !recs && (
          <p className="text-xs text-slate-400 text-center py-4">Select an asset to view recommendations</p>
        )}
        {!loading && recs && recs.recommendations.map(rec => {
          const Icon = TYPE_ICON[rec.rec_type] ?? Eye
          const done = accepted.has(rec.rec_id)
          return (
            <div
              key={rec.rec_id}
              className={`flex items-center gap-3 p-3 rounded-lg border transition-all ${
                done ? 'border-green-200 bg-green-50 opacity-70' : 'border-slate-100 bg-slate-50'
              }`}
            >
              <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${
                done ? 'bg-green-100' : 'bg-blue-100'
              }`}>
                {done ? <CheckCircle size={16} className="text-green-600" /> : <Icon size={16} className="text-blue-600" />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold text-slate-800">{rec.title}</div>
                <div className="text-xs text-slate-500 truncate">{rec.description}</div>
              </div>
              {!done ? (
                <button
                  onClick={() => handleAccept(rec.rec_id, rec.title)}
                  className={`text-xs px-2.5 py-1.5 rounded-lg font-semibold flex-shrink-0 transition-colors ${BTN_COLOR[rec.rec_type] ?? BTN_COLOR.MONITOR}`}
                >
                  {rec.action_label}
                </button>
              ) : (
                <span className="text-xs text-green-600 font-medium flex-shrink-0">Done</span>
              )}
            </div>
          )
        })}
      </div>

      {recs && (
        <button className="flex items-center justify-center gap-1 text-xs text-blue-600 hover:text-blue-800 py-3 border-t border-slate-100 transition-colors">
          View all recommendations <ChevronRight size={12} />
        </button>
      )}
    </div>
  )
}
