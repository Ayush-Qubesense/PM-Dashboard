import { CalendarClock, AlertTriangle } from 'lucide-react'
import type { PMJob } from '../../types'

interface Props {
  pmJob: PMJob | null
  loading: boolean
}

export function PMJobOptimization({ pmJob, loading }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-100 shadow-sm">
      <div className="px-4 pt-4 pb-3 border-b border-slate-100">
        <h2 className="font-semibold text-slate-800">PM Job Optimization</h2>
      </div>

      <div className="p-4">
        {loading && (
          <p className="text-xs text-slate-400 text-center py-3">Loading...</p>
        )}
        {!loading && !pmJob && (
          <div className="flex flex-col items-center gap-2 py-4">
            <CalendarClock size={28} className="text-slate-300" />
            <p className="text-xs text-slate-400">No PM adjustments for selected asset</p>
          </div>
        )}
        {!loading && pmJob && (
          <div>
            {/* Job header */}
            <div className="flex items-center gap-2 mb-3">
              <CalendarClock size={14} className="text-blue-600" />
              <span className="text-sm font-semibold text-slate-800">
                {pmJob.job_code} – {pmJob.job_name}
              </span>
            </div>

            {/* Date comparison */}
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div>
                <p className="text-xs text-slate-500 mb-1">Original Due</p>
                <p className="text-sm font-semibold text-slate-700">{pmJob.original_due_date}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500 mb-1">Adjusted Due</p>
                <p className="text-sm font-semibold text-amber-600">{pmJob.adjusted_due_date}</p>
              </div>
            </div>

            {/* Adjustment badge */}
            <div className="flex items-center gap-2 mb-3">
              <span className="flex items-center gap-1 text-xs bg-amber-50 border border-amber-200 text-amber-700 px-2 py-1 rounded-lg font-medium">
                <AlertTriangle size={11} />
                {pmJob.adjustment_pct}% shorter
              </span>
              <span className="text-xs text-slate-500">{pmJob.adjustment_reason}</span>
            </div>

            {/* Scope bars */}
            <div className="space-y-2">
              <div>
                <div className="flex justify-between text-xs text-slate-500 mb-1">
                  <span>Original Scope</span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-400 rounded-full" style={{ width: '100%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs text-slate-500 mb-1">
                  <span>Adjusted Scope</span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-amber-400 rounded-full"
                    style={{ width: `${100 - pmJob.adjustment_pct}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
