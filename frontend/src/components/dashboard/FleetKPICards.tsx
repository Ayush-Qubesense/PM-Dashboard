import { TrendingUp } from 'lucide-react'
import type { FleetKPIs } from '../../types'
import { FleetHealthGauge } from './FleetHealthGauge'

interface Props {
  kpis: FleetKPIs
}

export function FleetKPICards({ kpis }: Props) {
  return (
    <div className="grid grid-cols-5 gap-4 mb-4">
      {/* Fleet Health Index */}
      <div className="bg-white rounded-xl border border-slate-100 shadow-sm p-4 flex flex-col items-center">
        <span className="text-xs font-semibold text-slate-500 mb-1 self-start">Fleet Health Index</span>
        <FleetHealthGauge score={Math.round(kpis.health_index)} label={kpis.health_label} />
      </div>

      {/* At Risk Assets */}
      <KPICard
        title="At Risk Assets"
        value={kpis.at_risk_count.toLocaleString()}
        sub={
          <span className="flex items-center gap-1 text-amber-600 text-xs">
            <TrendingUp size={12} />
            {kpis.at_risk_delta} vs last 7 days
          </span>
        }
      />

      {/* Critical Alerts */}
      <KPICard
        title="Critical Alerts"
        value={kpis.critical_alert_count.toLocaleString()}
        sub={<span className="text-red-500 text-xs font-medium">Requires attention</span>}
      />

      {/* Predicted Failures */}
      <KPICard
        title="Predicted Failures (30d)"
        value={kpis.predicted_failures_30d.toLocaleString()}
        sub={<span className="text-amber-600 text-xs font-medium">High confidence</span>}
      />

      {/* PM Jobs */}
      <KPICard
        title="PM Jobs Adjusted"
        value={kpis.pm_jobs_adjusted_count.toLocaleString()}
        sub={<span className="text-blue-600 text-xs font-medium">This week</span>}
      />
    </div>
  )
}

function KPICard({
  title,
  value,
  sub,
}: {
  title: string
  value: string
  sub: React.ReactNode
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-100 shadow-sm p-4">
      <p className="text-xs font-semibold text-slate-500 mb-2">{title}</p>
      <p className="text-3xl font-bold text-slate-800 mb-1">{value}</p>
      {sub}
    </div>
  )
}
