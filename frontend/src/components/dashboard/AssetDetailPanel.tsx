import type { AssetDetail, RiskLevel } from '../../types'
import { MapPin, Clock, RefreshCw } from 'lucide-react'
import generatorImg from '/generator.png'

const RISK_BADGE: Record<RiskLevel, string> = {
  Critical: 'bg-red-600 text-white',
  High: 'bg-orange-500 text-white',
  Medium: 'bg-amber-500 text-white',
  Low: 'bg-green-500 text-white',
}

const PROB_COLOR = (fp: number) =>
  fp >= 0.8 ? 'bg-red-500'
  : fp >= 0.6 ? 'bg-orange-500'
  : fp >= 0.4 ? 'bg-amber-500'
  : 'bg-green-500'

function timeAgo(isoStr: string | null) {
  if (!isoStr) return 'Unknown'
  const diff = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000 / 60)
  if (diff < 1) return 'Just now'
  if (diff < 60) return `${diff} min ago`
  if (diff < 1440) return `${Math.floor(diff / 60)} hr ago`
  return `${Math.floor(diff / 1440)}d ago`
}

interface Props {
  detail: AssetDetail | null
  loading: boolean
}

export function AssetDetailPanel({ detail, loading }: Props) {
  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-100 shadow-sm flex items-center justify-center h-full">
        <div className="text-slate-400 text-sm">Loading asset details...</div>
      </div>
    )
  }

  if (!detail) {
    return (
      <div className="bg-white rounded-xl border border-slate-100 shadow-sm flex flex-col items-center justify-center h-full gap-3">
        <img src={generatorImg} alt="Generator" style={{ width: 64, opacity: 0.2 }} />
        <p className="text-slate-400 text-sm">Select an asset to view details</p>
      </div>
    )
  }

  const { prediction: pred, latest_telemetry: tel } = detail
  const fp = pred.failure_probability

  return (
    <div className="bg-white rounded-xl border border-slate-100 shadow-sm flex flex-col h-full overflow-auto">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-bold text-slate-800 text-base">{detail.display_name}</span>
          <span className={`text-xs px-2 py-0.5 rounded font-semibold ${RISK_BADGE[detail.risk_level]}`}>
            {detail.risk_level} Risk
          </span>
        </div>
      </div>

      {/* Generator image */}
      <div className="flex justify-center pt-4 pb-2">
        <img src={generatorImg} alt="Generator" style={{ width: 120 }} />
      </div>

      {/* Meta row */}
      <div className="grid grid-cols-3 gap-2 px-4 pb-3 text-xs text-slate-500 border-b border-slate-100">
        <div className="flex items-center gap-1">
          <MapPin size={12} />
          <span>{detail.location_city}, {detail.location_state}</span>
        </div>
        <div className="flex items-center gap-1">
          <Clock size={12} />
          <span>{tel.runtime_hours ? `${Math.round(tel.runtime_hours).toLocaleString()} hrs` : '—'}</span>
        </div>
        <div className="flex items-center gap-1">
          <RefreshCw size={12} />
          <span>{timeAgo(tel.recorded_at)}</span>
        </div>
      </div>

      {/* AI Risk Summary */}
      <div className="px-4 py-3 border-b border-slate-100">
        <h3 className="text-xs font-semibold text-slate-700 mb-1">AI Risk Summary</h3>
        <p className="text-xs text-slate-600 leading-relaxed">{detail.ai_risk_summary}</p>
      </div>

      {/* Metrics */}
      <div className="px-4 py-3 border-b border-slate-100 space-y-3">
        {/* Failure Probability */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-600 font-medium">Failure Probability</span>
            <span className={`font-bold ${fp >= 0.8 ? 'text-red-600' : fp >= 0.5 ? 'text-amber-600' : 'text-green-600'}`}>
              {Math.round(fp * 100)}%
            </span>
          </div>
          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${PROB_COLOR(fp)}`}
              style={{ width: `${Math.round(fp * 100)}%` }}
            />
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-slate-500">Predicted Time to Failure</span>
            <div className="font-semibold text-slate-800 mt-0.5">
              {pred.predicted_eta_label ?? 'No imminent failure'}
            </div>
          </div>
          <div>
            <span className="text-slate-500">Confidence</span>
            <div className={`font-semibold mt-0.5 ${
              pred.confidence_level === 'High' ? 'text-red-600'
              : pred.confidence_level === 'Medium' ? 'text-amber-600'
              : 'text-green-600'
            }`}>
              {pred.confidence_level}
            </div>
          </div>
        </div>
      </div>

      {/* Contributing Factors */}
      {pred.contributing_factors.length > 0 && (
        <div className="px-4 py-3 border-b border-slate-100">
          <h3 className="text-xs font-semibold text-slate-700 mb-2">Contributing Factors</h3>
          <ul className="space-y-1.5">
            {pred.contributing_factors.map((f, i) => (
              <li key={i} className="flex items-start gap-2 text-xs">
                <span className="mt-0.5 w-1.5 h-1.5 rounded-full bg-red-400 flex-shrink-0" />
                <span className="text-slate-600">
                  <span className="font-medium text-slate-700">{f.parameter}</span>
                  {' — '}
                  {f.deviation}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Live telemetry strip */}
      <div className="px-4 py-3">
        <h3 className="text-xs font-semibold text-slate-700 mb-2">Live Telemetry</h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
          <TelRow label="Coolant Temp" value={tel.coolant_temp_f ? `${tel.coolant_temp_f.toFixed(1)}°F` : null} />
          <TelRow label="Oil Pressure" value={tel.oil_pressure_psi ? `${tel.oil_pressure_psi.toFixed(1)} PSI` : null} />
          <TelRow label="Engine RPM" value={tel.rpm ? Math.round(tel.rpm).toLocaleString() : null} />
          <TelRow label="kW Output" value={tel.kw_output ? `${tel.kw_output.toFixed(0)} kW` : null} />
          <TelRow label="Voltage A" value={tel.voltage_a ? `${tel.voltage_a.toFixed(1)} V` : null} />
          <TelRow label="Frequency" value={tel.frequency_hz ? `${tel.frequency_hz.toFixed(2)} Hz` : null} />
          <TelRow label="Power Factor" value={tel.power_factor ? tel.power_factor.toFixed(3) : null} />
          <TelRow label="Fuel Type" value={tel.fuel_type} />
        </div>
      </div>
    </div>
  )
}

function TelRow({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="font-medium text-slate-700">{value ?? '—'}</span>
    </div>
  )
}
