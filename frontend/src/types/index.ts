export interface FleetKPIs {
  health_index: number
  health_label: string
  at_risk_count: number
  at_risk_delta: number
  critical_alert_count: number
  predicted_failures_30d: number
  pm_jobs_adjusted_count: number
}

export interface AssetHealthItem {
  asset_id: string
  display_name: string
  capacity_kw: number
  health_score: number
  risk_level: RiskLevel
  top_risk_parameter: string | null
  predicted_issue: string | null
  predicted_eta_hours: number | null
  predicted_eta_label: string | null
  sparkline: number[]
}

export interface AssetHealthList {
  total: number
  page: number
  limit: number
  assets: AssetHealthItem[]
}

export interface ContributingFactor {
  parameter: string
  deviation: string
  weight: number
}

export interface LatestTelemetry {
  coolant_temp_f: number | null
  oil_pressure_psi: number | null
  rpm: number | null
  kw_output: number | null
  voltage_a: number | null
  frequency_hz: number | null
  power_factor: number | null
  runtime_hours: number | null
  fuel_type: string | null
  latitude: number | null
  longitude: number | null
  recorded_at: string | null
}

export interface Prediction {
  failure_probability: number
  predicted_failure_hours: number | null
  predicted_eta_label: string | null
  confidence_level: string
  predicted_failure_mode: string
  contributing_factors: ContributingFactor[]
}

export interface AssetDetail {
  asset_id: string
  display_name: string
  capacity_kw: number
  health_score: number
  risk_level: RiskLevel
  location_city: string
  location_state: string
  ai_risk_summary: string
  latest_telemetry: LatestTelemetry
  prediction: Prediction
}

export interface RecommendationItem {
  rec_id: string
  rec_type: string
  priority: number
  title: string
  description: string
  action_label: string
}

export interface PMJob {
  job_code: string
  job_name: string
  original_due_date: string
  adjusted_due_date: string
  adjustment_pct: number
  adjustment_reason: string
}

export interface AssetRecommendations {
  asset_id: string
  recommendations: RecommendationItem[]
  pm_job: PMJob | null
}

export type RiskLevel = 'Low' | 'Medium' | 'High' | 'Critical'

export type SortOption = 'risk_level' | 'health_score' | 'eta'
