from pydantic import BaseModel
from typing import Optional


class FleetKPIResponse(BaseModel):
    health_index: float
    health_label: str
    at_risk_count: int
    at_risk_delta: int
    critical_alert_count: int
    predicted_failures_30d: int
    pm_jobs_adjusted_count: int


class SparklinePoint(BaseModel):
    ts: int
    score: float


class AssetHealthItem(BaseModel):
    asset_id: str
    display_name: str
    capacity_kw: int
    health_score: int
    risk_level: str
    top_risk_parameter: Optional[str]
    predicted_issue: Optional[str]
    predicted_eta_hours: Optional[float]
    predicted_eta_label: Optional[str]
    sparkline: list[float]


class AssetHealthListResponse(BaseModel):
    total: int
    page: int
    limit: int
    assets: list[AssetHealthItem]


class ContributingFactor(BaseModel):
    parameter: str
    deviation: str
    weight: float


class LatestTelemetry(BaseModel):
    coolant_temp_f: Optional[float]
    oil_pressure_psi: Optional[float]
    rpm: Optional[float]
    kw_output: Optional[float]
    voltage_a: Optional[float]
    frequency_hz: Optional[float]
    power_factor: Optional[float]
    runtime_hours: Optional[float]
    fuel_type: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    recorded_at: Optional[str]


class PredictionDetail(BaseModel):
    failure_probability: float
    predicted_failure_hours: Optional[float]
    predicted_eta_label: Optional[str]
    confidence_level: str
    predicted_failure_mode: str
    contributing_factors: list[ContributingFactor]


class AssetDetailResponse(BaseModel):
    asset_id: str
    display_name: str
    capacity_kw: int
    health_score: int
    risk_level: str
    location_city: str
    location_state: str
    ai_risk_summary: str
    latest_telemetry: LatestTelemetry
    prediction: PredictionDetail


class RecommendationItem(BaseModel):
    rec_id: str
    rec_type: str
    priority: int
    title: str
    description: str
    action_label: str


class PMJob(BaseModel):
    job_code: str
    job_name: str
    original_due_date: str
    adjusted_due_date: str
    adjustment_pct: int
    adjustment_reason: str


class RecommendationsResponse(BaseModel):
    asset_id: str
    recommendations: list[RecommendationItem]
    pm_job: Optional[PMJob]


class ModelInfo(BaseModel):
    name: str
    accuracy: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    train_ms: Optional[int]
    is_active: bool


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    active_model: str
