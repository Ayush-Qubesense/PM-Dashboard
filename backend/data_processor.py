import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import ml_engine


CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(os.path.dirname(__file__), "MaintenanceTelematicsData_synthetic_10k.csv"),
)

_asset_name_map: dict[str, str] = {}
_asset_df: pd.DataFrame = pd.DataFrame()

_SCENARIO_CONFIG: dict[str, dict] = {
    "normal": {
        "penalty": 0,
        "failure_mode": "None — Nominal Operation",
        "factors": [],
        "issue": None,
    },
    "overheating": {
        "penalty": 80,
        "failure_mode": "Overheat / Thermal Shutdown",
        "factors": [
            {"parameter": "Coolant Temperature", "deviation": "Above normal range — thermal stress", "weight": 0.50},
            {"parameter": "Coolant Pressure", "deviation": "Coolant pressure drop detected", "weight": 0.25},
            {"parameter": "Fan Speed", "deviation": "Lower than expected for load", "weight": 0.15},
            {"parameter": "Engine Load", "deviation": "High load for extended period", "weight": 0.10},
        ],
        "issue": "Overheat / Shutdown",
        "top_risk": "Coolant Temp",
    },
    "lubrication_failure": {
        "penalty": 80,
        "failure_mode": "Lubrication / Oil System Failure",
        "factors": [
            {"parameter": "Oil Pressure", "deviation": "Critically low — possible pump failure", "weight": 0.55},
            {"parameter": "Oil Level", "deviation": "Below minimum operating level", "weight": 0.25},
            {"parameter": "Engine RPM", "deviation": "Irregular variation under load", "weight": 0.20},
        ],
        "issue": "Lubrication System",
        "top_risk": "Oil Pressure",
    },
    "complete_shutdown": {
        "penalty": 95,
        "failure_mode": "Complete System Shutdown",
        "factors": [
            {"parameter": "Engine RPM", "deviation": "Dropped to 0 — unscheduled shutdown", "weight": 0.60},
            {"parameter": "Oil Pressure", "deviation": "Loss of pressure at shutdown", "weight": 0.25},
            {"parameter": "Battery Voltage", "deviation": "Voltage below threshold", "weight": 0.15},
        ],
        "issue": "Unscheduled Shutdown",
        "top_risk": "Engine RPM",
    },
    "electrical_failure": {
        "penalty": 55,
        "failure_mode": "Electrical / Generator Fault",
        "factors": [
            {"parameter": "Generator Voltage", "deviation": "Voltage deviation >5% from nominal", "weight": 0.45},
            {"parameter": "Frequency (Hz)", "deviation": "Frequency instability detected", "weight": 0.30},
            {"parameter": "Power Factor", "deviation": "Power factor degraded below 0.90", "weight": 0.25},
        ],
        "issue": "Generator Output",
        "top_risk": "Generator Voltage",
    },
    "sensor_comm_fault": {
        "penalty": 30,
        "failure_mode": "Sensor / Communication Fault",
        "factors": [
            {"parameter": "Sensor Data", "deviation": "Missing or implausible readings", "weight": 0.50},
            {"parameter": "Communication Link", "deviation": "Intermittent modem connectivity", "weight": 0.30},
            {"parameter": "ECU Signal", "deviation": "CAN bus errors detected", "weight": 0.20},
        ],
        "issue": "Sensor / Comm",
        "top_risk": "Sensor Data",
    },
    "partial_load_failure": {
        "penalty": 40,
        "failure_mode": "Partial Load / Output Degradation",
        "factors": [
            {"parameter": "kW Output", "deviation": "Below rated capacity by >20%", "weight": 0.45},
            {"parameter": "Engine Load %", "deviation": "Load imbalance across phases", "weight": 0.35},
            {"parameter": "Exhaust Temperature", "deviation": "Elevated exhaust temp at partial load", "weight": 0.20},
        ],
        "issue": "Power Output",
        "top_risk": "Load %",
    },
    "cascading_failure": {
        "penalty": 90,
        "failure_mode": "Cascading Multi-System Failure",
        "factors": [
            {"parameter": "Coolant Temperature", "deviation": "Thermal anomaly — root trigger", "weight": 0.35},
            {"parameter": "Oil Pressure", "deviation": "Secondary oil pressure drop", "weight": 0.30},
            {"parameter": "Generator Voltage", "deviation": "Electrical instability during failure", "weight": 0.20},
            {"parameter": "Engine RPM", "deviation": "RPM instability prior to shutdown", "weight": 0.15},
        ],
        "issue": "Multiple Systems",
        "top_risk": "Coolant Temp",
    },
    "fuel_starvation": {
        "penalty": 60,
        "failure_mode": "Fuel Starvation / Supply Fault",
        "factors": [
            {"parameter": "Fuel Pressure", "deviation": "Inlet pressure below minimum", "weight": 0.50},
            {"parameter": "Fuel Level", "deviation": "Low fuel level at regulator", "weight": 0.30},
            {"parameter": "Engine RPM", "deviation": "RPM drop under load — fuel starved", "weight": 0.20},
        ],
        "issue": "Fuel Supply",
        "top_risk": "Fuel Press",
    },
    "cooling_system_partial": {
        "penalty": 45,
        "failure_mode": "Partial Cooling System Degradation",
        "factors": [
            {"parameter": "Coolant Temperature", "deviation": "Trending upward — partial cooling loss", "weight": 0.40},
            {"parameter": "Fan Speed", "deviation": "Fan running below optimal speed", "weight": 0.35},
            {"parameter": "Coolant Level", "deviation": "Slight coolant level deficit", "weight": 0.25},
        ],
        "issue": "Cooling System",
        "top_risk": "Coolant Temp",
    },
    "voltage_fluctuation": {
        "penalty": 40,
        "failure_mode": "Voltage Fluctuation / Electrical Instability",
        "factors": [
            {"parameter": "Generator Voltage", "deviation": "Voltage fluctuating ±8% from nominal", "weight": 0.50},
            {"parameter": "Battery Voltage", "deviation": "Charging voltage irregular", "weight": 0.30},
            {"parameter": "Power Factor", "deviation": "Reactive power spike detected", "weight": 0.20},
        ],
        "issue": "Voltage Stability",
        "top_risk": "Generator Voltage",
    },
}


def _geo_to_city(lat: float, _lng: float) -> tuple[str, str]:
    if lat < 31.5:
        return "Pecos", "TX"
    if lat < 32.0:
        return "Odessa", "TX"
    if lat < 32.5:
        return "Midland", "TX"
    if lat < 33.0:
        return "Big Spring", "TX"
    return "Lubbock", "TX"


def _to_float(val) -> float | None:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _to_int(val) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


def _risk_level(score: int) -> str:
    if score < 25:
        return "Critical"
    if score < 50:
        return "High"
    if score < 75:
        return "Medium"
    return "Low"


def _compute_health_score(row: pd.Series) -> tuple[int, list[dict], str, str, float | None]:
    """Returns (health_score, factors, risk_level, scenario, fp_override).

    fp_override is the ML model's failure probability when an ML model is active,
    or None for rule-based (caller should use _failure_probability(health_score) instead).
    """
    active = ml_engine.get_active_model()
    scenario = str(row.get("failure_scenario", "normal")).strip().lower()
    failure_label = _to_int(row.get("failure_label", 0))

    if active != "Rule-Based":
        asset_id = str(int(float(row.get("asset_id", 0))))
        pred = ml_engine.get_ml_prediction(asset_id, active)
        if pred:
            scenario = pred["predicted_scenario"]
            fp = pred["failure_probability"]
            # Severity-weighted health from the ML engine (expected penalty across
            # the full class distribution). Fall back to fp only for older caches.
            health_score = pred.get("health_score")
            if health_score is None:
                health_score = max(0, min(100, int(round(100.0 - fp * 100.0))))
            return health_score, pred["top5_features"], _risk_level(health_score), scenario, fp

    # Rule-Based fallback
    cfg = _SCENARIO_CONFIG.get(scenario, _SCENARIO_CONFIG["normal"])
    penalty = cfg["penalty"]
    if scenario == "normal" and failure_label == 1:
        penalty = 45

    score = max(0.0, min(100.0, 100.0 - penalty))
    health_score = int(round(score))
    factors = [dict(f) for f in cfg["factors"]]
    return health_score, factors, _risk_level(health_score), scenario, None


def _failure_probability(health_score: int) -> float:
    raw = (100 - health_score) / 100.0 * 1.1
    return round(min(0.98, max(0.01, raw)), 2)


def _eta(risk_level: str) -> tuple[float | None, str | None]:
    if risk_level == "Critical":
        return 23.0, "< 24 hours"
    if risk_level == "High":
        return 120.0, "< 7 days"
    if risk_level == "Medium":
        return 360.0, "14–21 days"
    return None, None


def _confidence(failure_label: int, scenario: str) -> str:
    if failure_label == 1 and scenario not in ("normal", "sensor_comm_fault"):
        return "High"
    if scenario != "normal":
        return "Medium"
    return "Low"


def _ai_summary(risk_level: str, scenario: str, factors: list[dict]) -> str:
    cfg = _SCENARIO_CONFIG.get(scenario, _SCENARIO_CONFIG["normal"])
    if risk_level == "Critical":
        mode = cfg["failure_mode"]
        if factors:
            dev = factors[0]["deviation"].lower()
            return f"High probability of shutdown due to {dev}. {mode} imminent — immediate intervention required."
        return f"Critical failure risk: {mode}. Immediate intervention required."
    if risk_level == "High":
        return f"Elevated failure risk: {cfg['failure_mode']}. Multiple parameters trending outside normal operating bounds. Preventive action recommended within 7 days."
    if risk_level == "Medium":
        return f"Moderate risk indicators present. {cfg['failure_mode']} developing. Schedule maintenance within the next 2–3 weeks."
    return "Asset operating within normal parameters. No immediate action required. Continue routine monitoring."


def _generate_sparkline(health_score: int, scenario: str, seed: int, n: int = 24) -> list[float]:
    rng = np.random.default_rng(seed)
    if scenario in ("complete_shutdown", "cascading_failure", "overheating", "lubrication_failure"):
        trend = np.linspace(min(100, health_score + 30), health_score, n)
    elif scenario == "normal" and health_score >= 75:
        trend = np.full(n, health_score, dtype=float)
    else:
        trend = np.linspace(min(100, health_score + 15), health_score, n)
    noise = rng.normal(0, 2.5, n)
    sparkline = np.clip(trend + noise, 0, 100).tolist()
    return [round(v, 1) for v in sparkline]


def load_data() -> None:
    global _asset_df, _asset_name_map
    path = os.path.abspath(CSV_PATH)
    df = pd.read_csv(path, low_memory=False)
    df["latest_measurement_timestamp"] = pd.to_datetime(
        df["latest_measurement_timestamp"], errors="coerce"
    )
    df = df.sort_values("latest_measurement_timestamp")
    _asset_df = df.reset_index(drop=True)

    sorted_ids = sorted(df["asset_id"].unique())
    for i, aid in enumerate(sorted_ids):
        _asset_name_map[str(int(aid))] = f"NGG-{1001 + i:04d}"

    print("Training ML models...")
    ml_engine.train_models(_asset_df)
    print("ML models ready.")


def _get_latest_rows() -> pd.DataFrame:
    return (
        _asset_df
        .sort_values("latest_measurement_timestamp")
        .drop_duplicates(subset="asset_id", keep="last")
        .reset_index(drop=True)
    )


def get_fleet_kpis() -> dict:
    latest = _get_latest_rows()
    scores, risk_levels = [], []
    for _, row in latest.iterrows():
        hs, _, rl, _, _ = _compute_health_score(row)
        scores.append(hs)
        risk_levels.append(rl)

    health_index = round(float(np.mean(scores)), 1) if scores else 0.0
    if health_index >= 75:
        label = "Good"
    elif health_index >= 50:
        label = "Fair"
    else:
        label = "Poor"

    at_risk = sum(1 for rl in risk_levels if rl in ("High", "Critical"))
    critical = sum(1 for rl in risk_levels if rl == "Critical")
    predicted_30d = sum(1 for hs in scores if _failure_probability(hs) > 0.4)
    pm_adjusted = sum(
        1 for _, row in latest.iterrows()
        if str(row.get("failure_scenario", "normal")).lower() not in ("normal", "sensor_comm_fault")
        and _to_int(row.get("failure_label", 0)) == 1
    )

    return {
        "health_index": health_index,
        "health_label": label,
        "at_risk_count": at_risk,
        "at_risk_delta": 2,
        "critical_alert_count": critical,
        "predicted_failures_30d": predicted_30d,
        "pm_jobs_adjusted_count": pm_adjusted,
    }


def get_asset_health_list(
    sort: str = "risk_level",
    risk_filter: str | None = None,
    page: int = 1,
    limit: int = 5,
) -> dict:
    latest = _get_latest_rows()
    items = []
    for _, row in latest.iterrows():
        asset_id = str(int(row["asset_id"]))
        hs, factors, rl, scenario, _ = _compute_health_score(row)
        eta_hours, eta_label = _eta(rl)
        sparkline = _generate_sparkline(hs, scenario, seed=int(row["asset_id"]))
        kw = int(round(_to_float(row.get("ekWattsAvg")) or 500))
        cfg = _SCENARIO_CONFIG.get(scenario, _SCENARIO_CONFIG["normal"])

        items.append({
            "asset_id": asset_id,
            "display_name": _asset_name_map.get(asset_id, f"NGG-{asset_id}"),
            "capacity_kw": kw,
            "health_score": hs,
            "risk_level": rl,
            "top_risk_parameter": cfg.get("top_risk") or (factors[0]["parameter"] if factors else None),
            "predicted_issue": cfg.get("issue"),
            "predicted_eta_hours": eta_hours,
            "predicted_eta_label": eta_label,
            "sparkline": sparkline,
        })

    risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    if sort == "risk_level":
        items.sort(key=lambda x: (risk_order.get(x["risk_level"], 4), x["health_score"]))
    elif sort == "health_score":
        items.sort(key=lambda x: x["health_score"])
    elif sort == "eta":
        items.sort(key=lambda x: (x["predicted_eta_hours"] is None, x["predicted_eta_hours"] or 9999))

    if risk_filter:
        items = [i for i in items if i["risk_level"] == risk_filter]

    total = len(items)
    start = (page - 1) * limit
    end = start + limit

    return {"total": total, "page": page, "limit": limit, "assets": items[start:end]}


def get_asset_detail(asset_id: str) -> dict | None:
    mask = _asset_df["asset_id"].astype(str).str.strip() == asset_id.strip()
    rows = _asset_df[mask]
    if rows.empty:
        try:
            mask2 = _asset_df["asset_id"] == int(asset_id)
            rows = _asset_df[mask2]
        except ValueError:
            pass
    if rows.empty:
        return None

    row = rows.iloc[0]
    hs, factors, rl, scenario, fp_override = _compute_health_score(row)
    failure_label = _to_int(row.get("failure_label", 0))
    fp = fp_override if fp_override is not None else _failure_probability(hs)
    eta_hours, eta_label = _eta(rl)
    conf = _confidence(failure_label, scenario)
    cfg = _SCENARIO_CONFIG.get(scenario, _SCENARIO_CONFIG["normal"])
    mode = cfg["failure_mode"]
    summary = _ai_summary(rl, scenario, factors)

    lat = _to_float(row.get("Latitude"))
    lng = _to_float(row.get("Longitude"))
    city, state = _geo_to_city(lat or 32.0, lng or -102.0)

    ts = row.get("latest_measurement_timestamp")
    recorded_at = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

    kw = int(round(_to_float(row.get("ekWattsAvg")) or 500))

    return {
        "asset_id": asset_id,
        "display_name": _asset_name_map.get(str(int(float(asset_id))), f"NGG-{asset_id}"),
        "capacity_kw": kw,
        "health_score": hs,
        "risk_level": rl,
        "location_city": city,
        "location_state": state,
        "ai_risk_summary": summary,
        "latest_telemetry": {
            "coolant_temp_f": _to_float(row.get("ECT")),
            "oil_pressure_psi": _to_float(row.get("OILP_press")),
            "rpm": _to_float(row.get("rpm")) or _to_float(row.get("RPM")),
            "kw_output": _to_float(row.get("ekWattsAvg")),
            "voltage_a": _to_float(row.get("eGenVoltage_A")),
            "frequency_hz": _to_float(row.get("eGenFrequency")),
            "power_factor": _to_float(row.get("ePFAvg")),
            "runtime_hours": _to_float(row.get("HM_hours")),
            "fuel_type": str(row.get("fuel_type", "Natural Gas")).strip()
                if str(row.get("fuel_type", "")).strip() not in ("nan", "None", "") else "Natural Gas",
            "latitude": lat,
            "longitude": lng,
            "recorded_at": recorded_at,
        },
        "prediction": {
            "failure_probability": fp,
            "predicted_failure_hours": eta_hours,
            "predicted_eta_label": eta_label,
            "confidence_level": conf,
            "predicted_failure_mode": mode,
            "contributing_factors": factors[:5],
        },
    }


def get_asset_recommendations(asset_id: str) -> dict | None:
    detail = get_asset_detail(asset_id)
    if not detail:
        return None

    fp = detail["prediction"]["failure_probability"]
    mode = detail["prediction"]["predicted_failure_mode"]
    rl = detail["risk_level"]
    factors = detail["prediction"]["contributing_factors"]
    recs = []
    priority = 1

    if fp > 0.85 and rl in ("Critical", "High"):
        recs.append({
            "rec_id": f"{asset_id}-wo-1",
            "rec_type": "CREATE_WO",
            "priority": priority,
            "title": "Trigger Work Order",
            "description": mode,
            "action_label": "Create WO",
        })
        priority += 1

    if fp > 0.55:
        recs.append({
            "rec_id": f"{asset_id}-pm-1",
            "rec_type": "ADJUST_PM",
            "priority": priority,
            "title": "Adjust PM Job",
            "description": f"Shorten PM-{10000 + hash(asset_id) % 9000} by 40%",
            "action_label": "Adjust PM",
        })
        priority += 1

    if factors:
        top_param = factors[0]["parameter"]
        recs.append({
            "rec_id": f"{asset_id}-task-1",
            "rec_type": "ADD_TASK",
            "priority": priority,
            "title": f"Inspect {top_param}",
            "description": f"Check {top_param.lower()} — {factors[0]['deviation'].split('—')[0].strip()}",
            "action_label": "Add Task",
        })
        priority += 1

    if not recs:
        recs.append({
            "rec_id": f"{asset_id}-mon-1",
            "rec_type": "MONITOR",
            "priority": 1,
            "title": "Continue Monitoring",
            "description": "No immediate action required",
            "action_label": "Schedule Check",
        })

    today = datetime(2025, 6, 15)
    pm_job = None
    if fp > 0.4:
        orig_date = today + timedelta(days=30)
        adj_days = max(5, 30 - int(fp * 22))
        adj_date = today + timedelta(days=adj_days)
        pm_job = {
            "job_code": f"PM-{10000 + abs(hash(asset_id)) % 9000}",
            "job_name": "500 HR Service",
            "original_due_date": orig_date.strftime("%b %d, %Y"),
            "adjusted_due_date": adj_date.strftime("%b %d, %Y"),
            "adjustment_pct": int((1 - adj_days / 30) * 100),
            "adjustment_reason": "AI risk factors detected",
        }

    return {
        "asset_id": asset_id,
        "recommendations": recs[:3],
        "pm_job": pm_job,
    }
