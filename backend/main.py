from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data_processor import (
    load_data,
    get_fleet_kpis,
    get_asset_health_list,
    get_asset_detail,
    get_asset_recommendations,
)
import ml_engine
from models import (
    FleetKPIResponse,
    AssetHealthListResponse,
    AssetDetailResponse,
    RecommendationsResponse,
    ModelListResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading telematics data...")
    load_data()
    print("Data loaded.")
    yield


app = FastAPI(title="AI Health & Predictive Maintenance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/fleet/kpis", response_model=FleetKPIResponse)
def fleet_kpis():
    return get_fleet_kpis()


@app.get("/api/fleet/assets/health", response_model=AssetHealthListResponse)
def asset_health_list(
    sort: str = Query("risk_level", enum=["risk_level", "health_score", "eta", "random"]),
    risk_level: str | None = Query(None, alias="risk_level"),
    page: int = Query(1, ge=1),
    limit: int = Query(5, ge=1, le=50),
    seed: int | None = Query(None),
):
    return get_asset_health_list(sort=sort, risk_filter=risk_level, page=page, limit=limit, seed=seed)


@app.get("/api/assets/{asset_id}/detail", response_model=AssetDetailResponse)
def asset_detail(asset_id: str):
    data = get_asset_detail(asset_id)
    if not data:
        raise HTTPException(status_code=404, detail="Asset not found")
    return data


@app.get("/api/assets/{asset_id}/recommendations", response_model=RecommendationsResponse)
def asset_recommendations(asset_id: str):
    data = get_asset_recommendations(asset_id)
    if not data:
        raise HTTPException(status_code=404, detail="Asset not found")
    return data


@app.post("/api/recommendations/{rec_id}/accept")
def accept_recommendation(rec_id: str):
    return {"status": "accepted", "rec_id": rec_id, "message": "Work order created successfully"}


@app.post("/api/recommendations/{rec_id}/dismiss")
def dismiss_recommendation(rec_id: str):
    return {"status": "dismissed", "rec_id": rec_id}


@app.get("/api/models", response_model=ModelListResponse)
def list_models():
    return {"models": ml_engine.get_model_list(), "active_model": ml_engine.get_active_model()}


@app.post("/api/models/{model_name}/activate")
def activate_model(model_name: str):
    success = ml_engine.set_active_model(model_name)
    if not success:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    return {"status": "ok", "active_model": ml_engine.get_active_model()}


@app.get("/api/health")
def health():
    return {"status": "ok"}