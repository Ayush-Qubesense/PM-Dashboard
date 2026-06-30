# OpsFlo Predictive Maintenance Dashboard

AI-powered fleet health monitoring demo built on a 10,000-asset telematics dataset. Runs entirely from a single FastAPI server — no Node.js required.

---

## What it does

- **Fleet overview** — health index gauge, at-risk count, critical alerts, predicted failures in 30 days
- **Asset list** — sortable by risk level, health score, or predicted ETA; paginated; live sparklines
- **Asset detail** — latest telemetry, failure probability, predicted failure mode, contributing factors, AI recommendations, PM job optimization
- **ML model comparison** — switch between Rule-Based, Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and LSTM live; see how each model's predictions differ on the same fleet
- **SHAP explainability** — LR, Decision Tree, and XGBoost show per-asset contributing factors with actual sensor values vs fleet median (e.g. "Oil Pressure: 11.7 PSI vs fleet 87.8 PSI — 76.1 below ↑")

---

## Project structure

```
Predictive Maintenance Dashboard/
├── backend/
│   ├── main.py              # FastAPI app, API routes, serves frontend
│   ├── data_processor.py    # Health scoring, fleet KPIs, asset detail
│   ├── ml_engine.py         # Model training, SHAP, prediction cache
│   ├── models.py            # Pydantic response schemas
│   └── requirements.txt
└── frontend/
    └── index.html           # Standalone dashboard (Tailwind CDN, vanilla JS)
```

The frontend is a single HTML file served by FastAPI — no build step needed.

---

## Requirements

- **Python 3.10+**
- **The telematics CSV** — place it at `backend/MaintenanceTelematicsData_synthetic_10k.csv`

No Node.js, no npm, no Docker.

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, scikit-learn, XGBoost, SHAP, pandas, and uvicorn. First install takes ~2–3 minutes depending on your connection.

### 3. Place the data file

Copy the telematics CSV into the backend folder:

```
backend/MaintenanceTelematicsData_synthetic_10k.csv
```

---

## Starting the server

```bash
cd backend
python -m uvicorn main:app --port 8000
```

The server builds rolling window features, trains all five ML models, computes SHAP, and caches all predictions on startup. Expected output:

```
Loading telematics data...
Training ML models...
  [ML] Features: 81 total (22 snapshot + 59 rolling/ratio)
  [ML] Training Logistic Regression...
  [ML] Logistic Regression: acc=0.916 f1=0.911 (497ms)
  [ML] Computing SHAP for Logistic Regression...
  [ML] SHAP done in 111ms
  [ML] Training Decision Tree...
  [ML] Decision Tree: acc=0.982 f1=0.982 (347ms)
  [ML] Computing SHAP for Decision Tree...
  [ML] SHAP done in 192ms
  [ML] Training Random Forest...
  [ML] Random Forest: acc=0.992 f1=0.992 (845ms)
  [ML] Training XGBoost...
  [ML] XGBoost: acc=0.994 f1=0.993 (2409ms)
  [ML] Computing SHAP for XGBoost...
  [ML] SHAP done in 1818ms
  [ML] Training SVM...
  [ML] SVM: acc=0.961 f1=0.960 (~12000ms)
  [ML] Training LightGBM...
  [ML] LightGBM: acc=0.995 f1=0.994 (900ms)
  [ML] Computing SHAP for LightGBM...
  [ML] SHAP done in 2200ms
  [ML] Training LSTM...
    LSTM epoch 5/5 loss=1.83
  [ML] LSTM: acc=0.524 f1=0.378 (~15000ms)
  [ML] Computing LSTM attributions...
  [ML] Done in 610ms
  [ML] All models ready.
Data loaded.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Total startup time: ~35–45 seconds** (SVM calibration ~10s + LSTM training ~15s).

> Notes:
> - Random Forest skips SHAP (TreeSHAP on 100 trees × 10k rows takes ~140s). Uses global feature importances instead.
> - LSTM accuracy (~52%) is expected to be low — synthetic 24-hr sequences are linear interpolations from fleet median to the current snapshot, not real temporal data. Its unique value is in trend-based contributing factors (e.g. "Oil Pressure: 11.7 PSI — 24-hr shift: 76.1 PSI ↓").
> - All other models use per-asset SHAP values for contributing factors.

Open **http://localhost:8000** in your browser.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the dashboard UI |
| `GET` | `/api/fleet/kpis` | Fleet-level KPIs |
| `GET` | `/api/fleet/assets/health` | Paginated asset list with sort/filter |
| `GET` | `/api/assets/{id}/detail` | Full asset detail + prediction |
| `GET` | `/api/assets/{id}/recommendations` | AI recommendations + PM job |
| `GET` | `/api/models` | All models with metrics + active flag |
| `POST` | `/api/models/{name}/activate` | Switch active model |
| `POST` | `/api/recommendations/{id}/accept` | Accept a recommendation |
| `POST` | `/api/recommendations/{id}/dismiss` | Dismiss a recommendation |
| `GET` | `/api/health` | Health check |

---

## ML models

| Model | Accuracy | F1 | Explainability | Notes |
|---|---|---|---|---|
| Rule-Based | — | — | Scenario config | Baseline; maps `failure_scenario` → penalty |
| Logistic Regression | ~91.6% | ~91.1% | ✅ SHAP LinearExplainer | StandardScaler applied before training |
| Decision Tree | ~98.2% | ~98.1% | ✅ SHAP TreeExplainer | Max depth 8 |
| Random Forest | ~99.2% | ~99.2% | Global importances | 100 trees; SHAP skipped for startup speed |
| XGBoost | ~99.4% | ~99.3% | ✅ SHAP TreeExplainer | 100 estimators, max depth 6 |
| LightGBM | ~99.5% | ~99.4% | Global importances | 300 estimators, leaf-wise growth; SHAP skipped (23s on 10k rows); uses built-in feature_importances_ |
| SVM | ~96% | ~96% | Linear coef weights | LinearSVC + Platt calibration; class_weight=balanced; SHAP skipped (KernelExplainer too slow on 10k rows) |
| LSTM | ~52% | ~38% | ✅ 24-hr trend attribution | PyTorch 2-layer LSTM on synthetic sequences; shows trend shifts not snapshots |

All ML models train on **81 features**: 22 raw sensor readings + 59 engineered features (1hr/6hr/24hr mean, std, delta for 9 key columns; 4 cross-parameter ratios).

Switching models is instant — all predictions are pre-cached at startup.

---

## Troubleshooting

**Port already in use**

```bash
# Windows — find and kill the process on port 8000
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# macOS / Linux
lsof -ti :8000 | xargs kill -9
```

Then restart the server.

**CSV not found**

The server will fail at startup with a `FileNotFoundError`. Make sure the file is at `backend/MaintenanceTelematicsData_synthetic_10k.csv` (exact filename).

**SHAP import warning**

If `shap` is not installed, the server falls back to global feature importances for all models. Install it with:

```bash
pip install shap==0.46.0
```

**PyTorch / LSTM not available**

`torch` must be installed for the LSTM model to appear. Use the CPU-only build (smaller download):

```bash
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

If `torch` is missing, the server runs with 4 models (Rule-Based + LR + DT + RF + XGB) instead of 5.

**Logistic Regression `OptimizeWarning`**

A `sklearn` / `scipy` version mismatch may emit `OptimizeWarning: Unknown solver options: iprint`. This is cosmetic — the model trains correctly and converges.
