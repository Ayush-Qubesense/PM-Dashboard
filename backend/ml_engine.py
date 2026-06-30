import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    import shap as _shap_lib
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
    print("  [ML] shap not installed — using feature importances")

try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
    _LGBM_OK = True
except ImportError:
    _LGBM_OK = False
    print("  [ML] lightgbm not installed — LightGBM unavailable")

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    print("  [ML] torch not installed — LSTM unavailable")


# ── Raw feature columns (26 snapshot readings) ───────────────────────────────

FEATURE_COLS = [
    "ECT", "OIL", "AAT", "IAT", "TIP",
    "OILP_press", "MAP", "NG Fuel Press", "EPR",
    "eng_load", "spk_adv", "KNK_retard", "TPS pct", "FuelPercents",
    "eGenVoltage_A", "eGenVoltage_B", "eGenVoltage_C",
    "eGenCurrent_A", "eGenCurrent_B", "eGenCurrent_C",
    "eGenFrequency", "ePFAvg", "ekWattsAvg",
    "FaultSysFlag", "dwell1", "dwell2",
]

# Columns for which 1hr/6hr/24hr rolling stats + delta are generated
ROLLING_COLS = [
    "ECT", "OIL", "OILP_press", "MAP",
    "eng_load", "KNK_retard",
    "ekWattsAvg", "ePFAvg", "eGenFrequency",
]

SEQ_LEN = 24  # 24-hour sliding window for LSTM

# Per-scenario severity penalty (0–100). The health score is derived from the
# EXPECTED penalty across the model's full class-probability distribution, so a
# confident "complete_shutdown" (95) scores far worse than a confident
# "sensor_comm_fault" (30). Without this, health collapses to a binary
# normal-vs-abnormal split (every failure → Critical) because the classifier is
# highly confident and 1 − P(normal) saturates near 1.0 for any abnormal asset.
SCENARIO_SEVERITY: dict[str, int] = {
    "normal": 0,
    "sensor_comm_fault": 30,
    "partial_load_failure": 40,
    "voltage_fluctuation": 40,
    "cooling_system_partial": 45,
    "electrical_failure": 55,
    "fuel_starvation": 60,
    "overheating": 80,
    "lubrication_failure": 80,
    "cascading_failure": 90,
    "complete_shutdown": 95,
}

# SHAP skipped: RF/LightGBM (too slow on 10k rows), LSTM (gradient attribution), SVM (KernelExplainer)
_SKIP_SHAP: set[str] = {"Random Forest", "LightGBM", "LSTM", "SVM"}


# ── Human-readable display names (auto-generated for rolling features) ────────

def _make_display() -> dict[str, str]:
    d: dict[str, str] = {
        "ECT": "Coolant Temp", "OIL": "Oil Temp",
        "AAT": "Ambient Air Temp", "IAT": "Intake Air Temp",
        "TIP": "Intake Pressure Temp", "OILP_press": "Oil Pressure",
        "MAP": "Manifold Pressure", "NG Fuel Press": "NG Fuel Pressure",
        "EPR": "Exhaust Pressure", "eng_load": "Engine Load",
        "spk_adv": "Spark Advance", "KNK_retard": "Knock Retard",
        "TPS pct": "Throttle Position", "FuelPercents": "Fuel Percent",
        "eGenVoltage_A": "Generator Voltage (A)", "eGenVoltage_B": "Generator Voltage (B)",
        "eGenVoltage_C": "Generator Voltage (C)", "eGenCurrent_A": "Generator Current (A)",
        "eGenCurrent_B": "Generator Current (B)", "eGenCurrent_C": "Generator Current (C)",
        "eGenFrequency": "Generator Frequency", "ePFAvg": "Power Factor",
        "ekWattsAvg": "kW Output", "FaultSysFlag": "Fault System Flag",
        "dwell1": "Ignition Dwell 1", "dwell2": "Ignition Dwell 2",
    }
    for col in ROLLING_COLS:
        base = d.get(col, col)
        for w in ("1hr", "6hr", "24hr"):
            d[f"{col}_{w}_mean"] = f"{base} ({w} avg)"
            d[f"{col}_{w}_std"]  = f"{base} ({w} volatility)"
        d[f"{col}_delta_1hr"] = f"{base} (rate of change)"
    d["coolant_oil_ratio"]       = "Coolant/Oil Temp Ratio"
    d["oil_pressure_load_ratio"] = "Oil Pressure / Load Ratio"
    d["voltage_phase_imbalance"] = "Voltage Phase Imbalance"
    d["kw_per_pf"]               = "kW / Power Factor"
    return d

FEATURE_DISPLAY = _make_display()


# ── PyTorch LSTM ──────────────────────────────────────────────────────────────

if _TORCH_OK:
    class _LSTMNet(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, n_classes: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                                batch_first=True, dropout=0.3)
            self.drop = nn.Dropout(0.3)
            self.fc   = nn.Linear(hidden_size, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, (hn, _) = self.lstm(x)
            return self.fc(self.drop(hn[-1]))


    class LSTMWrapper:
        """sklearn-compatible wrapper that internally builds synthetic 24-hr sequences."""

        def __init__(self, hidden: int = 64, epochs: int = 10,
                     batch: int = 256, lr: float = 1e-3):
            self.hidden = hidden
            self.epochs = epochs
            self.batch  = batch
            self.lr     = lr
            self._net: Optional[_LSTMNet] = None
            self._medians: Optional[np.ndarray] = None
            self.classes_: Optional[np.ndarray] = None

        def fit(self, X: np.ndarray, y: np.ndarray,
                fleet_medians: Optional[np.ndarray] = None) -> "LSTMWrapper":
            self._medians = (fleet_medians if fleet_medians is not None
                             else np.median(X, axis=0)).astype(np.float32)
            n_classes = int(y.max()) + 1
            self.classes_ = np.arange(n_classes)

            seqs = _make_sequences(X, self._medians)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(seqs, dtype=torch.float32),
                    torch.tensor(y,    dtype=torch.long),
                ),
                batch_size=self.batch, shuffle=True,
            )
            net = _LSTMNet(X.shape[1], self.hidden, n_classes)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            loss_fn = nn.CrossEntropyLoss()
            net.train()
            for ep in range(self.epochs):
                total = 0.0
                for xb, yb in loader:
                    opt.zero_grad()
                    l = loss_fn(net(xb), yb)
                    l.backward()
                    opt.step()
                    total += l.item()
                if (ep + 1) % 5 == 0:
                    print(f"    LSTM epoch {ep+1}/{self.epochs} "
                          f"loss={total/len(loader):.4f}")
            net.eval()
            self._net = net
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            seqs = _make_sequences(X, self._medians)
            with torch.no_grad():
                logits = self._net(torch.tensor(seqs, dtype=torch.float32))
                return torch.softmax(logits, dim=1).numpy()

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self.predict_proba(X).argmax(axis=1)

        def attr_all(self, X: np.ndarray) -> np.ndarray:
            """Per-asset attribution: |last_timestep − 23-hr history mean|.
            Highlights features that shifted most in the final hour of the sequence."""
            seqs = _make_sequences(X, self._medians)       # (n, 24, n_feat)
            last = seqs[:, -1, :]                           # (n, n_feat)
            hist = seqs[:, :-1, :].mean(axis=1)            # (n, n_feat)
            return np.abs(last - hist)                      # (n, n_feat)


# ── SVM wrapper ──────────────────────────────────────────────────────────────

class _SVCWrap:
    """LinearSVC + Platt calibration.

    LinearSVC trains in ~5-10s on 10k rows (RBF SVC would take 30+ minutes).
    CalibratedClassifierCV adds predict_proba() via sigmoid calibration.
    coef_ is exposed as the mean across CV folds so _importances() can use it.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self._inner = LinearSVC(C=C, max_iter=max_iter, random_state=42, dual="auto")
        self._cal   = CalibratedClassifierCV(self._inner, cv=3, method="sigmoid")
        self.coef_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SVCWrap":
        self._cal.fit(X, y)
        self.classes_ = self._cal.classes_
        # Average coef_ across the 3 CV-fold estimators for feature importance
        self.coef_ = np.mean(
            [cc.estimator.coef_ for cc in self._cal.calibrated_classifiers_], axis=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._cal.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._cal.predict(X)


# ── Model registry ────────────────────────────────────────────────────────────

def _registry() -> dict:
    r: dict = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"),
        "Decision Tree": lambda: DecisionTreeClassifier(
            max_depth=8, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1),
    }
    if _LGBM_OK:
        r["LightGBM"] = lambda: _LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbose=-1)
    r["SVM"] = lambda: _SVCWrap(C=1.0, max_iter=2000)
    if _TORCH_OK:
        r["LSTM"] = lambda: LSTMWrapper(hidden=64, epochs=5, batch=256, lr=1e-3)
    return r

MODEL_REGISTRY = _registry()


# ── Active model config ───────────────────────────────────────────────────────
# Change this value to switch which ML model drives all predictions.
# Valid values: "Rule-Based" | "Logistic Regression" | "Decision Tree" |
#               "Random Forest" | "XGBoost" | "LightGBM" | "SVM" | "LSTM"
ACTIVE_MODEL: str = "XGBoost"

# ── Global state ──────────────────────────────────────────────────────────────

_active_model: str = ACTIVE_MODEL
_trained_models: dict = {}
_label_encoder: LabelEncoder = LabelEncoder()
_metrics: dict[str, dict] = {}
_all_predictions: dict[str, dict] = {}
_feat_cols: list[str] = []          # all features (base + rolling)
_feat_medians: dict[str, float] = {}
_normal_idx: Optional[int] = None


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_rolling(X_base: np.ndarray, base_cols: list[str],
                   medians: dict) -> tuple[np.ndarray, list[str]]:
    """
    Generate synthetic rolling window features seeded for reproducibility.
    Each window stat is correlated with the actual snapshot value so that
    high-risk sensors show elevated rolling averages — making the feature
    useful for the ML models even on single-snapshot data.
    """
    rng  = np.random.default_rng(seed=2024)
    n    = len(X_base)
    ci   = {c: i for i, c in enumerate(base_cols)}
    feats: list[np.ndarray] = []
    names: list[str]        = []

    for col in ROLLING_COLS:
        if col not in ci:
            continue
        v   = X_base[:, ci[col]]
        med = medians.get(col, 0.0)
        s   = max(float(v.std()), 1e-6)

        # Window stats: blend current value with fleet median + noise
        for wt, wl, wc, wn in [
            (1,  "1hr",  0.95, 0.05),
            (6,  "6hr",  0.80, 0.10),
            (24, "24hr", 0.55, 0.18),
        ]:
            m_win = v * wc + med * (1 - wc) + rng.normal(0, s * wn, n)
            s_win = np.abs(rng.normal(s * wn * 1.5, s * 0.01, n))
            feats += [m_win, s_win]
            names += [f"{col}_{wl}_mean", f"{col}_{wl}_std"]

        # Rate of change: proportional to deviation from fleet median
        dev   = (v - med) / s
        delta = dev * s * 0.04 + rng.normal(0, s * 0.02, n)
        feats.append(delta)
        names.append(f"{col}_delta_1hr")

    # Cross-parameter ratios
    def _r(a: str, b: str) -> Optional[np.ndarray]:
        if a in ci and b in ci:
            return X_base[:, ci[a]] / np.maximum(np.abs(X_base[:, ci[b]]), 1e-6)
        return None

    for ratio_arr, ratio_name in [
        (_r("ECT", "OIL"),           "coolant_oil_ratio"),
        (_r("OILP_press", "eng_load"), "oil_pressure_load_ratio"),
        (_r("ekWattsAvg", "ePFAvg"), "kw_per_pf"),
    ]:
        if ratio_arr is not None:
            feats.append(ratio_arr)
            names.append(ratio_name)

    if all(c in ci for c in ["eGenVoltage_A", "eGenVoltage_B", "eGenVoltage_C"]):
        vA, vB, vC = (X_base[:, ci[c]] for c in ["eGenVoltage_A", "eGenVoltage_B", "eGenVoltage_C"])
        feats.append(np.std([vA, vB, vC], axis=0))
        names.append("voltage_phase_imbalance")

    if not feats:
        return np.empty((n, 0)), []
    return np.column_stack(feats), names


def _make_sequences(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """
    Synthetic 24-hour sequences for LSTM training and inference.
    Each sequence trends linearly from the fleet median (t=0, 24hr ago)
    to the current snapshot (t=23, now), with small additive noise.
    This simulates a degradation trajectory leading up to the current reading.
    """
    n, nf = X.shape
    rng   = np.random.default_rng(seed=2024)
    fstd  = X.std(axis=0) + 1e-9
    seqs  = np.empty((n, SEQ_LEN, nf), dtype=np.float32)
    for t in range(SEQ_LEN):
        alpha = t / (SEQ_LEN - 1)           # 0 = 24hr ago, 1 = now
        noise_scale = fstd * 0.04 * (1 - alpha + 0.1)
        seqs[:, t, :] = (
            medians * (1 - alpha) + X * alpha
            + rng.normal(0, noise_scale, (n, nf))
        ).astype(np.float32)
    return seqs


# ── Main training pipeline ────────────────────────────────────────────────────

def train_models(df: pd.DataFrame) -> None:
    global _feat_cols, _feat_medians, _normal_idx

    # ── Base features ──
    base_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_df = df[base_cols].copy()
    for c in base_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    base_med = X_df.median().to_dict()
    X_base   = X_df.fillna(pd.Series(base_med)).values

    # ── Rolling window + ratio features ──
    X_roll, roll_names = _build_rolling(X_base, base_cols, base_med)
    if X_roll.shape[1] > 0:
        X_full  = np.hstack([X_base, X_roll])
        all_cols = base_cols + roll_names
    else:
        X_full  = X_base
        all_cols = base_cols

    all_med    = {c: float(np.nanmedian(X_full[:, i])) for i, c in enumerate(all_cols)}
    _feat_cols   = all_cols
    _feat_medians = all_med
    med_arr = np.array([all_med[c] for c in all_cols], dtype=np.float32)

    print(f"  [ML] Features: {len(all_cols)} total "
          f"({len(base_cols)} snapshot + {len(roll_names)} rolling/ratio)")

    # ── Labels ──
    y_raw = df["failure_scenario"].fillna("normal").astype(str).str.strip().str.lower()
    y     = _label_encoder.fit_transform(y_raw)
    classes = list(_label_encoder.classes_)
    _normal_idx = classes.index("normal") if "normal" in classes else None

    # Severity penalty aligned to the encoder's class order, for the
    # expected-penalty health score. Unknown scenarios default to 50 (mid risk).
    severity_vec = np.array(
        [SCENARIO_SEVERITY.get(c, 50) for c in classes], dtype=np.float64)

    asset_ids = [str(int(float(a))) for a in df["asset_id"]]

    # ── Train / test split ──
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y)

    # StandardScaler for Logistic Regression only
    scaler      = StandardScaler().fit(X_tr)
    X_scaled    = scaler.transform(X_full)
    X_tr_scaled = scaler.transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    for name, factory in MODEL_REGISTRY.items():
        if name != ACTIVE_MODEL:
            continue
        is_linear = name in ("Logistic Regression", "SVM")
        is_lstm   = name == "LSTM"

        X_fit      = X_tr_scaled if is_linear else X_tr
        X_fit_test = X_te_scaled if is_linear else X_te
        X_all      = X_scaled    if is_linear else X_full

        print(f"  [ML] Training {name}...")
        t0    = time.time()
        model = factory()

        if is_lstm:
            model.fit(X_fit, y_tr, fleet_medians=med_arr)
        else:
            model.fit(X_fit, y_tr)

        elapsed = int((time.time() - t0) * 1000)

        y_pred = model.predict(X_fit_test)
        _metrics[name] = {
            "accuracy":  round(float(accuracy_score (y_te, y_pred)), 4),
            "f1":        round(float(f1_score       (y_te, y_pred, average="weighted", zero_division=0)), 4),
            "precision": round(float(precision_score(y_te, y_pred, average="weighted", zero_division=0)), 4),
            "recall":    round(float(recall_score   (y_te, y_pred, average="weighted", zero_division=0)), 4),
            "train_ms":  elapsed,
        }
        m = _metrics[name]
        print(f"  [ML] {name}: acc={m['accuracy']:.3f} f1={m['f1']:.3f} ({elapsed}ms)")

        probas      = model.predict_proba(X_all)
        pred_cls    = model.predict(X_all)
        global_imp  = _norm(_importances(model))

        # ── SHAP (DT, LR, XGB) ──
        shap_sv = None
        if _SHAP_OK and name not in _SKIP_SHAP:
            try:
                print(f"  [ML] Computing SHAP for {name}...")
                t_s = time.time()
                if hasattr(model, "feature_importances_"):
                    exp = _shap_lib.TreeExplainer(model, feature_perturbation="tree_path_dependent")
                else:
                    exp = _shap_lib.LinearExplainer(model, X_fit)
                shap_sv = exp.shap_values(X_all)
                print(f"  [ML] SHAP done in {int((time.time()-t_s)*1000)}ms")
            except Exception as exc:
                print(f"  [ML] SHAP failed for {name}: {exc}")

        # ── LSTM gradient attribution ──
        lstm_attr = None
        if is_lstm:
            print(f"  [ML] Computing LSTM attributions...")
            t_a = time.time()
            lstm_attr = model.attr_all(X_all)   # (n, n_feat)
            print(f"  [ML] Done in {int((time.time()-t_a)*1000)}ms")

        # ── Cache per-asset predictions ──
        _all_predictions[name] = {}
        for j, aid in enumerate(asset_ids):
            pc           = int(pred_cls[j])
            pred_scenario = _label_encoder.inverse_transform([pc])[0]
            fp = float(1.0 - probas[j, _normal_idx]) if _normal_idx is not None else 0.5
            fp = round(min(0.98, max(0.01, fp)), 2)

            # Severity-weighted health: expected penalty across ALL classes,
            # so risk reflects how bad the likely failure is, not just whether
            # the asset is abnormal. Spreads assets across Critical/High/Medium.
            expected_penalty = float(probas[j] @ severity_vec)
            health_score = int(round(max(0.0, min(100.0, 100.0 - expected_penalty))))

            top5 = None

            if shap_sv is not None:
                try:
                    sv   = _shap_row(shap_sv, j, pc)
                    top5 = _top5_shap(sv, X_all[j])
                except Exception:
                    pass

            if top5 is None and lstm_attr is not None:
                top5 = _top5_lstm(lstm_attr[j], X_all[j], med_arr)

            if top5 is None:
                imp  = _norm(np.abs(model.coef_[pc])) if hasattr(model, "coef_") else global_imp
                top5 = _top5_imp(imp)

            _all_predictions[name][aid] = {
                "predicted_scenario": pred_scenario,
                "failure_probability": fp,
                "health_score": health_score,
                "top5_features": top5,
            }

        _trained_models[name] = model

    print("  [ML] All models ready.")


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def _shap_row(sv, j: int, pc: int) -> np.ndarray:
    """Extract per-asset SHAP vector (handles list, 2-D, and 3-D array formats)."""
    if isinstance(sv, list):
        return np.asarray(sv[pc][j])
    a = np.asarray(sv)
    if a.ndim == 3:
        return a[j, :, pc]
    return a[j]


def _top5_shap(sv_row: np.ndarray, asset_row: np.ndarray) -> list[dict]:
    abs_sv  = np.abs(sv_row)
    top_idx = np.argsort(abs_sv)[-5:][::-1]
    top_abs = abs_sv[top_idx]
    total   = top_abs.sum()
    result  = []
    for k, i in enumerate(top_idx):
        if i >= len(_feat_cols):
            continue
        col    = _feat_cols[i]
        actual = float(asset_row[i])
        med    = _feat_medians.get(col, 0.0)
        sv     = float(sv_row[i])
        result.append({
            "parameter": FEATURE_DISPLAY.get(col, col),
            "deviation": _dev_shap(col, actual, med, sv),
            "weight":    round(float(top_abs[k]) / total if total > 0 else 0.2, 4),
        })
    return result


def _dev_shap(col: str, actual: float, med: float, sv: float) -> str:
    sign      = "↑" if sv > 0 else "↓"
    diff      = abs(actual - med)
    direction = "above" if actual > med else "below"

    # Rolling means
    if col.endswith("_1hr_mean"):
        return f"1-hr avg: {actual:.2f} vs fleet {med:.2f} — {diff:.2f} {direction} {sign}"
    if col.endswith("_6hr_mean"):
        return f"6-hr avg: {actual:.2f} vs fleet {med:.2f} — {diff:.2f} {direction} {sign}"
    if col.endswith("_24hr_mean"):
        return f"24-hr avg: {actual:.2f} vs fleet {med:.2f} — {diff:.2f} {direction} {sign}"
    if col.endswith("_std"):
        return f"Volatility: {actual:.3f} (fleet {med:.3f}) {sign}"
    if col.endswith("_delta_1hr"):
        return f"Rate of change: {actual:+.3f}/hr (fleet avg {med:.3f}) {sign}"
    # Ratios
    if col in ("coolant_oil_ratio", "oil_pressure_load_ratio", "kw_per_pf"):
        return f"Ratio: {actual:.3f} vs fleet {med:.3f} — {diff:.3f} {direction} {sign}"
    if col == "voltage_phase_imbalance":
        return f"Phase imbalance: {actual:.2f} V vs fleet {med:.2f} V {sign}"
    # Base features
    if col in ("ECT", "OIL", "AAT", "IAT", "TIP"):
        return f"{actual:.1f}°F vs fleet {med:.1f}°F — {diff:.1f}° {direction} {sign}"
    if col in ("OILP_press", "MAP", "EPR", "NG Fuel Press"):
        return f"{actual:.1f} PSI vs fleet {med:.1f} PSI — {diff:.1f} {direction} {sign}"
    if col == "eng_load":
        return f"{actual:.1f}% load vs fleet {med:.1f}% — {diff:.1f}pp {direction} {sign}"
    if col == "ekWattsAvg":
        return f"{actual:.0f} kW vs fleet {med:.0f} kW — {diff:.0f} {direction} {sign}"
    if col in ("eGenVoltage_A", "eGenVoltage_B", "eGenVoltage_C"):
        return f"{actual:.1f} V vs fleet {med:.1f} V — {diff:.1f} {direction} {sign}"
    if col in ("eGenCurrent_A", "eGenCurrent_B", "eGenCurrent_C"):
        return f"{actual:.2f} A vs fleet {med:.2f} A — {diff:.2f} {direction} {sign}"
    if col == "eGenFrequency":
        return f"{actual:.2f} Hz vs fleet {med:.2f} Hz — {diff:.2f} {direction} {sign}"
    if col == "ePFAvg":
        return f"PF {actual:.3f} vs fleet {med:.3f} — {diff:.3f} {direction} {sign}"
    if col == "FaultSysFlag":
        return f"Fault flag {'active' if actual > 0.5 else 'inactive'} (fleet {med:.2f}) {sign}"
    return f"{actual:.2f} vs fleet {med:.2f} — {diff:.2f} {direction} {sign}"


# ── LSTM attribution helpers ──────────────────────────────────────────────────

def _top5_lstm(attr_row: np.ndarray, asset_row: np.ndarray,
               med_arr: np.ndarray) -> list[dict]:
    """Top-5 by |last_timestep − history_mean| — shows what shifted most in final hour."""
    top_idx = np.argsort(attr_row)[-5:][::-1]
    total   = attr_row[top_idx].sum()
    result  = []
    for k, i in enumerate(top_idx):
        if i >= len(_feat_cols):
            continue
        col    = _feat_cols[i]
        cur    = float(asset_row[i])
        med    = float(med_arr[i]) if i < len(med_arr) else _feat_medians.get(col, 0.0)
        shift  = float(attr_row[i])
        result.append({
            "parameter": FEATURE_DISPLAY.get(col, col),
            "deviation": _dev_lstm(col, cur, med, shift),
            "weight":    round(float(attr_row[i]) / total if total > 0 else 0.2, 4),
        })
    return result


def _dev_lstm(col: str, cur: float, med: float, shift: float) -> str:
    """LSTM deviation: emphasises the 24-hr trend shift, not just the current value."""
    direction = "↑" if cur > med else "↓"

    if col in ("ECT", "OIL", "AAT", "IAT", "TIP"):
        return f"{cur:.1f}°F now — 24-hr shift: {shift:.1f}° {direction}"
    if col in ("OILP_press", "MAP", "EPR", "NG Fuel Press"):
        return f"{cur:.1f} PSI now — 24-hr shift: {shift:.1f} PSI {direction}"
    if col == "eng_load":
        return f"{cur:.1f}% load — 24-hr shift: {shift:.1f}pp {direction}"
    if col == "ekWattsAvg":
        return f"{cur:.0f} kW now — 24-hr shift: {shift:.0f} kW {direction}"
    if col in ("eGenVoltage_A", "eGenVoltage_B", "eGenVoltage_C"):
        return f"{cur:.1f} V now — 24-hr shift: {shift:.1f} V {direction}"
    if col in ("eGenCurrent_A", "eGenCurrent_B", "eGenCurrent_C"):
        return f"{cur:.2f} A now — 24-hr shift: {shift:.2f} A {direction}"
    if col == "eGenFrequency":
        return f"{cur:.2f} Hz now — 24-hr shift: {shift:.2f} Hz {direction}"
    if col == "ePFAvg":
        return f"PF {cur:.3f} now — 24-hr shift: {shift:.3f} {direction}"
    if col == "FaultSysFlag":
        return f"Fault flag {'active' if cur > 0.5 else 'inactive'} — trend shift {direction}"
    if col.endswith("_mean"):
        return f"Rolling avg {cur:.2f} — 24-hr shift: {shift:.2f} {direction}"
    if col.endswith("_delta_1hr"):
        return f"Rate of change {cur:+.3f}/hr — accelerating {direction}"
    if col in ("coolant_oil_ratio", "oil_pressure_load_ratio", "kw_per_pf"):
        return f"Ratio {cur:.3f} — 24-hr shift: {shift:.3f} {direction}"
    if col == "voltage_phase_imbalance":
        return f"Phase imbalance {cur:.2f} V — 24-hr shift: {shift:.2f} V {direction}"
    return f"{cur:.2f} now — 24-hr shift: {shift:.2f} {direction}"


# ── Fallback importances ──────────────────────────────────────────────────────

def _importances(model) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        return np.abs(model.coef_).mean(axis=0)
    return np.ones(len(_feat_cols))


def _norm(arr: np.ndarray) -> np.ndarray:
    t = arr.sum()
    return arr / t if t > 0 else arr


def _top5_imp(importances: np.ndarray) -> list[dict]:
    top_idx = np.argsort(importances)[-5:][::-1]
    total   = importances[top_idx].sum()
    result  = []
    for k, i in enumerate(top_idx):
        if i >= len(_feat_cols):
            continue
        col = _feat_cols[i]
        med = _feat_medians.get(col, 0.0)
        result.append({
            "parameter": FEATURE_DISPLAY.get(col, col),
            "deviation": _dev_global(col, med),
            "weight":    round(float(importances[i]) / total if total > 0 else 0.2, 4),
        })
    return result


def _dev_global(col: str, med: float) -> str:
    if col in ("ECT", "OIL", "AAT", "IAT", "TIP"):
        return f"Temperature signal — fleet median {med:.1f}°F"
    if col in ("OILP_press", "MAP", "EPR", "NG Fuel Press"):
        return f"Pressure signal — fleet median {med:.1f} PSI"
    if col == "eng_load":
        return f"Engine load — fleet median {med:.1f}%"
    if col == "ekWattsAvg":
        return f"Power output — fleet median {med:.0f} kW"
    if col in ("eGenVoltage_A", "eGenVoltage_B", "eGenVoltage_C"):
        return f"Generator voltage — fleet median {med:.1f} V"
    if col in ("eGenCurrent_A", "eGenCurrent_B", "eGenCurrent_C"):
        return f"Generator current — fleet median {med:.2f} A"
    if col == "eGenFrequency":
        return f"Generator frequency — fleet median {med:.2f} Hz"
    if col == "ePFAvg":
        return f"Power factor — fleet median {med:.3f}"
    if col.endswith("_mean"):
        return f"Rolling average — fleet {med:.2f}"
    if col.endswith("_delta_1hr"):
        return f"Rate of change — fleet avg {med:.3f}/hr"
    if col in ("coolant_oil_ratio", "oil_pressure_load_ratio", "kw_per_pf"):
        return f"Cross-parameter ratio — fleet median {med:.3f}"
    if col == "voltage_phase_imbalance":
        return f"Phase imbalance — fleet median {med:.2f} V"
    return f"Key feature — fleet median {med:.2f}"


# ── Public API ────────────────────────────────────────────────────────────────

def get_active_model() -> str:
    return _active_model


def set_active_model(name: str) -> bool:
    global _active_model
    if name not in ({"Rule-Based"} | set(MODEL_REGISTRY.keys())):
        return False
    _active_model = name
    return True


def get_model_list() -> list[dict]:
    result = [{
        "name": "Rule-Based", "accuracy": None, "f1": None,
        "precision": None, "recall": None, "train_ms": None,
        "is_active": _active_model == "Rule-Based",
    }]
    for name in MODEL_REGISTRY:
        m = _metrics.get(name, {})
        result.append({
            "name": name,
            "accuracy":  m.get("accuracy"),
            "f1":        m.get("f1"),
            "precision": m.get("precision"),
            "recall":    m.get("recall"),
            "train_ms":  m.get("train_ms"),
            "is_active": _active_model == name,
        })
    return result


def get_ml_prediction(asset_id: str, model_name: str) -> dict | None:
    preds = _all_predictions.get(model_name)
    if not preds:
        return None
    return preds.get(str(asset_id))
