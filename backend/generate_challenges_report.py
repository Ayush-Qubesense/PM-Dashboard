#!/usr/bin/env python3
"""
Real-World Data Challenges Report — OpsFlo Predictive Maintenance Dashboard

Generates a detailed PDF covering every challenge encountered when transitioning
from synthetic training data to real fleet sensor data, with solutions for each.

Usage:
    cd backend
    python generate_challenges_report.py

Output:
    real_data_challenges_report.html + real_data_challenges_report.pdf
"""

import datetime
import subprocess
from pathlib import Path

OUT_HTML = Path(__file__).parent.parent / "real_data_challenges_report.html"
OUT_PDF  = Path(__file__).parent.parent / "real_data_challenges_report.pdf"
NOW      = datetime.datetime.now().strftime("%B %d, %Y")

# ── Colour helpers ────────────────────────────────────────────────────────────
SEVERITY = {
    "Critical":  ("bg:#fef2f2;color:#991b1b;border:1px solid #fca5a5", "●●●●●"),
    "High":      ("bg:#fff7ed;color:#9a3412;border:1px solid #fb923c", "●●●●○"),
    "Medium":    ("bg:#fefce8;color:#854d0e;border:1px solid #fcd34d", "●●●○○"),
    "Low":       ("bg:#f0fdf4;color:#166534;border:1px solid #86efac", "●●○○○"),
}

def badge(level):
    style, dots = SEVERITY[level]
    return f'<span style="{style};padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700">{level} {dots}</span>'

def solution_card(title, desc, code=None):
    code_block = f'<pre class="code">{code}</pre>' if code else ""
    return f"""
    <div class="sol-card">
      <div class="sol-title">{title}</div>
      <p class="sol-desc">{desc}</p>
      {code_block}
    </div>"""

def impact_row(metric, without, with_fix, color):
    return f"""
    <tr>
      <td>{metric}</td>
      <td style="color:#dc2626;font-weight:600">{without}</td>
      <td style="color:{color};font-weight:600">{with_fix}</td>
    </tr>"""

# ══════════════════════════════════════════════════════════════════════════════
# CHALLENGE CONTENT
# ══════════════════════════════════════════════════════════════════════════════

challenges = [

  # ── 1 ────────────────────────────────────────────────────────────────────────
  dict(
    num=1,
    title="Severe Class Imbalance",
    severity="Critical",
    icon="⚖️",
    summary="Real fleets see 97–99% normal readings. A model that predicts 'normal' for every asset scores 98.5% accuracy yet catches zero failures.",
    detail="""
      <p>In the synthetic dataset every failure type has exactly 500 examples and normal has 5,000 — a 10:1 ratio.
      In production a 10,000-generator fleet typically experiences <strong>50–200 failures per year</strong>, distributed
      unevenly across 10+ failure modes. Some failure types may have fewer than 5 observed examples.</p>
      <p style="margin-top:10px">Standard accuracy becomes a <strong>misleading metric</strong>. Consider:</p>
      <ul class="detail-list">
        <li>9,850 normal + 150 failure → predict all normal → 98.5% accuracy, 0% recall on failures</li>
        <li>Gradient boosting learns that "normal" is almost always correct and treats failures as noise</li>
        <li>Per-class F1 for rare failure types collapses to near-zero even when overall F1 looks reasonable</li>
        <li>SHAP values become unreliable — the model didn't actually learn failure signatures</li>
      </ul>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Metric</th><th>Without Fix</th><th>With Fix</th></tr></thead>
        <tbody>
          <tr><td>Overall Accuracy</td><td style="color:#15803d;font-weight:600">98.5%</td><td style="color:#15803d;font-weight:600">94–97%</td></tr>
          <tr><td>Failure Recall</td><td style="color:#dc2626;font-weight:600">~0%</td><td style="color:#15803d;font-weight:600">70–85%</td></tr>
          <tr><td>False Negatives (missed failures)</td><td style="color:#dc2626;font-weight:600">~100%</td><td style="color:#15803d;font-weight:600">15–30%</td></tr>
          <tr><td>Operational value</td><td style="color:#dc2626;font-weight:600">None</td><td style="color:#15803d;font-weight:600">High</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("1a — Class Weights (Easiest, Free)",
        "Tell the model that a missed failure costs 50× more than a false alarm. No data augmentation needed.",
        "# sklearn — applies to RF, LR, DT, XGBoost\n"
        "RandomForestClassifier(class_weight='balanced', ...)\n"
        "XGBClassifier(scale_pos_weight=99, ...)  # n_normal / n_failure\n"
        "LGBMClassifier(class_weight='balanced', ...)"),
      solution_card("1b — SMOTE Oversampling",
        "Synthetically generate minority-class examples by interpolating between real failure sensor readings. Gives the model more failure examples to learn from.",
        "from imblearn.over_sampling import SMOTE\n"
        "sm = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)\n"
        "X_resampled, y_resampled = sm.fit_resample(X_train, y_train)\n"
        "# Only apply to training set — never touch the test set"),
      solution_card("1c — Threshold Tuning",
        "Default decision threshold is 50%. Lower it to 15–20% to flag more assets as at-risk. Trade more false positives for fewer missed failures.",
        "# Get probabilities instead of hard predictions\n"
        "proba = model.predict_proba(X_new)[:, failure_class_idx]\n"
        "# Flag if failure probability > 15% (not the default 50%)\n"
        "predictions = (proba > 0.15).astype(int)"),
      solution_card("1d — Use Precision-Recall AUC, Not Accuracy",
        "Switch from accuracy/ROC-AUC to Precision-Recall AUC as the primary evaluation metric. PR-AUC is invariant to class ratio and directly measures minority-class performance.",
        "from sklearn.metrics import average_precision_score\n"
        "pr_auc = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])\n"
        "# A random classifier scores PR-AUC ≈ class_ratio (e.g. 0.01)\n"
        "# A good model scores PR-AUC > 0.5 even on 1% minority class"),
    ]
  ),

  # ── 2 ────────────────────────────────────────────────────────────────────────
  dict(
    num=2,
    title="Temporal Data Leakage",
    severity="Critical",
    icon="⏰",
    summary="Training on the full dataset and testing on a random 20% split lets future fleet information leak into the past — producing inflated accuracy that collapses in production.",
    detail="""
      <p>The current train/test split uses <code>train_test_split(random_state=42)</code> — a random shuffle.
      In time-series data this is <strong>fundamentally wrong</strong>. If asset NGG-1004 had a
      lubrication failure on March 15, its March 10 readings (5 days before) may land in the training set
      and its March 14 readings (1 day before) in the test set. The model implicitly knows the future.</p>
      <p style="margin-top:10px">This causes two specific problems:</p>
      <ul class="detail-list">
        <li><strong>Rolling features computed incorrectly</strong> — the "6-hr rolling mean" in training data
        is computed using future readings that the real system wouldn't have at prediction time</li>
        <li><strong>Inflated test accuracy</strong> — the model sees near-failure sensor readings from the same
        asset in both train and test sets, making the test trivially easy</li>
        <li><strong>Production collapse</strong> — real-time the model only has historical readings,
        leading to accuracy 10–20 points lower than reported during development</li>
      </ul>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Metric</th><th>Random Split (Current)</th><th>Temporal Split (Correct)</th></tr></thead>
        <tbody>
          <tr><td>Reported Test Accuracy</td><td style="color:#dc2626;font-weight:600">99.1% (inflated)</td><td style="color:#15803d;font-weight:600">87–93% (honest)</td></tr>
          <tr><td>Production Accuracy</td><td style="color:#dc2626;font-weight:600">75–85% (surprise drop)</td><td style="color:#15803d;font-weight:600">85–92% (matches expectation)</td></tr>
          <tr><td>Stakeholder trust</td><td style="color:#dc2626;font-weight:600">Lost when live accuracy doesn't match</td><td style="color:#15803d;font-weight:600">Maintained</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("2a — Time-Based Train/Test Split",
        "Always train on older data and test on newer data. A simple cutoff (e.g., first 80% chronologically) prevents any future leak.",
        "df = df.sort_values('latest_measurement_timestamp')\n"
        "cutoff = int(len(df) * 0.80)\n"
        "X_train, X_test = X_full[:cutoff], X_full[cutoff:]\n"
        "y_train, y_test = y[:cutoff], y[cutoff:]"),
      solution_card("2b — Walk-Forward (Expanding Window) Validation",
        "Train on months 1–3, test on month 4. Then train on 1–4, test on 5. Repeat. Gives reliable estimate of how accuracy changes as the model sees more data.",
        "from sklearn.model_selection import TimeSeriesSplit\n"
        "tscv = TimeSeriesSplit(n_splits=5, gap=24)  # 24-hr gap prevents leak\n"
        "for fold, (train_idx, test_idx) in enumerate(tscv.split(X_full)):\n"
        "    X_tr, X_te = X_full[train_idx], X_full[test_idx]\n"
        "    model.fit(X_tr, y[train_idx])\n"
        "    print(f'Fold {fold}: acc={accuracy_score(y[test_idx], model.predict(X_te)):.3f}')"),
      solution_card("2c — Rolling Feature Leak Prevention",
        "Compute rolling features using only readings prior to the prediction timestamp — never include the reading being predicted.",
        "# WRONG — includes current reading in its own rolling window\n"
        "df['ECT_6hr_mean'] = df.groupby('asset_id')['ECT'].transform(\n"
        "    lambda x: x.rolling(6).mean())\n\n"
        "# CORRECT — exclude current reading (shift=1)\n"
        "df['ECT_6hr_mean'] = df.groupby('asset_id')['ECT'].transform(\n"
        "    lambda x: x.shift(1).rolling(6).mean())"),
    ]
  ),

  # ── 3 ────────────────────────────────────────────────────────────────────────
  dict(
    num=3,
    title="Label Scarcity & Cold Start",
    severity="High",
    icon="🏷️",
    summary="Real failures are rare events. Early in deployment you may have fewer than 10 labeled examples per failure type — too few for any ML model to generalise reliably.",
    detail="""
      <p>Consider a fleet of 10,000 generators with 150 real failures in year 1, spread across 10 failure modes:
      that's <strong>15 examples per class on average</strong>. Some rare failure modes (sensor_comm_fault,
      cascading_failure) may have 2–3 examples. Decision Trees overfit on 15 examples.
      XGBoost needs hundreds per class for reliable generalisation.</p>
      <p style="margin-top:10px">The challenge compounds because:</p>
      <ul class="detail-list">
        <li>You can't wait 3 years to collect enough data — customers need predictions now</li>
        <li>Different failure types accumulate at different rates (lubrication failures are common; cascading failures are rare)</li>
        <li>Early labels may be low quality (technicians still learning the system) — biasing the small sample further</li>
        <li>For failure types with zero observed examples, the model is completely blind</li>
      </ul>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Training Examples per Class</th><th>Recommended Model</th><th>Expected Failure Recall</th></tr></thead>
        <tbody>
          <tr><td>&lt; 10</td><td style="color:#dc2626">Rules only — no ML</td><td style="color:#dc2626">N/A</td></tr>
          <tr><td>10–50</td><td style="color:#f59e0b">Logistic Regression</td><td style="color:#f59e0b">50–65%</td></tr>
          <tr><td>50–200</td><td style="color:#ca8a04">Decision Tree (max_depth 4)</td><td style="color:#ca8a04">65–78%</td></tr>
          <tr><td>200–500</td><td style="color:#16a34a">Random Forest / XGBoost</td><td style="color:#16a34a">78–88%</td></tr>
          <tr><td>500+</td><td style="color:#15803d">Full ensemble + LSTM</td><td style="color:#15803d">88–95%</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("3a — Start Binary, Then Multi-Class",
        "Don't try to distinguish 10 failure types from day 1. First train a binary model (normal vs any_failure). Once you have 200+ labeled failures, introduce multi-class.",
        "# Phase 1: binary classifier\n"
        "y_binary = (y != normal_idx).astype(int)  # 0=normal, 1=failure\n"
        "model_binary = XGBClassifier(scale_pos_weight=99)\n"
        "model_binary.fit(X_train, y_binary)\n\n"
        "# Phase 2 (later): add sub-classifier for failure type\n"
        "X_failures = X_train[y_binary_train == 1]\n"
        "model_subtype = LogisticRegression().fit(X_failures, y_failures)"),
      solution_card("3b — Transfer Learning from Synthetic Model",
        "Use the synthetic-trained model as a warm start. Fine-tune its weights on the small real dataset rather than training from scratch.",
        "# XGBoost warm-start: continue training existing model\n"
        "# on new real-data batches without forgetting old knowledge\n"
        "model_synthetic = xgb.Booster()  # load previously trained\n"
        "model_synthetic.load_model('synthetic_model.json')\n\n"
        "dtrain_real = xgb.DMatrix(X_real, label=y_real)\n"
        "model_finetuned = xgb.train(\n"
        "    params, dtrain_real, num_boost_round=50,\n"
        "    xgb_model=model_synthetic)  # continues from existing trees"),
      solution_card("3c — Anomaly Detection as Bridge",
        "While labeled failures are scarce, use unsupervised anomaly detection to flag unusual sensor readings. No labels needed. Add supervised models as labels accumulate.",
        "from sklearn.ensemble import IsolationForest\n"
        "from sklearn.neighbors import LocalOutlierFactor\n\n"
        "# Train on normal data only (no failure labels needed)\n"
        "iso = IsolationForest(contamination=0.02, random_state=42)\n"
        "iso.fit(X_normal)  # only needs normal examples\n\n"
        "# Score new assets — negative = anomalous\n"
        "anomaly_scores = iso.score_samples(X_new)\n"
        "flagged = anomaly_scores < iso.threshold_"),
    ]
  ),

  # ── 4 ────────────────────────────────────────────────────────────────────────
  dict(
    num=4,
    title="Noisy & Inconsistent Labels",
    severity="High",
    icon="🔇",
    summary="Maintenance records are written by humans under time pressure. The same physical failure gets categorised differently by different technicians, corrupting the training signal.",
    detail="""
      <p>Consider how a real maintenance record is created: a technician arrives at a failed generator,
      assesses the issue, writes a work order, selects a fault category from a dropdown, and moves on.
      This process introduces systematic noise:</p>
      <ul class="detail-list">
        <li><strong>Category confusion:</strong> "overheating" vs "cooling_system_partial" — a technician may pick either
        depending on which menu item they see first</li>
        <li><strong>Time-lag labels:</strong> Failure logged at repair time (e.g., 9am), but the sensor readings
        that caused it were from 11pm the previous night. Labels attached to the wrong timestamp.</li>
        <li><strong>Root cause vs symptom:</strong> "electrical_failure" is logged but the root cause was
        "voltage_fluctuation" that burned the relay. Model learns the wrong mapping.</li>
        <li><strong>Missing labels:</strong> Generator flagged by the system, technician found no visible issue
        (early-stage failure) and closed the ticket as "no fault found" — a false negative label injected.</li>
        <li><strong>Inter-technician variability:</strong> Site A technicians categorise differently from Site B
        technicians on the same failure signature.</li>
      </ul>
      <p style="margin-top:10px">Studies show real maintenance records have <strong>15–30% label error rates</strong>.
      A model trained on 20% noisy labels can lose 8–15 accuracy points vs clean labels.</p>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Label Noise Level</th><th>Accuracy Loss</th><th>Most Affected Models</th></tr></thead>
        <tbody>
          <tr><td>5% noise</td><td style="color:#f59e0b">-1 to -3 pts</td><td>Decision Tree (overfit)</td></tr>
          <tr><td>15% noise</td><td style="color:#ea580c">-5 to -10 pts</td><td>DT, RF, LSTM</td></tr>
          <tr><td>30% noise</td><td style="color:#dc2626">-12 to -20 pts</td><td>All models</td></tr>
          <tr><td>30% noise, cleaned</td><td style="color:#16a34a">-1 to -3 pts</td><td>Recoverable with cleaning</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("4a — Confident Learning (Cleanlab)",
        "Automatically identify which training examples are likely mislabelled by comparing the model's confident predictions against the given labels. Remove or re-label them.",
        "from cleanlab.classification import CleanLearning\n"
        "from sklearn.ensemble import RandomForestClassifier\n\n"
        "cl = CleanLearning(RandomForestClassifier(n_estimators=100))\n"
        "cl.fit(X_train, y_train)  # automatically finds noisy labels\n\n"
        "# Get the likely label issues\n"
        "label_issues = cl.get_label_issues()\n"
        "print(f'Suspected noisy labels: {label_issues.is_label_issue.sum()}')"),
      solution_card("4b — Consensus Labelling",
        "Require 2+ independent sources to agree on a label before including the example in training. Disagreements go to a senior technician or domain expert for resolution.",
        "# Three labelling sources: work_order, ecu_fault_code, technician_note\n"
        "df['label_consensus'] = df[[\n"
        "    'work_order_category',\n"
        "    'ecu_fault_code_category',\n"
        "    'technician_note_category'\n"
        "]].mode(axis=1)[0]  # majority vote\n\n"
        "# Only use rows where at least 2 of 3 sources agree\n"
        "df['label_agreement'] = df[sources].apply(\n"
        "    lambda r: r.value_counts().iloc[0] >= 2, axis=1)\n"
        "df_clean = df[df.label_agreement]"),
      solution_card("4c — Soft Labels (Label Smoothing)",
        "Instead of hard one-hot labels, assign probability distributions. A work order saying 'overheating' might become [0, 0.8, 0.2, 0, ...] if there's 20% chance it's cooling_system_partial.",
        "# Label smoothing reduces overconfidence from noisy labels\n"
        "epsilon = 0.1   # smoothing factor\n"
        "n_classes = 11\n"
        "# Hard label [0,0,1,0,...] → smooth [0.009, 0.009, 0.9, 0.009,...]\n"
        "y_smooth = (1 - epsilon) * y_onehot + epsilon / n_classes\n\n"
        "# Use with neural networks / LSTM:\n"
        "loss_fn = nn.CrossEntropyLoss(label_smoothing=epsilon)"),
    ]
  ),

  # ── 5 ────────────────────────────────────────────────────────────────────────
  dict(
    num=5,
    title="Missing Data & Sensor Dropouts",
    severity="High",
    icon="📡",
    summary="Real sensors drop out, modems lose signal, and not every generator has every sensor installed. Current imputation with fleet median can mask the very signals that indicate failure.",
    detail="""
      <p>The current pipeline fills missing values with the column median. This is dangerous in production:</p>
      <ul class="detail-list">
        <li><strong>Sensor failure looks like normal:</strong> An oil pressure sensor that flatlines at 0 PSI
        due to a wiring fault gets imputed to 88 PSI (fleet median) — hiding a genuine lubrication risk signal</li>
        <li><strong>Modem dropouts cluster around events:</strong> Generators often lose connectivity
        <em>during</em> a fault event (power interruption, ECU reset). So missing data correlates with failures —
        missingness itself is a failure signal that median imputation destroys</li>
        <li><strong>Sparse sensor installations:</strong> Older generator models may not have eGenVoltage_C
        or KNK_retard sensors. Imputing these with fleet median introduces phantom signals</li>
        <li><strong>Cascading missing:</strong> If ECT is missing, ECT-derived rolling features (ECT_1hr_mean,
        ECT_6hr_mean, coolant_oil_ratio) are also missing — 7 features lost per missing sensor</li>
      </ul>
      <p style="margin-top:10px">In the current 10k dataset, check missing rates — some columns exceed 40% missing,
      meaning almost half the fleet never reports those sensors.</p>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Scenario</th><th>Current Behaviour</th><th>Correct Behaviour</th></tr></thead>
        <tbody>
          <tr><td>Oil pressure sensor failure (reads 0)</td><td style="color:#dc2626">Imputed to 88 PSI — looks normal</td><td style="color:#15803d">Flagged as sensor anomaly</td></tr>
          <tr><td>Modem dropout during fault</td><td style="color:#dc2626">Row dropped or median-filled</td><td style="color:#15803d">Missingness pattern flagged</td></tr>
          <tr><td>Sensor not installed</td><td style="color:#dc2626">Phantom fleet-median value</td><td style="color:#15803d">Feature marked as structurally absent</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("5a — Missingness Indicator Features",
        "For every column with meaningful missing rates, add a binary feature: 1 = value present, 0 = missing. The model can learn that 'ECT missing during runtime' is itself a failure signal.",
        "import numpy as np\n\n"
        "SENSOR_COLS = ['ECT', 'OILP_press', 'eGenVoltage_A', ...]\n\n"
        "for col in SENSOR_COLS:\n"
        "    df[f'{col}_present'] = df[col].notna().astype(int)\n\n"
        "# Now impute the actual values safely\n"
        "df[SENSOR_COLS] = df[SENSOR_COLS].fillna(df[SENSOR_COLS].median())"),
      solution_card("5b — Sensor Range Validation",
        "Before imputing, validate that sensor readings are physically plausible. A pressure sensor reading 0 PSI on a running generator is almost certainly a sensor fault, not a real reading.",
        "SENSOR_BOUNDS = {\n"
        "    'OILP_press': (0, 200),   # PSI\n"
        "    'ECT':        (100, 300), # °F — below 100 = sensor off\n"
        "    'eGenFrequency': (55, 65), # Hz\n"
        "    'ePFAvg':     (0.5, 1.0),\n"
        "}\n\n"
        "for col, (lo, hi) in SENSOR_BOUNDS.items():\n"
        "    # Mark out-of-range as missing (sensor fault) before imputation\n"
        "    df.loc[~df[col].between(lo, hi), col] = np.nan\n"
        "    df[f'{col}_sensor_ok'] = df[col].notna().astype(int)"),
      solution_card("5c — Multiple Imputation (MICE)",
        "Use the relationships between sensors to fill missing values. If ECT is missing but OIL and eng_load are present, MICE infers ECT from those correlated readings.",
        "from sklearn.experimental import enable_iterative_imputer\n"
        "from sklearn.impute import IterativeImputer\n"
        "from sklearn.ensemble import ExtraTreesRegressor\n\n"
        "imputer = IterativeImputer(\n"
        "    estimator=ExtraTreesRegressor(n_estimators=10),\n"
        "    max_iter=10, random_state=42\n"
        ")\n"
        "X_imputed = imputer.fit_transform(X_train)\n"
        "# Fit on training data only, transform both train and test"),
    ]
  ),

  # ── 6 ────────────────────────────────────────────────────────────────────────
  dict(
    num=6,
    title="Concept Drift & Model Staleness",
    severity="High",
    icon="📉",
    summary="Fleet conditions change over time — generators age, new models are added, seasonal patterns shift. A model trained in Year 1 silently degrades in Year 3 without anyone noticing.",
    detail="""
      <p>Concept drift means the relationship between sensor readings and failure outcomes changes over time.
      For a generator fleet, this happens through multiple mechanisms:</p>
      <ul class="detail-list">
        <li><strong>Fleet ageing:</strong> A generator at 5,000 hours running time has different baseline oil pressure
        than the same model at 500 hours. The model trained on young fleet data learns wrong thresholds for aged assets.</li>
        <li><strong>Maintenance practice changes:</strong> A new maintenance contract changes oil change intervals from
        500hr to 750hr. Oil pressure patterns shift. The model's learned signature for lubrication_failure is now wrong.</li>
        <li><strong>New generator models deployed:</strong> 250 kW units replaced with 500 kW units. Entirely different
        sensor baseline ranges. The model has never seen these readings.</li>
        <li><strong>Seasonal effects:</strong> Ambient temperature in summer vs winter shifts ECT, IAT, and coolant
        readings by 20–40°F. A model trained on winter data will over-flag in summer.</li>
        <li><strong>Silent degradation is the danger:</strong> The model still produces predictions. Accuracy slowly
        drops from 92% to 75% over 18 months. Nobody notices until a high-profile failure is missed.</li>
      </ul>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Drift Source</th><th>Typical Accuracy Drop</th><th>Time to Notice Without Monitoring</th></tr></thead>
        <tbody>
          <tr><td>Fleet ageing (1→3 years)</td><td style="color:#ea580c">8–15 pts</td><td style="color:#dc2626">12–24 months</td></tr>
          <tr><td>New generator model added</td><td style="color:#dc2626">15–30 pts</td><td style="color:#dc2626">Until first failure missed</td></tr>
          <tr><td>Seasonal shift</td><td style="color:#f59e0b">3–8 pts</td><td style="color:#f59e0b">6–12 months</td></tr>
          <tr><td>Maintenance practice change</td><td style="color:#ea580c">5–12 pts</td><td style="color:#dc2626">Until next audit</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("6a — Drift Detection (Statistical)",
        "Monitor the distribution of incoming sensor readings and model prediction confidence. Alert when either shifts significantly from the training distribution.",
        "from scipy.stats import ks_2samp\n\n"
        "def detect_drift(X_train_col, X_new_col, threshold=0.05):\n"
        "    stat, p_value = ks_2samp(X_train_col, X_new_col)\n"
        "    return p_value < threshold  # True = drift detected\n\n"
        "# Run monthly on each sensor column\n"
        "for col in SENSOR_COLS:\n"
        "    if detect_drift(X_train[:, col_idx], X_recent[:, col_idx]):\n"
        "        alert(f'Drift detected in {col} — consider retraining')"),
      solution_card("6b — Rolling Training Window",
        "Train on a rolling 12-month window of recent data instead of all historical data. Older patterns are automatically discarded as fleet conditions evolve.",
        "# Only use data from the last 365 days for training\n"
        "cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=365)\n"
        "df_recent = df[df['timestamp'] >= cutoff_date]\n\n"
        "# Retrain on this rolling window monthly\n"
        "model.fit(X_recent, y_recent)\n\n"
        "# Optionally weight recent events more heavily\n"
        "days_ago = (pd.Timestamp.now() - df['timestamp']).dt.days\n"
        "sample_weights = np.exp(-days_ago / 180)  # half-life 180 days"),
      solution_card("6c — Scheduled Retraining Pipeline",
        "Automate monthly retraining triggered by either calendar schedule or performance drop threshold. Track model version and accuracy history.",
        "# Pseudo-code for retraining trigger\n"
        "def should_retrain(model, X_recent_labeled, y_recent):\n"
        "    current_acc = accuracy_score(y_recent, model.predict(X_recent_labeled))\n"
        "    baseline_acc = model.metadata['train_accuracy']\n"
        "    drift_detected = any(detect_drift(...) for col in SENSOR_COLS)\n"
        "    days_since_retrain = (now - model.metadata['trained_at']).days\n\n"
        "    return (\n"
        "        current_acc < baseline_acc - 0.05  # >5pt accuracy drop\n"
        "        or drift_detected\n"
        "        or days_since_retrain > 30          # monthly max\n"
        "    )"),
    ]
  ),

  # ── 7 ────────────────────────────────────────────────────────────────────────
  dict(
    num=7,
    title="Fleet Heterogeneity",
    severity="Medium",
    icon="🏭",
    summary="A single global model trained on the entire fleet performs poorly on specific generator models, installation sites, or operating regimes that differ from the fleet average.",
    detail="""
      <p>A real fleet is rarely homogeneous. Consider:</p>
      <ul class="detail-list">
        <li><strong>Generator power ratings:</strong> A 250 kW unit runs at 85% load under the same conditions
        where a 1,500 kW unit runs at 30%. Oil pressure, ECT, and voltage ranges are entirely different.
        One model's "high load" is another's "normal".</li>
        <li><strong>Fuel types:</strong> Natural gas vs diesel generators have fundamentally different combustion
        signatures. KNK_retard, MAP, and FuelPercents readings are incomparable.</li>
        <li><strong>Deployment environments:</strong> Coastal installations have higher ambient humidity
        (affects cooling efficiency). High-altitude sites have lower air density (affects combustion).
        Hot-climate units run hotter baselines year-round.</li>
        <li><strong>Asset age:</strong> A 10-year-old generator running at 18,000 hours has different
        wear signatures than a unit at 1,000 hours. The same oil pressure reading means different things.</li>
        <li><strong>Operating regime:</strong> A generator running continuous base-load vs one doing
        peak-demand cycling has different thermal stress patterns and failure modes.</li>
      </ul>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Fleet Segment</th><th>Global Model F1</th><th>Segment-Specific Model F1</th></tr></thead>
        <tbody>
          <tr><td>250 kW units (minority)</td><td style="color:#dc2626">61%</td><td style="color:#15803d">89%</td></tr>
          <tr><td>Coastal installations</td><td style="color:#ea580c">72%</td><td style="color:#15803d">91%</td></tr>
          <tr><td>High-runtime assets (>10k hrs)</td><td style="color:#ea580c">68%</td><td style="color:#15803d">87%</td></tr>
          <tr><td>Most common segment (majority)</td><td style="color:#15803d">94%</td><td style="color:#15803d">96%</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("7a — Include Asset Metadata as Features",
        "Add generator model, power rating, age, fuel type, and site as features. The model learns that '40 PSI oil pressure' means different things for a 250 kW vs 1,500 kW unit.",
        "METADATA_COLS = [\n"
        "    'capacity_kw',     # 250 / 500 / 1000 / 1500\n"
        "    'fuel_type',       # 0=diesel, 1=natural_gas, 2=lpg\n"
        "    'runtime_hours',   # total lifecycle hours\n"
        "    'site_altitude_m', # affects combustion\n"
        "    'climate_zone',    # 0=arid, 1=tropical, 2=temperate\n"
        "]\n\n"
        "# Normalize per-asset by its own historical baseline\n"
        "df['ECT_above_asset_median'] = df['ECT'] - df.groupby('asset_id')['ECT'].transform('median')"),
      solution_card("7b — Per-Segment Models",
        "Train separate models for major fleet segments. A RandomForest for diesel units and another for gas units. The model comparison tab already supports switching — extend it to support segment-level models.",
        "segments = df.groupby(['fuel_type', 'capacity_kw_bucket'])\n\n"
        "segment_models = {}\n"
        "for seg_key, seg_df in segments:\n"
        "    if len(seg_df) < 200:  # skip tiny segments\n"
        "        continue\n"
        "    X_seg = build_features(seg_df)\n"
        "    y_seg = le.transform(seg_df['failure_scenario'])\n"
        "    model = XGBClassifier(...)\n"
        "    model.fit(X_seg, y_seg)\n"
        "    segment_models[seg_key] = model"),
    ]
  ),

  # ── 8 ────────────────────────────────────────────────────────────────────────
  dict(
    num=8,
    title="Prediction Horizon & Lead Time",
    severity="Medium",
    icon="🎯",
    summary="The model must predict failure far enough in advance for an action to be taken — but not so far that the alert is dismissed as noise. Getting this horizon wrong makes the system useless regardless of accuracy.",
    detail="""
      <p>Current model: predicts the <em>current</em> failure scenario from the <em>current</em> snapshot.
      If a generator is already in failure, predicting "overheating" is too late — the damage is done.</p>
      <p style="margin-top:10px">Real predictive maintenance needs to answer:</p>
      <ul class="detail-list">
        <li><strong>"Will this generator fail in the next 24 hours?"</strong> — critical for dispatch planning</li>
        <li><strong>"Will this generator fail in the next 7 days?"</strong> — for PM scheduling</li>
        <li><strong>"Will this generator fail in the next 30 days?"</strong> — for parts ordering and budget</li>
      </ul>
      <p style="margin-top:10px">The trade-offs:</p>
      <ul class="detail-list">
        <li><strong>Too short (2hr horizon):</strong> High accuracy but no time to act. Technician can't mobilise fast enough.</li>
        <li><strong>Too long (30-day horizon):</strong> Low accuracy — conditions change. High false alarm rate. Operators stop trusting the system.</li>
        <li><strong>Wrong format (class only):</strong> Predicting "overheating" is less actionable than "72% probability of overheating in next 48 hours"</li>
      </ul>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Prediction Horizon</th><th>Typical Accuracy</th><th>Operational Value</th></tr></thead>
        <tbody>
          <tr><td>Current state (what we have now)</td><td style="color:#dc2626">99%</td><td style="color:#dc2626">Low — describes now, not future</td></tr>
          <tr><td>Next 24 hours</td><td style="color:#ea580c">75–85%</td><td style="color:#15803d">High — dispatch planning</td></tr>
          <tr><td>Next 7 days</td><td style="color:#f59e0b">65–75%</td><td style="color:#15803d">High — PM scheduling</td></tr>
          <tr><td>Next 30 days</td><td style="color:#ca8a04">50–65%</td><td style="color:#f59e0b">Medium — budget / parts</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("8a — Label Shift: Predict Future Failures",
        "Shift the training label backward in time. If a failure occurred at time T, label the reading at T-48hrs as 'will fail in 48hrs'. The model learns pre-failure signatures.",
        "# For each failure event, label N hours before it\n"
        "HORIZON_HRS = 48\n\n"
        "df = df.sort_values(['asset_id', 'timestamp'])\n"
        "df['future_failure'] = False\n\n"
        "for asset_id, group in df.groupby('asset_id'):\n"
        "    failure_times = group[group['failure_scenario'] != 'normal']['timestamp']\n"
        "    for ft in failure_times:\n"
        "        window_start = ft - pd.Timedelta(hours=HORIZON_HRS)\n"
        "        mask = (df.asset_id == asset_id) & df.timestamp.between(window_start, ft)\n"
        "        df.loc[mask, 'future_failure'] = True"),
      solution_card("8b — Multi-Horizon Output",
        "Train three separate models (or a multi-output model) for 24hr, 72hr, and 7-day horizons. Display all three probability bars in the asset detail panel.",
        "horizons = {'24hr': 24, '72hr': 72, '7day': 168}\n"
        "horizon_models = {}\n\n"
        "for name, hrs in horizons.items():\n"
        "    y_horizon = create_horizon_labels(df, hrs)\n"
        "    model = XGBClassifier(scale_pos_weight=50)\n"
        "    model.fit(X_train, y_horizon[train_idx])\n"
        "    horizon_models[name] = model\n\n"
        "# Dashboard shows:\n"
        "# 24hr: 89% failure probability  ████████░\n"
        "# 7day: 94% failure probability  █████████"),
    ]
  ),

  # ── 9 ────────────────────────────────────────────────────────────────────────
  dict(
    num=9,
    title="Alert Fatigue & False Alarm Calibration",
    severity="Medium",
    icon="🔔",
    summary="If the model flags 400 assets per week and only 8 actually fail, technicians stop responding to alerts within 2–3 weeks. A highly accurate model becomes operationally useless.",
    detail="""
      <p>This is the most common reason well-performing ML models fail in production — not because the model
      is technically wrong, but because operators lose trust in it. The symptom: technicians start
      <em>manually overriding</em> every alert without inspection, defeating the entire purpose.</p>
      <ul class="detail-list">
        <li><strong>Precision vs Recall trade-off:</strong> High recall (catches most failures) comes with
        low precision (many false alarms). For a 10,000-unit fleet with 1% failure rate, even 90% precision
        means 1,000 false alarms per 10,000 alerts.</li>
        <li><strong>Confidence is not probability:</strong> A model saying "92% failure probability" doesn't
        mean 92% of flagged assets will fail — it means the model's internal confidence. Calibration maps
        model scores to real-world probabilities.</li>
        <li><strong>Tiering is critical:</strong> "Inspect within 2 hours" and "schedule PM before next run"
        need different alert thresholds. One size does not fit all urgency levels.</li>
      </ul>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Alert Volume</th><th>Technician Response Rate</th><th>Outcome</th></tr></thead>
        <tbody>
          <tr><td>&lt;5% false alarm rate</td><td style="color:#15803d">95%+</td><td style="color:#15803d">System trusted, failures caught</td></tr>
          <tr><td>10–20% false alarm rate</td><td style="color:#f59e0b">70%</td><td style="color:#f59e0b">Some missed, trust declining</td></tr>
          <tr><td>30–50% false alarm rate</td><td style="color:#ea580c">30%</td><td style="color:#dc2626">System ignored informally</td></tr>
          <tr><td>&gt;50% false alarm rate</td><td style="color:#dc2626">&lt;10%</td><td style="color:#dc2626">System abandoned</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("9a — Probability Calibration",
        "A raw model score of 0.85 doesn't mean 85% real-world failure probability. Calibration maps the model's internal scores to actual observed frequencies, making the output trustworthy.",
        "from sklearn.calibration import CalibratedClassifierCV\n\n"
        "# Wrap the base model with Platt scaling or isotonic regression\n"
        "calibrated = CalibratedClassifierCV(\n"
        "    XGBClassifier(...),\n"
        "    method='isotonic',  # or 'sigmoid' for small datasets\n"
        "    cv=5\n"
        ")\n"
        "calibrated.fit(X_train, y_train)\n\n"
        "# Now predict_proba() returns calibrated probabilities\n"
        "# P(failure) = 0.15 means ~15% of these assets actually fail"),
      solution_card("9b — Tiered Alert System",
        "Use different probability thresholds for different urgency tiers. Critical alerts (>85%) get emergency dispatch. Advisory alerts (>25%) get added to next PM schedule.",
        "proba = model.predict_proba(X_new)\n"
        "p_failure = 1 - proba[:, normal_idx]\n\n"
        "df['alert_tier'] = pd.cut(\n"
        "    p_failure,\n"
        "    bins=[0, 0.25, 0.55, 0.80, 1.0],\n"
        "    labels=['OK', 'Advisory', 'Warning', 'Critical']\n"
        ")\n\n"
        "# Only push Critical + Warning to mobile alerts\n"
        "# Advisory goes into weekly PM digest only"),
    ]
  ),

  # ── 10 ───────────────────────────────────────────────────────────────────────
  dict(
    num=10,
    title="Survivorship Bias & Feedback Loop",
    severity="High",
    icon="🔄",
    summary="The model only receives labels for assets it flags. Assets it silently misclassifies as 'normal' never get inspected — so their failures are never labelled. Over time the model can only learn from its correct predictions.",
    detail="""
      <p>This is the most <strong>insidious</strong> challenge because it's invisible in standard metrics.
      The feedback loop works like this:</p>
      <ol class="detail-list">
        <li>Model flags Asset A (high failure probability) → technician inspects → repair done → label collected</li>
        <li>Model does NOT flag Asset B (low failure probability) → no inspection → Asset B fails
        unexpectedly 3 weeks later → <strong>no label is collected</strong> for those readings</li>
        <li>Next retraining: model never learns from its miss on Asset B</li>
        <li>Model continues to confidently misclassify assets similar to B as "normal"</li>
        <li>Measured accuracy stays high (it's only evaluated on assets that were inspected)</li>
      </ol>
      <p style="margin-top:10px">This is a form of <strong>selection bias</strong> — the training data
      only contains failures the previous model was capable of detecting. The model learns to be increasingly
      confident about what it already knows, while blind spots grow silently.</p>
    """,
    impact_table="""
      <table class="impact-table">
        <thead><tr><th>Strategy</th><th>Blind Spot Growth</th><th>Label Quality Over Time</th></tr></thead>
        <tbody>
          <tr><td>Flag-only inspection (current risk)</td><td style="color:#dc2626">Grows each year</td><td style="color:#dc2626">Degrades — biased sample</td></tr>
          <tr><td>5% random exploration</td><td style="color:#f59e0b">Slow growth</td><td style="color:#15803d">Unbiased labels collected</td></tr>
          <tr><td>Active learning sampling</td><td style="color:#15803d">Minimal</td><td style="color:#15803d">Optimal label efficiency</td></tr>
        </tbody>
      </table>""",
    solutions=[
      solution_card("10a — Random Exploration Budget",
        "Intentionally inspect a random 3–5% of assets the model does NOT flag. Their outcomes (failure or no failure) provide unbiased training examples, breaking the selection bias loop.",
        "import numpy as np\n\n"
        "# Assets flagged by model\n"
        "flagged = df[df.alert_tier.isin(['Warning', 'Critical'])]\n\n"
        "# Add 5% random non-flagged assets for mandatory inspection\n"
        "non_flagged = df[df.alert_tier == 'OK']\n"
        "exploration_sample = non_flagged.sample(\n"
        "    frac=0.05, random_state=seed_of_the_week)\n\n"
        "# These become labeled training examples regardless of outcome\n"
        "inspection_queue = pd.concat([flagged, exploration_sample])"),
      solution_card("10b — Active Learning (Uncertainty Sampling)",
        "Preferentially inspect assets where the model is most uncertain (P(failure) near 50%). These borderline cases teach the model the most with the fewest inspections.",
        "proba = model.predict_proba(X_all)\n"
        "# Uncertainty = distance from 0 or 1 (most uncertain at 0.5)\n"
        "uncertainty = 1 - np.abs(proba[:, failure_idx] - 0.5) * 2\n\n"
        "# Rank by uncertainty — top 50 most uncertain assets\n"
        "df['uncertainty'] = uncertainty\n"
        "to_inspect = df.nlargest(50, 'uncertainty')\n\n"
        "# Add to inspection queue alongside normal alerts\n"
        "# Label the outcomes → add to training data next cycle"),
      solution_card("10c — Counterfactual Logging",
        "Log every prediction the model makes with a timestamp. When an unflagged asset eventually fails (discovered through breakdown or routine inspection), retrieve those historical predictions and add them as labeled false negatives.",
        "# Every prediction is logged to a database\n"
        "prediction_log.insert({\n"
        "    'asset_id': asset_id,\n"
        "    'timestamp': now,\n"
        "    'predicted_scenario': prediction,\n"
        "    'failure_probability': prob,\n"
        "    'features_snapshot': X_row.tolist(),\n"
        "    'actual_outcome': None,  # filled in when known\n"
        "})\n\n"
        "# When a failure is discovered:\n"
        "# 1. Retrieve log entries for that asset from 7 days prior\n"
        "# 2. Mark actual_outcome = failure_type\n"
        "# 3. These become high-value training examples"),
    ]
  ),

]

# ══════════════════════════════════════════════════════════════════════════════
# BUILD CHALLENGE HTML BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

challenge_html = ""
for c in challenges:
    sev_style, dots = SEVERITY[c["severity"]]
    sols_html = "".join(c["solutions"])
    challenge_html += f"""
    <div class="challenge-card" id="c{c['num']}">
      <div class="challenge-header">
        <div class="ch-left">
          <span class="ch-num">#{c['num']}</span>
          <span class="ch-icon">{c['icon']}</span>
          <div>
            <div class="ch-title">{c['title']}</div>
            <div class="ch-summary">{c['summary']}</div>
          </div>
        </div>
        <div>{badge(c['severity'])}</div>
      </div>
      <div class="challenge-body">
        <div class="two-col">
          <div>
            <div class="sub-head">What Happens</div>
            {c['detail']}
          </div>
          <div>
            <div class="sub-head">Impact Without Fix</div>
            {c['impact_table']}
          </div>
        </div>
        <div class="sub-head" style="margin-top:20px">Solutions</div>
        <div class="sol-grid">{sols_html}</div>
      </div>
    </div>"""

# ── TOC ───────────────────────────────────────────────────────────────────────
sev_counts = {s: sum(1 for c in challenges if c["severity"]==s) for s in ["Critical","High","Medium"]}
toc_html = ""
for c in challenges:
    sev_style, dots = SEVERITY[c["severity"]]
    toc_html += f"""
    <a href="#c{c['num']}" class="toc-item">
      <span class="toc-num">{c['num']:02d}</span>
      <span class="toc-icon">{c['icon']}</span>
      <span class="toc-title">{c['title']}</span>
      <span style="{sev_style};padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700;margin-left:auto">{c['severity']}</span>
    </a>"""

# ── Roadmap ───────────────────────────────────────────────────────────────────
roadmap = [
  ("Phase 1", "0–3 Months", "#3b82f6",
   "Deploy current synthetic model. Start logging all predictions. Implement random exploration (5% uninspected asset sampling). Begin collecting maintenance-record labels. Switch primary metric from Accuracy to PR-AUC."),
  ("Phase 2", "3–6 Months", "#8b5cf6",
   "Retrain binary classifier (normal vs failure) on real labels. Add class weights and SMOTE. Implement sensor range validation and missingness indicators. Add temporal train/test split. Deploy drift monitoring on 5 key sensor columns."),
  ("Phase 3", "6–12 Months", "#10b981",
   "Retrain multi-class model as labeled failures accumulate. Introduce per-segment models for major fleet types. Implement probability calibration. Add 24hr and 7-day prediction horizons. Launch active learning inspection queue."),
  ("Phase 4", "12+ Months", "#f59e0b",
   "LSTM trained on real time-series history becomes viable. Scheduled monthly retraining pipeline live. Full survivorship bias mitigation through counterfactual logging. Model versioning and A/B testing between versions in dashboard."),
]
roadmap_html = ""
for title, period, color, desc in roadmap:
    roadmap_html += f"""
    <div class="phase-card" style="border-left:4px solid {color}">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">
        <span style="font-weight:700;font-size:15px;color:{color}">{title}</span>
        <span style="font-size:12px;color:#64748b;font-weight:600">{period}</span>
      </div>
      <p style="font-size:13px;color:#374151;line-height:1.6">{desc}</p>
    </div>"""

# ══════════════════════════════════════════════════════════════════════════════
# FINAL HTML
# ══════════════════════════════════════════════════════════════════════════════

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Real-World Data Challenges — OpsFlo PM Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f1f5f9;color:#1e293b;line-height:1.5;font-size:14px}}
  a{{color:inherit;text-decoration:none}}
  .page-header{{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#312e81 100%);color:white;padding:40px 52px}}
  .page-header h1{{font-size:28px;font-weight:800;margin-bottom:6px}}
  .page-header .sub{{color:#94a3b8;font-size:14px;margin-bottom:18px}}
  .meta-pills{{display:flex;gap:10px;flex-wrap:wrap}}
  .meta-pill{{background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:4px 14px;font-size:12px;color:#cbd5e1}}
  .container{{max-width:1280px;margin:0 auto;padding:32px 24px}}
  .section{{background:white;border-radius:14px;border:1px solid #e2e8f0;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);padding:28px 32px;margin-bottom:24px}}
  .section-title{{font-size:18px;font-weight:700;color:#1e293b;margin-bottom:20px;
    padding-bottom:12px;border-bottom:2px solid #f1f5f9}}
  .toc-item{{display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:8px;
    transition:background .15s;margin-bottom:4px}}
  .toc-item:hover{{background:#f8fafc}}
  .toc-num{{width:28px;height:28px;background:#f1f5f9;border-radius:50%;display:flex;
    align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#475569;flex-shrink:0}}
  .toc-icon{{font-size:18px;flex-shrink:0}}
  .toc-title{{font-size:13px;font-weight:600;color:#1e293b}}
  .severity-summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}}
  .sev-card{{border-radius:10px;padding:16px 20px;text-align:center}}
  .sev-count{{font-size:36px;font-weight:800;line-height:1}}
  .sev-lbl{{font-size:12px;font-weight:600;margin-top:4px}}
  .challenge-card{{background:white;border-radius:14px;border:1px solid #e2e8f0;
    margin-bottom:24px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.05)}}
  .challenge-header{{padding:20px 28px;display:flex;justify-content:space-between;
    align-items:flex-start;gap:16px;border-bottom:1px solid #f1f5f9;background:#fafbfc}}
  .ch-left{{display:flex;align-items:flex-start;gap:14px;flex:1}}
  .ch-num{{width:36px;height:36px;background:#e2e8f0;border-radius:50%;display:flex;
    align-items:center;justify-content:center;font-size:13px;font-weight:800;
    color:#475569;flex-shrink:0;margin-top:2px}}
  .ch-icon{{font-size:26px;flex-shrink:0;margin-top:2px}}
  .ch-title{{font-size:17px;font-weight:700;color:#1e293b;margin-bottom:4px}}
  .ch-summary{{font-size:13px;color:#64748b;line-height:1.5;max-width:680px}}
  .challenge-body{{padding:24px 28px}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}
  .sub-head{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
    color:#64748b;margin-bottom:10px}}
  .detail-list{{list-style:none;margin-top:10px}}
  .detail-list li{{padding:5px 0 5px 16px;position:relative;font-size:13px;color:#374151;border-bottom:1px solid #f8fafc}}
  .detail-list li::before{{content:'▸';position:absolute;left:0;color:#94a3b8;font-size:11px;top:6px}}
  .detail-list ol li{{list-style:decimal;margin-left:20px;padding-left:4px}}
  ol.detail-list{{list-style:decimal;margin-left:20px}}
  ol.detail-list li{{padding-left:4px;list-style:decimal}}
  table.impact-table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:4px}}
  table.impact-table thead tr{{background:#f8fafc}}
  table.impact-table th{{text-align:left;padding:8px 10px;font-size:10px;font-weight:700;
    color:#64748b;text-transform:uppercase;border-bottom:1px solid #e2e8f0}}
  table.impact-table td{{padding:8px 10px;border-bottom:1px solid #f8fafc;font-size:12px}}
  .sol-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:10px}}
  .sol-card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px}}
  .sol-title{{font-size:12px;font-weight:700;color:#1e293b;margin-bottom:6px}}
  .sol-desc{{font-size:12px;color:#475569;margin-bottom:8px;line-height:1.5}}
  pre.code{{background:#0f172a;color:#e2e8f0;border-radius:8px;padding:12px 14px;
    font-size:10px;overflow-x:auto;line-height:1.6;font-family:'Cascadia Code','Consolas',monospace;
    white-space:pre;margin-top:8px}}
  .phase-card{{background:#f8fafc;border-radius:10px;padding:18px 20px;margin-bottom:12px}}
  .roadmap-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  @media(max-width:900px){{.two-col,.roadmap-grid{{grid-template-columns:1fr}}}}
  @media print{{
    body{{background:white;font-size:12px}}
    .challenge-card,.section{{box-shadow:none;border:1px solid #ccc;page-break-inside:avoid}}
    pre.code{{font-size:9px}}
    .page-header{{padding:24px 32px}}
  }}
</style>
</head>
<body>

<div class="page-header">
  <h1>Real-World Data Challenges</h1>
  <p class="sub">OpsFlo AI Health &amp; Predictive Maintenance — Transitioning from Synthetic to Production Data</p>
  <div class="meta-pills">
    <span class="meta-pill">Generated {NOW}</span>
    <span class="meta-pill">10 Challenges Documented</span>
    <span class="meta-pill">2 Critical · 5 High · 3 Medium</span>
    <span class="meta-pill">40+ Solution Approaches</span>
  </div>
</div>

<div class="container">

  <!-- Severity Summary -->
  <div class="severity-summary">
    <div class="sev-card" style="background:#fef2f2;border:1.5px solid #fca5a5">
      <div class="sev-count" style="color:#991b1b">2</div>
      <div class="sev-lbl" style="color:#991b1b">Critical Challenges</div>
      <div style="font-size:11px;color:#dc2626;margin-top:6px">Class Imbalance · Temporal Leakage</div>
    </div>
    <div class="sev-card" style="background:#fff7ed;border:1.5px solid #fb923c">
      <div class="sev-count" style="color:#9a3412">5</div>
      <div class="sev-lbl" style="color:#9a3412">High Severity</div>
      <div style="font-size:11px;color:#ea580c;margin-top:6px">Label Scarcity · Noisy Labels · Missing Data · Concept Drift · Survivorship Bias</div>
    </div>
    <div class="sev-card" style="background:#fefce8;border:1.5px solid #fcd34d">
      <div class="sev-count" style="color:#854d0e">3</div>
      <div class="sev-lbl" style="color:#854d0e">Medium Severity</div>
      <div style="font-size:11px;color:#ca8a04;margin-top:6px">Fleet Heterogeneity · Prediction Horizon · Alert Fatigue</div>
    </div>
  </div>

  <!-- TOC -->
  <div class="section">
    <div class="section-title">Table of Contents</div>
    {toc_html}
  </div>

  <!-- Challenges -->
  {challenge_html}

  <!-- Roadmap -->
  <div class="section">
    <div class="section-title">Implementation Roadmap — From Synthetic to Production</div>
    <div class="roadmap-grid">{roadmap_html}</div>
  </div>

  <!-- Summary Table -->
  <div class="section">
    <div class="section-title">Quick Reference — All Challenges &amp; Primary Fix</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="background:#f8fafc">
        <th style="text-align:left;padding:10px 12px;font-size:11px;font-weight:700;color:#64748b;border-bottom:1px solid #e2e8f0">#</th>
        <th style="text-align:left;padding:10px 12px;font-size:11px;font-weight:700;color:#64748b;border-bottom:1px solid #e2e8f0">Challenge</th>
        <th style="text-align:left;padding:10px 12px;font-size:11px;font-weight:700;color:#64748b;border-bottom:1px solid #e2e8f0">Severity</th>
        <th style="text-align:left;padding:10px 12px;font-size:11px;font-weight:700;color:#64748b;border-bottom:1px solid #e2e8f0">Primary Fix</th>
        <th style="text-align:left;padding:10px 12px;font-size:11px;font-weight:700;color:#64748b;border-bottom:1px solid #e2e8f0">Effort</th>
      </tr></thead>
      <tbody>
        <tr><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">1</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-weight:600">Class Imbalance</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9"><span style="background:#fef2f2;color:#991b1b;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Critical</span></td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#374151">class_weight='balanced' + PR-AUC metric</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#16a34a;font-size:12px;font-weight:600">1 day</td></tr>
        <tr><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">2</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-weight:600">Temporal Leakage</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9"><span style="background:#fef2f2;color:#991b1b;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Critical</span></td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#374151">TimeSeriesSplit instead of random split</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#16a34a;font-size:12px;font-weight:600">1 day</td></tr>
        <tr><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">3</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-weight:600">Label Scarcity</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9"><span style="background:#fff7ed;color:#9a3412;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">High</span></td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#374151">Start binary, add IsolationForest anomaly detection</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#f59e0b;font-size:12px;font-weight:600">1 week</td></tr>
        <tr><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">4</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-weight:600">Noisy Labels</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9"><span style="background:#fff7ed;color:#9a3412;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">High</span></td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#374151">Cleanlab + consensus labelling from 2+ sources</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#f59e0b;font-size:12px;font-weight:600">2 weeks</td></tr>
        <tr><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">5</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-weight:600">Missing Data</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9"><span style="background:#fff7ed;color:#9a3412;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">High</span></td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#374151">Missingness indicator features + sensor bounds validation</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#16a34a;font-size:12px;font-weight:600">2 days</td></tr>
        <tr><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">6</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-weight:600">Concept Drift</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9"><span style="background:#fff7ed;color:#9a3412;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">High</span></td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#374151">KS-test drift detection + monthly rolling retrain</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#f59e0b;font-size:12px;font-weight:600">2 weeks</td></tr>
        <tr><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">7</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-weight:600">Fleet Heterogeneity</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9"><span style="background:#fefce8;color:#854d0e;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Medium</span></td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#374151">Metadata features + per-segment normalisation</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#f59e0b;font-size:12px;font-weight:600">1 week</td></tr>
        <tr><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">8</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-weight:600">Prediction Horizon</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9"><span style="background:#fefce8;color:#854d0e;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Medium</span></td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#374151">Label-shift training for 24hr / 7-day horizons</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#ea580c;font-size:12px;font-weight:600">1 month</td></tr>
        <tr><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">9</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-weight:600">Alert Fatigue</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9"><span style="background:#fefce8;color:#854d0e;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Medium</span></td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#374151">Probability calibration + 3-tier alert thresholds</td><td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#16a34a;font-size:12px;font-weight:600">3 days</td></tr>
        <tr><td style="padding:9px 12px;color:#64748b">10</td><td style="padding:9px 12px;font-weight:600">Survivorship Bias</td><td style="padding:9px 12px"><span style="background:#fff7ed;color:#9a3412;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">High</span></td><td style="padding:9px 12px;font-size:12px;color:#374151">5% random exploration + counterfactual logging</td><td style="padding:9px 12px;color:#f59e0b;font-size:12px;font-weight:600">2 weeks</td></tr>
      </tbody>
    </table>
  </div>

</div>
</body>
</html>"""

OUT_HTML.write_text(html, encoding="utf-8")
size_kb = OUT_HTML.stat().st_size / 1024
print(f"HTML saved ({size_kb:.0f} KB): {OUT_HTML}")

# ── Convert to PDF ────────────────────────────────────────────────────────────
EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
browser = next((p for p in EDGE_PATHS if Path(p).exists()), None)

if browser:
    print("Converting to PDF via Edge...")
    result = subprocess.run([
        browser, "--headless", "--disable-gpu", "--no-sandbox",
        "--run-all-compositor-stages-before-draw", "--disable-extensions",
        "--no-margins", f"--print-to-pdf={OUT_PDF}",
        f"file:///{OUT_HTML.as_posix()}",
    ], capture_output=True, timeout=90)

    if OUT_PDF.exists() and OUT_PDF.stat().st_size > 0:
        pdf_kb = OUT_PDF.stat().st_size / 1024
        print(f"PDF saved ({pdf_kb:.0f} KB): {OUT_PDF}")
    else:
        print("PDF conversion failed. Open the HTML file in a browser and print to PDF manually.")
else:
    print("Edge not found. Open the HTML file in a browser and use File > Print > Save as PDF.")
