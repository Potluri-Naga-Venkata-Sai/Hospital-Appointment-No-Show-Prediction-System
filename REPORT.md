# Algions AI Labs — Technical Assessment Report
## Hospital Appointment No-Show Prediction System
**Author:** Senior AI & Machine Learning Engineer  
**Role:** AI & Machine Learning  
**Assigned Domain:** Hospital Management & Clinic Operations  
**Date:** November 2026  
**Pipeline Version:** `v1.0.0`

---

## Executive Summary

Outpatient appointment no-shows represent one of the most critical operational inefficiencies in healthcare administration, costing health systems billions of dollars annually in idle physician time, wasted diagnostic capacity, and delayed access to care for waitlisted patients. 

This technical report documents the end-to-end design, implementation, defensible evaluation, and operationalization of a **Decision-Support Machine Learning Solution** developed for **Algions AI Labs**. The system predicts the likelihood of an appointment being missed at booking confirmation time, enabling clinical operations staff to deploy tiered, proactive reminders (e.g., automated SMS, two-way confirmations, and personal coordinator outreach) and plan dynamic scheduling buffers.

### Key Architectural Highlights:
1. **Strict Non-Clinical Scope:** In compliance with regulatory and ethical safeguards, the model strictly excludes clinical diagnoses, ICD codes, prescriptions, symptoms, or medical acuity. It operates purely on administrative, temporal, scheduling, and historical attendance metadata.
2. **Zero Temporal Leakage:** Data ingestion and feature transformations enforce strict chronological causality. Historical attendance rates are computed via expanding historical windows prior to booking time. Splits follow a chronological partition: **Train (70%, Jan–Sep 2025)**, **Validation (15%, Sep–Nov 2025)**, and **Test (15%, Nov–Dec 2025)**.
3. **Calibrated Probability Estimation:** Raw gradient-boosted tree scores are calibrated via isotonic regression, driving the Brier score down to **0.1431** and aligning predicted risks with empirical frequencies.
4. **Cost-Optimal Thresholding:** Rather than defaulting to an arbitrary 0.50 cutoff, the decision threshold was mathematically optimized against a healthcare operational cost matrix ($C_{\text{FN}} = \$120$ for an unutilized doctor slot vs. $C_{\text{FP}} = \$15$ for staff reminder outreach). The optimal threshold ($t^* = 0.12$) captured **93.1% of true no-shows**, reducing total operational loss by **43.8% ($34,755 saved on 3,750 test appointments)** compared to standard thresholding.
5. **Human-in-the-Loop Governance:** The model serves exclusively as a supportive triage mechanism. It includes local SHAP feature attributions and prohibits automated slot cancellations or care deprivation.

---

## 1. Problem Formulation & Operational Framing

### 1.1 Business Objective
The operational objective is to maximize hospital outpatient capacity utilization by predicting which scheduled appointments are at risk of being missed, allowing hospital administrative coordinators to execute targeted reminders and adjust booking overbooking allowances without degrading patient satisfaction.

### 1.2 Mathematical Formulation
We frame this as a supervised binary classification task under class imbalance:
$$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N, \quad y_i \in \{0, 1\}$$
- $y_i = 1$: Patient misses the scheduled appointment (No-Show).
- $y_i = 0$: Patient arrives for the scheduled appointment (Attended).
- $x_i \in \mathbb{R}^d$: Feature vector available at prediction point $T_{\text{pred}}$.
- Goal: Learn a calibrated conditional probability estimator $\hat{P}(Y=1 \mid X=x)$.

### 1.3 Prediction Horizon and Temporal Point-in-Time
- **Prediction Unit:** Single scheduled outpatient appointment.
- **Point in Time ($T_{\text{pred}}$):** The exact moment an appointment booking is confirmed in the Hospital Information System (HIS / EHR scheduling module).
- **Available Information:** Only metadata known at or before $T_{\text{pred}}$ (booking timestamp, scheduled appointment timestamp, patient administrative demographic proxies, clinic department, booking channel, and past visit history completed strictly prior to $T_{\text{pred}}$).
- **Prohibited Information:** Any clinical test results, post-booking outcomes, doctor notes, or future visit occurrences.

---

## 2. Dataset Sourcing & Leakage Analysis

### 2.1 Dataset Specifications & Synthetic Generation
To guarantee strict patient privacy (HIPAA / GDPR compliance) while avoiding restricted personal health identifiers, a statistically grounded, realistic synthetic scheduling generator was developed (`src/data_generator.py`). The dataset generates **25,000 appointment records across 8,000 distinct patients** over the calendar year 2025.

#### Distributional Parameters:
- **Baseline No-Show Prevalence:** **18.49%** (4,622 / 25,000), precisely mirroring outpatient benchmarks in public healthcare datasets (e.g., Kaggle Medical Appointment No-Shows, UK NHS outpatient records).
- **Lead Time ($T_{\text{lead}} = T_{\text{appointment}} - T_{\text{booking}}$):** Exponentially distributed with a mean of 11 days (median 7 days, max capped at 90 days). Same-day appointments ($T_{\text{lead}} = 0$) occur with low no-show propensity.
- **Patient Age:** Multi-modal distribution representing pediatric ($12\%$), adult ($58\%$), and geriatric ($30\%$) populations ($1 \le \text{age} \le 94$).
- **Distance to Clinic:** Log-normally distributed ($\mu = 1.8, \sigma = 0.65$), bounded between 0.5 and 45.0 miles.
- **Administrative Clinic Departments:** 8 outpatient services (`General Medicine`, `Orthopedics`, `Ophthalmology`, `Dermatology`, `Cardiology Consultation`, `Physiotherapy`, `Pediatrics`, `Dental Care`).
- **Scheduling Channels:** `Online Portal` (38%), `Call Center` (34%), `Walk-in Reception` (12%), `Mobile App` (16%).
- **Financial Subsidy / Welfare Assistance:** Binary indicator (15% prevalence).

### 2.2 Dynamic Chronological Feature Calculation (Zero-Leakage Guarantee)
A primary failure mode in published no-show models is calculating patient historical no-show rates across the entire dataset (including future appointments). In our pipeline:
1. Records are sorted strictly chronologically by $(T_{\text{appointment}}, T_{\text{booking}})$.
2. For appointment $k$ of patient $p$, historical metrics (`prior_appointments_count`, `prior_no_shows_count`, and `prior_no_show_rate`) are computed using an expanding window over visits occurring **strictly prior** to $T_{\text{booking}}$.
3. Bayesian smoothing with a prior pseudo-count $\text{Beta}(1, 4)$ is applied to regularize small sample histories:
$$\text{Smoothed Prior Rate} = \frac{\text{Prior No-Shows} + 1.0}{\text{Prior Appointments} + 5.0}$$

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Preprocessing Architecture
Feature transformation is encapsulated inside an exportable Scikit-Learn `Pipeline` coupled with a `ColumnTransformer`:
- **DomainFeatureEngineer Transformer:**
  - `lead_time_bucket`: Categorizes lead times into `Same-Day (0d)`, `Short (1-3d)`, `Medium (4-14d)`, `Long (15-30d)`, and `Extended (>30d)`.
  - `age_group`: Categorizes age into `Pediatric (<18)`, `Young Adult (18-34)`, `Middle Aged (35-64)`, and `Senior (65+)`.
  - `transit_barrier_flag`: Interaction feature triggering when `distance_miles > 15` AND `financial_assistance == 1`.
  - `is_same_day`: Binary indicator for zero lead-time bookings.
- **Imputation:**
  - Numeric variables imputed via `SimpleImputer(strategy='median')`.
  - Categorical variables imputed via `SimpleImputer(strategy='constant', fill_value='Unknown')`.
- **Scaling & Encoding:**
  - Continuous numeric features scaled via `StandardScaler()`.
  - Categorical features encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- Total transformed feature dimensionality: **45 numerical inputs**.

---

## 4. Model Development & Validation Strategy

### 4.1 Chronological Split Strategy
Random $k$-fold cross-validation inflates performance metrics on time-series operational data by training on future patterns to predict past events. To replicate production deployment:
- **Training Set (Earliest 70%):** 17,500 records (2025-01-01 to 2025-09-15) | No-show rate: 18.57%
- **Validation Set (Middle 15%):** 3,750 records (2025-09-15 to 2025-11-07) | No-show rate: 18.53%
- **Held-Out Test Set (Final 15%):** 3,750 records (2025-11-07 to 2025-12-30) | No-show rate: 18.08%

The held-out test set remained strictly sequestered during preprocessing fitting, model training, hyperparameter exploration, probability calibration, and threshold tuning.

### 4.2 Candidate Classifiers
Four architectures were trained and compared:
1. **Dummy Baseline:** Majority class / prior probability classifier ($\hat{P}(Y=1) = \bar{y}_{\text{train}}$).
2. **Logistic Regression (L2 Regularized):** Linear model with balanced class weighting (`class_weight='balanced'`), serving as a transparent, linear baseline.
3. **Random Forest Classifier:** Non-linear ensemble with 150 trees, `max_depth=12`, `min_samples_split=10`, and balanced class weights.
4. **LightGBM Classifier (Champion):** Gradient Boosted Decision Tree (GBDT) with 220 estimators, `learning_rate=0.04`, `num_leaves=31`, `max_depth=6`, `subsample=0.85`, and `colsample_bytree=0.85`.

---

## 5. Experimental Results & Performance Comparison

All models were evaluated on the held-out test partition ($N=3,750$).

### 5.1 Comprehensive Test Metrics Table

| Model Architecture | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Brier Score | Total Operational Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 0.5000 | 0.1808 | 0.0000 | 0.0000 | 0.0000 | 0.1481 | \$81,360 |
| **Logistic Regression (L2)** | **0.6500** | **0.3107** | **0.3556** | 0.2576 | 0.5737 | 0.2307 | \$51,495 |
| **Random Forest** | 0.6145 | 0.2644 | 0.2954 | 0.2743 | 0.3201 | 0.2005 | \$63,930 |
| **LightGBM (Raw)** | 0.6306 | 0.2879 | 0.3411 | 0.2543 | 0.5177 | 0.2229 | \$54,675 |
| **LightGBM + Isotonic Calibration ($t=0.50$)** | 0.6307 | 0.2716 | 0.0533 | **0.5429** | 0.0280 | **0.1431** | \$79,320 |
| **LightGBM + Calibration ($t^*=0.12$ Cost-Tuned)** | 0.6307 | 0.2716 | 0.3233 | 0.1956 | **0.9307** | **0.1431** | **\$44,565** |

### 5.2 Key Diagnostic Visualizations
All diagnostic figures are automatically generated by the training pipeline and archived under `artifacts/figures/`:
1. `roc_pr_curves.png`: Compares true positive rate vs. false positive rate and precision-recall trade-offs.
2. `calibration_curve.png`: Reliability diagram confirming that post-calibration probabilities match observed no-show frequencies.
3. `cost_curve_thresholds.png`: Dual-axis curve plotting total operational cost ($) and F1-score across the entire decision threshold spectrum ($0.05 \le t \le 0.95$).
4. `confusion_matrices.png`: Side-by-side comparison of error distributions between default $t=0.50$ and cost-optimal $t=0.12$.
5. `shap_summary.png`: Global beeswarm feature importance distribution.
6. `subgroup_fairness.png`: Metric parity across age demographics and financial assistance groups.

---

## 6. Threshold Optimization & Healthcare Operational Cost Matrix

### 6.1 Asymmetric Healthcare Costs
In hospital scheduling, classification errors carry vastly asymmetric financial and operational costs:
- **Cost of False Negative ($C_{\text{FN}} = \$120$):** A patient fails to show up without prior notice. The allocated doctor, specialized examination room, and nursing staff sit idle. The clinic slot is completely unrecoverable, directly destroying capacity and revenue.
- **Cost of False Positive ($C_{\text{FP}} = \$15$):** A patient is flagged as high-risk, but attends normally. The cost is limited to an automated multi-channel SMS plus a 90-second phone check-in by an administrative coordinator.
- **Benefit of True Positive ($B_{\text{TP}} = \$90$):** An at-risk patient is identified early. Proactive outreach confirms attendance, reschedules the patient if necessary, or activates an overbooking buffer to backfill the slot from the urgent waitlist.

The operational objective function minimizes total expected cost:
$$\mathcal{C}(t) = C_{\text{FN}} \cdot \text{FN}(t) + C_{\text{FP}} \cdot \text{FP}(t)$$

### 6.2 Empirical Threshold Optimization Results
Evaluating thresholds across validation data revealed that the default threshold ($t=0.50$) is completely misaligned with hospital operations:
- **Default Cutoff ($t = 0.50$):**
  - Caught only 19 no-shows, missing **659 true no-shows** (Recall = 2.80%).
  - Total operational loss on test set: **\$79,320.00** (\$21.15 per scheduled appointment).
- **Cost-Optimal Cutoff ($t^* = 0.12$):**
  - Identified **631 out of 678 true no-shows** (Recall = **93.07%**).
  - Reduced False Negatives from 659 down to **47**.
  - Total operational cost: **\$44,565.00** (\$11.88 per scheduled appointment).
  - **Net Operational Savings: \$34,755.00 (a 43.82% cost reduction on 3,750 appointments).**

---

## 7. Model Explainability & Interpretability (SHAP Analysis)

To ensure operations staff understand model recommendations, SHAP (`shap.TreeExplainer`) was integrated for both global system governance and local instance-level explainability.

### 7.1 Global Feature Importance
The top influential factors driving outpatient no-show probabilities across the population are:
1. **`lead_time_days` (+):** The single strongest predictor. Every additional 7 days between booking and appointment increases no-show log-odds. Patients who book same-day or 1-2 days ahead demonstrate strong attendance.
2. **`prior_no_show_rate` (+):** Patients with a demonstrated history of missed appointments are substantially more likely to miss future slots.
3. **`sms_reminder_sent` (-):** Dispatching an SMS confirmation notice reduces the odds of a missed visit.
4. **`patient_age` (-):** Senior patients (65+) demonstrate the highest adherence, whereas young adults (18–34) exhibit higher cancellation and absence volatility.
5. **`distance_miles` (+):** Patients living greater than 15–20 miles from the clinic experience transit barriers that elevate missed visits.
6. **`booking_channel_Walk-in Reception` (-):** Appointments booked in person at clinic reception have significantly lower no-show rates compared to web portal bookings.

### 7.2 Local Instance-Level Risk Attribution
For every prediction served by the API or Streamlit dashboard, the system computes exact SHAP feature attributions. For example, for an at-risk appointment:
- `lead_time_days = 24` $\to$ **+0.42 log-odds (Increases Risk)**
- `sms_reminder_sent = 0` $\to$ **+0.38 log-odds (Increases Risk)**
- `distance_miles = 19.2` $\to$ **+0.21 log-odds (Increases Risk)**
- `booking_channel = Walk-in` $\to$ **-0.28 log-odds (Decreases Risk)**

This transparency allows clinic staff to see *why* an appointment was flagged and choose the appropriate intervention (e.g., dispatching an SMS vs. offering transportation coordination).

---

## 8. Responsible AI, Subgroup Parity & Ethical Guardrails

### 8.1 Subgroup Parity Audit
A model audit was performed across demographic proxies on the held-out test set:

| Subgroup Category | Cohort | Cohort Size | True No-Show Rate | Precision | Recall ($t=0.12$) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Age Bracket** | Pediatric (<18) | 476 | 17.86% | 0.1963 | 87.06% | 0.3203 |
| **Age Bracket** | Young Adult (18-34) | 909 | 20.24% | 0.2202 | 98.37% | 0.3598 |
| **Age Bracket** | Middle Aged (35-64) | 1,784 | 18.39% | 0.1913 | 95.43% | 0.3187 |
| **Age Bracket** | Senior (65+) | 581 | 13.94% | 0.1611 | 77.78% | 0.2669 |
| **Financial Assistance** | No Assistance | 3,147 | 17.25% | 0.1865 | 92.45% | 0.3105 |
| **Financial Assistance** | Subsidy / Welfare | 603 | 22.39% | 0.2411 | 95.56% | 0.3851 |

#### Analysis:
- The model maintains high recall ($\ge 87\%$) across all adult and pediatric groups, and 77.8% among seniors (who already exhibit the lowest baseline no-show rate of 13.9%).
- Patients receiving financial subsidies have slightly higher no-show prevalence (22.39% vs. 17.25%), driven by transit and work scheduling hurdles. The model achieves **95.56% recall** for this group, ensuring these vulnerable patients receive proactive assistance rather than falling through administrative cracks.

### 8.2 Mandatory Decision-Support Safeguards
1. **Strictly Non-Clinical Decision Support:** Predictions are restricted to operational administrative workflows. The model never makes diagnostic, urgency, or treatment recommendations.
2. **Zero Care Deprivation Guarantee:** Under hospital policy, an at-risk prediction must **NEVER** be used to cancel an appointment, delay care, or deprioritize a patient.
3. **Supportive Intervention Spectrum:**
   - **Low Risk ($p < 0.20$):** Standard automated SMS 48 hours prior.
   - **Medium Risk ($0.20 \le p < 0.45$):** Two-way interactive confirmation SMS.
   - **High Risk ($p \ge 0.45$):** Personal phone call from clinic scheduling coordinator to resolve scheduling or transportation hurdles, with eligibility for flexible backfill buffer slots.
4. **Human-in-the-Loop:** All phone outreach and schedule adjustments are conducted by human administrative personnel.

---

## 9. Production Architecture & Delivery Package

### 9.1 Software Architecture
The repository is structured following clean, modular software engineering patterns:
```
├── README.md                          # Production operations & acceptance documentation
├── REPORT.md                          # Full Senior ML Engineer technical report
├── requirements.txt                   # Fixed dependency specifications
├── Makefile                           # Unified automation targets (setup, train, test, api)
├── data/
│   ├── raw/appointments.csv          # Generated baseline scheduling dataset (25,000 records)
├── src/
│   ├── __init__.py
│   ├── config.py                      # System configuration, paths, features, cost parameters
│   ├── data_generator.py              # Reproducible, leak-free synthetic appointment generator
│   ├── data_loader.py                 # Ingestion, data-quality cleaning, chronological splitting
│   ├── features.py                    # Domain feature engineering & Scikit-Learn transformers
│   ├── train.py                       # Training, calibration, threshold search, artifact export
│   ├── evaluate.py                    # Metric computations, curves, cost analysis, fairness audit
│   ├── explainability.py              # SHAP TreeExplainer, local and global attribution
│   ├── schemas.py                     # Pydantic v2 schemas for API validation
│   └── api.py                         # Production FastAPI ASGI service
├── app/
│   └── streamlit_app.py               # Streamlit Hospital Operations Command Center
├── artifacts/
│   ├── model.joblib                   # Serialized LightGBM champion model
│   ├── calibrated_model.joblib        # Serialized Isotonic Calibrated model
│   ├── preprocessor.joblib            # Serialized feature engineering pipeline
│   ├── explainer.joblib               # Serialized SHAP explainer
│   ├── metadata.json                  # Experiment metadata, metrics, and threshold settings
│   └── figures/                       # High-resolution diagnostic plots
├── tests/
│   ├── test_data.py                   # Data integrity, non-clinical constraints, leak-free splits
│   ├── test_pipeline.py               # Transformer array transformations, feature shapes
│   ├── test_inference.py              # Artifact loading, non-hardcoded dynamic scoring tests
│   └── test_api.py                    # FastAPI endpoint integration tests
└── submission/
    ├── SUBMISSION_FORM.md             # Completed Algions assessment submission form
    └── DEMO_SCRIPT.md                 # 3-5 minute video presentation script
```

### 9.2 API Endpoints
The REST API is implemented in FastAPI (`src/api.py`) with full Pydantic v2 data validation:
- `GET /health`: Health status, model version, and readiness flag.
- `GET /model/metadata`: Deployment metadata, evaluation metrics, and operational cost matrix.
- `POST /predict`: Real-time scoring for single appointment payload; returns calibrated probability, risk tier, action recommendation, and top 5 SHAP factors.
- `POST /predict/batch`: High-throughput scoring for daily clinic schedules.

### 9.3 Automated Acceptance Test Suite
All 12 automated unit and integration tests execute cleanly via `pytest`:
- `test_data.py`: Validates non-negative lead times, age bounds, non-clinical constraints, and chronological split boundaries.
- `test_pipeline.py`: Validates numerical transformation, imputer behavior, and feature expansion.
- `test_inference.py`: Proves genuine model execution (low-risk vs. high-risk inputs produce distinct calibrated probabilities) and validates SHAP local outputs.
- `test_api.py`: Validates root, health, metadata, single prediction, batch prediction, and input validation rejection ($422$ on invalid negative lead time).

---

## 10. Failure Modes, Limitations & Monitoring Strategy

### 10.1 Known Limitations
1. **Unobserved External Factors:** Extreme weather events, local transportation strikes, and acute personal emergencies cannot be anticipated from administrative booking data.
2. **Behavioral Interventions Feedback Loop:** If phone outreach succeeds in convincing an at-risk patient to attend, the model's prediction may appear as a "False Positive" in future retraining data, artificially depressing observed precision.
3. **Department-Specific Variance:** Specialized clinics (e.g., surgical consultations vs. routine dental checkups) exhibit differing patient commitment levels.

### 10.2 Ongoing Production Monitoring Plan
- **Data Drift Tracking:** Monitor population shifts in lead time, booking channel adoption, and distance via Evidently AI or Kolmogorov-Smirnov / Population Stability Index (PSI).
- **Concept Drift Tracking:** Monthly tracking of rolling Brier score and calibration curve stability.
- **Quarterly Retraining:** Retrain quarterly on an expanding chronological window with automated acceptance tests before champion artifact promotion.

---

## Conclusion

The Hospital Appointment No-Show Prediction system delivers a production-grade, mathematically defensible machine learning solution for hospital outpatient management. By rejecting naive 0.50 thresholds in favor of cost-optimal calibration, enforcing strict temporal split hygiene, and providing actionable SHAP explainability, the system provides hospital staff with a reliable decision-support tool that protects revenue, recovers capacity, and respects patient equity.
