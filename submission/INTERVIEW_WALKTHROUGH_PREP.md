# Algions AI Labs — Technical Walkthrough & Live Interview Preparation Guide
## Hospital Appointment No-Show Prediction System

This document is prepared for **Naga Venkata Sai Potluri** for the **15-minute technical walkthrough** and **10-minute live change/debugging task** conducted by Algions AI Labs evaluators.

---

## Part 1: 15-Minute Technical Walkthrough Structure

Evaluators will ask you to walk through your architecture, justify technical decisions, and explain trade-offs.

### 1. Business Problem & Operational Framing (2–3 mins)
- **Question:** *"Why did you frame this problem as decision-support rather than automated decision-making?"*
  - **Answer:** In healthcare operations, automated actions like unilaterally cancelling a patient's slot create immense clinical and legal liability. Our system provides administrative decision support—identifying at-risk slots to allocate reminder interventions (automated SMS, two-way confirmation, priority coordinator phone call) and manage buffer slots without denying or delaying patient care.
- **Question:** *"How did you ensure compliance with non-clinical constraints?"*
  - **Answer:** We strictly excluded any medical variables—no ICD-10 diagnosis codes, prescriptions, or clinical notes. We relied exclusively on administrative scheduling metadata (lead time, channel, time slot, appointment history, travel distance proxy), ensuring zero clinical bias.

### 2. Temporal Partitioning & Leakage Prevention (3–4 mins)
- **Question:** *"Why not use stratified 5-fold cross-validation?"*
  - **Answer:** Standard random k-fold cross-validation introduces severe temporal data leakage in appointment scheduling. A patient booking in December would be used to predict an appointment in January, allowing future operational trends and patient habits to leak backwards. Instead, we used a strict chronological split:
    - Train (70%): Jan 1 to Sep 15, 2025 (17,500 records)
    - Validation (15%): Sep 15 to Nov 7, 2025 (3,750 records)
    - Test (15%): Nov 7 to Dec 30, 2025 (3,750 records)
- **Question:** *"How did you calculate patient historical no-show rates without leakage?"*
  - **Answer:** In `src/data_generator.py` and `src/data_loader.py`, we maintained an expanding temporal window. For appointment $k$ of patient $p$, prior visits and no-shows were calculated strictly using records completed before the booking confirmation date, regularized with Bayesian smoothing:
    $$\text{Smoothed Rate} = \frac{\text{Prior No-Shows} + 1.0}{\text{Prior Visits} + 5.0}$$

### 3. Model Benchmark & Probability Calibration (3–4 mins)
- **Question:** *"Why did you calibrate the model using Isotonic Regression?"*
  - **Answer:** Tree ensembles with class weighting (`class_weight='balanced'`) push probabilities toward the extremes, distorting true probabilities. For operational decision-making, staff need reliable risk probabilities. We applied post-hoc Isotonic Regression on the validation set, which dropped our Brier score to **0.1431**, ensuring that a predicted 30% probability matches an empirical 30% failure rate.

### 4. Threshold Optimization & Economic Matrix (3–4 mins)
- **Question:** *"Why did you change the threshold from 0.50 to 0.12?"*
  - **Answer:** In healthcare scheduling, the cost matrix is heavily asymmetric:
    - False Negative ($C_{\text{FN}} = \$120$): Doctor and room sit idle; slot is wasted.
    - False Positive ($C_{\text{FP}} = \$15$): Staff makes an extra reminder call.
  - At the default 0.50 threshold, the model misses 97% of no-shows, resulting in an operational loss of \$79,320. At the cost-optimal threshold of $t^* = 0.12$, the model catches 93.1% of no-shows, reducing total operational cost to \$44,565—saving **\$34,755 (a 43.8% cost reduction)**.

---

## Part 2: 10-Minute Live Change / Debugging Tasks (Prepared Solutions)

The brief notes: *"Shortlisted candidates will attend a 10-minute live change/debugging task. You must explain your own architecture and modify your own work."*

Here are the 4 most likely live tasks and how to execute them in under 2 minutes:

### Task 1: "The hospital CFO says False Negatives cost $200 and False Positives cost $25. Recalculate the optimal threshold."
- **File to Edit:** `src/config.py`
```python
# Change lines in src/config.py:
COST_FALSE_NEGATIVE = 200.0
COST_FALSE_POSITIVE = 25.0
```
- **Execute:**
```bash
python src/train.py
```
- **Explanation:** The threshold search will automatically evaluate the new cost surface and update `artifacts/metadata.json`.

---

### Task 2: "Add a new engineered feature: `is_weekend_appointment` (Saturday booking)."
- **File to Edit:** `src/features.py` inside `DomainFeatureEngineer.transform()`:
```python
# Add under transform():
if "appointment_day_of_week" in df.columns:
    df["is_weekend"] = df["appointment_day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
```
- And update `numeric_cols` in `build_preprocessor()`:
```python
numeric_cols = NUMERIC_FEATURES + ["transit_barrier_flag", "is_same_day", "is_weekend"]
```
- **Run Tests:** `pytest -v tests/` to confirm all assertions pass.

---

### Task 3: "Add an endpoint `/predict/summary` to FastAPI that returns high-level risk breakdown for a batch."
- **File to Edit:** `src/api.py`:
```python
@app.post("/predict/summary", tags=["Inference"])
def predict_summary(payload: BatchPredictionRequest):
    res = predict_batch_appointments(payload)
    return {
        "total": res.total_appointments,
        "high_risk_percentage": round((res.high_risk_count / res.total_appointments) * 100, 1),
        "recommended_calls_needed": res.high_risk_count + res.medium_risk_count,
    }
```
- **Test in Browser:** Open `http://localhost:8000/docs` to show the new endpoint running live!

---

### Task 4: "Change the risk tier thresholds from (0.20, 0.45) to (0.15, 0.40)."
- **File to Edit:** `src/config.py`:
```python
RISK_THRESHOLD_LOW = 0.15
RISK_THRESHOLD_HIGH = 0.40
```
- Both `src/api.py` and `app/streamlit_app.py` import these constants directly, so restarting uvicorn or refreshing Streamlit instantly applies the change.
