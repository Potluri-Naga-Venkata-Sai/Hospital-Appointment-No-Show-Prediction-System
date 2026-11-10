# How to Run the Project: Step-by-Step Guide
### Hospital Appointment No-Show Prediction System (Algions AI Labs)

This guide provides complete instructions to set up, train, test, and run the Hospital Appointment No-Show Prediction system from scratch.

---

## 📋 System Requirements
- **Operating System:** macOS, Linux, or Windows (WSL recommended)
- **Python Version:** Python 3.9+ (Python 3.9, 3.10, 3.11, or 3.12)
- **Virtual Environment Tool:** `venv` (standard Python library)
- **Disk Space:** ~500 MB for virtual environment and training artifacts

---

## ⚡ Option 1: Quick Start (Using Makefile)

If you have `make` installed, you can run all steps using simple commands:

```bash
# 1. Setup virtual environment and install dependencies
make setup

# 2. Generate the synthetic appointment dataset (25,000 records)
make data

# 3. Train models, calibrate probabilities, optimize threshold, and save artifacts
make train

# 4. Run automated acceptance test suite (12 tests)
make test

# 5. Launch the Streamlit Operations Command Center (Dashboard)
make dashboard

# 6. (In a separate terminal) Launch the FastAPI REST service
make api
```

---

## 🛠️ Option 2: Step-by-Step Manual Setup

If you prefer running standard Python / Bash commands without `make`, follow these steps:

### Step 1: Open Terminal & Navigate to Project Directory
```bash
cd "Algions AI -Project"
```

### Step 2: Initialize Virtual Environment & Install Dependencies
```bash
# Create Python virtual environment (.venv)
python3 -m venv .venv

# Activate the virtual environment:
# On macOS / Linux:
source .venv/bin/activate
# On Windows (Command Prompt):
# .venv\Scripts\activate.bat
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### Step 3: (Optional) Set Up Environment Variables
Copy the example environment configuration:
```bash
cp .env.example .env
```
*(Default settings work out-of-the-box without modifying `.env`)*

---

## 🔄 Pipeline Execution Steps

### Step 4: Generate Synthetic Dataset
Generates 25,000 appointment records with 8,000 distinct patient histories, zero clinical leakage, and dynamic expanding-window attendance metrics:
```bash
PYTHONPATH=. .venv/bin/python src/data_generator.py
```
- **Output:** `data/raw/appointments.csv`

### Step 5: Train Models, Calibrate & Generate Artifacts
Executes chronological splitting (70% train, 15% val, 15% test), trains baseline and tree models, applies Isotonic probability calibration, optimizes decision threshold based on hospital economics ($C_{\text{FN}}=\$120, C_{\text{FP}}=\$15$), generates SHAP TreeExplainer, and exports diagnostic plots:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python src/train.py
```
- **Generated Artifacts:**
  - `artifacts/preprocessor.joblib` (Feature transformer pipeline)
  - `artifacts/calibrated_model.joblib` (Champion Calibrated LightGBM)
  - `artifacts/explainer.joblib` (SHAP TreeExplainer artifact)
  - `artifacts/metadata.json` (Experiment metrics, optimal threshold $t^*=0.12$, cost savings)
  - `artifacts/figures/*.png` (ROC/PR curves, Calibration curve, Cost curve, Confusion matrices, SHAP summary, Fairness audit)

### Step 6: Verify Acceptance Criteria & Run Tests
Run the comprehensive verification runner and full Pytest suite:
```bash
# 1. Run full acceptance verification script
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python verify_acceptance.py

# 2. Or run standard Pytest suite
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib .venv/bin/pytest -v tests/
```
All **12 unit and integration tests** should pass.

---

## 🌐 Running the User & API Interfaces

### Step 7: Launch the Streamlit Operations Command Center
Launch the interactive web application designed for clinic operations staff and coordinators:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib .venv/bin/streamlit run app/streamlit_app.py
```
- **Dashboard URL:** [http://localhost:8501](http://localhost:8501)
- **Features:**
  - **Single Appointment Triage:** Live sliders for patient parameters, dynamic risk gauge, tier recommendation, and interactive SHAP waterfall attribution.
  - **Daily Schedule Batch Triage:** View clinic roster, filter by risk level, and export CSV call list.
  - **Model Performance & Calibration:** Inspect ROC-AUC, PR-AUC, Calibration curves, and interactive operational cost calculator.
  - **Subgroup Fairness & Ethics:** Inspect error slices across age groups and financial assistance tiers.

### Step 8: Launch the Production FastAPI REST API
In a new terminal window (with the virtual environment activated), start the FastAPI ASGI server:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib .venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
- **API Base URL:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check Endpoint:** `GET http://localhost:8000/health`

---

## 🧪 Testing the API via cURL

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "explainer_loaded": true,
  "optimal_threshold": 0.12,
  "champion_model": "Calibrated LightGBM (Isotonic)"
}
```

### 2. Single Appointment Risk Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "appointment_id": "APT-7721",
       "patient_id": "PAT-4021",
       "booking_date": "2025-06-01",
       "appointment_date": "2025-06-20",
       "lead_time_days": 19,
       "patient_age": 27,
       "distance_miles": 18.5,
       "financial_assistance": 1,
       "clinic_department": "Orthopedics",
       "booking_channel": "Online Portal",
       "appointment_time_slot": "Late Afternoon (14:00-17:00)",
       "appointment_day_of_week": "Friday",
       "sms_reminder_sent": 0,
       "prior_appointments_count": 3,
       "prior_no_shows_count": 2,
       "prior_no_show_rate": 0.50,
       "is_first_appointment": 0
     }'
```
**Expected Response:**
```json
{
  "appointment_id": "APT-7721",
  "patient_id": "PAT-4021",
  "no_show_probability": 0.3959,
  "is_at_risk": true,
  "risk_tier": "Medium",
  "decision_threshold": 0.12,
  "recommended_action": "Standard automated SMS reminder 48h prior + Two-way SMS confirmation.",
  "top_factors": [
    {
      "feature": "prior_no_show_rate",
      "shap_value": 0.4846,
      "direction": "Increases Risk"
    },
    {
      "feature": "patient_age",
      "shap_value": 0.2680,
      "direction": "Increases Risk"
    },
    {
      "feature": "appointment_day_of_week_Friday",
      "shap_value": 0.2148,
      "direction": "Increases Risk"
    }
  ],
  "disclaimer": "DECISION-SUPPORT NOTICE: This prediction is for operations planning (reminders, scheduling buffers) only. It must NEVER be used to withhold care, cancel slots, or alter clinical treatment."
}
```

---

## ❓ Troubleshooting & FAQs

1. **Port already in use (`Address already in use: 8501` or `8000`)**:
   - Check which process is using the port:
     ```bash
     lsof -i :8501 -i :8000
     ```
   - Either terminate the existing process or launch on a different port:
     ```bash
     # Run Streamlit on port 8502
     streamlit run app/streamlit_app.py --server.port 8502

     # Run FastAPI on port 8001
     uvicorn src.api:app --host 0.0.0.0 --port 8001
     ```

2. **Matplotlib Font / Cache Warnings on Headless Environments**:
   - Prefix commands with `MPLCONFIGDIR=/tmp/matplotlib`:
     ```bash
     MPLCONFIGDIR=/tmp/matplotlib python src/train.py
     ```

3. **ModuleNotFoundError: No module named 'src'**:
   - Ensure you are in the root project directory and prepend `PYTHONPATH=.`:
     ```bash
     PYTHONPATH=. python src/train.py
     ```

4. **Cleaning temporary files**:
   ```bash
   make clean
   # Or manually:
   find . -type d -name "__pycache__" -exec rm -rf {} +
   rm -rf .pytest_cache
   ```

