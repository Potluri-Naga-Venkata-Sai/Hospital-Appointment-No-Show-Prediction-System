# 3–5 Minute Video Demonstration Walkthrough Script
## Hospital Appointment No-Show Prediction System
**Candidate Presentation Guide for Algions AI Labs Evaluators**

---

### Video Demonstration Checklist
- **Duration:** 3 minutes 30 seconds to 4 minutes 30 seconds.
- **Audio:** Clear voiceover with screen recording.
- **Visuals:** Split between VS Code / Terminal, Streamlit Dashboard, and FastAPI Swagger Docs.

---

### Step-by-Step Script & On-Screen Actions

#### ⏱️ 0:00 – 0:45 | Scene 1: Introduction & Problem Framing
- **On Screen:** Title slide or `README.md` repository view.
- **Presenter Script:**
  > *"Hello, everyone. My name is [Your Name], and this is my individual project assessment for Algions AI Labs in the AI & Machine Learning role. My assigned domain is Hospital Management, focusing on Hospital Appointment No-Show Prediction.*
  >
  > *Outpatient no-shows severely hurt healthcare operations, leaving costly clinical staff and examination rooms idle while inflating patient waitlists. Our mission is to build a decision-support ML system that predicts the probability of an appointment being missed at booking confirmation time.*
  >
  > *Crucially, in compliance with regulatory and clinical ethics, this model strictly excludes clinical diagnoses, ICD codes, and medical urgency. It operates purely on administrative, scheduling, and past attendance metadata to support human-reviewed reminder allocation."*

---

#### ⏱️ 0:45 – 1:30 | Scene 2: Data Engineering, Zero Leakage & Splitting Strategy
- **On Screen:** Show terminal running `pytest` or code in `src/data_loader.py` and `src/train.py`.
- **Presenter Script:**
  > *"Let's look at the data engineering pipeline. We generated 25,000 realistic, de-identified appointment records across 8,000 patients. 
  >
  > To ensure zero future leakage, patient attendance histories are computed dynamically using an expanding window strictly prior to each appointment's booking time.
  >
  > Furthermore, instead of a naive random k-fold split that leaks future patterns into the past, we enforce a strict chronological partition: the earliest 70% of appointments from January to September 2025 form our training set; the next 15% through November serve as our validation and threshold tuning set; and the final 15% in November and December are sequestered as our true held-out test set."*

---

#### ⏱️ 1:30 – 2:30 | Scene 3: Model Evaluation, Calibration & Threshold Optimization
- **On Screen:** Open the Streamlit dashboard on the **"Model Performance & Calibration"** tab, displaying the ROC/PR curves, reliability diagram, and cost curve.
- **Presenter Script:**
  > *"Now turning to our experimental results. We benchmarked a Dummy Baseline, L2-Regularized Logistic Regression, Random Forest, and LightGBM.
  >
  > LightGBM achieved the strongest discriminatory performance with a PR-AUC of 0.272 and ROC-AUC of 0.631. But raw tree scores are often miscalibrated. We applied Isotonic Probability Calibration, which brought our Brier score down to 0.1431, ensuring that a predicted 30% risk truly corresponds to a 30% empirical event rate.
  >
  > More importantly, in hospital scheduling, errors are asymmetric. A False Negative—an empty doctor slot—costs approximately $120 in lost clinical capacity, whereas a False Positive—an extra reminder call—costs only $15.
  >
  > If we used the standard 0.50 threshold, the model would miss 97% of true no-shows, costing $79,320. By optimizing our threshold mathematically against the cost matrix, we discovered an optimal cutoff of t = 0.12. At this threshold, our model catches 93.1% of all true no-shows, slashing total operational losses by 43.8%—saving $34,755 on 3,750 test appointments."*

---

#### ⏱️ 2:30 – 3:30 | Scene 4: Live Inference & Local SHAP Explainability
- **On Screen:** Switch to the Streamlit **"Single Appointment Triage"** tab.
- **Presenter Script:**
  > *"Now let's demonstrate real-time inference on a live appointment. Notice this is not a hardcoded stub; it loads our serialized preprocessor and calibrated LightGBM artifact directly from disk.
  >
  > Let's simulate an at-risk scenario: a 27-year-old patient who booked 19 days in advance for an Orthopedics consultation on a Friday afternoon, who hasn't received an SMS reminder, lives 18 miles away, and missed 2 out of 3 previous visits.
  >
  > When I click 'Evaluate Appointment Risk', the model processes the 45 transformed features and estimates a 52.8% no-show probability. It categorizes this as HIGH RISK and recommends our highest operational tier: a priority phone outreach by a scheduling coordinator 3 days prior, making the slot eligible for a flexible overbooking buffer.
  >
  > Right below, our integrated SHAP TreeExplainer breaks down the exact drivers: the long lead time, absence of an SMS reminder, prior missed visits, and distance are clearly shown pushing the risk upward.
  >
  > Conversely, if we change the profile to a 72-year-old senior booking a walk-in appointment on the same day with an SMS reminder and a clean visit history, the probability plummets to 4.2%—a LOW RISK requiring only standard automated SMS reminders."*

---

#### ⏱️ 3:30 – 4:15 | Scene 5: API, Daily Batch Operations & Responsible AI Governance
- **On Screen:** Briefly show the **FastAPI Swagger UI** at `http://localhost:8000/docs`, execute a `/predict` call, and show the **"Subgroup Fairness & Ethics"** tab on Streamlit.
- **Presenter Script:**
  > *"For production integration into Hospital Information Systems, we also exposed a complete FastAPI REST API supporting single and batch scoring with Pydantic validation.
  >
  > Finally, responsible AI and governance are built into the architecture. We audited subgroup parity across age brackets and financial subsidy tiers, confirming high recall across all adult and pediatric cohorts.
  >
  > Most importantly, the system includes a strict ethical guarantee: this model is a supportive decision tool for reminders and buffer planning only. Under no circumstances is it ever used to cancel appointments, alter medical urgency, or withhold healthcare from patients.
  >
  > All 12 unit and integration tests are passing cleanly. Thank you for your time, and I look forward to the technical walkthrough!"*

---

### Tips for Recording:
1. **Screen Resolution:** 1080p (1920x1080).
2. **Audio:** Use a headset mic to minimize background noise.
3. **Pacing:** Speak at a steady, confident pace; pause briefly when switching tabs.
