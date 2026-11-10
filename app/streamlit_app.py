"""
Streamlit Decision-Support Dashboard for Hospital Outpatient Operations.
Provides live appointment risk scoring, local SHAP explainability, batch schedule triage,
and model governance analytics.
"""

import json
from datetime import date, timedelta
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Hospital No-Show Decision Support | Algions AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODEL_PATH = ARTIFACTS_DIR / "calibrated_model.joblib"
RAW_MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
EXPLAINER_PATH = ARTIFACTS_DIR / "explainer.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DATA_PATH = BASE_DIR / "data" / "raw" / "appointments.csv"


@st.cache_resource
def load_ml_assets():
    """Load serialized preprocessing, model, explainer, and metadata."""
    if not PREPROCESSOR_PATH.exists() or not (MODEL_PATH.exists() or RAW_MODEL_PATH.exists()):
        return None, None, None, None

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH if MODEL_PATH.exists() else RAW_MODEL_PATH)
    
    explainer_dict = None
    if EXPLAINER_PATH.exists():
        explainer_dict = joblib.load(EXPLAINER_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

    return preprocessor, model, explainer_dict, metadata


@st.cache_data
def load_sample_schedule():
    """Load sample appointments for batch demonstration."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # Take the most recent 100 appointments from the test window
        return df.tail(100).copy()
    return pd.DataFrame()


preprocessor, model, explainer_dict, metadata = load_ml_assets()

# Sidebar: Governance & System Info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital-3.png", width=64)
    st.title("Algions AI Labs")
    st.caption("Hospital Management ML Operations")
    st.divider()

    optimal_threshold = metadata.get("optimal_threshold", 0.12)
    pipeline_version = metadata.get("pipeline_version", "1.0.0")

    st.markdown(f"**Model:** `{metadata.get('champion_model', 'Calibrated LightGBM')}`")
    st.markdown(f"**Version:** `{pipeline_version}`")
    st.markdown(f"**Decision Threshold:** `{optimal_threshold:.2f}`")
    st.markdown(f"**Cost Reduction:** `43.8% vs Default`")
    st.divider()

    st.markdown("### ⚠️ Decision Support Protocol")
    st.info(
        "This tool supports operational reminder allocation and buffer planning only. "
        "It must **NEVER** be used to cancel appointments, prioritize clinical urgency, "
        "or withhold healthcare services."
    )


# Main Application Header
st.title("🏥 Hospital Appointment No-Show Prediction System")
st.markdown(
    "**Operational Decision-Support Tool** for clinic coordinators to proactively identify "
    "unattended appointment risks, allocate tailored reminders, and minimize idle provider capacity."
)

if preprocessor is None or model is None:
    st.error("Model artifacts not found! Please execute `python src/train.py` to train and build artifacts.")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Appointment Triage",
    "📋 Daily Schedule Batch Triage",
    "📊 Model Performance & Calibration",
    "⚖️ Subgroup Fairness & Ethics",
])

# ==========================================================
# TAB 1: Single Appointment Real-Time Triage
# ==========================================================
with tab1:
    st.subheader("Interactive Appointment Risk Scoring")
    st.write("Enter scheduling parameters to estimate no-show probability and inspect contributing factors.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📅 Scheduling Details")
        booking_date = st.date_input("Booking Confirmation Date", value=date(2025, 6, 1))
        lead_time = st.slider("Lead Time (Days between booking & visit)", min_value=0, max_value=60, value=14)
        appointment_date = booking_date + timedelta(days=lead_time)
        st.caption(f"Calculated Appointment Date: **{appointment_date.strftime('%Y-%m-%d')}**")

        clinic_dept = st.selectbox(
            "Clinic Department (Administrative)",
            [
                "General Medicine",
                "Orthopedics",
                "Ophthalmology",
                "Dermatology",
                "Cardiology Consultation",
                "Physiotherapy",
                "Pediatrics",
                "Dental Care",
            ],
        )

        booking_channel = st.selectbox(
            "Booking Channel",
            ["Online Portal", "Call Center", "Walk-in Reception", "Mobile App"],
        )

        time_slot = st.selectbox(
            "Appointment Time Slot",
            [
                "Early Morning (08:00-10:00)",
                "Late Morning (10:00-12:00)",
                "Early Afternoon (12:00-14:00)",
                "Late Afternoon (14:00-17:00)",
            ],
        )

        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            index=2,
        )

    with col2:
        st.markdown("#### 👤 Operational Patient Profile")
        patient_age = st.slider("Patient Age (Years)", min_value=1, max_value=100, value=38)
        distance = st.slider("Distance to Hospital (Miles)", min_value=0.5, max_value=40.0, value=7.2, step=0.5)

        fin_assist = st.radio(
            "Healthcare Subsidy / Financial Assistance?",
            options=[0, 1],
            format_func=lambda x: "Enrolled in Subsidy/Assistance" if x == 1 else "Standard/Self-Pay",
            horizontal=True,
        )

        sms_sent = st.radio(
            "SMS Confirmation Reminder Dispatched?",
            options=[1, 0],
            format_func=lambda x: "Yes (SMS Delivered)" if x == 1 else "No / Pending",
            horizontal=True,
        )

        st.markdown("##### Prior Attendance Record")
        c_a, c_b = st.columns(2)
        with c_a:
            prior_appts = st.number_input("Prior Total Visits", min_value=0, max_value=30, value=2)
        with c_b:
            prior_no_shows = st.number_input("Prior Missed Visits", min_value=0, max_value=prior_appts, value=0)

        is_first = 1 if prior_appts == 0 else 0
        prior_rate = round((prior_no_shows + 1.0) / (prior_appts + 5.0), 4)

    st.markdown("---")
    if st.button("🔍 Evaluate Appointment Risk", type="primary", use_container_width=True):
        # Assemble payload
        single_dict = {
            "appointment_id": "APT-SIMULATED",
            "patient_id": "PAT-SIMULATED",
            "booking_date": booking_date.strftime("%Y-%m-%d"),
            "appointment_date": appointment_date.strftime("%Y-%m-%d"),
            "lead_time_days": int(lead_time),
            "patient_age": int(patient_age),
            "distance_miles": float(distance),
            "financial_assistance": int(fin_assist),
            "clinic_department": clinic_dept,
            "booking_channel": booking_channel,
            "appointment_time_slot": time_slot,
            "appointment_day_of_week": day_of_week,
            "sms_reminder_sent": int(sms_sent),
            "prior_appointments_count": int(prior_appts),
            "prior_no_shows_count": int(prior_no_shows),
            "prior_no_show_rate": float(prior_rate),
            "is_first_appointment": int(is_first),
        }

        # Run inference
        df_input = pd.DataFrame([single_dict])
        X_trans = preprocessor.transform(df_input)
        prob = float(model.predict_proba(X_trans)[0, 1])

        # Tier & Action
        if prob < 0.20:
            tier = "Low"
            tier_color = "green"
            action = "Automated standard SMS reminder 48 hours prior. Standard slot allocation."
        elif prob < 0.45:
            tier = "Medium"
            tier_color = "orange"
            action = "Two-way confirmation SMS with 24hr response window. Automated reminder call if unconfirmed."
        else:
            tier = "High"
            tier_color = "red"
            action = "Priority phone outreach by hospital scheduling coordinator 3 days prior. Eligible for flexible scheduling buffer."

        # Display results in cards
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Predicted No-Show Probability", f"{prob * 100:.1f}%")
            st.caption(f"Cost-Optimal Decision Threshold: **{optimal_threshold * 100:.1f}%**")
        with r2:
            st.markdown(f"**Risk Tier:** :{tier_color}[**{tier} RISK**]")
            st.progress(min(prob, 1.0))
        with r3:
            is_flagged = prob >= optimal_threshold
            st.markdown(f"**Operational Triage:** {'🚨 **Action Required**' if is_flagged else '✅ **Normal Workflow**'}")
            st.caption("Based on hospital capacity loss trade-off ($120 FN vs $15 FP)")

        st.info(f"📋 **Recommended Staff Action:** {action}")

        # SHAP Local Explainability
        if explainer_dict is not None:
            st.markdown("#### 🔬 Explainability: Key Drivers for This Score")
            from src.explainability import explain_instance
            contribs = explain_instance(explainer_dict, X_trans[0], top_k=6)
            
            # Bar chart of SHAP values
            feat_df = pd.DataFrame(contribs)
            
            fig, ax = plt.subplots(figsize=(8, 3.5))
            colors = ["#e07a5f" if v > 0 else "#2b5c8f" for v in feat_df["shap_value"]]
            ax.barh(feat_df["feature"], feat_df["shap_value"], color=colors)
            ax.axvline(0, color="black", lw=0.8, linestyle="--")
            ax.set_xlabel("SHAP Impact on No-Show Log-Odds (Red = Increases Risk, Blue = Decreases Risk)")
            ax.set_title("Local Feature Contribution (Why the model scored this appointment)", fontweight="bold")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ==========================================================
# TAB 2: Daily Operations Schedule & Batch Triage
# ==========================================================
with tab2:
    st.subheader("Daily Clinic Schedule Operational Triage")
    st.write(
        "Upload or inspect an upcoming clinic appointment roster to prioritize staff outreach and buffer slots."
    )

    sample_df = load_sample_schedule()
    if not sample_df.empty:
        # Score sample appointments
        X_batch_trans = preprocessor.transform(sample_df)
        batch_probs = model.predict_proba(X_batch_trans)[:, 1]

        scored_df = sample_df.copy()
        scored_df["no_show_probability"] = np.round(batch_probs, 3)
        scored_df["risk_tier"] = [
            "Low" if p < 0.20 else "Medium" if p < 0.45 else "High" for p in batch_probs
        ]
        scored_df["action_required"] = scored_df["no_show_probability"] >= optimal_threshold
        scored_df["priority_rank"] = scored_df["no_show_probability"].rank(ascending=False).astype(int)

        # Overview KPI metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Upcoming Appointments", len(scored_df))
        m2.metric("High-Risk Outliers", (scored_df["risk_tier"] == "High").sum())
        m3.metric("Medium-Risk Followups", (scored_df["risk_tier"] == "Medium").sum())
        m4.metric("Estimated At-Risk Slots", int(scored_df["no_show_probability"].sum()))

        st.markdown("#### Filtered Call Outreach Roster")
        tier_filter = st.multiselect(
            "Filter by Risk Tier",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
        )
        filtered = scored_df[scored_df["risk_tier"].isin(tier_filter)].sort_values(
            by="no_show_probability", ascending=False
        )

        display_cols = [
            "priority_rank",
            "appointment_id",
            "patient_id",
            "clinic_department",
            "appointment_time_slot",
            "lead_time_days",
            "no_show_probability",
            "risk_tier",
        ]
        st.dataframe(
            filtered[display_cols].head(30),
            use_container_width=True,
            hide_index=True,
        )

        # Download call sheet
        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Prioritized Staff Call Sheet (CSV)",
            data=csv_data,
            file_name="hospital_prioritized_reminder_calls.csv",
            mime="text/csv",
        )


# ==========================================================
# TAB 3: Model Performance & Calibration Audit
# ==========================================================
with tab3:
    st.subheader("Defensible Evaluation & Probability Calibration")
    st.write(
        "Rigorous verification of candidate classifiers on the strictly held-out test set."
    )

    c1, c2 = st.columns(2)
    roc_fig = FIGURES_DIR / "roc_pr_curves.png"
    calib_fig = FIGURES_DIR / "calibration_curve.png"
    cost_fig = FIGURES_DIR / "cost_curve_thresholds.png"
    cm_fig = FIGURES_DIR / "confusion_matrices.png"

    if roc_fig.exists():
        with c1:
            st.image(str(roc_fig), caption="ROC & Precision-Recall Curves across Models")
    if calib_fig.exists():
        with c2:
            st.image(str(calib_fig), caption="Reliability Diagram (Probability Calibration)")

    c3, c4 = st.columns(2)
    if cost_fig.exists():
        with c3:
            st.image(str(cost_fig), caption="Expected Cost vs Decision Threshold Curve")
    if cm_fig.exists():
        with c4:
            st.image(str(cm_fig), caption="Confusion Matrix: Default (0.50) vs Optimal (0.12)")

    st.markdown("### 💰 Operational Cost Model Calculator")
    st.write(
        "Demonstrates how balancing False Negatives (idle clinic slots) against False Positives (reminder costs) "
        "drives optimal operational decision-making."
    )
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        fn_val = st.slider("Cost of False Negative ($ - Unattended Slot)", 50, 300, 120, step=10)
    with col_k2:
        fp_val = st.slider("Cost of False Positive ($ - Reminder Outreach)", 5, 50, 15, step=5)

    test_opt = metadata.get("champion_test_metrics_optimal", {})
    cm_opt = test_opt.get("confusion_matrix", {"fn": 47, "fp": 2595})
    test_def = metadata.get("champion_test_metrics_default", {})
    cm_def = test_def.get("confusion_matrix", {"fn": 659, "fp": 16})

    opt_cost = (cm_opt["fn"] * fn_val) + (cm_opt["fp"] * fp_val)
    def_cost = (cm_def["fn"] * fn_val) + (cm_def["fp"] * fp_val)
    saved = def_cost - opt_cost

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Default Threshold Cost", f"${def_cost:,.0f}")
    kpi2.metric("Optimal Threshold Cost", f"${opt_cost:,.0f}")
    kpi3.metric("Projected Savings", f"${saved:,.0f}", delta=f"{saved:,.0f}")


# ==========================================================
# TAB 4: Subgroup Fairness & Governance
# ==========================================================
with tab4:
    st.subheader("Responsible AI, Subgroup Parity & Ethical Boundaries")
    st.write(
        "Audit of model performance across demographic brackets to ensure equitable service delivery."
    )

    fair_fig = FIGURES_DIR / "subgroup_fairness.png"
    shap_fig = FIGURES_DIR / "shap_summary.png"

    col_f1, col_f2 = st.columns(2)
    if fair_fig.exists():
        with col_f1:
            st.image(str(fair_fig), caption="Subgroup Parity Audit (Age & Financial Assistance)")
    if shap_fig.exists():
        with col_f2:
            st.image(str(shap_fig), caption="Global SHAP Feature Importance Summary")

    st.markdown("#### Model Limitations & Risk Mitigation Guidelines")
    st.markdown(
        """
        1. **Non-Clinical Scope Guarantee:** The model does not include medical diagnosis, prescription history, 
           or health conditions. This guarantees no clinical bias creeps into operational decisions.
        2. **Human-in-the-Loop Oversight:** Staff coordinators review all high-risk flags before placing telephone calls.
        3. **Equitable Outreach:** Patients flagged with financial subsidy or transportation barriers receive supportive transit 
           assistance and appointment rescheduling assistance, rather than penalties.
        4. **Zero Penalty Policy:** A high-risk score never deprioritizes care or auto-cancels a patient slot.
        """
    )

