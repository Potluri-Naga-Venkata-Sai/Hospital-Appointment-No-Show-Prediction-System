"""
Synthetic Data Generator for Hospital Appointment No-Show Prediction.
Generates realistic, de-identified scheduling data with strict temporal ordering
and zero clinical or diagnostic variables.
"""

import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.config import RANDOM_SEED, RAW_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_appointments(
    n_appointments: int = 25000,
    n_patients: int = 8000,
    start_date_str: str = "2025-01-01",
    end_date_str: str = "2025-12-31",
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic hospital appointments with realistic operational patterns,
    patient behavioral histories, and non-clinical factors.
    Strictly avoids clinical diagnosis, urgency, or medical outcome fields.
    """
    np.random.seed(random_seed)
    logger.info("Initializing synthetic appointment generator (n=%d, patients=%d)...", n_appointments, n_patients)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    date_range_days = (end_date - start_date).days

    # 1. Patient Base Demographics (Non-sensitive)
    patient_ids = [f"PAT-{i:05d}" for i in range(1, n_patients + 1)]
    
    # Age distribution: mix of pediatric, working adult, and geriatric peaks
    ages = np.concatenate([
        np.random.normal(9, 3, int(n_patients * 0.12)),      # Pediatric
        np.random.normal(36, 10, int(n_patients * 0.58)),    # Adults
        np.random.normal(68, 8, int(n_patients * 0.30)),     # Seniors
    ])
    ages = np.clip(ages, 1, 94).astype(int)
    np.random.shuffle(ages)
    
    # Distance in miles: log-normal distribution (mostly within 15 miles)
    distances = np.round(np.random.lognormal(mean=1.8, sigma=0.65, size=n_patients), 1)
    distances = np.clip(distances, 0.5, 45.0)

    # Financial assistance / subsidy enrollment (15% prevalence)
    assistance = np.random.binomial(n=1, p=0.15, size=n_patients)

    # Base patient intrinsic adherence propensity (latent trait: mean 0, std 0.5)
    patient_adherence_trait = np.random.normal(0, 0.5, size=n_patients)

    patient_df = pd.DataFrame({
        "patient_id": patient_ids,
        "patient_age": ages,
        "distance_miles": distances,
        "financial_assistance": assistance,
        "adherence_trait": patient_adherence_trait,
    })

    # 2. Appointment Generation
    departments = [
        "General Medicine",
        "Orthopedics",
        "Ophthalmology",
        "Dermatology",
        "Cardiology Consultation",
        "Physiotherapy",
        "Pediatrics",
        "Dental Care",
    ]
    dept_weights = [0.28, 0.15, 0.12, 0.11, 0.10, 0.10, 0.08, 0.06]

    booking_channels = ["Online Portal", "Call Center", "Walk-in Reception", "Mobile App"]
    channel_weights = [0.38, 0.34, 0.12, 0.16]

    time_slots = [
        "Early Morning (08:00-10:00)",
        "Late Morning (10:00-12:00)",
        "Early Afternoon (12:00-14:00)",
        "Late Afternoon (14:00-17:00)",
    ]
    slot_weights = [0.25, 0.35, 0.15, 0.25]

    # Sample patients with Pareto-like distribution (some frequent visitors)
    patient_weights = np.random.pareto(a=1.5, size=n_patients)
    patient_weights /= patient_weights.sum()
    sampled_patient_indices = np.random.choice(n_patients, size=n_appointments, p=patient_weights)

    records = []
    
    # Generate dates
    for i in range(n_appointments):
        p_idx = sampled_patient_indices[i]
        pat = patient_df.iloc[p_idx]

        # Appointment date sampled across the year
        appt_offset = np.random.randint(0, date_range_days)
        appt_date = start_date + timedelta(days=appt_offset)
        # Avoid Sunday appointments
        if appt_date.weekday() == 6:
            appt_date += timedelta(days=1)

        # Lead time: days between booking and appointment (exponential distribution)
        # e.g., median ~ 7 days, max ~ 90 days
        lead_time = int(np.random.exponential(scale=11))
        lead_time = min(lead_time, 90)
        book_date = appt_date - timedelta(days=lead_time)

        dept = np.random.choice(departments, p=dept_weights)
        channel = np.random.choice(booking_channels, p=channel_weights)
        slot = np.random.choice(time_slots, p=slot_weights)

        # SMS reminder policy: more likely if booked >= 3 days ahead
        if lead_time >= 3:
            sms_sent = int(np.random.binomial(n=1, p=0.82))
        else:
            sms_sent = 0

        records.append({
            "appointment_id": f"APT-{i+1:06d}",
            "patient_id": pat["patient_id"],
            "booking_date": book_date.strftime("%Y-%m-%d"),
            "appointment_date": appt_date.strftime("%Y-%m-%d"),
            "lead_time_days": lead_time,
            "patient_age": int(pat["patient_age"]),
            "distance_miles": float(pat["distance_miles"]),
            "financial_assistance": int(pat["financial_assistance"]),
            "adherence_trait": float(pat["adherence_trait"]),
            "clinic_department": dept,
            "booking_channel": channel,
            "appointment_time_slot": slot,
            "sms_reminder_sent": sms_sent,
        })

    raw_df = pd.DataFrame(records)

    # 3. Chronological sorting to compute historical features with ZERO leakage
    raw_df["appointment_date_dt"] = pd.to_datetime(raw_df["appointment_date"])
    raw_df["booking_date_dt"] = pd.to_datetime(raw_df["booking_date"])
    raw_df.sort_values(by=["appointment_date_dt", "booking_date_dt"], inplace=True)
    raw_df.reset_index(drop=True, inplace=True)

    # Day of week
    raw_df["appointment_day_of_week"] = raw_df["appointment_date_dt"].dt.day_name()

    # Track historical appointment outcomes dynamically
    patient_history = {pid: {"total": 0, "no_shows": 0} for pid in patient_ids}

    prior_counts = []
    prior_no_shows = []
    prior_rates = []
    is_first_appts = []
    outcomes = []

    logger.info("Simulating realistic appointment attendance and dynamic patient histories...")

    for _, row in raw_df.iterrows():
        pid = row["patient_id"]
        hist = patient_history[pid]
        p_total = hist["total"]
        p_missed = hist["no_shows"]

        prior_counts.append(p_total)
        prior_no_shows.append(p_missed)
        is_first_appts.append(1 if p_total == 0 else 0)

        # Bayesian smoothed prior rate with mild global prior (Beta(1, 4) ~ 20%)
        smoothed_rate = (p_missed + 1.0) / (p_total + 5.0)
        prior_rates.append(round(smoothed_rate, 4))

        # Causal behavioral formula for log-odds of No-Show
        # Baseline log-odds around -1.6 (approx ~17% base rate)
        log_odds = -1.60

        # Lead time impact: +0.025 per day (~ +0.5 for 20 days lead)
        log_odds += 0.025 * row["lead_time_days"]
        if row["lead_time_days"] == 0:
            log_odds -= 0.85  # Same-day appointments rarely missed

        # Historical attendance influence
        if p_total > 0:
            log_odds += 2.2 * (smoothed_rate - 0.20)
        else:
            log_odds += 0.15  # First-time patient uncertainty

        # Age factors
        if row["patient_age"] < 12:
            log_odds -= 0.25  # Parents take kids consistently
        elif 18 <= row["patient_age"] <= 30:
            log_odds += 0.35  # Young adults higher churn
        elif row["patient_age"] >= 65:
            log_odds -= 0.35  # Retirees have reliable attendance

        # Distance factor
        if row["distance_miles"] > 20.0:
            log_odds += 0.30
        elif row["distance_miles"] < 3.0:
            log_odds -= 0.15

        # Financial assistance factor (often faces transit/work flexibility hurdles)
        if row["financial_assistance"] == 1:
            log_odds += 0.20

        # SMS reminder mitigation
        if row["sms_reminder_sent"] == 1:
            log_odds -= 0.45

        # Booking channel effect
        if row["booking_channel"] == "Walk-in Reception":
            log_odds -= 0.40
        elif row["booking_channel"] == "Online Portal":
            log_odds += 0.10

        # Time slot and day effects
        if row["appointment_day_of_week"] == "Friday" and "Late Afternoon" in row["appointment_time_slot"]:
            log_odds += 0.35
        elif row["appointment_day_of_week"] == "Monday" and "Early Morning" in row["appointment_time_slot"]:
            log_odds += 0.25

        # Intrinsic patient adherence trait
        log_odds -= 1.0 * row["adherence_trait"]

        # Convert log-odds to probability
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        # Add slight stochastic noise
        prob = float(np.clip(prob + np.random.normal(0, 0.02), 0.01, 0.99))

        # Bernoulli trial for outcome
        no_show_outcome = int(np.random.binomial(n=1, p=prob))
        outcomes.append(no_show_outcome)

        # Update patient history strictly for subsequent dates
        patient_history[pid]["total"] += 1
        patient_history[pid]["no_shows"] += no_show_outcome

    # Attach engineered history columns
    raw_df["prior_appointments_count"] = prior_counts
    raw_df["prior_no_shows_count"] = prior_no_shows
    raw_df["prior_no_show_rate"] = prior_rates
    raw_df["is_first_appointment"] = is_first_appts
    raw_df["no_show"] = outcomes

    # Drop unobservable latent trait & temp datetimes
    final_df = raw_df.drop(columns=["adherence_trait", "appointment_date_dt", "booking_date_dt"])

    logger.info("Dataset generated successfully. Total records: %d", len(final_df))
    logger.info("Overall No-Show rate: %.2f%% (%d / %d)", 
                final_df["no_show"].mean() * 100, 
                final_df["no_show"].sum(), 
                len(final_df))
    
    return final_df


def save_dataset(df: pd.DataFrame, file_path=RAW_DATA_PATH) -> None:
    """Save dataset to target CSV path."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info("Saved raw dataset to: %s", file_path)


if __name__ == "__main__":
    df = generate_synthetic_appointments()
    save_dataset(df)

