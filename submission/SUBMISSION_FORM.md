# Algions AI Labs — Project Assessment Submission Form
## Advanced Individual Project Assessment — AI & Machine Learning

---

### Section 1: Candidate Identification
| Field | Details |
| :--- | :--- |
| **Candidate Full Name** | Naga Venkata Sai Potluri |
| **Registration Number** | [Enter Registration Number] |
| **University / Institutional Email** | pnvsai@example.com |

---

### Section 2: Assignment Confirmation
| Field | Details |
| :--- | :--- |
| **Internship Role** | AI & Machine Learning |
| **Assigned Project Title** | Hospital Appointment No-Show Prediction |
| **Assigned Domain** | Hospital Management & Clinic Operations |
| **Submission Date** | November 2026 |

---

### Section 3: Code and Version
| Field | Details |
| :--- | :--- |
| **Repository URL** | `https://github.com/pnvsai/hospital-noshow-prediction` |
| **Primary Branch** | `main` |
| **Final Commit ID** | `e2d0da0` |
| **Access Permissions** | Public or reviewer access granted |

---

### Section 4: Demonstration
| Field | Details |
| :--- | :--- |
| **3–5 Minute Video Demonstration Link** | `https://youtu.be/sample-demo-link` *(walkthrough following DEMO_SCRIPT.md)* |
| **Live Web App / Demo URL** | Declared Local-Only Demo: `http://localhost:8501` (Streamlit) & `http://localhost:8000` (FastAPI) |
| **Interactive API Documentation** | `http://localhost:8000/docs` (FastAPI OpenAPI Swagger UI) |

---

### Section 5: Documentation & System Architecture
| Document | Location / Link |
| :--- | :--- |
| **Technical Report** | [`REPORT.md`](file:///Users/pnvsai/Projects/Algions%20AI%20-Project/REPORT.md) |
| **Production README** | [`README.md`](file:///Users/pnvsai/Projects/Algions%20AI%20-Project/README.md) |
| **Video Demonstration Script** | [`submission/DEMO_SCRIPT.md`](file:///Users/pnvsai/Projects/Algions%20AI%20-Project/submission/DEMO_SCRIPT.md) |
| **Architecture Diagram** | Rendered Mermaid diagram in `README.md` and `REPORT.md` |

---

### Section 6: Technology Stack and Limitations
| Category | Details |
| :--- | :--- |
| **Programming Language** | Python 3.9+ |
| **Machine Learning Core** | Scikit-Learn 1.6+, LightGBM 4.6+, NumPy 2.0+, Pandas 2.3+, SciPy 1.13+ |
| **Explainability & Diagnostics** | SHAP 0.49+ (TreeExplainer), Matplotlib 3.9+, Seaborn 0.13+ |
| **Inference Interfaces** | FastAPI 0.128+ (ASGI), Uvicorn 0.39+, Pydantic 2.13+, Streamlit 1.50+ |
| **Automated Testing** | Pytest 8.4+, HTTPX 0.28+ (12/12 acceptance tests passing) |
| **Setup Notes** | `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| **Assumptions & Known Incomplete Features** | 1. Weather and transit strike disruptions are unobserved in administrative data. <br>2. Behavioral intervention feedback loops may cause successfully reminded patients to appear as False Positives in future retrospective retraining logs. <br>3. Model is restricted strictly to non-clinical decision support; automated cancellation of appointments is explicitly barred. |

---

### Section 7: AI and Attribution
| Field | Details |
| :--- | :--- |
| **AI Tools Used** | Antigravity AI Assistant |
| **Purpose of Use** | Accelerated scaffolding of boilerplate code, test fixtures, docstring formatting, and initial draft of markdown reports. |
| **Attribution & Sources** | 1. Synthetic dataset generator modeled after distributions from Kaggle Medical Appointment No-Shows (Javni, 2016) and NHS outpatient benchmark statistics. <br>2. Scikit-learn Pipeline and ColumnTransformer design patterns. <br>3. SHAP (Lundberg & Lee, 2017) for unified game-theoretic tree explainability. |

---

### Section 8: Role-Specific Assessment Fields
| Assessment Metric / Artifact | Recorded Value / Location |
| :--- | :--- |
| **Dataset Source / License** | Synthetic generator in `src/data_generator.py` (MIT License, zero PII / zero clinical data). Saved to `data/raw/appointments.csv`. |
| **Saved Model Artifact Location** | `artifacts/calibrated_model.joblib` and `artifacts/preprocessor.joblib` |
| **SHAP Explainer Artifact** | `artifacts/explainer.joblib` |
| **Test Set ROC-AUC** | **0.6307** |
| **Test Set PR-AUC (Average Precision)** | **0.2716** |
| **Test Set Brier Score (Calibration)** | **0.1431** (Post-Isotonic Calibration) |
| **Default Threshold ($t=0.50$) Recall / Cost** | Recall: **2.80%** \| Total Cost: **\$79,320.00** |
| **Cost-Optimal Threshold ($t^*=0.12$) Recall / Cost** | Recall: **93.07%** \| Total Cost: **\$44,565.00** |
| **Net Operational Savings** | **\$34,755.00 (43.82% cost reduction)** |

---

### Section 9: Formal Declarations
- [x] **Originality:** I declare that the submitted code, architecture, pipeline, and report were developed by me with complete engineering ownership.
- [x] **Walkthrough Readiness:** I confirm my readiness to walk through any line of code, explain the trade-offs, and complete the live 10-minute live modification task during the technical interview.
- [x] **No Restricted / Sensitive Data:** I declare that no real patient identifiers, HIPAA-restricted PHI, passwords, access keys, or secrets are committed.
- [x] **Link and Code Verification:** All local links, test suites (`pytest`), and interfaces (`FastAPI` & `Streamlit`) have been executed and verified in the environment.
