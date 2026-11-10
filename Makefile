# Algions AI Labs - Hospital Appointment No-Show Prediction
.PHONY: help setup data train test api dashboard clean

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
UVICORN := .venv/bin/uvicorn
STREAMLIT := .venv/bin/streamlit

help:
	@echo "Hospital Appointment No-Show Prediction — Command Automation"
	@echo "------------------------------------------------------------"
	@echo "make setup      : Initialize virtual environment and install requirements"
	@echo "make data       : Generate leak-free synthetic appointment dataset (25,000 rows)"
	@echo "make train      : Run end-to-end model training, calibration, and evaluation"
	@echo "make test       : Execute pytest test suite (12 acceptance tests)"
	@echo "make api        : Launch FastAPI REST inference server on port 8000"
	@echo "make dashboard  : Launch Streamlit Operations Command Center on port 8501"
	@echo "make clean      : Clean Python bytecode and temp files"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	PYTHONPATH=. $(PYTHON) src/data_generator.py

train:
	PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib $(PYTHON) src/train.py

test:
	PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib $(PYTEST) -v tests/

api:
	PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib $(UVICORN) src.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib $(STREAMLIT) run app/streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

