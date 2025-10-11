# PROJECT VARIABLES
PYTHON ?= python
VENV ?= .venv
REQS ?= requirements.txt
PY := $(VENV)/Scripts/python.exe
PIP := $(VENV)/Scripts/pip.exe
UV ?= uv

ifeq ($(OS),Linux)
    PY := $(VENV)/bin/python
    PIP := $(VENV)/bin/pip
endif

# PROJECT FOLDER PATHS
SRC := src
INGESTPY := $(SRC)/data/ingest_and_eda.py
TRAIN := data/raw/train.csv
TEST := data/raw/test.csv
MANIFEST := data/processed/manifest.csv

# PROJECT COMMANDS
setup: $(VENV)
$(VENV):
	@echo ">> CREATING VIRTUAL ENVIRONMENT AT $(VENV) USING $(UV)"
	$(UV) venv $(VENV)

ifeq ($(OS),Windows_NT)
	@echo ">> UPGRADING PIP IN VIRTUAL ENV"
	$(VENV)/Scripts/python.exe -m ensurepip --upgrade
	$(VENV)/Scripts/python.exe -m pip install --upgrade pip
else
	$(VENV)/bin/python -m ensurepip --upgrade
	$(VENV)/bin/python -m pip install --upgrade pip
endif

	@echo ">> INSTALLING REQUIRED PACKAGES FROM $(REQS)"
	@if [ -f "$(REQS)" ]; then $(UV) pip install -r $(REQS); else echo ">> NO REQUIREMENTS FILE FOUND"; fi

ifeq ($(OS),Windows_NT)
	@echo ">> VIRTUAL ENV CREATED, ACTIVATE IT WITH: source $(VENV)/Scripts/activate"
else
	@echo ">> VIRTUAL ENV CREATED, ACTIVATE IT WITH: source $(VENV)/bin/activate"
endif

ingest:
	$(PY) $(INGESTPY) --train $(TRAIN) --test $(TEST) --out $(MANIFEST) --sample_download 300

clean-ingest:
	rm -rf $(MANIFEST)
	rm -rf reports/

extract-features:
	@echo ">> EXTRACTING FEATURES FROM MANIFEST"
	$(PY) $(SRC)/preprocessing/feature_extraction.py