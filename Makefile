VENV ?= .venv
PY := $(VENV)/bin/python
NB := $(VENV)/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400

.PHONY: help setup test lint scrape clean-data warehouse analysis all

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

setup:  ## Create .venv and install everything (analysis, scraping, dev tools, this package)
	python3 -m venv $(VENV)
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install -e .

test:  ## Run unit tests (data checks run only if data/cleaned exists)
	$(PY) -m pytest -q

lint:  ## Lint the package and tests
	$(PY) -m ruff check src tests

scrape:  ## Scrape into data/ (SOURCES="goodreads rottentomatoes"; takes hours; run playwright install chromium first)
	$(PY) -m story_reception.scrape $(SOURCES)

clean-data:  ## Rebuild data/cleaned/ from the raw CSVs in data/ (about 1.5 min)
	$(NB) data-cleaning.ipynb

warehouse:  ## Build data/reception.duckdb from data/cleaned/ and run the SQL quality checks (about 5 s)
	$(PY) -m story_reception.warehouse

analysis:  ## Rerun main.ipynb and regenerate figures/ (about 15 min)
	$(NB) main.ipynb

all: clean-data warehouse analysis test  ## Clean, build the warehouse, analyze, then test
