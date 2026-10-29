VENV ?= .venv
PY := $(VENV)/bin/python
NB := $(VENV)/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400

.PHONY: help setup test lint scrape clean-data warehouse analysis gap books train-app app space all

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

gap:  ## Rerun gap-analysis.ipynb (critic-audience gap models, RQ2)
	$(NB) gap-analysis.ipynb

books:  ## Rerun divisive-books.ipynb (what makes a book divisive, RQ3)
	$(NB) divisive-books.ipynb

train-app:  ## Retrain the app models from data/cleaned/ and write app/artifacts/ (about 5 min)
	$(PY) app/train.py

app:  ## Run the Streamlit app locally at http://localhost:8501
	$(VENV)/bin/streamlit run app/app.py

space:  ## Build the Hugging Face Space bundle in build/space/ (publish: see app/build_space.py)
	$(PY) app/build_space.py

all: clean-data warehouse analysis gap books train-app test  ## Clean, build the warehouse, run all three analyses, retrain the app, then test
