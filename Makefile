.DEFAULT_GOAL := all

.PHONY: setup data build train evaluate analyze test lint all clean serve

setup:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py

build:
	python scripts/build_dataset.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

analyze:
	python scripts/error_analysis.py

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

all: setup data build train evaluate analyze
	@echo "Pipeline complete."

clean:
	rm -rf data/raw/* data/interim/* data/processed/*
	@echo "Cleaned data directories."

serve:
	uvicorn mmrerank.app:app --host 127.0.0.1 --port 8000 --reload

