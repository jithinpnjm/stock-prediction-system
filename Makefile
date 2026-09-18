.PHONY: setup lint test clean pipeline

setup:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

pipeline:
	python pipelines/01_ingest.py
	python pipelines/02_validate.py
	python pipelines/03_aggregate_5m.py
	python pipelines/04_features.py
	python pipelines/05_labels.py
	python pipelines/06_build_dataset.py
	python pipelines/07_train.py
	python pipelines/08_validate.py
	python pipelines/09_backtest.py
	python pipelines/10_report.py
