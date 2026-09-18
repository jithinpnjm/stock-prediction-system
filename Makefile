.PHONY: setup lint test test-cov pipeline data features labels train validate backtest report ci

setup:
	python -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=term-missing --cov-fail-under=70

data:
	python scripts/download_banknifty.py --start 2020-01-01 --end $$(date +%Y-%m-%d)

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

features:
	python pipelines/03_aggregate_5m.py && python pipelines/04_features.py

labels:
	python pipelines/05_labels.py && python pipelines/06_build_dataset.py

train:
	python pipelines/07_train.py

validate:
	python pipelines/08_validate.py

backtest:
	python pipelines/09_backtest.py && python pipelines/10_report.py

ci: lint test
