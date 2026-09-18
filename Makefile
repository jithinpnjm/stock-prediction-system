.PHONY: setup lint typecheck test ci pipeline dvc-status sequence-tcn sequence-transformer sweep targets

setup:
	python -m pip install -e ".[dev,ml,mlops,ingest,research]"
	pre-commit install

lint:
	ruff check src pipelines tests

typecheck:
	mypy src

test:
	pytest

ci: lint test

pipeline:
	python -m pipelines

sequence-tcn:
	python pipelines/07_sequence_train.py tcn

sequence-transformer:
	python pipelines/07_sequence_train.py transformer

sweep:
	python pipelines/07_sweep_lgbm.py --trials 100

targets:
	python pipelines/05_target_ladder.py

dvc-status:
	dvc status
