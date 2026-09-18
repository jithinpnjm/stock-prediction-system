.PHONY: setup lint typecheck test ci pipeline dvc-status

setup:
	python -m pip install -e ".[dev,ingest,research]"
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

dvc-status:
	dvc status
