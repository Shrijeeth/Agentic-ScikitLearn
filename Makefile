install:
	python -m pip install ".[dev]"

lint:
	python -m ruff check

auto-lint:
	python -m ruff format

test:
	ENVIRONMENT=test python -m pytest -v -s -p no:warnings

test-report:
	ENVIRONMENT=test python -m pytest -v -s -p no:warnings --cov=. --cov-report=html:coverage