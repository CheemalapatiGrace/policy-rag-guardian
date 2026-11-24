PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install test lint eval run bench docker clean

install:  ## create .venv and install the package with dev tools
	python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:  ## unit + integration tests (fully offline)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

eval:  ## golden-set evaluation; exits non-zero when a gate fails (CI gate)
	$(PY) -m rag_guardian.evals

run:  ## API + dashboard on http://127.0.0.1:8000
	$(PY) -m rag_guardian.api

bench:  ## latency / throughput report (in-process)
	$(PY) scripts/bench.py --requests 500 --concurrency 8

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .data build dist src/*.egg-info
