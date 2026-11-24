FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

COPY data ./data

# run as an unprivileged user; state (SQLite + persisted index) lives in /data
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown -R app /data
USER app
ENV HOST=0.0.0.0 PORT=8000 RAG_DB_PATH=/data/guardrail.db RAG_INDEX_DIR=/data/index \
    RAG_CORPUS_DIR=/app/data/corpus RAG_REGRESSION_PATH=/data/regression.jsonl RAG_DEMO_KEYS=1
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status==200 else 1)"
CMD ["python", "-m", "rag_guardian.api"]
