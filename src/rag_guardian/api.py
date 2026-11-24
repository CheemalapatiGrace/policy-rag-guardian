"""FastAPI application: async query API, review/audit/trace endpoints, health probes and the dashboard."""


import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .auth import KeyStore, Principal
from .config import Settings
from .hitl import ReviewConflict, ReviewError, ReviewNotFound
from .pipeline import RAGPipeline, build_pipeline
from .ratelimit import TokenBucketLimiter
from .tracing import configure_json_logging

DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"
log = logging.getLogger("rag_guardian.api")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class DecisionRequest(BaseModel):
    action: str = Field(pattern="^(approve|edit|reject)$")
    edited_answer: str | None = Field(default=None, max_length=8000)
    note: str = Field(default="", max_length=1000)


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    s = settings or (pipeline.settings if pipeline else Settings.from_env())
    pipe = pipeline or build_pipeline(s)
    keys = KeyStore(s.api_keys, demo=s.demo_keys)
    limiter = TokenBucketLimiter(s.rate_limit_per_min, s.rate_limit_burst)
    semaphore = asyncio.Semaphore(s.max_concurrency)

    app = FastAPI(title="policy-rag-guardian", version="1.0.0",
                  description="Policy Q&A with RAG, guardrails, observability and human review.")
    app.state.pipeline = pipe
    app.state.limiter = limiter
    if s.demo_keys and not s.api_keys:
        log.warning("demo API keys are enabled; set RAG_DEMO_KEYS=0 and RAG_API_KEYS for real deployments")

    # ---- auth dependencies --------------------------------------------------------------------
    def principal(x_api_key: Annotated[str | None, Header()] = None) -> Principal:
        p = keys.authenticate(x_api_key)
        if p is None:
            pipe.db.append_audit("anonymous", "none", "auth_failure", "", {})
            raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})
        return p

    def staff(p: Annotated[Principal, Depends(principal)]) -> Principal:
        if not p.has_any("reviewer", "admin"):
            raise HTTPException(403, "reviewer or admin role required")
        return p

    def admin(p: Annotated[Principal, Depends(principal)]) -> Principal:
        if not p.has_any("admin"):
            raise HTTPException(403, "admin role required")
        return p

    # ---- probes -------------------------------------------------------------------------------
    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        info = pipe.health()
        ready = info["index"]["chunks"] > 0 and pipe.db.ping()
        degraded = [n for n, st in info["breakers"].items() if st != "closed"]
        body = {"status": "ready" if ready else "not_ready", "degraded": degraded, **info}
        return JSONResponse(body, status_code=200 if ready else 503)

    # ---- query --------------------------------------------------------------------------------
    @app.post("/v1/query")
    async def query(req: QueryRequest, request: Request, p: Annotated[Principal, Depends(principal)]) -> dict[str, Any]:
        allowed, retry_after = limiter.check(p.user_id)
        if not allowed:
            pipe.db.append_audit(p.user_id, p.label, "rate_limited", "", {})
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(max(1, int(retry_after + 0.999)))})
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=2.0)  # bulkhead: shed load instead of queueing forever
        except TimeoutError as exc:
            raise HTTPException(503, "server busy", headers={"Retry-After": "1"}) from exc
        try:
            result = await asyncio.wait_for(pipe.aanswer(req.question, p, req.top_k), timeout=s.request_timeout_s)
        except TimeoutError as exc:
            raise HTTPException(504, "request timed out") from exc
        finally:
            semaphore.release()
        return result.public()

    @app.get("/v1/queries/{trace_id}")
    async def query_status(trace_id: str, p: Annotated[Principal, Depends(principal)]) -> dict[str, Any]:
        t = pipe.db.get_trace(trace_id)
        if t is None or (t["user_id"] != p.user_id and not p.has_any("reviewer", "admin")):
            raise HTTPException(404, "unknown trace")
        released = t["status"] in ("answered", "review_approved", "review_edited")
        message = {"pending_review": "Still awaiting human review.",
                   "review_rejected": "A reviewer declined to release an answer to this question."}.get(
            t["status"], t["final_answer"] or "")
        return {"trace_id": trace_id, "status": t["status"], "answer": t["final_answer"] if released else None,
                "message": message, "citations": t["citations"] if released else []}

    # ---- observability ------------------------------------------------------------------------
    @app.get("/v1/traces")
    async def traces(_: Annotated[Principal, Depends(staff)], limit: int = Query(50, ge=1, le=500),
                     status: str | None = None) -> dict[str, Any]:
        return {"traces": pipe.db.list_traces(limit, status), "stats": pipe.db.stats()}

    @app.get("/v1/traces/{trace_id}")
    async def trace_detail(trace_id: str, _: Annotated[Principal, Depends(staff)]) -> dict[str, Any]:
        t = pipe.db.get_trace(trace_id)
        if t is None:
            raise HTTPException(404, "unknown trace")
        return t

    @app.get("/v1/stats")
    async def stats(_: Annotated[Principal, Depends(staff)]) -> dict[str, Any]:
        return {**pipe.db.stats(), **pipe.health()}

    # ---- human in the loop --------------------------------------------------------------------
    @app.get("/v1/reviews")
    async def reviews(_: Annotated[Principal, Depends(staff)], status: str = "pending") -> dict[str, Any]:
        return {"reviews": pipe.db.list_reviews(status)}

    @app.post("/v1/reviews/{review_id}/decision")
    async def decide(review_id: int, req: DecisionRequest, p: Annotated[Principal, Depends(staff)]) -> dict[str, Any]:
        try:
            return pipe.reviews.decide(review_id, p, req.action, req.edited_answer, req.note)
        except ReviewNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except ReviewConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ReviewError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/audit")
    async def audit(_: Annotated[Principal, Depends(staff)], limit: int = Query(100, ge=1, le=1000),
                    action: str | None = None) -> dict[str, Any]:
        return {"events": pipe.db.list_audit(limit, action)}

    @app.get("/v1/audit/verify")
    async def audit_verify(_: Annotated[Principal, Depends(staff)]) -> dict[str, Any]:
        return pipe.db.verify_audit()

    # ---- admin --------------------------------------------------------------------------------
    @app.post("/v1/ingest")
    async def ingest(p: Annotated[Principal, Depends(admin)]) -> dict[str, Any]:
        report = await asyncio.to_thread(pipe.reingest)
        pipe.db.append_audit(p.user_id, p.label, "ingest", "corpus", report.to_dict())
        return report.to_dict()

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        return DASHBOARD.read_text(encoding="utf-8")

    return app


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import uvicorn

    configure_json_logging()
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")),
                log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
