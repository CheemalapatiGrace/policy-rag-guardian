"""SQLite persistence: traces, review queue and a hash-chained (tamper-evident) audit log."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY, created_at TEXT, user_id TEXT, roles TEXT, question TEXT, status TEXT, reason TEXT,
  latency_ms REAL, tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, confidence REAL, degraded TEXT,
  final_answer TEXT, citations TEXT, spans TEXT);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC);
CREATE TABLE IF NOT EXISTS reviews (
  review_id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, created_at TEXT, reasons TEXT, question TEXT,
  asker_roles TEXT, draft_answer TEXT, citations TEXT, status TEXT, reviewer TEXT, final_answer TEXT, note TEXT,
  decided_at TEXT);
CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, roles TEXT, action TEXT, target TEXT, details TEXT,
  prev_hash TEXT, hash TEXT);
"""

GENESIS = "0" * 64


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)

    def ping(self) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1").fetchone()[0] == 1

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- traces -------------------------------------------------------------------------------
    def save_trace(self, t: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t["trace_id"], t["created_at"], t["user_id"], t["roles"], t["question"], t["status"], t["reason"],
                 t["latency_ms"], t["tokens_in"], t["tokens_out"], t["cost_usd"], t["confidence"],
                 json.dumps(t["degraded"]), t["final_answer"], json.dumps(t["citations"]), json.dumps(t["spans"])))
            self._conn.commit()

    @staticmethod
    def _trace_row(r: sqlite3.Row, with_spans: bool) -> dict[str, Any]:
        d = {k: r[k] for k in r.keys()}  # noqa: SIM118
        d["degraded"] = json.loads(d["degraded"] or "[]")
        d["citations"] = json.loads(d["citations"] or "[]")
        spans = json.loads(d.pop("spans") or "[]")
        if with_spans:
            d["spans"] = spans
        else:
            d["span_summary"] = [{"name": s["name"], "duration_ms": s["duration_ms"], "status": s["status"]} for s in spans]
        return d

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
        return self._trace_row(r, True) if r else None

    def list_traces(self, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM traces", []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(max(1, min(limit, 500)))
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._trace_row(r, False) for r in rows]

    def update_trace_release(self, trace_id: str, status: str, final_answer: str | None) -> None:
        with self._lock:
            self._conn.execute("UPDATE traces SET status=?, final_answer=? WHERE trace_id=?", (status, final_answer, trace_id))
            self._conn.commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) c, AVG(latency_ms) l, SUM(cost_usd) cost, "
                                      "SUM(tokens_in) ti, SUM(tokens_out) tout FROM traces GROUP BY status").fetchall()
            pending = self._conn.execute("SELECT COUNT(*) FROM reviews WHERE status='pending'").fetchone()[0]
        by = {r["status"]: {"count": r["c"], "avg_latency_ms": round(r["l"] or 0, 2)} for r in rows}
        return {"by_status": by, "total": sum(v["count"] for v in by.values()), "pending_reviews": pending,
                "cost_usd": round(sum(r["cost"] or 0 for r in rows), 6),
                "tokens_in": sum(r["ti"] or 0 for r in rows), "tokens_out": sum(r["tout"] or 0 for r in rows)}

    # ---- reviews ------------------------------------------------------------------------------
    def create_review(self, trace_id: str, reasons: list[str], question: str, asker_roles: str, draft: str,
                      citations: list[dict]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO reviews (trace_id, created_at, reasons, question, asker_roles, draft_answer, citations, status)"
                " VALUES (?,?,?,?,?,?,?, 'pending')",
                (trace_id, now_iso(), json.dumps(reasons), question, asker_roles, draft, json.dumps(citations)))
            self._conn.commit()
            return int(cur.lastrowid)

    @staticmethod
    def _review_row(r: sqlite3.Row) -> dict[str, Any]:
        d = {k: r[k] for k in r.keys()}  # noqa: SIM118
        d["reasons"] = json.loads(d["reasons"] or "[]")
        d["citations"] = json.loads(d["citations"] or "[]")
        return d

    def get_review(self, review_id: int) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM reviews WHERE review_id=?", (review_id,)).fetchone()
        return self._review_row(r) if r else None

    def list_reviews(self, status: str | None = "pending", limit: int = 100) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM reviews", []
        if status and status != "all":
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY review_id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [self._review_row(r) for r in self._conn.execute(q, args).fetchall()]

    def decide_review(self, review_id: int, status: str, reviewer: str, final_answer: str | None, note: str) -> bool:
        """Atomically move a *pending* review to a terminal state. Returns False if already decided."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE reviews SET status=?, reviewer=?, final_answer=?, note=?, decided_at=? "
                "WHERE review_id=? AND status='pending'", (status, reviewer, final_answer, note, now_iso(), review_id))
            self._conn.commit()
            return cur.rowcount == 1

    # ---- audit log ----------------------------------------------------------------------------
    @staticmethod
    def _digest(prev: str, ts: str, actor: str, roles: str, action: str, target: str, details: str) -> str:
        return hashlib.sha256("|".join([prev, ts, actor, roles, action, target, details]).encode()).hexdigest()

    def append_audit(self, actor: str, roles: str, action: str, target: str = "", details: dict | None = None) -> int:
        det = json.dumps(details or {}, sort_keys=True, default=str)
        with self._lock:
            row = self._conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS
            ts = now_iso()
            h = self._digest(prev, ts, actor, roles, action, target, det)
            cur = self._conn.execute(
                "INSERT INTO audit_log (ts, actor, roles, action, target, details, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?)",
                (ts, actor, roles, action, target, det, prev, h))
            self._conn.commit()
            return int(cur.lastrowid)

    def list_audit(self, limit: int = 100, action: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM audit_log", []
        if action:
            q += " WHERE action=?"
            args.append(action)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = {k: r[k] for k in r.keys()}  # noqa: SIM118
            d["details"] = json.loads(d["details"] or "{}")
            out.append(d)
        return out

    def verify_audit(self) -> dict[str, Any]:
        """Recompute the hash chain. Any edited/deleted row breaks it."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
        prev = GENESIS
        for r in rows:
            expect = self._digest(prev, r["ts"], r["actor"], r["roles"], r["action"], r["target"], r["details"])
            if r["prev_hash"] != prev or r["hash"] != expect:
                return {"valid": False, "checked": r["id"], "broken_at": r["id"], "total": len(rows)}
            prev = r["hash"]
        return {"valid": True, "checked": len(rows), "broken_at": None, "total": len(rows)}

    def _raw_execute(self, sql: str, args: tuple = ()) -> None:  # used by tests to simulate tampering
        with self._lock:
            self._conn.execute(sql, args)
            self._conn.commit()
