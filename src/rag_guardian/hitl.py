"""Human-in-the-loop review queue.

Answers that are low-confidence, fail grounding, touch sensitive topics or leak PII are *held*
(status ``pending_review``) and never shown to the asker until a reviewer approves, edits or
rejects them. Every decision is audit-logged and appended to a regression eval set.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from . import pii
from .auth import Principal
from .storage import Database

ACTIONS = {"approve": "approved", "edit": "edited", "reject": "rejected"}
TRACE_STATUS = {"approved": "review_approved", "edited": "review_edited", "rejected": "review_rejected"}


class ReviewError(Exception):
    pass


class ReviewNotFound(ReviewError):
    pass


class ReviewConflict(ReviewError):
    pass


class ReviewService:
    def __init__(self, db: Database, regression_path: Path | None, allowed_email_domains: tuple[str, ...] = ()):
        self.db = db
        self.regression_path = Path(regression_path) if regression_path else None
        self.allowed_email_domains = allowed_email_domains
        self._lock = threading.Lock()

    def enqueue(self, trace_id: str, reasons: list[str], question: str, asker: Principal, draft: str,
                citations: list[dict]) -> int:
        rid = self.db.create_review(trace_id, reasons, question, asker.label, draft, citations)
        self.db.append_audit("system", "system", "review_enqueued", f"review:{rid}",
                             {"trace_id": trace_id, "reasons": reasons})
        return rid

    def decide(self, review_id: int, reviewer: Principal, action: str, edited_answer: str | None = None,
               note: str = "") -> dict[str, Any]:
        if action not in ACTIONS:
            raise ReviewError(f"action must be one of {sorted(ACTIONS)}")
        review = self.db.get_review(review_id)
        if review is None:
            raise ReviewNotFound(f"review {review_id} not found")
        status = ACTIONS[action]
        final: str | None = None
        if action == "approve":
            final = review["draft_answer"]
        elif action == "edit":
            if not edited_answer or not edited_answer.strip():
                raise ReviewError("edited_answer is required for action=edit")
            if pii.find_pii(edited_answer, self.allowed_email_domains):
                raise ReviewError("edited_answer contains PII; remove it before release")
            final = edited_answer.strip()
        if not self.db.decide_review(review_id, status, reviewer.user_id, final, note):
            raise ReviewConflict(f"review {review_id} was already decided")
        self.db.update_trace_release(review["trace_id"], TRACE_STATUS[status], final)
        self.db.append_audit(reviewer.user_id, reviewer.label, f"review_{action}", f"review:{review_id}",
                             {"trace_id": review["trace_id"], "reasons": review["reasons"], "note": note,
                              "edited": action == "edit"})
        updated = self.db.get_review(review_id)
        assert updated is not None
        self._append_regression(updated, action)
        return updated

    def _append_regression(self, review: dict[str, Any], action: str) -> None:
        if not self.regression_path:
            return
        docs = list(dict.fromkeys(c["doc_id"] for c in review["citations"]))
        case = {
            "id": f"reg-{review['review_id']}",
            "category": "hitl_regression",
            "question": review["question"],
            "role": (review["asker_roles"] or "employee").split(",")[0],
            "expect": "not_answered" if action == "reject" else "releasable",
            "expected_docs": [] if action == "reject" else docs,
            "reference_answer": review["final_answer"],
            "source": {"review_id": review["review_id"], "reviewer": review["reviewer"], "action": action},
        }
        with self._lock:
            self.regression_path.parent.mkdir(parents=True, exist_ok=True)
            with self.regression_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(case, sort_keys=True) + "\n")
