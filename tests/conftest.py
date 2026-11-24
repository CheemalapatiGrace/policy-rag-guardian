from __future__ import annotations

from dataclasses import replace

import pytest

from rag_guardian.auth import Principal
from rag_guardian.config import Settings
from rag_guardian.pipeline import build_pipeline
from rag_guardian.storage import Database


def P(user: str, *roles: str) -> Principal:
    return Principal(user, frozenset(roles))


EMPLOYEE = P("erin", "employee")
GUEST = P("gus", "guest")
HR = P("hannah", "hr")
MANAGER = P("max", "manager")
SECURITY = P("sam", "security")
REVIEWER = P("rita", "reviewer")


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), db_path=tmp_path / "t.db", regression_path=tmp_path / "reg.jsonl",
                   retry_attempts=2, retry_base_delay_s=0.001, retry_max_delay_s=0.002, embed_timeout_s=1.0)


@pytest.fixture
def pipeline(settings):
    return build_pipeline(settings, db=Database(settings.db_path))
