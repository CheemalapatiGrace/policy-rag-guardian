"""PII detection and redaction (regex based, deterministic, no external service)."""

from __future__ import annotations

import re
from dataclasses import dataclass

SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")
SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})\b")


@dataclass(frozen=True)
class PIIFinding:
    kind: str
    value: str


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_pii(text: str, allowed_email_domains: tuple[str, ...] = ()) -> list[PIIFinding]:
    out: list[PIIFinding] = []
    allowed = {d.lower() for d in allowed_email_domains}
    out += [PIIFinding("secret", m.group()) for m in SECRET.finditer(text)]
    out += [PIIFinding("ssn", m.group()) for m in SSN.finditer(text)]
    out += [PIIFinding("credit_card", m.group().strip()) for m in CARD.finditer(text) if _luhn(m.group())]
    out += [PIIFinding("email", m.group()) for m in EMAIL.finditer(text) if m.group(1).lower() not in allowed]
    out += [PIIFinding("phone", m.group()) for m in PHONE.finditer(text)]
    return out


def redact(text: str, allowed_email_domains: tuple[str, ...] = ()) -> tuple[str, dict[str, int]]:
    """Replace PII with [REDACTED_<KIND>] tokens. Returns (clean_text, {kind: count})."""
    counts: dict[str, int] = {}
    # longest values first so that overlapping matches don't corrupt each other
    for f in sorted(find_pii(text, allowed_email_domains), key=lambda f: -len(f.value)):
        if f.value in text:
            text = text.replace(f.value, f"[REDACTED_{f.kind.upper()}]")
            counts[f.kind] = counts.get(f.kind, 0) + 1
    return text, counts
