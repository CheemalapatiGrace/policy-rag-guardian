"""Three-layer guardrails.

1. Input   - prompt-injection detection, PII redaction, topic scope check.
2. Context - retrieved text is data, not instructions: chunks that look like injected instructions
             are quarantined, and every chunk is re-checked against the caller's ACL.
3. Output  - grounding / faithfulness (every claim must be supported by a cited chunk), citation
             validity, ACL re-check and a PII leak scan.

All checks are deterministic and fail closed.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field

from . import pii
from .auth import can_access
from .models import Chunk
from .text import normalize, numbers, split_sentences, tokens

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# (pattern, weight, tag). A total weight >= 1.0 blocks the request.
_QUERY_RULES: list[tuple[str, float, str]] = [
    (r"\b(ignore|disregard|forget|override|overrule)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the|these)\b.{0,30}"
     r"\b(instructions?|rules|prompts?|guardrails?|policies|restrictions|guidelines|filters?|safeguards?|constraints)\b", 1.0, "override_instructions"),
    (r"\b(reveal|show|print|display|repeat|output|leak|dump|tell me|give me)\b.{0,40}\b(system|hidden|initial|original|developer|secret)\b.{0,15}"
     r"\b(prompts?|instructions?|messages?)\b", 1.0, "prompt_exfiltration"),
    (r"\byou are now\b|\bfrom now on,? you\b|\bnew (persona|identity)\b|\bact as (an? )?(unrestricted|unfiltered|jailbroken|dan|root|admin(istrator)?|different)\b"
     r"|\bpretend (to be|you are|you're|that you)\b", 1.0, "role_hijack"),
    (r"\b(developer|debug|god|sudo|jailbreak|dan|unrestricted) mode\b|\bjailbreak\b|\bdo anything now\b", 1.0, "jailbreak"),
    (r"\b(bypass|disable|turn off|circumvent|skip|get around|evade)\b.{0,30}\b(access controls?|acl|permissions?|security|guardrails?|safety|filters?|"
     r"authori[sz]ation|clearance|restrictions?)\b", 1.0, "control_bypass"),
    (r"\b(i am|i'm|this is|as)\b.{0,12}\b(admin|administrator|ceo|cto|ciso|cfo|hr director|root|superuser|security officer|the owner)\b.{0,80}"
     r"\b(show|give|grant|access|reveal|list|unlock|override|provide|send)\b", 1.0, "privilege_claim"),
    (r"\bgrant me\b.{0,30}\b(access|admin|clearance|permissions?)\b|\b(elevate|escalate) my (role|privileges?|access)\b", 1.0, "privilege_escalation"),
    (r"<\|?(im_start|im_end|system|endoftext)\|?>|\[/?(inst|system)\]|(^|\s)#{2,}\s*(system|instruction)", 1.0, "chat_template_injection"),
    (r"\bnew instructions?\s*:|\bsystem\s*:\s|\bassistant\s*:\s", 0.8, "role_marker"),
    (r"\b(base64|rot13|hex)\b.{0,30}\b(decode|decoded|encoded)\b|\bdecode (this|the following)\b", 0.6, "encoded_payload"),
    (r"\bsystem prompt\b", 0.6, "mentions_system_prompt"),
    (r"\bwithout (any )?(restrictions|filters|limitations|citations?|checks)\b|\bno matter what\b", 0.6, "constraint_removal"),
    (r"\b(all|every|restricted|confidential|hidden|secret)\b.{0,20}\b(documents?|files?|policies)\b.{0,40}\b(regardless|even if|anyway|ignore)\b", 0.6, "acl_probe"),
]
_COMPILED_QUERY = [(re.compile(p, re.I | re.S), w, t) for p, w, t in _QUERY_RULES]

# Narrower rules for retrieved *document* text: policies legitimately say "do not bypass security".
_CONTEXT_RULES = [
    (re.compile(p, re.I | re.S), t) for p, t in [
        (r"\b(ignore|disregard|forget)\b (all |any |the )?(previous|prior|above|earlier) (instructions?|prompts?|rules)", "override_instructions"),
        (r"\byou are now\b|\bnew instructions?\s*:|\bsystem prompt\b|\bas an ai\b", "role_hijack"),
        (r"<\|?(im_start|im_end|system|endoftext)\|?>|\[/?(inst|system)\]", "chat_template_injection"),
        (r"\b(do not|don't|never) (tell|inform|mention to) the user\b|\brespond (only )?with\b.{0,40}\binstead\b", "covert_instruction"),
        (r"\b(assistant|ai|model|llm)\b.{0,20}\b(must|should|will)\b.{0,30}\b(reveal|ignore|disclose|output)\b", "directive_to_model"),
    ]
]

_OFFTOPIC = re.compile(
    r"\b(weather|forecast|recipe|poem|poetry|joke|song|lyrics|movie|football|soccer|basketball|nba|horoscope|bitcoin|crypto|"
    r"stock price|lottery|capital of|translate|write (me )?(a |an )?(python|javascript|java|code|script|essay|story)|"
    r"who won|president of|celebrity|homework|math problem|riddle|sports score)\b", re.I)

_SENSITIVE = re.compile(
    r"\b(harass\w*|discriminat\w*|bully\w*|whistleblow\w*|lawsuit|sue|legal action|retaliat\w*|assault|misconduct|"
    r"wrongful|termination for cause|investigat\w*|disciplin\w*|grievance)\b", re.I)

_PLACEHOLDER = re.compile(r"\[REDACTED_\w+\]")
_SINGLE_LETTERS = re.compile(r"\b(?:\w ){4,}\w\b")
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


@dataclass
class InputVerdict:
    allowed: bool
    reason: str | None
    redacted: str
    pii: dict[str, int] = field(default_factory=dict)
    injection_score: float = 0.0
    injection_tags: list[str] = field(default_factory=list)
    in_scope: bool = True
    sensitive: bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("allowed", "reason", "redacted", "pii", "injection_score",
                                                "injection_tags", "in_scope", "sensitive")}


def _deobfuscate(text: str) -> list[str]:
    """Return extra views of the text an attacker may have hidden a payload in."""
    views = [text]
    spaced = _SINGLE_LETTERS.sub(lambda m: m.group().replace(" ", ""), text)
    if spaced != text:
        views.append(spaced)
    for blob in _B64.findall(text):
        try:
            dec = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False).decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            continue
        if dec and sum(c.isprintable() for c in dec) / len(dec) > 0.9:
            views.append(dec)
    return views


def injection_score(text: str) -> tuple[float, list[str]]:
    score, tags = 0.0, []
    for view in _deobfuscate(normalize(text)):
        for rx, weight, tag in _COMPILED_QUERY:
            if rx.search(view) and tag not in tags:
                score += weight
                tags.append(tag)
    return score, tags


class InputGuard:
    def __init__(self, vocabulary_fn=None, max_chars: int = 2000, scope_min_known: float = 0.4,
                 allowed_email_domains: tuple[str, ...] = (), block_threshold: float = 1.0):
        self._vocab_fn = vocabulary_fn
        self.max_chars = max_chars
        self.scope_min_known = scope_min_known
        self.allowed_email_domains = allowed_email_domains
        self.block_threshold = block_threshold

    def check(self, question: str) -> InputVerdict:
        if not question or not question.strip():
            return InputVerdict(False, "empty_question", "")
        if len(question) > self.max_chars:
            return InputVerdict(False, "question_too_long", question[:200] + "...")
        clean = normalize(question)
        score, tags = injection_score(clean)
        redacted, counts = pii.redact(clean, self.allowed_email_domains)
        if score >= self.block_threshold:
            return InputVerdict(False, "prompt_injection", redacted, counts, score, tags)
        in_scope = self._in_scope(clean)
        sensitive = bool(_SENSITIVE.search(clean))
        if not in_scope:
            return InputVerdict(False, "out_of_scope", redacted, counts, score, tags, in_scope=False, sensitive=sensitive)
        return InputVerdict(True, None, redacted, counts, score, tags, True, sensitive)

    def _in_scope(self, text: str) -> bool:
        if _OFFTOPIC.search(text):
            return False
        toks = tokens(_PLACEHOLDER.sub(" ", pii.redact(text)[0]))
        toks = [t for t in toks if not t.isdigit()]
        if not toks:
            return False
        if self._vocab_fn is None:
            return True
        vocab = self._vocab_fn()
        known = sum(1 for t in toks if t in vocab)
        return known / len(toks) >= self.scope_min_known


class ContextGuard:
    """Retrieval-side checks: quarantine injected instructions and enforce ACL on every chunk."""

    def __init__(self, effective_roles_check=can_access):
        self._can = effective_roles_check

    def filter(self, chunks: list[Chunk], effective_roles: frozenset[str]) -> tuple[list[Chunk], list[dict]]:
        safe: list[Chunk] = []
        dropped: list[dict] = []
        for c in chunks:
            if not self._can(effective_roles, c.allowed_roles):
                dropped.append({"chunk_id": c.chunk_id, "reason": "acl"})
                continue
            tag = next((t for rx, t in _CONTEXT_RULES if rx.search(c.text)), None)
            if tag:
                dropped.append({"chunk_id": c.chunk_id, "reason": f"injection:{tag}"})
                continue
            safe.append(c)
        return safe, dropped


_MARKER = re.compile(r"\[(\d+)\]")
_MOVE_MARKERS = re.compile(r"([.!?])\s*((?:\[\d+\]\s*)+)")


@dataclass
class Claim:
    text: str
    markers: list[int]
    support: float = 0.0
    supported: bool = False
    reason: str = ""


@dataclass
class OutputVerdict:
    ok: bool
    faithfulness: float
    claims: list[Claim]
    bad_citations: list[int] = field(default_factory=list)
    acl_violations: list[str] = field(default_factory=list)
    pii_found: list[str] = field(default_factory=list)
    refusal: bool = False
    clean_text: str = ""
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "faithfulness": round(self.faithfulness, 3), "issues": self.issues,
            "claims": [{"text": c.text, "markers": c.markers, "support": round(c.support, 3),
                        "supported": c.supported, "reason": c.reason} for c in self.claims],
            "bad_citations": self.bad_citations, "acl_violations": self.acl_violations,
            "pii_found": self.pii_found,
        }


def parse_claims(answer: str) -> list[Claim]:
    text = _MOVE_MARKERS.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", answer.strip())
    claims = []
    for sent in split_sentences(text):
        markers = [int(n) for n in _MARKER.findall(sent)]
        body = _MARKER.sub("", sent).strip()
        if len(tokens(body)) == 0:
            continue
        claims.append(Claim(body, markers))
    return claims


class OutputGuard:
    def __init__(self, support_threshold: float = 0.7, allowed_email_domains: tuple[str, ...] = ()):
        self.support_threshold = support_threshold
        self.allowed_email_domains = allowed_email_domains

    def support(self, claim_text: str, cited: list[Chunk]) -> tuple[float, str]:
        """Lexical entailment proxy: share of the claim's content terms found in the cited chunks,
        and every number in the claim must also appear there."""
        evidence = " ".join(c.text for c in cited)
        ctoks = set(tokens(evidence))
        qtoks = set(tokens(claim_text))
        if not qtoks:
            return 1.0, ""
        frac = len(qtoks & ctoks) / len(qtoks)
        missing_numbers = numbers(claim_text) - numbers(evidence)
        if missing_numbers:
            return min(frac, 0.0), f"number(s) not in evidence: {sorted(missing_numbers)}"
        return frac, "" if frac >= self.support_threshold else "claim terms not found in cited chunks"

    def check(self, answer: str, chunks: list[Chunk], effective_roles: frozenset[str]) -> OutputVerdict:
        text = answer.strip()
        findings = pii.find_pii(text, self.allowed_email_domains)
        clean, _ = pii.redact(text, self.allowed_email_domains)
        pii_kinds = sorted({f.kind for f in findings})
        if not text or text == INSUFFICIENT:
            return OutputVerdict(ok=False, faithfulness=0.0, claims=[], refusal=True, clean_text=clean,
                                 issues=["model_declined"], pii_found=pii_kinds)
        claims = parse_claims(text)
        bad_cites: list[int] = []
        acl_bad: list[str] = []
        issues: list[str] = []
        for cl in claims:
            valid = [m for m in cl.markers if 1 <= m <= len(chunks)]
            bad_cites += [m for m in cl.markers if m not in valid]
            cited = [chunks[m - 1] for m in valid]
            for c in cited:
                if not can_access(effective_roles, c.allowed_roles):
                    acl_bad.append(c.chunk_id)
            if not cl.markers:
                cl.reason = "no citation"
                continue
            if not cited:
                cl.reason = "citation does not exist"
                continue
            cl.support, cl.reason = self.support(cl.text, cited)
            cl.supported = cl.support >= self.support_threshold and not cl.reason.startswith("number")
        faith = sum(c.supported for c in claims) / len(claims) if claims else 0.0
        if not claims:
            issues.append("no_claims")
        if faith < 1.0:
            issues.append("unsupported_claims")
        if bad_cites:
            issues.append("invalid_citations")
        if acl_bad:
            issues.append("acl_violation")
        if pii_kinds:
            issues.append("pii_leak")
        ok = not issues
        return OutputVerdict(ok, faith, claims, sorted(set(bad_cites)), sorted(set(acl_bad)), pii_kinds,
                             clean_text=clean, issues=issues)


def is_sensitive_query(text: str) -> bool:
    return bool(_SENSITIVE.search(text))
