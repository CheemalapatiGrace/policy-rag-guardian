"""Text normalisation shared by chunking, lexical retrieval, embeddings and guardrails."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being do does did
    can could should would will may might must shall i me my we our us you your they their them it its this
    that these those what which who whom how when where why many much any some there here about into than
    then so not no yes also just get got have has had please tell explain policy policies company employee
    employees need needs per each every other long take takes quickly quick fast soon often minimum maximum
    length amount number answer question ask email far advance""".split()
)

# Tiny domain thesaurus: maps surface forms to one canonical word before stemming. It gives the
# deterministic local embedder and BM25 a little paraphrase tolerance without any model.
SYNONYMS = {
    "vacation": "leave", "vacations": "leave", "pto": "leave", "holiday": "leave", "holidays": "leave",
    "wfh": "remote", "telework": "remote", "telecommute": "remote", "telecommuting": "remote",
    "reimburse": "reimburse", "reimbursed": "reimburse", "reimbursement": "reimburse", "refund": "reimburse",
    "salary": "pay", "salaries": "pay", "compensation": "pay", "wage": "pay", "wages": "pay",
    "laptop": "device", "laptops": "device", "phone": "device", "phones": "device",
    "hack": "breach", "hacked": "breach", "intrusion": "breach",
    "fired": "terminat", "dismissal": "terminat", "dismissed": "terminat",
    "hire": "hire", "hiring": "hire", "recruitment": "hire",
    "ill": "sick", "illness": "sick", "unwell": "sick",
    "baby": "parental", "maternity": "parental", "paternity": "parental", "newborn": "parental",
    "passwords": "password", "passphrase": "password", "2fa": "mfa", "multifactor": "mfa",
    "delete": "destroy", "deleted": "destroy", "deletion": "destroy", "purge": "destroy",
    "keep": "retain", "kept": "retain", "store": "retain", "stored": "retain", "storage": "retain",
    "gift": "gift", "gifts": "gift", "nightly": "night",
}

_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def normalize(text: str) -> str:
    """NFKC-normalise, strip zero-width/control characters and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"\s+", " ", text).strip()


def stem(word: str) -> str:
    """Very light suffix stripper (deterministic, dependency free)."""
    word = SYNONYMS.get(word, word)
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ly") and len(word) > 6:
        word = word[:-2]
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    else:
        stripped = False
        for suf in ("ing", "ed"):
            if word.endswith(suf) and len(word) > len(suf) + 3:
                word = word[: -len(suf)]
                stripped = True
                if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                    word = word[:-1]
                break
        if not stripped and word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokens(text: str, *, drop_stop: bool = True) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if drop_stop and w in STOPWORDS:
            continue
        st = stem(w)
        if drop_stop and st in STOPWORDS:
            continue
        out.append(st)
    return out


def numbers(text: str) -> set[str]:
    """Numeric literals (commas stripped) found in text, used to verify quantitative claims."""
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def clean_markdown(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?m)^\s*(?:[-*+]|\d+\.)\s+", "", text)
    return text


def approx_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token) used for cost accounting."""
    return max(1, (len(text) + 3) // 4)
