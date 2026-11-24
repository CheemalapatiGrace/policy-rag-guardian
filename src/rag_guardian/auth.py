"""Principals, role expansion and API-key authentication.

Access control is *role based at document level*: every document declares ``allowed_roles`` and a
principal may see it iff the principal's effective roles intersect that list. The role comes from
the authenticated API key, never from anything the user types in the question.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable
from dataclasses import dataclass

ALL_ROLES = frozenset({"guest", "employee", "manager", "hr", "security", "reviewer", "admin"})

# role -> roles it also holds. ``admin`` can read everything.
ROLE_EXPANSION: dict[str, frozenset[str]] = {
    "guest": frozenset({"guest"}),
    "employee": frozenset({"guest", "employee"}),
    "manager": frozenset({"guest", "employee", "manager"}),
    "hr": frozenset({"guest", "employee", "hr"}),
    "security": frozenset({"guest", "employee", "security"}),
    "reviewer": frozenset({"guest", "employee", "reviewer"}),
    "admin": ALL_ROLES,
}


def effective_roles(roles: Iterable[str]) -> frozenset[str]:
    out: set[str] = set()
    for r in roles:
        out |= ROLE_EXPANSION.get(r, frozenset())  # unknown roles grant nothing (fail closed)
    return frozenset(out)


def can_access(effective: Iterable[str], allowed_roles: Iterable[str]) -> bool:
    return bool(set(effective) & set(allowed_roles))


@dataclass(frozen=True)
class Principal:
    user_id: str
    roles: frozenset[str]

    @property
    def effective(self) -> frozenset[str]:
        return effective_roles(self.roles)

    @property
    def label(self) -> str:
        return ",".join(sorted(self.roles))

    def has_any(self, *roles: str) -> bool:
        return bool(self.roles & set(roles))


DEMO_KEYS: dict[str, tuple[str, tuple[str, ...]]] = {
    "key-guest": ("guest-user", ("guest",)),
    "key-employee": ("erin.employee", ("employee",)),
    "key-manager": ("max.manager", ("manager",)),
    "key-hr": ("hannah.hr", ("hr",)),
    "key-security": ("sam.security", ("security",)),
    "key-admin": ("ada.admin", ("admin",)),
    "key-reviewer": ("rita.reviewer", ("reviewer",)),
}


class KeyStore:
    """Constant-time API key lookup. Keys are stored hashed in memory."""

    def __init__(self, api_keys: dict | None = None, demo: bool = True):
        self._entries: list[tuple[bytes, Principal]] = []
        if demo:
            for key, (user, roles) in DEMO_KEYS.items():
                self.add(key, user, roles)
        for key, spec in (api_keys or {}).items():
            self.add(key, spec["user"], tuple(spec["roles"]))

    @staticmethod
    def _h(key: str) -> bytes:
        return hashlib.sha256(key.encode()).digest()

    def add(self, key: str, user: str, roles: Iterable[str]) -> None:
        self._entries.append((self._h(key), Principal(user, frozenset(roles))))

    def authenticate(self, key: str | None) -> Principal | None:
        if not key:
            return None
        digest = self._h(key)
        found = None
        for stored, principal in self._entries:  # no early exit: constant-time-ish
            if hmac.compare_digest(stored, digest):
                found = principal
        return found
