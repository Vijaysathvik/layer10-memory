"""
canonicalize.py — Entity deduplication and canonicalization.

Strategy:
  Person: cluster by email-address username similarity + name token overlap.
  Project/Component: lowercase + punctuation-strip exact match; manual overrides.

All merges are logged to entity_merge_log (reversible).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..extraction.ontology import (
    EntityType,
    MergeLogEntry,
    Person,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _email_username(email: str) -> str:
    return email.split("@")[0].lower().replace(".", "").replace("_", "").replace("-", "")


def _name_tokens(name: str) -> set[str]:
    return set(_normalize_name(name).split())


def _email_overlap(emails_a: list[str], emails_b: list[str]) -> bool:
    """True if any email address is shared."""
    set_a = {e.lower() for e in emails_a}
    set_b = {e.lower() for e in emails_b}
    return bool(set_a & set_b)


def _username_overlap(emails_a: list[str], emails_b: list[str]) -> bool:
    """True if normalized usernames overlap."""
    ua = {_email_username(e) for e in emails_a}
    ub = {_email_username(e) for e in emails_b}
    return bool(ua & ub)


def _name_similarity(name_a: str, name_b: str) -> float:
    """Jaccard similarity of name tokens."""
    ta = _name_tokens(name_a)
    tb = _name_tokens(name_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Person canonicalization
# ---------------------------------------------------------------------------

class PersonCanonicalizer:
    """
    Clusters Person entities by email / username / name similarity.
    Maintains a union-find structure; merges are logged for reversibility.
    """

    def __init__(self, override_path: Optional[str | Path] = None):
        # parent[id] = canonical_id (union-find)
        self._parent: dict[str, str] = {}
        # persons by id
        self._persons: dict[str, Person] = {}
        # all emails seen → person id
        self._email_index: dict[str, str] = {}
        self._merge_log: list[MergeLogEntry] = []

        self._overrides: dict[str, str] = {}
        if override_path:
            self._load_overrides(Path(override_path))

    def _load_overrides(self, path: Path) -> None:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self._overrides = data.get("person_aliases", {})

    def find(self, person_id: str) -> str:
        """Path-compressed union-find root."""
        while self._parent.get(person_id, person_id) != person_id:
            grandparent = self._parent.get(
                self._parent.get(person_id, person_id),
                self._parent.get(person_id, person_id),
            )
            self._parent[person_id] = grandparent
            person_id = grandparent
        return person_id

    def add(self, person: Person) -> str:
        """
        Register a Person. If it matches an existing canonical entity, merge.
        Returns the canonical_id.
        """
        if person.id is None:
            person.id = person.compute_id()

        pid = person.id
        self._parent[pid] = pid
        self._persons[pid] = person
        for email in person.email_addresses:
            self._email_index[email.lower()] = pid

        # Check manual overrides first
        for email in person.email_addresses:
            if email.lower() in self._overrides:
                target_email = self._overrides[email.lower()]
                target_pid = self._email_index.get(target_email.lower())
                if target_pid:
                    self._merge(pid, target_pid, reason="manual_override")
                    return self.find(pid)

        # Auto-cluster: email exact match
        for email in person.email_addresses:
            existing_pid = self._email_index.get(email.lower())
            if existing_pid and existing_pid != pid:
                self._merge(pid, existing_pid, reason="email_exact_match")
                return self.find(pid)

        # Auto-cluster: username overlap + name similarity
        for existing_pid, existing in self._persons.items():
            if existing_pid == pid:
                continue
            canonical_existing = self.find(existing_pid)
            if canonical_existing == self.find(pid):
                continue

            if _username_overlap(person.email_addresses, existing.email_addresses):
                sim = _name_similarity(person.name, existing.name)
                if sim >= 0.4:
                    self._merge(pid, canonical_existing, reason="username_name_cluster")
                    return self.find(pid)

        return self.find(pid)

    def _merge(self, source_id: str, target_id: str, reason: str) -> None:
        """Union source into target. Log the merge."""
        source_root = self.find(source_id)
        target_root = self.find(target_id)
        if source_root == target_root:
            return

        # target_root becomes canonical
        self._parent[source_root] = target_root

        # Merge email lists onto target
        target_person = self._persons.get(target_root)
        source_person = self._persons.get(source_root)
        if target_person and source_person:
            merged_emails = list(
                dict.fromkeys(target_person.email_addresses + source_person.email_addresses)
            )
            target_person.email_addresses = merged_emails

        log = MergeLogEntry(
            source_id=source_root,
            target_id=target_root,
            entity_type=EntityType.PERSON,
            reason=reason,
            merged_at=datetime.now(timezone.utc).replace(tzinfo=None),
            merged_by="pipeline_canonicalizer",
        )
        self._merge_log.append(log)

    def canonical_id(self, person_id: str) -> str:
        return self.find(person_id)

    def get_merge_log(self) -> list[MergeLogEntry]:
        return list(self._merge_log)

    def undo_merge(self, source_id: str) -> bool:
        """Undo a merge by restoring source_id as its own root."""
        log_entry = next(
            (e for e in reversed(self._merge_log) if e.source_id == source_id and e.undone_at is None),
            None,
        )
        if not log_entry:
            return False
        self._parent[source_id] = source_id
        log_entry.undone_at = datetime.now(timezone.utc).replace(tzinfo=None)
        return True


# ---------------------------------------------------------------------------
# Generic name canonicalization (Project, Component, Concept)
# ---------------------------------------------------------------------------

class NameCanonicalizer:
    """
    Simple exact-match-after-normalization canonicalization for non-person entities.
    """

    def __init__(self, entity_type: str):
        self.entity_type = entity_type
        self._canonical: dict[str, str] = {}  # normalized_name → canonical_id
        self._id_to_name: dict[str, str] = {}

    def add(self, entity_id: str, name: str) -> str:
        """
        Register an entity name. Returns the canonical_id to use.
        If the normalized name was seen before, returns existing canonical_id.
        """
        key = _normalize_name(name)
        if key in self._canonical:
            return self._canonical[key]
        self._canonical[key] = entity_id
        self._id_to_name[entity_id] = key
        return entity_id

    def resolve(self, name: str) -> Optional[str]:
        return self._canonical.get(_normalize_name(name))
