"""
ontology.py — Entity, Claim, Evidence schemas for Layer10 memory graph.

Every extracted object carries grounding (source_id + excerpt + offsets).
Schema version is embedded so extractions can be backfilled when ontology evolves.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    PERSON = "Person"
    PROJECT = "Project"
    COMPONENT = "Component"
    DECISION = "Decision"
    ISSUE = "Issue"
    CONCEPT = "Concept"


class ClaimType(str, Enum):
    AUTHORED = "AUTHORED"
    DECIDED = "DECIDED"
    REVERSED = "REVERSED"
    ASSIGNED = "ASSIGNED"
    DEPENDS_ON = "DEPENDS_ON"
    MENTIONS = "MENTIONS"
    SUPPORTS = "SUPPORTS"
    OPPOSES = "OPPOSES"
    IMPLEMENTED_BY = "IMPLEMENTED_BY"
    QUOTES = "QUOTES"


class VoteResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# Evidence — every claim must carry at least one
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """Verbatim grounding pointer back to the source corpus."""

    source_id: str = Field(
        description="SHA-256 of the raw message bytes"
    )
    message_id: str = Field(
        description="RFC 2822 Message-ID header value"
    )
    excerpt: str = Field(
        max_length=600,
        description="Verbatim text span from the source supporting this claim"
    )
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    timestamp: datetime
    extraction_version: str = Field(
        description="{model}:{schema_version}:{prompt_hash}"
    )

    @model_validator(mode="after")
    def offsets_ordered(self) -> "Evidence":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be > char_start")
        return self

    @property
    def span_length(self) -> int:
        return self.char_end - self.char_start


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

class BaseEntity(BaseModel):
    id: Optional[str] = None           # assigned at write time
    canonical_id: Optional[str] = None # set after canonicalization
    schema_version: str = SCHEMA_VERSION
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    evidence: list[Evidence] = Field(min_length=1)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None  # soft-delete for redactions

    def stable_id(self, *key_fields: str) -> str:
        """Deterministic id from key fields for idempotent upsert."""
        blob = "|".join(key_fields).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class Person(BaseEntity):
    type: EntityType = EntityType.PERSON
    name: str
    email_addresses: list[str] = Field(min_length=1)
    org: Optional[str] = None
    display_name: Optional[str] = None

    @field_validator("email_addresses")
    @classmethod
    def normalize_emails(cls, v: list[str]) -> list[str]:
        return [e.strip().lower() for e in v]

    def compute_id(self) -> str:
        return self.stable_id(self.email_addresses[0])


class Project(BaseEntity):
    type: EntityType = EntityType.PROJECT
    name: str
    repo_url: Optional[str] = None
    description: Optional[str] = None

    def compute_id(self) -> str:
        return self.stable_id(self.name.lower())


class Component(BaseEntity):
    type: EntityType = EntityType.COMPONENT
    name: str
    project_id: Optional[str] = None

    def compute_id(self) -> str:
        return self.stable_id(self.name.lower(), self.project_id or "")


class Decision(BaseEntity):
    type: EntityType = EntityType.DECISION
    summary: str
    vote_result: VoteResult = VoteResult.PENDING
    participants: list[str] = Field(default_factory=list)
    thread_subject: Optional[str] = None
    supersedes_id: Optional[str] = None  # points to reversed Decision

    def compute_id(self) -> str:
        return self.stable_id(self.summary[:80])


class Issue(BaseEntity):
    type: EntityType = EntityType.ISSUE
    external_id: str        # e.g. "BZ-12345"
    tracker_url: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = None  # open/closed/resolved/wontfix

    def compute_id(self) -> str:
        return self.stable_id(self.external_id)


class Concept(BaseEntity):
    type: EntityType = EntityType.CONCEPT
    label: str
    definition: Optional[str] = None

    def compute_id(self) -> str:
        return self.stable_id(self.label.lower())


# ---------------------------------------------------------------------------
# Claims / Relations
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    id: Optional[str] = None
    type: ClaimType
    subject_id: str         # entity id
    object_id: str          # entity id
    text: str               # human-readable statement, e.g. "Justin proposed removing SSLv3"
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None   # None = currently true
    support_count: int = 1
    evidence: list[Evidence] = Field(min_length=1)
    schema_version: str = SCHEMA_VERSION
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def is_current(self) -> bool:
        return self.valid_to is None

    def stable_id(self) -> str:
        blob = f"{self.type}|{self.subject_id}|{self.object_id}|{self.text[:60]}".encode()
        return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Extraction output envelope
# ---------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    """Complete output from one LLM extraction call over one message."""

    message_id: str
    source_id: str
    extraction_version: str
    entities: list[Person | Project | Component | Decision | Issue | Concept] = Field(
        default_factory=list
    )
    claims: list[Claim] = Field(default_factory=list)
    parse_errors: list[str] = Field(default_factory=list)
    retries: int = 0

    @property
    def is_clean(self) -> bool:
        return len(self.parse_errors) == 0

    def grounding_complete(self) -> bool:
        """Every claim must have at least one evidence pointer."""
        return all(len(c.evidence) >= 1 for c in self.claims)


# ---------------------------------------------------------------------------
# Merge log entry (for reversibility)
# ---------------------------------------------------------------------------

class MergeLogEntry(BaseModel):
    id: Optional[str] = None
    source_id: str
    target_id: str          # canonical entity after merge
    entity_type: EntityType
    reason: str             # e.g. "email_alias_cluster", "manual_override"
    merged_at: datetime
    merged_by: str          # "pipeline_v1" or user id
    undone_at: Optional[datetime] = None
