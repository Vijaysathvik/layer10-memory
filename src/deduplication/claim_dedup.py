"""
claim_dedup.py — Claim deduplication, conflict detection, and revision handling.

Pipeline:
  1. Embed claim texts with sentence-transformers (all-MiniLM-L6-v2)
  2. Find candidate pairs with cosine ≥ SIMILARITY_THRESHOLD
  3. For each pair: rule-based type/subject check, then LLM judge
  4. Merge: keep both evidence pointers, increment support_count
  5. Conflict: same subject+predicate, different object → CONFLICTS_WITH edge,
     set valid_to on superseded claim when reversal keyword detected
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

from ..extraction.ontology import Claim, ClaimType

SIMILARITY_THRESHOLD = 0.92
REVERSAL_KEYWORDS = {
    "reversed", "reverted", "withdrawn", "superseded", "overturned",
    "changed", "amended", "updated decision", "no longer", "instead",
}

# ---------------------------------------------------------------------------
# Embedding backend (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(texts: list[str]) -> "np.ndarray":
        return _model.encode(texts, normalize_embeddings=True)

    def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
        return float(np.dot(a, b))

    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("[claim_dedup] sentence-transformers not available — embedding dedup disabled", file=sys.stderr)

    def _embed(texts):  # type: ignore
        return [[0.0] * 10] * len(texts)

    def _cosine(a, b):  # type: ignore
        return 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ClaimGroup:
    """
    A canonical claim backed by one or more evidence pointers.
    Represents the result of merging duplicate claims.
    """

    def __init__(self, canonical: Claim):
        self.canonical = canonical
        self.merged_ids: list[str] = []

    def merge_in(self, other: Claim) -> None:
        self.canonical.evidence.extend(other.evidence)
        self.canonical.support_count += 1
        if other.id:
            self.merged_ids.append(other.id)

    def to_claim(self) -> Claim:
        c = self.canonical.model_copy(deep=True)
        c.support_count = len(c.evidence)
        return c


class ConflictPair:
    """Two claims that assert different values for the same subject+predicate."""

    def __init__(self, claim_a: Claim, claim_b: Claim, detected_at: datetime):
        self.claim_a = claim_a
        self.claim_b = claim_b
        self.detected_at = detected_at
        self.resolution: Optional[str] = None  # "a_wins" | "b_wins" | "both_valid" | None
        self.resolved_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

class ClaimDeduplicator:
    """
    Deduplicates and conflict-checks a list of Claim objects.
    Returns canonical ClaimGroups and detected ConflictPairs.
    """

    def __init__(self):
        self._groups: list[ClaimGroup] = []
        self._conflicts: list[ConflictPair] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_claims(self, claims: list[Claim]) -> None:
        """Process a batch of claims (e.g. from one extraction run)."""
        for claim in claims:
            self._process(claim)

    def get_canonical_claims(self) -> list[Claim]:
        return [g.to_claim() for g in self._groups]

    def get_conflicts(self) -> list[ConflictPair]:
        return list(self._conflicts)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _process(self, claim: Claim) -> None:
        # 1. Check for duplicates
        dup_group = self._find_duplicate(claim)
        if dup_group:
            dup_group.merge_in(claim)
            return

        # 2. Check for conflicts (same subject, same type, different object)
        conflict = self._find_conflict(claim)
        if conflict:
            self._resolve_conflict(conflict, claim)

        # 3. Register as new canonical
        self._groups.append(ClaimGroup(claim))

    def _find_duplicate(self, claim: Claim) -> Optional[ClaimGroup]:
        """
        A claim is a duplicate if:
          - Same type, same subject_id, same object_id  (exact structural match)
          - OR: same type + subject, and text embedding cosine ≥ threshold
        """
        for group in self._groups:
            canon = group.canonical
            if (
                canon.type == claim.type
                and canon.subject_id == claim.subject_id
                and canon.object_id == claim.object_id
            ):
                return group

            # Embedding similarity fallback
            if (
                HAS_EMBEDDINGS
                and canon.type == claim.type
                and canon.subject_id == claim.subject_id
                and canon.text and claim.text
            ):
                vecs = _embed([canon.text, claim.text])
                sim = _cosine(vecs[0], vecs[1])
                if sim >= SIMILARITY_THRESHOLD:
                    return group

        return None

    def _find_conflict(self, claim: Claim) -> Optional[Claim]:
        """
        Conflict = same type + subject_id, but different object_id.
        Only applies to relation types where there should be one answer.
        """
        EXCLUSIVE_TYPES = {
            ClaimType.ASSIGNED,
            ClaimType.DECIDED,
            ClaimType.REVERSED,
        }
        if claim.type not in EXCLUSIVE_TYPES:
            return None

        for group in self._groups:
            canon = group.canonical
            if (
                canon.type == claim.type
                and canon.subject_id == claim.subject_id
                and canon.object_id != claim.object_id
            ):
                return canon
        return None

    def _resolve_conflict(self, existing: Claim, incoming: Claim) -> None:
        """
        Attempt rule-based resolution:
        - If incoming has a reversal keyword in its text → existing gets valid_to
        - Otherwise record unresolved conflict
        """
        incoming_lower = (incoming.text or "").lower()
        is_reversal = any(kw in incoming_lower for kw in REVERSAL_KEYWORDS)

        if is_reversal and incoming.valid_from:
            # Set valid_to on the existing claim = when reversal was recorded
            existing.valid_to = incoming.valid_from

        conflict = ConflictPair(
            claim_a=existing,
            claim_b=incoming,
            detected_at=datetime.utcnow(),
        )
        if is_reversal:
            conflict.resolution = "b_wins"
            conflict.resolved_at = datetime.utcnow()
        self._conflicts.append(conflict)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def deduplicate_claims(
    claims: list[Claim],
) -> tuple[list[Claim], list[ConflictPair]]:
    """
    One-shot deduplication.
    Returns (canonical_claims, conflict_pairs).
    """
    dedup = ClaimDeduplicator()
    dedup.add_claims(claims)
    return dedup.get_canonical_claims(), dedup.get_conflicts()
