"""
retriever.py — Hybrid retrieval (BM25 + embedding) over the memory graph.

Returns a ContextPack: ranked evidence snippets + linked entities/claims + citations.
Every returned item is grounded in evidence; ambiguities/conflicts are surfaced.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Graceful fallbacks
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

from ..graph.store import MemoryStore


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class EvidenceSnippet:
    source_id: str
    message_id: str
    excerpt: str
    timestamp: str
    extraction_version: str
    claim_id: str
    claim_text: str
    claim_type: str
    score: float = 0.0


@dataclass
class Citation:
    number: int
    message_id: str
    timestamp: str
    subject: Optional[str] = None
    from_addr: Optional[str] = None
    excerpt: str = ""


@dataclass
class ContextPack:
    question: str
    snippets: list[EvidenceSnippet] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)

    def format_answer_context(self) -> str:
        """Human-readable context string for feeding to an LLM."""
        lines = [f"Question: {self.question}\n"]

        if self.entities:
            lines.append("## Relevant Entities")
            for e in self.entities[:5]:
                lines.append(f"  [{e['type']}] {e['name']} (id={e['id']}, confidence={e.get('confidence',1):.2f})")

        if self.snippets:
            lines.append("\n## Evidence Snippets (ranked)")
            for i, s in enumerate(self.snippets[:8]):
                lines.append(
                    f"  [{i+1}] ({s.claim_type}) {s.claim_text}\n"
                    f"       Excerpt: \"{s.excerpt[:200]}\"\n"
                    f"       Source: {s.message_id} @ {s.timestamp}"
                )

        if self.ambiguities:
            lines.append("\n## Ambiguities / Conflicts")
            for a in self.ambiguities:
                lines.append(f"  ⚠ {a}")

        if self.citations:
            lines.append("\n## Citations")
            for c in self.citations:
                lines.append(
                    f"  [{c.number}] {c.message_id} | {c.timestamp}"
                    + (f" | {c.subject}" if c.subject else "")
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, db_path: str | Path):
        self.store = MemoryStore(db_path)
        self._conn: sqlite3.Connection = self.store._conn
        self._bm25: Optional["BM25Okapi"] = None
        self._bm25_docs: list[dict] = []
        self._embeddings: Optional["np.ndarray"] = None
        self._build_index()

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #

    def _build_index(self) -> None:
        rows = self._conn.execute("""
            SELECT c.id, c.type, c.text, c.subject_id, c.object_id,
                   c.confidence, c.valid_from, c.valid_to, c.support_count,
                   e.excerpt, e.source_id, e.message_id, e.timestamp,
                   e.extraction_version
            FROM claims c
            JOIN evidence e ON e.claim_id = c.id
            WHERE c.text IS NOT NULL AND c.text != ''
            ORDER BY c.support_count DESC, c.confidence DESC
        """).fetchall()

        self._bm25_docs = [dict(r) for r in rows]
        if not self._bm25_docs:
            return

        corpus_tokens = [_tokenize(d["text"] + " " + (d["excerpt"] or "")) for d in self._bm25_docs]

        if HAS_BM25 and corpus_tokens:
            self._bm25 = BM25Okapi(corpus_tokens)

        if HAS_EMBEDDINGS and self._bm25_docs:
            texts = [d["text"] for d in self._bm25_docs]
            self._embeddings = _st_model.encode(texts, normalize_embeddings=True)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: int = 10,
        min_confidence: float = 0.3,
        current_only: bool = True,
    ) -> ContextPack:
        if not self._bm25_docs:
            return ContextPack(question=question, ambiguities=["No indexed claims found."])

        scores = self._score_docs(question)

        # Filter by confidence + optionally current-only
        filtered = [
            (score, doc)
            for score, doc in zip(scores, self._bm25_docs)
            if doc["confidence"] >= min_confidence
            and (not current_only or doc["valid_to"] is None)
        ]
        filtered.sort(key=lambda x: x[0], reverse=True)

        # Diversity penalty: cap per-source snippets at 3
        source_count: dict[str, int] = {}
        selected: list[tuple[float, dict]] = []
        for score, doc in filtered:
            sid = doc["source_id"]
            if source_count.get(sid, 0) >= 3:
                continue
            source_count[sid] = source_count.get(sid, 0) + 1
            selected.append((score, doc))
            if len(selected) >= top_k:
                break

        pack = self._build_pack(question, selected)
        return pack

    def _score_docs(self, question: str) -> list[float]:
        n = len(self._bm25_docs)
        bm25_scores = [0.0] * n
        emb_scores = [0.0] * n

        # BM25
        if HAS_BM25 and self._bm25:
            query_tokens = _tokenize(question)
            bm25_raw = self._bm25.get_scores(query_tokens)
            bm25_max = max(bm25_raw) if bm25_raw.max() > 0 else 1.0
            bm25_scores = [s / bm25_max for s in bm25_raw]

        # Embedding cosine
        if HAS_EMBEDDINGS and self._embeddings is not None:
            q_vec = _st_model.encode([question], normalize_embeddings=True)[0]
            emb_scores = list(float(self._embeddings[i] @ q_vec) for i in range(n))

        # Combine: BM25 * 0.5 + embedding * 0.3 + support * 0.1 + recency * 0.1
        max_support = max((d["support_count"] for d in self._bm25_docs), default=1)
        combined = []
        for i, doc in enumerate(self._bm25_docs):
            support_norm = doc["support_count"] / max_support
            score = (
                0.5 * bm25_scores[i]
                + 0.3 * emb_scores[i]
                + 0.1 * support_norm
                + 0.1 * float(doc["valid_to"] is None)  # recency: current > historical
            )
            combined.append(score)
        return combined

    def _build_pack(
        self, question: str, scored_docs: list[tuple[float, dict]]
    ) -> ContextPack:
        snippets: list[EvidenceSnippet] = []
        seen_claim_ids: set[str] = set()
        entity_ids: set[str] = set()

        for score, doc in scored_docs:
            snippets.append(EvidenceSnippet(
                source_id=doc["source_id"],
                message_id=doc["message_id"],
                excerpt=doc["excerpt"] or "",
                timestamp=doc["timestamp"] or "",
                extraction_version=doc["extraction_version"] or "",
                claim_id=doc["id"],
                claim_text=doc["text"] or "",
                claim_type=doc["type"] or "",
                score=round(score, 4),
            ))
            if doc["id"] not in seen_claim_ids:
                seen_claim_ids.add(doc["id"])
                entity_ids.add(doc["subject_id"])
                entity_ids.add(doc["object_id"])

        # Fetch claim rows
        claims = []
        for cid in seen_claim_ids:
            row = self._conn.execute("SELECT * FROM claims WHERE id = ?", (cid,)).fetchone()
            if row:
                claims.append(dict(row))

        # Fetch entity rows
        entities = []
        for eid in entity_ids:
            if eid and not eid.startswith("unresolved:"):
                row = self._conn.execute(
                    "SELECT * FROM entities WHERE id = ? AND deleted_at IS NULL", (eid,)
                ).fetchone()
                if row:
                    entities.append(dict(row))

        # Conflicts / ambiguities
        ambiguities = []
        conflict_rows = self._conn.execute("""
            SELECT cf.*, ca.text AS text_a, cb.text AS text_b
            FROM claim_conflicts cf
            JOIN claims ca ON ca.id = cf.claim_a_id
            JOIN claims cb ON cb.id = cf.claim_b_id
            WHERE cf.resolution IS NULL
              AND (cf.claim_a_id IN ({0}) OR cf.claim_b_id IN ({0}))
        """.format(",".join("?" * len(seen_claim_ids))),
            list(seen_claim_ids) + list(seen_claim_ids),
        ).fetchall()

        for row in conflict_rows:
            ambiguities.append(
                f"Conflicting claims:\n"
                f"  A: {row['text_a']}\n"
                f"  B: {row['text_b']}"
            )

        # Citations
        citations: list[Citation] = []
        seen_msg_ids: dict[str, int] = {}
        for s in snippets:
            if s.message_id not in seen_msg_ids:
                num = len(seen_msg_ids) + 1
                seen_msg_ids[s.message_id] = num
                msg_row = self._conn.execute(
                    "SELECT subject, from_addr, timestamp FROM messages WHERE message_id = ?",
                    (s.message_id,),
                ).fetchone()
                citations.append(Citation(
                    number=num,
                    message_id=s.message_id,
                    timestamp=s.timestamp,
                    subject=msg_row["subject"] if msg_row else None,
                    from_addr=msg_row["from_addr"] if msg_row else None,
                    excerpt=s.excerpt[:200],
                ))

        return ContextPack(
            question=question,
            snippets=snippets,
            entities=entities,
            claims=claims,
            citations=citations,
            ambiguities=ambiguities,
        )
