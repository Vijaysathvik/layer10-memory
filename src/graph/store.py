"""
store.py — SQLite-backed memory graph.

Tables:
  messages        — raw ingested messages (dedup fingerprints)
  entities        — Person, Project, Component, Decision, Issue, Concept
  claims          — typed relations between entities
  evidence        — grounding pointers (one claim → many evidence rows)
  entity_merge_log — reversible merge audit trail
  claim_conflicts  — detected conflicts between claims
  extraction_runs  — model/schema/prompt versioning

All writes are idempotent upserts keyed on stable IDs.
Soft-delete: set deleted_at (never hard-delete for redaction safety).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from ..extraction.ontology import (
    Claim,
    Evidence,
    ExtractionResult,
    MergeLogEntry,
)
from ..deduplication.claim_dedup import ConflictPair
from ..deduplication.canonicalize import PersonCanonicalizer


DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    message_id  TEXT UNIQUE NOT NULL,
    thread_id   TEXT,
    from_addr   TEXT,
    from_name   TEXT,
    subject     TEXT,
    body_clean  TEXT,
    timestamp   TEXT,
    simhash     INTEGER,
    duplicate_of TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    name            TEXT NOT NULL,
    canonical_id    TEXT,
    metadata_json   TEXT,
    confidence      REAL DEFAULT 1.0,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),
    deleted_at      TEXT
);

CREATE TABLE IF NOT EXISTS claims (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    subject_id      TEXT NOT NULL,
    object_id       TEXT NOT NULL,
    text            TEXT,
    confidence      REAL DEFAULT 0.8,
    valid_from      TEXT,
    valid_to        TEXT,
    support_count   INTEGER DEFAULT 1,
    schema_version  TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS evidence (
    id                  TEXT PRIMARY KEY,
    claim_id            TEXT NOT NULL REFERENCES claims(id),
    source_id           TEXT NOT NULL,
    message_id          TEXT NOT NULL,
    excerpt             TEXT,
    char_start          INTEGER,
    char_end            INTEGER,
    timestamp           TEXT,
    extraction_version  TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS entity_merge_log (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    reason      TEXT,
    merged_at   TEXT NOT NULL,
    merged_by   TEXT,
    undone_at   TEXT
);

CREATE TABLE IF NOT EXISTS claim_conflicts (
    id          TEXT PRIMARY KEY,
    claim_a_id  TEXT NOT NULL,
    claim_b_id  TEXT NOT NULL,
    resolution  TEXT,
    resolved_at TEXT,
    detected_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    id              TEXT PRIMARY KEY,
    model           TEXT,
    schema_version  TEXT,
    prompt_hash     TEXT,
    message_count   INTEGER,
    error_count     INTEGER DEFAULT 0,
    run_at          TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_id);
CREATE INDEX IF NOT EXISTS idx_claims_object  ON claims(object_id);
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source_id);
CREATE INDEX IF NOT EXISTS idx_entities_type  ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_id);
"""


def _dt(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _gen_id(prefix: str, *parts: str) -> str:
    import hashlib
    blob = "|".join(parts).encode()
    return f"{prefix}_{hashlib.sha256(blob).hexdigest()[:12]}"


class MemoryStore:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(DDL)
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------ #
    # Messages
    # ------------------------------------------------------------------ #

    def upsert_message(self, msg_data: dict) -> None:
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO messages
                    (id, message_id, thread_id, from_addr, from_name,
                     subject, body_clean, timestamp, simhash, duplicate_of)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at = datetime('now')
            """, (
                msg_data["id"],
                msg_data["message_id"],
                msg_data.get("thread_id"),
                msg_data.get("from_addr"),
                msg_data.get("from_name"),
                msg_data.get("subject"),
                msg_data.get("body_clean"),
                _dt(msg_data.get("timestamp")),
                msg_data.get("simhash"),
                msg_data.get("duplicate_of"),
            ))

    # ------------------------------------------------------------------ #
    # Entities
    # ------------------------------------------------------------------ #

    def upsert_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        canonical_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        confidence: float = 1.0,
    ) -> None:
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO entities
                    (id, type, name, canonical_id, metadata_json, confidence)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    canonical_id  = excluded.canonical_id,
                    metadata_json = excluded.metadata_json,
                    confidence    = excluded.confidence,
                    updated_at    = datetime('now')
            """, (
                entity_id,
                entity_type,
                name,
                canonical_id or entity_id,
                json.dumps(metadata or {}),
                confidence,
            ))

    def soft_delete_entity(self, entity_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE entities SET deleted_at = datetime('now') WHERE id = ?",
                (entity_id,),
            )

    def get_entity(self, entity_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_entities_by_type(self, entity_type: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE type = ? AND deleted_at IS NULL",
            (entity_type,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_entities(self, query: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE name LIKE ? AND deleted_at IS NULL LIMIT 50",
            (f"%{query}%",),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Claims
    # ------------------------------------------------------------------ #

    def upsert_claim(self, claim: Claim) -> None:
        cid = claim.id or claim.stable_id()
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO claims
                    (id, type, subject_id, object_id, text, confidence,
                     valid_from, valid_to, support_count, schema_version)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    support_count = excluded.support_count,
                    valid_to      = excluded.valid_to,
                    confidence    = excluded.confidence,
                    updated_at    = datetime('now')
            """, (
                cid,
                claim.type.value,
                claim.subject_id,
                claim.object_id,
                claim.text,
                claim.confidence,
                _dt(claim.valid_from),
                _dt(claim.valid_to),
                claim.support_count,
                claim.schema_version,
            ))
            for ev in claim.evidence:
                self._upsert_evidence(cid, ev, conn)

    def _upsert_evidence(
        self, claim_id: str, ev: Evidence, conn: sqlite3.Connection
    ) -> None:
        eid = _gen_id("ev", claim_id, ev.source_id, str(ev.char_start))
        conn.execute("""
            INSERT OR IGNORE INTO evidence
                (id, claim_id, source_id, message_id, excerpt,
                 char_start, char_end, timestamp, extraction_version)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            eid,
            claim_id,
            ev.source_id,
            ev.message_id,
            ev.excerpt,
            ev.char_start,
            ev.char_end,
            _dt(ev.timestamp),
            ev.extraction_version,
        ))

    def get_claims_for_entity(
        self,
        entity_id: str,
        current_only: bool = True,
    ) -> list[dict]:
        sql = """
            SELECT c.*, GROUP_CONCAT(e.excerpt, ' || ') AS excerpts
            FROM claims c
            LEFT JOIN evidence e ON e.claim_id = c.id
            WHERE (c.subject_id = ? OR c.object_id = ?)
        """
        params: list[Any] = [entity_id, entity_id]
        if current_only:
            sql += " AND c.valid_to IS NULL"
        sql += " GROUP BY c.id ORDER BY c.confidence DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_evidence_for_claim(self, claim_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE claim_id = ? ORDER BY timestamp",
            (claim_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_current_claims(
        self,
        claim_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 500,
    ) -> list[dict]:
        sql = """
            SELECT * FROM claims
            WHERE valid_to IS NULL AND confidence >= ?
        """
        params: list[Any] = [min_confidence]
        if claim_type:
            sql += " AND type = ?"
            params.append(claim_type)
        sql += f" ORDER BY support_count DESC, confidence DESC LIMIT {limit}"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Merge log
    # ------------------------------------------------------------------ #

    def write_merge_log(self, entry: MergeLogEntry) -> None:
        eid = _gen_id("ml", entry.source_id, entry.target_id, _dt(entry.merged_at) or "")
        with self._tx() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO entity_merge_log
                    (id, source_id, target_id, entity_type, reason,
                     merged_at, merged_by, undone_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                eid,
                entry.source_id,
                entry.target_id,
                entry.entity_type.value,
                entry.reason,
                _dt(entry.merged_at),
                entry.merged_by,
                _dt(entry.undone_at),
            ))

    def get_merge_log(self, entity_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM entity_merge_log WHERE source_id = ? OR target_id = ?",
            (entity_id, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Conflicts
    # ------------------------------------------------------------------ #

    def write_conflict(self, conflict: ConflictPair) -> None:
        cid = _gen_id(
            "cf",
            conflict.claim_a.id or "",
            conflict.claim_b.id or "",
            _dt(conflict.detected_at) or "",
        )
        with self._tx() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO claim_conflicts
                    (id, claim_a_id, claim_b_id, resolution, resolved_at, detected_at)
                VALUES (?,?,?,?,?,?)
            """, (
                cid,
                conflict.claim_a.id,
                conflict.claim_b.id,
                conflict.resolution,
                _dt(conflict.resolved_at),
                _dt(conflict.detected_at),
            ))

    def get_unresolved_conflicts(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM claim_conflicts WHERE resolution IS NULL"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Extraction run logging
    # ------------------------------------------------------------------ #

    def log_extraction_run(
        self,
        model: str,
        schema_version: str,
        prompt_hash: str,
        message_count: int,
        error_count: int,
    ) -> None:
        rid = _gen_id("run", model, schema_version, str(datetime.utcnow()))
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO extraction_runs
                    (id, model, schema_version, prompt_hash, message_count, error_count)
                VALUES (?,?,?,?,?,?)
            """, (rid, model, schema_version, prompt_hash, message_count, error_count))

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def export_json(self, output_path: str | Path) -> None:
        """Write full graph as portable JSON for the visualization layer."""
        entities = self._conn.execute(
            "SELECT * FROM entities WHERE deleted_at IS NULL"
        ).fetchall()
        claims = self._conn.execute("SELECT * FROM claims").fetchall()
        evidence = self._conn.execute("SELECT * FROM evidence").fetchall()
        merges = self._conn.execute("SELECT * FROM entity_merge_log").fetchall()
        conflicts = self._conn.execute("SELECT * FROM claim_conflicts").fetchall()

        data = {
            "entities": [dict(r) for r in entities],
            "claims": [dict(r) for r in claims],
            "evidence": [dict(r) for r in evidence],
            "merge_log": [dict(r) for r in merges],
            "conflicts": [dict(r) for r in conflicts],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[store] Exported graph → {output_path}")

    def stats(self) -> dict:
        def count(table: str, where: str = "") -> int:
            sql = f"SELECT COUNT(*) FROM {table}"
            if where:
                sql += f" WHERE {where}"
            return self._conn.execute(sql).fetchone()[0]

        return {
            "entities": count("entities", "deleted_at IS NULL"),
            "claims_current": count("claims", "valid_to IS NULL"),
            "claims_historical": count("claims", "valid_to IS NOT NULL"),
            "evidence": count("evidence"),
            "messages": count("messages"),
            "merges": count("entity_merge_log"),
            "conflicts_open": count("claim_conflicts", "resolution IS NULL"),
        }
