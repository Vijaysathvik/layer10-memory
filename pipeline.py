"""
pipeline.py — End-to-end pipeline: ingest → extract → canonicalize → dedup → store → export.

Usage:
    python pipeline.py --corpus data/raw/corpus.mbox
    python pipeline.py --sample   (uses bundled sample data)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.extraction.ingest import MboxIngester, generate_sample_mbox
from src.extraction.extractor import Extractor
from src.extraction.ontology import (
    SCHEMA_VERSION,
    Component,
    Concept,
    Decision,
    Issue,
    Person,
    Project,
)
from src.deduplication.canonicalize import NameCanonicalizer, PersonCanonicalizer
from src.deduplication.claim_dedup import deduplicate_claims
from src.graph.store import MemoryStore
from src.retrieval.retriever import Retriever


SAMPLE_MBOX = "data/raw/sample_httpd_dev.mbox"
DEFAULT_DB = "outputs/graph/memory.db"
DEFAULT_EXPORT = "outputs/graph/memory_export.json"
CONTEXT_PACKS_DIR = Path("outputs/context_packs")

EXAMPLE_QUESTIONS = [
    "Who proposed removing SSLv3 support and was there opposition?",
    "What was decided about the worker MPM thread limit?",
    "Which bugs were assigned to Justin Erenkrantz?",
    "Were there any reversed or amended decisions?",
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(corpus_path: str, db_path: str, model: str, dry_run: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Layer10 Memory Pipeline")
    print(f"Corpus  : {corpus_path}")
    print(f"DB      : {db_path}")
    print(f"Model   : {model}")
    print(f"{'='*60}\n")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(db_path)

    ingester = MboxIngester(corpus_path)
    extractor = Extractor(model=model)
    person_canon = PersonCanonicalizer(
        override_path="data/canonical_overrides.json"
    )
    project_canon = NameCanonicalizer("Project")
    component_canon = NameCanonicalizer("Component")

    all_claims = []
    message_count = 0
    error_count = 0

    # ------------------------------------------------------------------ #
    # Phase 1: Ingest + Extract
    # ------------------------------------------------------------------ #
    print("[1/4] Ingesting and extracting...")

    for msg in ingester.ingest():
        # Store raw message
        store.upsert_message({
            "id": msg.source_id,
            "message_id": msg.message_id,
            "thread_id": msg.thread_id,
            "from_addr": msg.from_addr,
            "from_name": msg.from_name,
            "subject": msg.subject,
            "body_clean": msg.body_clean,
            "timestamp": msg.timestamp,
            "simhash": msg.simhash,
            "duplicate_of": msg.duplicate_of,
        })

        # Skip extraction on duplicates (but keep message record for QUOTES edges)
        if msg.duplicate_of:
            print(f"  ↩ Skipping duplicate: {msg.message_id}")
            continue

        message_count += 1
        print(f"  Extracting: {msg.message_id} ({msg.subject[:50]})")

        if dry_run:
            # Dry run: generate synthetic extraction for demo purposes
            result = _synthetic_extraction(msg, extractor.extraction_version)
        else:
            result = extractor.extract(msg)

        if not result.is_clean:
            error_count += len(result.parse_errors)
            print(f"    ⚠ {len(result.parse_errors)} parse error(s): {result.parse_errors[0][:80]}")

        # ---------------------------------------------------------------- #
        # Phase 2: Canonicalize entities
        # ---------------------------------------------------------------- #
        for entity in result.entities:
            if isinstance(entity, Person):
                canonical_id = person_canon.add(entity)
                entity.canonical_id = canonical_id
                meta = {
                    "emails": entity.email_addresses,
                    "org": entity.org,
                    "display_name": entity.display_name,
                }
                store.upsert_entity(
                    entity.id, "Person", entity.name,
                    canonical_id=canonical_id,
                    metadata=meta,
                    confidence=entity.confidence,
                )

            elif isinstance(entity, Decision):
                entity.id = entity.id or entity.compute_id()
                meta = {
                    "vote_result": entity.vote_result.value,
                    "participants": entity.participants,
                    "thread_subject": entity.thread_subject,
                    "supersedes_id": entity.supersedes_id,
                }
                store.upsert_entity(
                    entity.id, "Decision", entity.summary,
                    metadata=meta, confidence=entity.confidence,
                )

            elif isinstance(entity, Issue):
                entity.id = entity.id or entity.compute_id()
                meta = {
                    "external_id": entity.external_id,
                    "tracker_url": entity.tracker_url,
                    "status": entity.status,
                }
                store.upsert_entity(
                    entity.id, "Issue", entity.external_id,
                    metadata=meta, confidence=entity.confidence,
                )

            elif isinstance(entity, Project):
                entity.id = entity.id or entity.compute_id()
                canonical_id = project_canon.add(entity.id, entity.name)
                store.upsert_entity(
                    entity.id, "Project", entity.name,
                    canonical_id=canonical_id, confidence=entity.confidence,
                )

            elif isinstance(entity, Component):
                entity.id = entity.id or entity.compute_id()
                canonical_id = component_canon.add(entity.id, entity.name)
                store.upsert_entity(
                    entity.id, "Component", entity.name,
                    canonical_id=canonical_id, confidence=entity.confidence,
                )

            elif isinstance(entity, Concept):
                entity.id = entity.id or entity.compute_id()
                store.upsert_entity(
                    entity.id, "Concept", entity.label,
                    confidence=entity.confidence,
                )

        all_claims.extend(result.claims)

    # ------------------------------------------------------------------ #
    # Phase 3: Claim dedup + conflict resolution
    # ------------------------------------------------------------------ #
    print(f"\n[2/4] Deduplicating {len(all_claims)} claims...")
    canonical_claims, conflicts = deduplicate_claims(all_claims)
    print(f"  → {len(canonical_claims)} canonical claims, {len(conflicts)} conflict(s)")

    for claim in canonical_claims:
        store.upsert_claim(claim)

    for conflict in conflicts:
        store.write_conflict(conflict)

    # ------------------------------------------------------------------ #
    # Phase 4: Write merge log
    # ------------------------------------------------------------------ #
    print("\n[3/4] Writing merge log...")
    for entry in person_canon.get_merge_log():
        store.write_merge_log(entry)
    print(f"  → {len(person_canon.get_merge_log())} person merge(s) recorded")

    # Log extraction run
    store.log_extraction_run(
        model=model,
        schema_version=SCHEMA_VERSION,
        prompt_hash=extractor._ph,
        message_count=message_count,
        error_count=error_count,
    )

    # ------------------------------------------------------------------ #
    # Phase 5: Export + retrieval demo
    # ------------------------------------------------------------------ #
    print("\n[4/4] Exporting and running example queries...")
    store.export_json(DEFAULT_EXPORT)

    stats = store.stats()
    print(f"\n📊 Graph stats: {json.dumps(stats, indent=2)}")

    # Example context packs
    CONTEXT_PACKS_DIR.mkdir(parents=True, exist_ok=True)
    retriever = Retriever(db_path)

    for question in EXAMPLE_QUESTIONS:
        pack = retriever.query(question)
        safe_name = question[:40].replace(" ", "_").replace("?", "").replace("/", "_")
        out_path = CONTEXT_PACKS_DIR / f"{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump({
                "question": pack.question,
                "snippets": [
                    {
                        "score": s.score,
                        "claim_type": s.claim_type,
                        "claim_text": s.claim_text,
                        "excerpt": s.excerpt,
                        "message_id": s.message_id,
                        "timestamp": s.timestamp,
                    }
                    for s in pack.snippets
                ],
                "entities": [{"id": e["id"], "type": e["type"], "name": e["name"]} for e in pack.entities],
                "citations": [
                    {"number": c.number, "message_id": c.message_id, "timestamp": c.timestamp, "subject": c.subject}
                    for c in pack.citations
                ],
                "ambiguities": pack.ambiguities,
            }, f, indent=2)
        print(f"\n  Q: {question}")
        print(f"  → {len(pack.snippets)} snippet(s), {len(pack.citations)} citation(s)")
        if pack.snippets:
            best = pack.snippets[0]
            print(f"  Top result [{best.score:.3f}]: {best.claim_text}")
            print(f"  Evidence: \"{best.excerpt[:120]}\"")

    print(f"\n✅ Done. Graph: {db_path} | Export: {DEFAULT_EXPORT}")


# ---------------------------------------------------------------------------
# Synthetic extraction (dry-run / demo without API key)
# ---------------------------------------------------------------------------

def _synthetic_extraction(msg, extraction_version: str):
    """
    Generate deterministic synthetic extractions from the sample messages.
    Used when --dry-run is set or no API key is configured.
    """
    from src.extraction.ontology import (
        Claim, ClaimType, Component, Decision, Evidence,
        ExtractionResult, Issue, Person, VoteResult,
    )

    entities = []
    claims = []

    def ev(excerpt: str, start: int = 0) -> Evidence:
        return Evidence(
            source_id=msg.source_id,
            message_id=msg.message_id,
            excerpt=excerpt,
            char_start=start,
            char_end=start + len(excerpt),
            timestamp=msg.timestamp,
            extraction_version=extraction_version,
        )

    # Sender is always a Person
    sender = Person(
        name=msg.from_name or msg.from_addr,
        email_addresses=[msg.from_addr],
        evidence=[ev(f"From: {msg.from_name} <{msg.from_addr}>", 0)],
        confidence=1.0,
    )
    sender.id = sender.compute_id()
    entities.append(sender)

    body = msg.body_clean
    subj = msg.subject

    # VOTE threads → Decision
    if "[VOTE]" in subj or "VOTE" in subj.upper():
        summary = subj.replace("[VOTE]", "").replace("[vote]", "").strip()
        vote_result = VoteResult.PENDING
        if "RESOLVED" in body.upper() or "+2" in body:
            vote_result = VoteResult.PASSED

        participants = [msg.from_addr]
        decision = Decision(
            summary=summary,
            vote_result=vote_result,
            participants=participants,
            thread_subject=subj,
            evidence=[ev(body[:200], 0)],
            confidence=0.9,
        )
        decision.id = decision.compute_id()
        entities.append(decision)

        claim_type = ClaimType.SUPPORTS if "+1" in body else (
            ClaimType.OPPOSES if "-1" in body else ClaimType.DECIDED
        )
        claims.append(Claim(
            type=claim_type,
            subject_id=sender.id,
            object_id=decision.id,
            text=f"{sender.name} {'supports' if '+1' in body else 'opposes' if '-1' in body else 'proposed'}: {summary}",
            confidence=0.85,
            valid_from=msg.timestamp,
            evidence=[ev(body[:200], 0)],
        ))

    # Bug references → Issue
    import re
    for match in re.finditer(r"BZ-(\d+)", body):
        bz_id = f"BZ-{match.group(1)}"
        issue = Issue(
            external_id=bz_id,
            tracker_url=f"https://bz.apache.org/bugzilla/show_bug.cgi?id={match.group(1)}",
            status="open",
            evidence=[ev(match.group(0), match.start())],
            confidence=0.95,
        )
        issue.id = issue.compute_id()
        entities.append(issue)

        if "assigned" in body.lower():
            claims.append(Claim(
                type=ClaimType.ASSIGNED,
                subject_id=sender.id,
                object_id=issue.id,
                text=f"Bug {bz_id} mentioned by {sender.name}",
                confidence=0.8,
                valid_from=msg.timestamp,
                evidence=[ev(match.group(0), match.start())],
            ))

    # Component mentions
    for comp_name in ["mod_ssl", "worker MPM", "httpd-worker", "mod_rewrite"]:
        if comp_name.lower() in body.lower() or comp_name.lower() in subj.lower():
            comp = Component(
                name=comp_name,
                evidence=[ev(comp_name, body.lower().find(comp_name.lower()))],
                confidence=0.9,
            )
            comp.id = comp.compute_id()
            entities.append(comp)
            claims.append(Claim(
                type=ClaimType.MENTIONS,
                subject_id=sender.id,
                object_id=comp.id,
                text=f"{sender.name} mentions {comp_name}",
                confidence=0.75,
                valid_from=msg.timestamp,
                evidence=[ev(comp_name, 0)],
            ))

    for c in claims:
        c.id = c.stable_id()

    return ExtractionResult(
        message_id=msg.message_id,
        source_id=msg.source_id,
        extraction_version=extraction_version,
        entities=entities,
        claims=claims,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer10 Memory Pipeline")
    parser.add_argument("--corpus", default=SAMPLE_MBOX)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--sample", action="store_true", help="Generate + use sample corpus")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use synthetic extraction (no API key needed)"
    )
    args = parser.parse_args()

    corpus = args.corpus
    if args.sample or not Path(corpus).exists():
        print(f"[pipeline] Generating sample corpus → {SAMPLE_MBOX}")
        generate_sample_mbox(SAMPLE_MBOX)
        corpus = SAMPLE_MBOX

    # Auto dry-run if no API key
    dry_run = args.dry_run or not os.environ.get("ANTHROPIC_API_KEY")
    if dry_run and not args.dry_run:
        print("[pipeline] No ANTHROPIC_API_KEY found — using synthetic extraction (--dry-run)")

    run(corpus, args.db, args.model, dry_run=dry_run)
