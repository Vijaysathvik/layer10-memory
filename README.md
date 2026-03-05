# Layer10 Take-Home: Grounded Long-Term Memory

> Structured extraction, deduplication, and a context graph over the Apache Software Foundation dev mailing list archives.

---

## Corpus

**Dataset:** Apache Software Foundation mailing list archives (public MBOX format)  
**Source:** https://mail-archives.apache.org/mod_mbox/ — specifically `httpd-dev` (Apache HTTP Server dev list)  
**Reproduction:**
```bash
python src/extraction/download_corpus.py --list httpd-dev --months 6
# Downloads ~6 months of MBOX archives to data/raw/
```
Or use the bundled sample:
```bash
cp data/raw/sample_httpd_dev.mbox data/raw/corpus.mbox
```

The Apache dev lists are chosen because they contain:
- Long-running technical decisions with reversals
- Identity/alias resolution challenges (same person, many email addresses)
- Threaded discussions with quoting/forwarding (artifact dedup challenge)
- References to external artifacts (Bugzilla issues, SVN commits, RFCs)
- Explicit decision records ("VOTE", "+1/-1", "RESOLVED")

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download or use sample corpus
python src/extraction/download_corpus.py --sample

# 3. Run full pipeline
python pipeline.py --corpus data/raw/corpus.mbox

# 4. Launch visualization
cd visualization && python -m http.server 8080
# open http://localhost:8080
```

---

## Architecture Overview

```
corpus (MBOX)
    │
    ▼
[1] Ingestion & Artifact Dedup      src/extraction/ingest.py
    • parse MBOX → Message objects
    • fingerprint (SimHash) → deduplicate quoted/forwarded copies
    • normalize bodies (strip signatures, strip quoted blocks)
    │
    ▼
[2] Structured Extraction           src/extraction/extractor.py
    • LLM call (claude-haiku free tier / local Ollama mistral)
    • schema: Entity | Claim | Event  (see ontology.py)
    • grounding: every claim carries source_id + excerpt + char offsets
    • validation & repair loop (pydantic + retry)
    │
    ▼
[3] Entity Canonicalization         src/deduplication/canonicalize.py
    • person aliases → canonical Person node
    • project/component name normalization
    • merge log (reversible, audit trail)
    │
    ▼
[4] Claim Dedup & Conflict Res.     src/deduplication/claim_dedup.py
    • embedding similarity → candidate pairs → LLM judge
    • validity time: [valid_from, valid_to] on each claim
    • conflict representation: competing claims with provenance
    │
    ▼
[5] Memory Graph                    src/graph/store.py
    • SQLite + adjacency tables (portable, no server needed)
    • queryable by entity / claim type / time window / confidence
    • idempotent upsert; soft-delete for redactions
    │
    ▼
[6] Retrieval API                   src/retrieval/retriever.py
    • hybrid: BM25 keyword + embedding cosine
    • context pack: ranked evidence + linked entities/claims + citations
    │
    ▼
[7] Visualization                   visualization/
    • single-page app (vanilla JS + D3 force graph)
    • evidence panel, merge inspector, time filter
```

---

## Ontology

See [`src/extraction/ontology.py`](src/extraction/ontology.py) for full Pydantic schemas.

### Entity Types
| Type | Fields | Notes |
|------|--------|-------|
| `Person` | name, email_addresses[], org, canonical_id | alias-resolved |
| `Project` | name, repo_url, component | e.g. "httpd-core" |
| `Component` | name, project_id | e.g. "mod_ssl" |
| `Decision` | summary, vote_result, participants[] | VOTE threads |
| `Issue` | external_id, tracker_url, status | Bugzilla refs |
| `Concept` | label, definition | recurring technical terms |

### Claim / Relation Types
| Type | Subject → Object | Meaning |
|------|-----------------|---------|
| `AUTHORED` | Person → Message | wrote this message |
| `DECIDED` | Decision → Concept/Project | group resolved to do X |
| `REVERSED` | Decision → Decision | supersedes earlier decision |
| `ASSIGNED` | Person → Issue | owns this ticket |
| `DEPENDS_ON` | Issue → Issue | blocking relationship |
| `MENTIONS` | Message → Entity | co-occurrence reference |
| `SUPPORTS` | Message → Decision | +1 vote |
| `OPPOSES` | Message → Decision | -1 vote |

### Evidence Schema
```python
class Evidence(BaseModel):
    source_id: str          # SHA-256 of raw message
    message_id: str         # RFC 2822 Message-ID header
    excerpt: str            # verbatim text span (≤ 500 chars)
    char_start: int
    char_end: int
    timestamp: datetime
    extraction_version: str # "{model}:{schema_version}:{prompt_hash}"
```

---

## Deduplication Strategy

### Artifact Dedup
1. **Exact dedup**: SHA-256 of normalized body → skip if seen
2. **Near-dedup**: 4-gram SimHash with Hamming distance ≤ 3 → mark as duplicate, keep earliest
3. **Quote stripping**: lines starting with `>` are stripped before fingerprinting; the original quoted message is linked via `QUOTES` edge rather than re-extracted

### Entity Canonicalization
- **Person**: email addresses clustered by domain-normalized username + name token overlap. Merge produces a `canonical_id`; all aliases kept in `email_addresses[]`. Merges logged to `entity_merge_log` table with `reason` and `merged_at`.
- **Projects/Components**: lowercased, punctuation-normalized, then exact match. Manual override file at `data/canonical_overrides.json`.
- **Reversibility**: merges never delete source rows. `entity_merge_log` stores `(source_id, target_id, reason, merged_at, merged_by)`. Undo = set `canonical_id = source_id` on source row, delete merge log entry.

### Claim Dedup
1. Embed claim text with `sentence-transformers/all-MiniLM-L6-v2`
2. Candidate pairs: cosine ≥ 0.92
3. LLM judge (yes/no): "Are these two claims saying the same thing about the same subject?"
4. Merge: keep both evidence pointers under one canonical claim node; set `support_count`
5. Conflicts: claims with same subject+predicate but different object → stored as `CONFLICTS_WITH` edge; `valid_to` set on superseded claim when reversal is detected

---

## Memory Graph Schema (SQLite)

```sql
-- Core tables
entities(id, type, name, canonical_id, metadata_json, created_at, updated_at, deleted_at)
claims(id, type, subject_id, object_id, text, confidence, valid_from, valid_to,
       support_count, created_at, updated_at)
evidence(id, claim_id, source_id, message_id, excerpt, char_start, char_end,
         timestamp, extraction_version)
messages(id, message_id, thread_id, from_addr, subject, body_clean, timestamp,
         simhash, duplicate_of)
entity_merge_log(id, source_id, target_id, reason, merged_at, merged_by)
claim_conflicts(id, claim_a, claim_b, resolution, resolved_at)
extraction_runs(id, model, schema_version, prompt_hash, run_at, message_count)
```

**Time semantics:**
- `valid_from` / `valid_to`: *validity time* — when the claim was true in the world
- `created_at` / `updated_at`: *system time* — when we recorded it
- "Current" = `valid_to IS NULL AND deleted_at IS NULL`
- Historical queries: `WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)`

---

## Retrieval

`src/retrieval/retriever.py` exposes:

```python
retriever = Retriever(db_path="outputs/graph/memory.db")
pack = retriever.query("Who decided to deprecate SSLv2 support and when?")
```

Returns a `ContextPack`:
```python
@dataclass
class ContextPack:
    question: str
    entities: list[Entity]
    claims: list[RankedClaim]     # score, claim, evidence[]
    snippets: list[EvidenceSnippet]
    citations: list[Citation]     # [1] Author, Date, Subject, message_id
    ambiguities: list[str]        # flagged conflicts or low-confidence items
```

**Ranking:**
1. BM25 over claim text + entity names
2. Embedding cosine (MiniLM) over claim text
3. Score = 0.5·BM25_norm + 0.3·cosine + 0.1·support_count_norm + 0.1·recency_norm
4. Hard cap: top-10 evidence snippets, diversity penalty for same-source items

---

## Example Context Packs

See [`outputs/context_packs/`](outputs/context_packs/) for pre-generated JSON.

Questions answered:
1. `"Who proposed removing SSLv3 support and was there opposition?"`
2. `"What was decided about the worker MPM thread limit?"`
3. `"Which bugs were assigned to Justin Erenkrantz in 2003?"`
4. `"Were there any reversed decisions about the 2.0 release timeline?"`

---

## Visualization

Single-page app at `visualization/index.html`. No build step needed.

Features:
- **Force-directed graph** (D3 v7): entities as nodes, claims as edges
- **Filter bar**: by entity type, claim type, confidence threshold, date range
- **Evidence panel**: click any edge → see exact excerpt(s), source metadata, extraction version
- **Merge inspector**: click any Person node → see all aliases and merge history
- **Conflict viewer**: red edges = conflicting claims; panel shows both sides
- **Timeline scrubber**: drag to see memory state at any point in time

Screenshots: `visualization/screenshots/`

---

## Layer10 Adaptation

See [`LAYER10_ADAPTATION.md`](LAYER10_ADAPTATION.md) for the full write-up on adapting this to Layer10's production environment (email + Slack + Jira/Linear).

---

## Evaluation & Observability

- `tests/test_extraction.py` — schema validation, grounding completeness
- `tests/test_dedup.py` — merge correctness, reversibility
- `tests/test_retrieval.py` — recall@k on labeled question set
- Logged per extraction run: `extraction_runs` table tracks model + schema version + prompt hash → enables regression detection when ontology changes
- Quality gates: claims with `confidence < 0.5` quarantined in `claims_quarantine` table pending human review

---

## Reproducing End-to-End

```bash
git clone <this-repo>
cd layer10-memory
pip install -r requirements.txt

# Option A: use bundled 50-message sample
python pipeline.py --corpus data/raw/sample_httpd_dev.mbox --sample

# Option B: download live corpus (requires internet)
python src/extraction/download_corpus.py --list httpd-dev --months 3
python pipeline.py --corpus data/raw/corpus.mbox

# Outputs written to:
#   outputs/graph/memory.db          (SQLite memory graph)
#   outputs/graph/memory_export.json (portable JSON export)
#   outputs/context_packs/*.json     (example retrieval results)

# Launch visualization
cd visualization && python -m http.server 8080
```

---

## Dependencies

```
pydantic>=2.0
sentence-transformers
rank_bm25
mailbox (stdlib)
simhash
anthropic          # or: ollama (local)
sqlite3 (stdlib)
d3 (CDN, visualization only)
```
