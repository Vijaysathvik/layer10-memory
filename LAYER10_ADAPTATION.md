# Layer10 Adaptation Write-Up

## How I Would Adapt This System for Layer10's Environment

Layer10's target environment — email, Slack/Teams, docs, and structured systems like Jira/Linear — is meaningfully more complex than a single Apache mailing list. Below I walk through what would change and why.

---

## 1. Ontology Changes

The mailing-list ontology captures the right conceptual primitives, but needs expansion:

**New entity types:**
- `Channel` / `Workspace` (Slack): messages have a channel + workspace scope, which matters for permissions
- `Thread` (Slack): ephemeral discussions that may or may not become durable memory
- `Document` (Google Docs, Notion, Confluence): long-form artifacts with revision history
- `Ticket` (Jira/Linear): replaces our informal Issue, now with structured fields (status, sprint, priority, story points)
- `Customer` / `Account`: in many orgs, decisions reference specific customers
- `Meeting`: calendar events that generate transcripts/notes

**New claim types:**
- `MENTIONED_IN_TICKET` (Message → Ticket): connects Slack discussion to the formal ticket
- `BLOCKED_BY`, `CLOSES` (Ticket → Ticket): richer issue dependency graph
- `SUPERSEDES_DOC` (Document → Document): version chains
- `COMMITTED_TO` (Person → Ticket): promise/assignment in chat, not yet a formal Jira assignment

**What stays the same:** The `Evidence` schema (source_id + excerpt + offsets + timestamp) works across all connectors. Every claim must be traceable regardless of origin.

---

## 2. Extraction Contract Changes

The current extraction prompt is tuned for threading, vote semantics, and Bugzilla references. For Layer10:

**Per-connector extraction:**
- **Email**: existing approach works well. Add RFC 2822 thread reconstruction, DKIM sender verification for identity confidence.
- **Slack**: messages are short and noisy. Extraction should be at the *thread* level, not individual messages. System-generated messages (join/leave, bot posts) filtered pre-extraction. Emoji reactions (+1 / ✅) treated as implicit SUPPORTS claims with lower confidence.
- **Jira/Linear**: structured fields → deterministic extraction (no LLM needed for status, assignee, priority). LLM only for comment bodies and description text. Field changes (status transitions, reassignments) auto-generate REVISED claims with timestamps.
- **Docs**: chunk by section/heading, not whole-document. Extract decisions and action items. Version diffs → detect reversals by diffing adjacent versions.

**Extraction versioning** becomes critical when connectors update their API schemas. Extraction runs must be tagged with `{connector}:{api_version}:{schema_version}:{model}:{prompt_hash}` and stored in `extraction_runs`.

---

## 3. Deduplication Strategy Changes

**Artifact dedup** is harder in the real environment:
- Email threads are often forwarded into Slack ("FYI, see below") → cross-platform near-dedup needed. SimHash still works but must normalize across line-ending conventions and email quoting styles.
- Jira comments frequently quote Slack discussions verbatim. Cross-source SimHash with a shared normalization step catches this.
- Meeting transcripts overlap heavily with follow-up emails/docs → section-level SimHash rather than document-level.

**Entity canonicalization** gains a new dimension: **org directory integration**. With access to LDAP / Google Workspace / Okta, person aliases are resolved with high confidence using canonical employee IDs. This reduces the need for heuristic email-username matching.

**Claim dedup** needs a time-decay component in the real environment. A decision made 3 years ago and restated last week should have the old claim's `valid_to` set, not a merge. The current approach handles this via reversal keywords; in production, a recency-weighted conflict resolution policy is more robust.

---

## 4. Grounding Requirements

In a production system, grounding requirements become compliance requirements:

- **Permalink stability**: every `Evidence` pointer must include a stable URL (Slack message permalink, Jira comment URL, email Message-ID), not just internal source_id. These can be invalidated by workspace migrations.
- **Redaction handling**: when a Slack message is deleted, all claims grounded *only* in that message get `valid_to` set and a `REDACTED` flag. Claims with multiple evidence sources remain but with reduced confidence. This is why reversibility matters.
- **Permission-scoped evidence**: a claim grounded in a private Slack DM must carry the DM channel ID in its evidence metadata. Retrieval layer enforces that the requesting user has access to the channel. This means evidence rows need an `access_scope` field (e.g., `{"slack_channel": "D0123456", "workspace": "acme"}`).

---

## 5. Long-Term Memory Behavior

Not everything should become durable memory. The current pipeline extracts from every message; in production you need a **durability policy**:

| Signal | Policy |
|--------|--------|
| Explicit VOTE / DECISION / RESOLVED marker | Always durable |
| Cross-referenced in 3+ messages over 7+ days | Durable (high cross-evidence support) |
| Single mention, low confidence | Ephemeral (TTL 30 days, promote if re-referenced) |
| Slack casual chat, no entities | Not extracted |
| Ticket status transition | Always durable (structured source) |
| Meeting transcript action item | Durable if assigned with due date |

**Drift prevention**: scheduled jobs re-evaluate claims against recent evidence. If a claim hasn't been reinforced in 6 months and the related entities are stale, confidence decays. If confidence drops below threshold, claim moves to `claims_quarantine` for human review.

---

## 6. Permissions

The current system has no permission model; Layer10 needs one at every layer:

- **Ingestion**: connector auth tokens are scoped per user/service account. The pipeline records `access_scope` on every message at ingest time.
- **Entity graph**: entities derived from private sources carry the union of their evidence's access scopes. A user who can't see Slack channel #exec-offsite should not learn that "Q3 target was revised" even if that claim has public evidence too — unless the public evidence alone supports it (this requires evidence-level access checks, not claim-level).
- **Retrieval**: the retriever filters evidence by `access_scope ∩ user_permissions`. Claims where all evidence is filtered become invisible to that user.
- **Memory summaries / LLM answers**: the LLM context pack is constructed from access-filtered evidence only. No leakage through model reasoning over privately-grounded claims.

---

## 7. Operational Reality

**Scaling**: the current SQLite store works for thousands of messages. For Layer10's scale (millions of messages across an org), the store layer should be:
- **Postgres** for the relational tables (entities, claims, evidence) — JSONB for metadata, GIN indexes for full-text search
- **pgvector** for claim embeddings (replacing in-process numpy) — enables ANN retrieval at scale
- **Object storage** (S3/GCS) for raw message bodies — evidence table stores offsets + object keys, not inline excerpts

**Incremental ingestion**: each connector runs on a per-source watermark (last processed event ID / timestamp). Idempotent upserts (already implemented) ensure re-processing is safe. Schema migrations are handled with Alembic; old extraction runs are flagged for backfill when ontology changes.

**Cost**: LLM extraction is the most expensive step. Mitigation:
- Route short/structured messages (Jira field changes, emoji reactions) through deterministic extractors, not LLM
- Cache extraction results keyed on `(source_id, extraction_version)` — re-extraction only on schema/model change
- Batch low-priority messages (archived tickets, old email) through a cheaper model tier

**Evaluation / regression testing**: every ontology change triggers a regression suite over a labeled held-out set of messages. `extraction_runs` table enables comparison of extraction quality across model/schema versions. A CI check blocks deployment if `grounding_completeness` or `entity_recall` drops by > 5%.

---

## Summary of Key Changes

| Dimension | This prototype | Layer10 production |
|-----------|---------------|-------------------|
| Connectors | Single MBOX | Email + Slack + Jira + Docs + Calendar |
| Entity resolution | Email heuristics | Org directory (LDAP/Okta) |
| Extraction routing | LLM for all | Deterministic for structured, LLM for unstructured |
| Durability policy | Extract everything | Signal-based promote/decay |
| Permissions | None | Evidence-scoped, retrieval-enforced |
| Store | SQLite | Postgres + pgvector + S3 |
| Observability | Extraction run log | Full metrics pipeline (confidence drift, recall@k, latency) |
