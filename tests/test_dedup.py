"""
tests/test_dedup.py — Deduplication, canonicalization, merge reversibility.
tests/test_retrieval.py — Retrieval recall and grounding.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
import pytest

from src.extraction.ontology import (
    Evidence, Person, Claim, ClaimType, Decision, VoteResult,
)
from src.deduplication.canonicalize import PersonCanonicalizer, NameCanonicalizer
from src.deduplication.claim_dedup import ClaimDeduplicator, deduplicate_claims
from src.graph.store import MemoryStore


def ev(msg_id="<t@t>"):
    return Evidence(
        source_id="sha_test",
        message_id=msg_id,
        excerpt="test excerpt",
        char_start=0, char_end=12,
        timestamp=datetime(2003,1,14),
        extraction_version="test:1.0.0:abc",
    )


def make_person(name, emails, msg_id="<t@t>"):
    p = Person(name=name, email_addresses=emails, evidence=[ev(msg_id)])
    p.id = p.compute_id()
    return p


def make_claim(ctype, subj, obj, text, conf=0.8, valid_from=None, valid_to=None):
    c = Claim(
        type=ctype, subject_id=subj, object_id=obj, text=text,
        confidence=conf, valid_from=valid_from, valid_to=valid_to,
        evidence=[ev()],
    )
    c.id = c.stable_id()
    return c


# ============================================================
# CANONICALIZATION TESTS
# ============================================================

class TestPersonCanonicalizer:
    def test_same_email_merges(self):
        canon = PersonCanonicalizer()
        p1 = make_person("Justin E", ["justin@apache.org"])
        p2 = make_person("J. Erenkrantz", ["justin@apache.org"])
        id1 = canon.add(p1)
        id2 = canon.add(p2)
        assert id1 == id2

    def test_different_email_no_merge(self):
        canon = PersonCanonicalizer()
        p1 = make_person("Alice", ["alice@a.org"])
        p2 = make_person("Bob", ["bob@b.org"])
        id1 = canon.add(p1)
        id2 = canon.add(p2)
        assert id1 != id2

    def test_username_overlap_merges(self):
        canon = PersonCanonicalizer()
        p1 = make_person("Greg Stein", ["gstein@apache.org"])
        p2 = make_person("Greg Stein", ["gstein@google.com"])
        id1 = canon.add(p1)
        id2 = canon.add(p2)
        # username "gstein" matches → should merge
        assert id1 == id2

    def test_merge_log_recorded(self):
        canon = PersonCanonicalizer()
        p1 = make_person("Justin", ["jerenkrantz@apache.org"])
        p2 = make_person("Justin E", ["jerenkrantz@apache.org"])
        canon.add(p1)
        canon.add(p2)
        log = canon.get_merge_log()
        assert len(log) >= 1

    def test_merge_reversible(self):
        canon = PersonCanonicalizer()
        p1 = make_person("Alice", ["alice@x.org"])
        p2 = make_person("Alice Smith", ["alice@x.org"])
        id1 = canon.add(p1)
        id2 = canon.add(p2)
        assert id1 == id2

        # Undo merge
        result = canon.undo_merge(p2.id)
        assert result or canon.undo_merge(p1.id)

        # After undo, log entry has undone_at set
        log = canon.get_merge_log()
        undone = [e for e in log if e.undone_at is not None]
        assert len(undone) >= 1


class TestNameCanonicalizer:
    def test_same_normalized_name(self):
        nc = NameCanonicalizer("Component")
        id1 = nc.add("comp_1", "mod_ssl")
        id2 = nc.add("comp_2", "Mod SSL")
        # Both normalize to "mod ssl"
        assert id1 == id2

    def test_different_names_separate(self):
        nc = NameCanonicalizer("Component")
        id1 = nc.add("c1", "mod_ssl")
        id2 = nc.add("c2", "mod_rewrite")
        assert id1 != id2

    def test_resolve(self):
        nc = NameCanonicalizer("Component")
        nc.add("c1", "mod_ssl")
        assert nc.resolve("Mod SSL") == "c1"
        assert nc.resolve("nonexistent") is None


# ============================================================
# CLAIM DEDUP TESTS
# ============================================================

class TestClaimDedup:
    def test_exact_duplicate_merged(self):
        c1 = make_claim(ClaimType.DECIDED, "s1", "o1", "X decided Y")
        c2 = make_claim(ClaimType.DECIDED, "s1", "o1", "X decided Y")
        canonical, _ = deduplicate_claims([c1, c2])
        assert len(canonical) == 1
        assert canonical[0].support_count == 2

    def test_different_type_not_merged(self):
        c1 = make_claim(ClaimType.SUPPORTS, "s1", "o1", "s1 supports o1")
        c2 = make_claim(ClaimType.OPPOSES, "s1", "o1", "s1 opposes o1")
        canonical, _ = deduplicate_claims([c1, c2])
        assert len(canonical) == 2

    def test_conflict_detected(self):
        c1 = make_claim(ClaimType.ASSIGNED, "issue1", "alice", "issue1 assigned to alice")
        c2 = make_claim(ClaimType.ASSIGNED, "issue1", "bob", "issue1 assigned to bob")
        _, conflicts = deduplicate_claims([c1, c2])
        assert len(conflicts) == 1

    def test_reversal_sets_valid_to(self):
        dt_old = datetime(2003, 1, 14)
        dt_new = datetime(2003, 1, 15)
        c1 = make_claim(ClaimType.DECIDED, "s", "o", "DECIDED: do X", valid_from=dt_old)
        c2 = make_claim(ClaimType.DECIDED, "s", "o2",
                        "REVERSED previous decision, now do Y instead",
                        valid_from=dt_new)
        canonical, conflicts = deduplicate_claims([c1, c2])
        # c1 should have valid_to set
        c1_out = next((c for c in canonical if c.id == c1.id), None)
        if c1_out:
            assert c1_out.valid_to is not None

    def test_evidence_merged(self):
        c1 = make_claim(ClaimType.MENTIONS, "s", "o", "s mentions o")
        c2 = make_claim(ClaimType.MENTIONS, "s", "o", "s mentions o")
        canonical, _ = deduplicate_claims([c1, c2])
        assert len(canonical[0].evidence) == 2


# ============================================================
# STORE TESTS
# ============================================================

class TestMemoryStore:
    def test_upsert_entity_idempotent(self):
        store = MemoryStore(":memory:")
        store.upsert_entity("eid1", "Person", "Alice", confidence=0.9)
        store.upsert_entity("eid1", "Person", "Alice", confidence=0.95)
        ent = store.get_entity("eid1")
        assert ent is not None
        assert ent["confidence"] == 0.95

    def test_upsert_claim_with_evidence(self):
        store = MemoryStore(":memory:")
        c = make_claim(ClaimType.DECIDED, "s1", "o1", "Test claim")
        store.upsert_entity("s1", "Person", "S")
        store.upsert_entity("o1", "Decision", "D")
        store.upsert_claim(c)
        evs = store.get_evidence_for_claim(c.id)
        assert len(evs) == 1
        assert evs[0]["excerpt"] == "test excerpt"

    def test_soft_delete(self):
        store = MemoryStore(":memory:")
        store.upsert_entity("eid2", "Person", "Bob")
        store.soft_delete_entity("eid2")
        results = store.search_entities("Bob")
        assert len(results) == 0

    def test_stats(self):
        store = MemoryStore(":memory:")
        store.upsert_entity("e1", "Person", "Alice")
        stats = store.stats()
        assert stats["entities"] == 1
        assert stats["claims_current"] == 0

    def test_export_json(self, tmp_path):
        store = MemoryStore(":memory:")
        store.upsert_entity("e1", "Person", "Alice")
        out = tmp_path / "export.json"
        store.export_json(str(out))
        import json
        data = json.loads(out.read_text())
        assert len(data["entities"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
