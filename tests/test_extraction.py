"""
tests/test_extraction.py — Extraction schema, grounding completeness, validation/repair.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
import pytest

from src.extraction.ontology import (
    Evidence, Person, Decision, Issue, Component, Claim,
    ClaimType, VoteResult, ExtractionResult, SCHEMA_VERSION,
)
from src.extraction.ingest import MboxIngester, generate_sample_mbox
from src.extraction.extractor import Extractor, _extract_json
import tempfile, os


# ---- Evidence schema ----

def make_ev(**kwargs):
    defaults = dict(
        source_id="sha_abc",
        message_id="<001@test>",
        excerpt="some verbatim text",
        char_start=0,
        char_end=18,
        timestamp=datetime(2003,1,14,10,0,0),
        extraction_version=f"test:{SCHEMA_VERSION}:deadbeef",
    )
    defaults.update(kwargs)
    return Evidence(**defaults)


def test_evidence_offsets_ordered():
    with pytest.raises(Exception):
        make_ev(char_start=100, char_end=50)


def test_evidence_span_length():
    ev = make_ev(char_start=10, char_end=40)
    assert ev.span_length == 30


# ---- Entity schema ----

def test_person_email_normalized():
    ev = make_ev()
    p = Person(name="Justin E", email_addresses=["JUSTIN@Apache.ORG"], evidence=[ev])
    assert p.email_addresses == ["justin@apache.org"]


def test_person_id_stable():
    ev = make_ev()
    p1 = Person(name="Justin E", email_addresses=["justin@apache.org"], evidence=[ev])
    p2 = Person(name="Justin Erenkrantz", email_addresses=["justin@apache.org"], evidence=[ev])
    p1.id = p1.compute_id()
    p2.id = p2.compute_id()
    assert p1.id == p2.id  # keyed on email


def test_decision_fields():
    ev = make_ev()
    d = Decision(
        summary="Deprecate SSLv3",
        vote_result=VoteResult.PASSED,
        participants=["a@b.com"],
        evidence=[ev],
    )
    d.id = d.compute_id()
    assert d.id is not None
    assert d.vote_result == VoteResult.PASSED


# ---- ExtractionResult ----

def test_grounding_complete():
    ev = make_ev()
    p = Person(name="A", email_addresses=["a@b.com"], evidence=[ev])
    p.id = p.compute_id()
    d = Decision(summary="Do X", evidence=[ev])
    d.id = d.compute_id()
    c = Claim(
        type=ClaimType.DECIDED,
        subject_id=p.id, object_id=d.id,
        text="A decided to do X",
        confidence=0.9,
        evidence=[ev],
    )
    result = ExtractionResult(
        message_id="<001@test>",
        source_id="sha_abc",
        extraction_version="test:1.0.0:abc",
        entities=[p, d],
        claims=[c],
    )
    assert result.grounding_complete()


def test_grounding_incomplete_fails():
    ev = make_ev()
    p = Person(name="A", email_addresses=["a@b.com"], evidence=[ev])
    p.id = p.compute_id()
    with pytest.raises(Exception):
        # claim with no evidence should fail pydantic min_length=1
        Claim(
            type=ClaimType.DECIDED,
            subject_id=p.id, object_id="some-id",
            text="A decided something",
            confidence=0.9,
            evidence=[],  # invalid
        )


# ---- JSON extraction ----

def test_extract_json_plain():
    raw = '{"entities": [], "claims": []}'
    assert _extract_json(raw) == {"entities": [], "claims": []}


def test_extract_json_fenced():
    raw = '```json\n{"entities": [], "claims": []}\n```'
    assert _extract_json(raw) == {"entities": [], "claims": []}


def test_extract_json_bad():
    import json
    with pytest.raises(json.JSONDecodeError):
        _extract_json("not json at all")


# ---- Ingest + synthetic extraction ----

def test_sample_mbox_roundtrip(tmp_path):
    mbox_path = tmp_path / "test.mbox"
    generate_sample_mbox(str(mbox_path))
    ingester = MboxIngester(str(mbox_path))
    messages = list(ingester.ingest())
    assert len(messages) > 0
    # All messages have source_id
    assert all(m.source_id for m in messages)
    # No two non-duplicate messages share a source_id
    non_dup_ids = [m.source_id for m in messages if not m.duplicate_of]
    assert len(non_dup_ids) == len(set(non_dup_ids))


def test_ingest_dedup(tmp_path):
    """Exact duplicate message should be flagged."""
    import mailbox
    from email.mime.text import MIMEText

    mbox_path = tmp_path / "dup.mbox"
    mbox = mailbox.mbox(str(mbox_path), create=True)
    body = "Hello world, this is a test message."
    for i in range(2):
        msg = MIMEText(body)
        msg["Message-ID"] = f"<00{i}@test.org>"
        msg["From"] = "test@test.org"
        msg["Date"] = "Mon, 14 Jan 2003 10:00:00 -0800"
        msg["Subject"] = "Test"
        mbox.add(msg)
    mbox.flush(); mbox.close()

    ingester = MboxIngester(str(mbox_path))
    msgs = list(ingester.ingest())
    dups = [m for m in msgs if m.duplicate_of]
    assert len(dups) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
