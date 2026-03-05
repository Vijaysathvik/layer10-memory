"""
extractor.py — LLM-based structured extraction with validation, repair, and grounding.

Supports:
  - Anthropic claude-haiku (free tier / low cost)
  - Local Ollama (mistral / llama3)

Every extracted claim carries at least one Evidence pointer back to the source message.
Invalid/partial outputs trigger a repair loop (up to MAX_RETRIES attempts).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from pydantic import ValidationError

from .ingest import ParsedMessage
from .ontology import (
    SCHEMA_VERSION,
    Claim,
    ClaimType,
    Component,
    Concept,
    Decision,
    EntityType,
    Evidence,
    ExtractionResult,
    Issue,
    MergeLogEntry,
    Person,
    Project,
    VoteResult,
)

MAX_RETRIES = 3
DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # free-tier / low-cost


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are a structured-extraction engine for a long-term memory system.
Given a mailing-list message, extract all entities and claims.

OUTPUT FORMAT: Return ONLY valid JSON matching this exact schema:
{
  "entities": [
    {
      "type": "Person|Project|Component|Decision|Issue|Concept",
      "name": "...",
      "email_addresses": ["..."],   // Person only
      "org": "...",                 // Person only, optional
      "vote_result": "passed|failed|withdrawn|pending",  // Decision only
      "participants": ["email1"],   // Decision only
      "external_id": "BZ-XXXXX",   // Issue only
      "tracker_url": "...",         // Issue optional
      "status": "open|closed|resolved|wontfix",  // Issue optional
      "label": "...",               // Concept only
      "definition": "...",          // Concept optional
      "confidence": 0.0-1.0,
      "excerpt": "verbatim text span (≤400 chars) that mentions this entity",
      "char_start": 0,
      "char_end": 100
    }
  ],
  "claims": [
    {
      "type": "AUTHORED|DECIDED|REVERSED|ASSIGNED|DEPENDS_ON|MENTIONS|SUPPORTS|OPPOSES|IMPLEMENTED_BY|QUOTES",
      "subject_name": "name of subject entity",
      "object_name": "name of object entity",
      "text": "human-readable statement of this claim",
      "confidence": 0.0-1.0,
      "valid_from": "ISO datetime or null",
      "valid_to": "ISO datetime or null",
      "excerpt": "verbatim text span from message supporting this claim (≤400 chars)",
      "char_start": 0,
      "char_end": 100
    }
  ]
}

Rules:
- Every entity and claim MUST have an excerpt and char offsets pointing into the message body.
- Excerpt must be verbatim text from the message.
- For Person entities, always include all email addresses visible in the message.
- For Decision entities, extract the vote outcome if visible.
- char_start/char_end are character offsets into the combined "Subject: ... \\nBody: ..." text.
- Do NOT include markdown, code fences, or any text outside the JSON object.
- If nothing interesting is present, return {"entities": [], "claims": []}.
""").strip()


def _build_user_prompt(msg: ParsedMessage) -> tuple[str, str]:
    """
    Returns (prompt_text, combined_text).
    combined_text is what char offsets reference.
    """
    combined = f"Subject: {msg.subject}\n\nBody:\n{msg.body_clean}"
    prompt = (
        f"Message-ID: {msg.message_id}\n"
        f"From: {msg.from_name} <{msg.from_addr}>\n"
        f"Date: {msg.timestamp.isoformat()}\n\n"
        f"--- MESSAGE TEXT (use this for char offsets) ---\n"
        f"{combined}\n"
        f"--- END ---\n\n"
        f"Extract all entities and claims. Return only JSON."
    )
    return prompt, combined


def _prompt_hash(system: str, user_prefix: str) -> str:
    blob = (system + user_prefix[:200]).encode()
    return hashlib.md5(blob).hexdigest()[:8]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_anthropic(user_prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def _call_ollama(user_prompt: str, model: str = "mistral") -> str:
    import urllib.request
    payload = json.dumps({
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data.get("response", "")


def _llm_call(user_prompt: str, model: str) -> str:
    if model.startswith("claude"):
        return _call_anthropic(user_prompt, model)
    else:
        return _call_ollama(user_prompt, model)


# ---------------------------------------------------------------------------
# JSON parsing & repair
# ---------------------------------------------------------------------------

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(raw: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    m = JSON_FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    return json.loads(raw)


def _repair_prompt(original_prompt: str, raw_output: str, error: str) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Your previous output failed validation:\n{error}\n\n"
        f"Previous output (for reference):\n{raw_output[:800]}\n\n"
        f"Please return corrected JSON only."
    )


# ---------------------------------------------------------------------------
# Evidence construction
# ---------------------------------------------------------------------------

def _make_evidence(
    msg: ParsedMessage,
    excerpt: str,
    char_start: int,
    char_end: int,
    extraction_version: str,
) -> Evidence:
    # Clamp offsets to body length
    combined_len = len(f"Subject: {msg.subject}\n\nBody:\n{msg.body_clean}")
    char_end = min(char_end, combined_len)
    char_start = max(0, char_start)
    if char_end <= char_start:
        char_end = min(char_start + len(excerpt), combined_len)

    return Evidence(
        source_id=msg.source_id,
        message_id=msg.message_id,
        excerpt=excerpt[:600],
        char_start=char_start,
        char_end=char_end,
        timestamp=msg.timestamp,
        extraction_version=extraction_version,
    )


# ---------------------------------------------------------------------------
# Main Extractor
# ---------------------------------------------------------------------------

class Extractor:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._ph = _prompt_hash(SYSTEM_PROMPT, "v1")
        self.extraction_version = f"{model}:{SCHEMA_VERSION}:{self._ph}"

    def extract(self, msg: ParsedMessage) -> ExtractionResult:
        user_prompt, combined_text = _build_user_prompt(msg)
        parse_errors: list[str] = []
        raw_output = ""
        retries = 0

        for attempt in range(MAX_RETRIES):
            try:
                raw_output = _llm_call(user_prompt, self.model)
                raw_dict = _extract_json(raw_output)
                result = self._build_result(msg, raw_dict, combined_text)
                result.retries = attempt
                result.parse_errors = parse_errors
                return result
            except json.JSONDecodeError as e:
                err = f"JSONDecodeError: {e}"
                parse_errors.append(err)
                user_prompt = _repair_prompt(user_prompt, raw_output, err)
                retries += 1
            except ValidationError as e:
                err = f"ValidationError: {e.error_count()} errors — {str(e)[:300]}"
                parse_errors.append(err)
                user_prompt = _repair_prompt(user_prompt, raw_output, err)
                retries += 1
            except Exception as e:
                err = f"Unexpected error: {e}"
                parse_errors.append(err)
                break

        # Return empty result after exhausted retries
        return ExtractionResult(
            message_id=msg.message_id,
            source_id=msg.source_id,
            extraction_version=self.extraction_version,
            entities=[],
            claims=[],
            parse_errors=parse_errors,
            retries=retries,
        )

    def _build_result(
        self, msg: ParsedMessage, raw: dict, combined_text: str
    ) -> ExtractionResult:
        entities = []
        entity_name_map: dict[str, object] = {}

        for e_raw in raw.get("entities", []):
            etype = e_raw.get("type", "")
            excerpt = e_raw.get("excerpt", "")[:600]
            cs = int(e_raw.get("char_start", 0))
            ce = int(e_raw.get("char_end", cs + len(excerpt)))
            ev = _make_evidence(msg, excerpt, cs, ce, self.extraction_version)
            conf = float(e_raw.get("confidence", 0.8))

            try:
                if etype == "Person":
                    emails = e_raw.get("email_addresses") or [msg.from_addr]
                    ent = Person(
                        name=e_raw.get("name", "Unknown"),
                        email_addresses=emails,
                        org=e_raw.get("org"),
                        confidence=conf,
                        evidence=[ev],
                    )
                    ent.id = ent.compute_id()
                elif etype == "Decision":
                    ent = Decision(
                        summary=e_raw.get("name", "Unnamed decision"),
                        vote_result=VoteResult(e_raw.get("vote_result", "pending")),
                        participants=e_raw.get("participants", []),
                        thread_subject=msg.subject,
                        confidence=conf,
                        evidence=[ev],
                    )
                    ent.id = ent.compute_id()
                elif etype == "Issue":
                    ent = Issue(
                        external_id=e_raw.get("external_id", "BZ-unknown"),
                        tracker_url=e_raw.get("tracker_url"),
                        title=e_raw.get("name"),
                        status=e_raw.get("status"),
                        confidence=conf,
                        evidence=[ev],
                    )
                    ent.id = ent.compute_id()
                elif etype == "Component":
                    ent = Component(
                        name=e_raw.get("name", "unknown"),
                        confidence=conf,
                        evidence=[ev],
                    )
                    ent.id = ent.compute_id()
                elif etype == "Project":
                    ent = Project(
                        name=e_raw.get("name", "unknown"),
                        confidence=conf,
                        evidence=[ev],
                    )
                    ent.id = ent.compute_id()
                else:  # Concept / fallback
                    ent = Concept(
                        label=e_raw.get("label", e_raw.get("name", "unknown")),
                        definition=e_raw.get("definition"),
                        confidence=conf,
                        evidence=[ev],
                    )
                    ent.id = ent.compute_id()

                entities.append(ent)
                name_key = e_raw.get("name", getattr(ent, "label", ""))
                entity_name_map[name_key.lower()] = ent

            except (ValidationError, KeyError, ValueError) as ex:
                # Skip malformed entity, don't fail entire extraction
                print(f"[extractor] skip entity {etype}: {ex}", file=sys.stderr)
                continue

        claims = []
        for c_raw in raw.get("claims", []):
            excerpt = c_raw.get("excerpt", "")[:600]
            cs = int(c_raw.get("char_start", 0))
            ce = int(c_raw.get("char_end", cs + len(excerpt)))
            ev = _make_evidence(msg, excerpt, cs, ce, self.extraction_version)
            conf = float(c_raw.get("confidence", 0.7))

            # Resolve entity ids by name
            subj_name = c_raw.get("subject_name", "").lower()
            obj_name = c_raw.get("object_name", "").lower()
            subj_ent = entity_name_map.get(subj_name)
            obj_ent = entity_name_map.get(obj_name)

            try:
                claim_type = ClaimType(c_raw.get("type", "MENTIONS"))
            except ValueError:
                claim_type = ClaimType.MENTIONS

            claim = Claim(
                type=claim_type,
                subject_id=subj_ent.id if subj_ent else f"unresolved:{subj_name}",
                object_id=obj_ent.id if obj_ent else f"unresolved:{obj_name}",
                text=c_raw.get("text", ""),
                confidence=conf,
                valid_from=msg.timestamp,
                evidence=[ev],
            )
            claim.id = claim.stable_id()
            claims.append(claim)

        return ExtractionResult(
            message_id=msg.message_id,
            source_id=msg.source_id,
            extraction_version=self.extraction_version,
            entities=entities,
            claims=claims,
        )
