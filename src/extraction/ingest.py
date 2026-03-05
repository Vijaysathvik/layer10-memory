"""
ingest.py — Parse MBOX corpus, normalize messages, deduplicate artifacts.

Dedup strategy:
  1. Exact:   SHA-256 of normalized body → skip if seen
  2. Near:    4-gram SimHash, Hamming ≤ 3 → mark duplicate_of = earliest seen
  3. Quotes:  strip >-quoted lines before fingerprinting; linked via QUOTES edge
"""

from __future__ import annotations

import email
import hashlib
import mailbox
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

# Graceful fallback if simhash not installed
try:
    from simhash import Simhash
    HAS_SIMHASH = True
except ImportError:
    HAS_SIMHASH = False
    print("[ingest] simhash not available — near-dedup disabled", file=sys.stderr)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedMessage:
    source_id: str          # SHA-256 of raw bytes
    message_id: str         # RFC 2822 Message-ID
    thread_id: str          # In-Reply-To chain root
    from_addr: str
    from_name: str
    subject: str
    body_raw: str           # full body including quotes
    body_clean: str         # quotes stripped
    timestamp: datetime
    references: list[str] = field(default_factory=list)
    quoted_ids: list[str] = field(default_factory=list)  # message-ids quoted
    simhash: Optional[int] = None
    duplicate_of: Optional[str] = None  # source_id of canonical copy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUOTE_LINE_RE = re.compile(r"^>+\s?", re.MULTILINE)
SIG_RE = re.compile(r"\n-- ?\n.*$", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_body(raw: str) -> str:
    """Strip signatures, collapse whitespace."""
    body = SIG_RE.sub("", raw)
    body = body.strip()
    return body


def _strip_quotes(body: str) -> str:
    """Remove >-quoted lines, return clean body."""
    lines = body.splitlines()
    clean = [l for l in lines if not QUOTE_LINE_RE.match(l)]
    return "\n".join(clean).strip()


def _extract_quoted_ids(raw: str) -> list[str]:
    """
    Heuristic: quoted blocks often start with 'On <date>, <person> wrote:'
    followed by >-lines. We also look for embedded Message-ID headers in quotes.
    """
    ids = []
    for m in re.finditer(r"<([^>]+@[^>]+)>", raw):
        candidate = m.group(1)
        if "@" in candidate and "." in candidate:
            ids.append(f"<{candidate}>")
    return list(dict.fromkeys(ids))  # deduplicate, preserve order


def _parse_date(msg: email.message.Message) -> datetime:
    date_str = msg.get("Date", "")
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return datetime.utcnow()


def _get_text_body(msg: email.message.Message) -> str:
    """Extract plain-text body, handling multipart."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    )
                except Exception:
                    return ""
        return ""
    try:
        payload = msg.get_payload(decode=True)
        if payload is None:
            return str(msg.get_payload())
        charset = msg.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    except Exception:
        return ""


def _addr_parts(msg: email.message.Message) -> tuple[str, str]:
    from email.utils import parseaddr
    name, addr = parseaddr(msg.get("From", ""))
    return (name.strip(), addr.strip().lower())


def _message_id(msg: email.message.Message) -> str:
    mid = msg.get("Message-ID", "").strip()
    return mid if mid else f"<unknown-{hashlib.md5(str(msg).encode()).hexdigest()[:8]}>"


def _thread_root(msg: email.message.Message, message_id: str) -> str:
    """Use References header to find thread root; fall back to own id."""
    refs = msg.get("References", "").split()
    if refs:
        return refs[0].strip()
    in_reply = msg.get("In-Reply-To", "").strip()
    return in_reply if in_reply else message_id


# ---------------------------------------------------------------------------
# SimHash helpers
# ---------------------------------------------------------------------------

def _compute_simhash(text: str) -> Optional[int]:
    if not HAS_SIMHASH or not text.strip():
        return None
    tokens = WHITESPACE_RE.split(text.lower())
    ngrams = [" ".join(tokens[i:i+4]) for i in range(max(1, len(tokens)-3))]
    return Simhash(ngrams).value


def _hamming(a: int, b: int) -> int:
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


NEAR_DEDUP_THRESHOLD = 3   # Hamming distance


# ---------------------------------------------------------------------------
# Main ingestion class
# ---------------------------------------------------------------------------

class MboxIngester:
    def __init__(self, mbox_path: str | Path):
        self.mbox_path = Path(mbox_path)
        self._seen_exact: dict[str, str] = {}       # sha256 → source_id
        self._seen_simhash: list[tuple[int, str]] = []  # [(simhash, source_id)]

    def ingest(self) -> Iterator[ParsedMessage]:
        """
        Yield ParsedMessage objects; duplicates are yielded with duplicate_of set
        (so we can build QUOTES edges) but callers should skip re-extraction.
        """
        mbox = mailbox.mbox(str(self.mbox_path))
        for raw_msg in mbox:
            parsed = self._parse_one(raw_msg)
            if parsed is not None:
                yield parsed

    def _parse_one(self, msg: email.message.Message) -> Optional[ParsedMessage]:
        body_raw = _get_text_body(msg)
        body_norm = _normalize_body(body_raw)
        body_clean = _strip_quotes(body_norm)

        # --- exact dedup ---
        sha = hashlib.sha256(body_norm.encode()).hexdigest()
        if sha in self._seen_exact:
            # Return with duplicate_of so we can record the edge
            mid = _message_id(msg)
            name, addr = _addr_parts(msg)
            return ParsedMessage(
                source_id=sha,
                message_id=mid,
                thread_id=_thread_root(msg, mid),
                from_addr=addr,
                from_name=name,
                subject=msg.get("Subject", "").strip(),
                body_raw=body_raw,
                body_clean=body_clean,
                timestamp=_parse_date(msg),
                references=msg.get("References", "").split(),
                quoted_ids=_extract_quoted_ids(body_raw),
                simhash=_compute_simhash(body_clean),
                duplicate_of=self._seen_exact[sha],
            )
        self._seen_exact[sha] = sha

        # --- near dedup (SimHash) ---
        sh = _compute_simhash(body_clean)
        duplicate_of = None
        if sh is not None:
            for prev_sh, prev_id in self._seen_simhash:
                if _hamming(sh, prev_sh) <= NEAR_DEDUP_THRESHOLD:
                    duplicate_of = prev_id
                    break
            if duplicate_of is None:
                self._seen_simhash.append((sh, sha))

        mid = _message_id(msg)
        name, addr = _addr_parts(msg)
        return ParsedMessage(
            source_id=sha,
            message_id=mid,
            thread_id=_thread_root(msg, mid),
            from_addr=addr,
            from_name=name,
            subject=msg.get("Subject", "").strip(),
            body_raw=body_raw,
            body_clean=body_clean,
            timestamp=_parse_date(msg),
            references=msg.get("References", "").split(),
            quoted_ids=_extract_quoted_ids(body_raw),
            simhash=sh,
            duplicate_of=duplicate_of,
        )


# ---------------------------------------------------------------------------
# Sample corpus generator (for demo / tests)
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    {
        "Message-ID": "<001@httpd.apache.org>",
        "From": "Justin Erenkrantz <jerenkrantz@apache.org>",
        "Date": "Mon, 14 Jan 2003 10:22:00 -0800",
        "Subject": "[VOTE] Remove SSLv3 support from mod_ssl",
        "Body": (
            "I propose we remove SSLv3 support from mod_ssl entirely.\n"
            "It is obsolete and CVE-2014-3566 (POODLE) demonstrates why.\n"
            "Casting my vote: +1\n\n-- \nJustin Erenkrantz\njustine@apache.org"
        ),
    },
    {
        "Message-ID": "<002@httpd.apache.org>",
        "From": "Greg Stein <gstein@apache.org>",
        "Date": "Mon, 14 Jan 2003 12:05:00 -0800",
        "Subject": "Re: [VOTE] Remove SSLv3 support from mod_ssl",
        "In-Reply-To": "<001@httpd.apache.org>",
        "Body": (
            "On Mon, Jan 14, Justin Erenkrantz wrote:\n"
            "> I propose we remove SSLv3 support from mod_ssl entirely.\n"
            "> +1\n\n"
            "+1 from me as well. Long overdue.\n\n-- \nGreg Stein"
        ),
    },
    {
        "Message-ID": "<003@httpd.apache.org>",
        "From": "William Rowe <wrowe@apache.org>",
        "Date": "Mon, 14 Jan 2003 14:30:00 -0800",
        "Subject": "Re: [VOTE] Remove SSLv3 support from mod_ssl",
        "In-Reply-To": "<001@httpd.apache.org>",
        "Body": (
            "-1 for now. Some enterprise customers still require SSLv3.\n"
            "We should deprecate first with a 2-release warning, not remove.\n\n"
            "-- \nWilliam Rowe"
        ),
    },
    {
        "Message-ID": "<004@httpd.apache.org>",
        "From": "Justin Erenkrantz <jerenkrantz@apache.org>",
        "Date": "Tue, 15 Jan 2003 09:00:00 -0800",
        "Subject": "Re: [VOTE] Remove SSLv3 support from mod_ssl",
        "In-Reply-To": "<003@httpd.apache.org>",
        "Body": (
            "William raises a fair point. Let's amend: deprecate in 2.4, remove in 2.6.\n"
            "RESOLVED: deprecate SSLv3 in 2.4, remove in 2.6. Vote: +2, -0, abstain 0.\n\n"
            "-- \nJustin Erenkrantz"
        ),
    },
    {
        "Message-ID": "<005@httpd.apache.org>",
        "From": "wrowe@apache.org",
        "Date": "Wed, 16 Jan 2003 08:45:00 -0800",
        "Subject": "[BUG] BZ-12301 assigned to Justin",
        "Body": (
            "Bug BZ-12301 (mod_ssl segfault on renegotiation) has been assigned to Justin.\n"
            "See https://bz.apache.org/bugzilla/show_bug.cgi?id=12301\n\n"
            "-- \nBill"
        ),
    },
    {
        "Message-ID": "<006@httpd.apache.org>",
        "From": "Justin Erenkrantz <justin@erenkrantz.com>",
        "Date": "Thu, 17 Jan 2003 11:00:00 -0800",
        "Subject": "Re: [BUG] BZ-12301 assigned to Justin",
        "In-Reply-To": "<005@httpd.apache.org>",
        "Body": (
            "Confirmed. I'll look at BZ-12301 this week.\n"
            "Also related: BZ-12288 (same renegotiation path).\n\n"
            "-- \nJustin"
        ),
    },
    {
        "Message-ID": "<007@httpd.apache.org>",
        "From": "Greg Stein <gstein@google.com>",
        "Date": "Fri, 18 Jan 2003 15:30:00 -0800",
        "Subject": "[VOTE] Worker MPM: raise default ThreadsPerChild to 64",
        "Body": (
            "Current default is 25. Modern hardware easily handles 64.\n"
            "This would affect httpd-worker MPM component.\n"
            "Vote: +1\n\n-- \nGreg Stein (gstein@google.com)"
        ),
    },
    {
        "Message-ID": "<008@httpd.apache.org>",
        "From": "Justin Erenkrantz <jerenkrantz@apache.org>",
        "Date": "Fri, 18 Jan 2003 16:00:00 -0800",
        "Subject": "Re: [VOTE] Worker MPM: raise default ThreadsPerChild to 64",
        "In-Reply-To": "<007@httpd.apache.org>",
        "Body": (
            "+1. Benchmark data supports this. See BZ-12310.\n\n-- \nJustin"
        ),
    },
]


def generate_sample_mbox(output_path: str | Path) -> None:
    """Write SAMPLE_MESSAGES to an mbox file for testing."""
    from email.mime.text import MIMEText
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mbox = mailbox.mbox(str(output_path), create=True)
    mbox.clear()

    for m in SAMPLE_MESSAGES:
        msg = MIMEText(m["Body"])
        msg["Message-ID"] = m["Message-ID"]
        msg["From"] = m["From"]
        msg["Date"] = m["Date"]
        msg["Subject"] = m["Subject"]
        if "In-Reply-To" in m:
            msg["In-Reply-To"] = m["In-Reply-To"]
        mbox.add(msg)

    mbox.flush()
    mbox.close()
    print(f"[ingest] Sample mbox written: {output_path} ({len(SAMPLE_MESSAGES)} messages)")


if __name__ == "__main__":
    generate_sample_mbox("../../data/raw/sample_httpd_dev.mbox")
