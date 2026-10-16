"""Shared loaders for the Nevis dataset. Used by the pipeline, the scorer and the analysis tools."""
import json
import re
from datetime import date
from pathlib import Path

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
LONG_RE = re.compile(rf"\b({MONTHS}) (\d{{1,2}}), (\d{{4}})\b")


def load_meeting(folder):
    folder = Path(folder)
    out = {
        "meetingId": folder.name,
        "org": json.loads((folder / "org.json").read_text()),
        "templates": json.loads((folder / "templates.json").read_text()),
        "summary": (folder / "summary.md").read_text(),
    }
    ref = folder / "tickets.json"
    out["reference"] = json.loads(ref.read_text()) if ref.exists() else None
    return out


def meeting_order(meeting_id):
    """Sort key for meeting ids: numeric ids in numeric order, then any other ids alphabetically."""
    return (0, int(meeting_id), "") if meeting_id.isdigit() else (1, 0, meeting_id)


def load_dataset(root):
    root = Path(root)
    ids = sorted((p.name for p in root.iterdir() if p.is_dir()), key=meeting_order)
    return {i: load_meeting(root / i) for i in ids}


US_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
NOYEAR_RE = re.compile(rf"\b({MONTHS}) (\d{{1,2}})\b(?!,? \d{{4}})")


def dates_in_text(text, default_year=None):
    """All dates mentioned in text, normalised to YYYY-MM-DD.
    Handles 2026-06-10, 'June 10, 2026', 6/10/2026 and (with default_year) 'June 10'."""
    found = set(ISO_RE.findall(text))
    month_idx = {m: i + 1 for i, m in enumerate(MONTHS.split("|"))}

    def add(y, mo, d):
        try:
            found.add(date(int(y), int(mo), int(d)).isoformat())
        except ValueError:
            pass

    for mon, d, y in LONG_RE.findall(text):
        add(y, month_idx[mon], d)
    for mo, d, y in US_RE.findall(text):
        add(y, mo, d)
    if default_year:
        for mon, d in NOYEAR_RE.findall(text):
            add(default_year, month_idx[mon], d)
    return found


def meeting_date(summary):
    """First date in the summary; summaries open with 'On <date>, ...'."""
    head = summary[:300]
    m = ISO_RE.search(head)
    if m:
        return m.group(1)
    ds = sorted(dates_in_text(head))
    return ds[0] if ds else None


# ---------------------------------------------------------------------------- near-copies
# The dataset contains reworded variants of the same meeting (same firm, same household, same follow-ups; sometimes
# the follow-ups removed): 129 same-firm pairs share > 40% of their summary words, many > 90%. A near-copy of a
# meeting must never be shown to the model as a few-shot example of it, and the scorer reports results with and
# without meetings that have a near-copy in tune.
NEAR_COPY = 0.5          # share of distinct words in common (Jaccard) above which two summaries are near-copies


def summary_overlap(a, b):
    """Jaccard overlap of the two summaries' distinct lower-cased words (0..1)."""
    wa, wb = set(a["summary"].lower().split()), set(b["summary"].lower().split())
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def is_near_copy(a, b):
    return a["org"]["id"] == b["org"]["id"] and summary_overlap(a, b) > NEAR_COPY
