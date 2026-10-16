"""Scorer trust tests: damage the reference on purpose in ONE known way and check that the score moves exactly
where it should and nowhere else. If one of these fails, the scorer measures something other than what we claim.

    .venv/bin/python -m pytest -q tests/test_scorer_trust.py      (offline, no API calls; ~1 minute)

Runs on a fixed sample of tune meetings, without the LLM judge (text similarity = word overlap), so it is free
and deterministic.
"""
import copy
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from nevis.data import load_dataset
from nevis.postprocess import snap_evidence
from nevis.score import BLANK_RE, parse_judge_score, score_folder, template_of

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "dataset").is_dir():
    pytest.skip("needs the home-task dataset in ./dataset", allow_module_level=True)
DATA = load_dataset(ROOT / "dataset")
SPLITS = json.loads((ROOT / "splits.json").read_text())
IDS = sorted(SPLITS["tune"], key=int)[:150]            # fixed sample: the first 150 tune meetings
DECLINED = re.compile(r"\bdeclined\b|\bpostponed\b|without a (?:current )?commitment", re.I)


def sentences(s):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", s)


def score(tmp_path, name, make, invalid="meeting"):
    """Write make(meeting) for every sample meeting to tmp_path/name, then score it."""
    root = tmp_path / name
    for mid in IDS:
        (root / mid).mkdir(parents=True)
        p = make(copy.deepcopy(DATA[mid]))
        (root / mid / "tickets.json").write_text(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False))
    overall, pm, pairs, *_ = score_folder(root, data=DATA, meeting_ids=IDS, invalid=invalid, n_boot=0)
    return overall, pm, pairs


def n_ref():
    return sum(len(DATA[m]["reference"]["tickets"]) for m in IDS)


# ---------------------------------------------------------------------------------------------------- sanity
def test_reference_vs_itself_is_perfect(tmp_path):
    o, _, _ = score(tmp_path, "self", lambda m: m["reference"])
    assert o["north_star"] == pytest.approx(1.0) and o["f1"] == 1.0


def test_empty_output_scores_zero(tmp_path):
    o, _, _ = score(tmp_path, "empty", lambda m: {"meetingId": m["meetingId"], "tickets": []})
    assert o["f1"] == 0 and o["empty_meeting_acc"] == 1.0          # empty meetings are still answered correctly


def test_order_does_not_matter_and_each_ticket_finds_its_own(tmp_path):
    order = {}
    def shuffled(m):
        t = m["reference"]["tickets"]; idx = list(range(len(t))); random.Random(1).shuffle(idx)
        order[m["meetingId"]] = idx
        return {"meetingId": m["meetingId"], "tickets": [t[i] for i in idx]}
    o, _, pairs = score(tmp_path, "shuffled", shuffled)
    assert o["north_star"] == pytest.approx(1.0)
    own = sum(order[r.meetingId][r.pred_idx] == r.ref_idx for r in pairs.itertuples())
    assert own == len(pairs) == n_ref()


# ---------------------------------------------------------------------------------------------------- detection
def test_dropping_one_ticket_per_meeting_lowers_recall_exactly(tmp_path):
    def drop_one(m):
        t = m["reference"]["tickets"]
        if t: t.pop(random.Random(int(m["meetingId"])).randrange(len(t)))
        return m["reference"]
    o, _, _ = score(tmp_path, "drop", drop_one)
    nonempty = sum(1 for m in IDS if DATA[m]["reference"]["tickets"])
    assert o["recall"] == pytest.approx((n_ref() - nonempty) / n_ref()) and o["precision"] == 1.0


def test_ticket_from_a_declined_item_counts_as_extra(tmp_path):
    added = []
    def add_declined(m):
        for s in sentences(m["summary"]):
            if DECLINED.search(s) and len(s) <= 300 and not any(s in t["evidence"] or t["evidence"] in s for t in m["reference"]["tickets"]):
                m["reference"]["tickets"].append({"templateId": None, "title": "Follow up on declined item", "text": s,
                                                  "priority": None, "dueDate": None, "assigneeId": None, "subTasks": [], "evidence": s})
                added.append(m["meetingId"]); break
        return m["reference"]
    o, _, _ = score(tmp_path, "declined", add_declined)
    assert added and o["precision"] < 1 and o["recall"] == 1 and o["extra_from_declined"] == len(added)


# ---------------------------------------------------------------------------------------------------- fields
def test_shifted_due_dates_only_hit_date_accuracy(tmp_path):
    def shift(m):
        for t in m["reference"]["tickets"]:
            if t["dueDate"]: t["dueDate"] = (date.fromisoformat(t["dueDate"]) + timedelta(days=1)).isoformat()
        return m["reference"]
    o, _, _ = score(tmp_path, "dates", shift)
    assert o["f1"] == 1 and o["dueDate_ok"] < 0.05 and o["template_ok"] == 1


def test_dropping_templates_is_counted_as_missed_templates(tmp_path):
    def drop(m):
        for t in m["reference"]["tickets"]:
            if t["templateId"]: t["templateId"] = None; t["subTasks"] = []
        return m["reference"]
    o, _, _ = score(tmp_path, "templates", drop)
    n_tmpl = sum(t["templateId"] is not None for m in IDS for t in DATA[m]["reference"]["tickets"])
    assert o["f1"] == 1 and o.get("template_missed", 0) == n_tmpl


def test_wider_evidence_quote_still_pairs(tmp_path):
    def widen(m):
        for t in m["reference"]["tickets"]:
            for s in sentences(m["summary"]):
                if t["evidence"] in s and len(s) <= 300: t["evidence"] = s; break
        return m["reference"]
    o, _, _ = score(tmp_path, "widen", widen)
    assert o["f1"] == 1


def test_shorter_text_lowers_text_quality_only(tmp_path):
    def first_sentence(m):
        for t in m["reference"]["tickets"]: t["text"] = sentences(t["text"])[0]
        return m["reference"]
    o, _, _ = score(tmp_path, "short", first_sentence)
    assert o["f1"] == 1 and o["text_good"] < 1 and o["template_ok"] == 1 and o["dueDate_ok"] == 1


def _form_tickets():
    return sum(1 for m in IDS for t in DATA[m]["reference"]["tickets"]
               if BLANK_RE.search((template_of(t, DATA[m]) or {}).get("task", {}).get("text") or ""))


def test_raw_form_left_unfilled_is_caught(tmp_path):
    # paste the form template unfilled (every blank left) -> blank-slot F1 drops, those texts are not good; nothing else moves
    def unfilled(m):
        for t in m["reference"]["tickets"]:
            tt = (template_of(t, m) or {}).get("task", {}).get("text") or ""
            if BLANK_RE.search(tt): t["text"] = tt
        return m["reference"]
    o, _, pairs = score(tmp_path, "unfilled", unfilled)
    forms = pairs[pairs.blanks_f1.notna()]
    assert _form_tickets() > 0 and len(forms) == _form_tickets()
    assert o["f1"] == 1 and o["template_ok"] == 1 and forms.blanks_f1.mean() < 0.5 and not forms.template_used_ok.mean() > 0.5


def test_template_wording_dropped_is_caught(tmp_path):
    # replace the text of template tickets by the evidence sentence (template wording gone) -> template not used like the reference
    def dropped(m):
        for t in m["reference"]["tickets"]:
            if ((template_of(t, m) or {}).get("task", {}).get("text") or "").strip(): t["text"] = t["evidence"]
        return m["reference"]
    o, _, pairs = score(tmp_path, "dropped", dropped)
    tmpl = pairs[pairs.template_kept.notna()]
    assert len(tmpl) > 0 and tmpl.template_used_ok.mean() < 0.05 and o["f1"] == 1 and o["template_ok"] == 1


# ---------------------------------------------------------------------------------------------------- validity
def test_one_broken_file_only_costs_that_meeting(tmp_path):
    broken = next(m for m in IDS if len(DATA[m]["reference"]["tickets"]) >= 3)
    o, pm, _ = score(tmp_path, "broken", lambda m: "{not json" if m["meetingId"] == broken else m["reference"])
    assert pm.set_index("meetingId").loc[broken, "matched"] == 0 and o["precision"] == 1 and o["valid_rate"] < 1


def _straight_quote(m):
    for t in m["reference"]["tickets"]:
        if "’" in t["evidence"]:
            t["evidence"] = t["evidence"].replace("’", "'"); break
    return m["reference"]


def test_near_miss_evidence_strict_vs_lenient_vs_repaired(tmp_path):
    strict, _, _ = score(tmp_path, "strict", _straight_quote)
    lenient, _, _ = score(tmp_path, "lenient", _straight_quote, invalid="ticket")
    def repaired(m):
        ref = _straight_quote(m)
        for t in ref["tickets"]: t["evidence"] = snap_evidence(t["evidence"], m["summary"])
        return ref
    fixed, _, _ = score(tmp_path, "repaired", repaired)
    assert strict["recall"] < lenient["recall"] < 1 and fixed["north_star"] == pytest.approx(1.0)


def test_invented_amount_and_name_are_counted(tmp_path):
    def invent(m):
        if m["reference"]["tickets"]:
            t = copy.deepcopy(m["reference"]["tickets"][0])
            t["text"] = "Transfer $987,654 to Zorblax Vexington before the review."
            m["reference"]["tickets"].append(t)
        return m["reference"]
    o, _, _ = score(tmp_path, "invent", invent)
    nonempty = sum(1 for m in IDS if DATA[m]["reference"]["tickets"])
    assert o["invented_facts"] >= nonempty and o["invented_names"] >= nonempty


def test_judge_reply_parser():
    nl = chr(10)
    cases = {'{"score": 4, "reason": "x"}': 4, '{"score": 4}' + nl + '{"score": 5}': 4,
             "```json" + nl + '{"score": 3, "reason": "a {b}"}' + nl + "```": 3,
             'Rating: {"score": 5, "reason": "same"} ok {:)}': 5, "score: 2": 2, "no score here": None}
    for reply, expected in cases.items():
        assert parse_judge_score(reply) == expected, reply
