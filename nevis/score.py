"""Scorer: how close is a folder of predicted tickets.json to the reference? Works for ANY model's output.

    python -m nevis.score --pred outputs/<run> --ref <meetings with reference tickets.json> --out scores/<run> [--judge]
                          [--invalid meeting|ticket] [--bootstrap 1000]

INPUT  --pred folder: <meetingId>/tickets.json   (required; the only thing the scorer needs)
                      <meetingId>/meta.json      (optional; latency/tokens/cost, see nevis/runmeta.py)
       --ref  folder: the dataset (summary.md, org.json, templates.json, tickets.json per meeting)
OUTPUT --out folder:  summary.json, report.md, per_meeting.csv, pairs.csv, errors.csv, slices.csv, inventions.csv

What it measures (DESIGN.md (b)):
  step 0   validity of every ticket (schema, ids, verbatim evidence, limits).
           --invalid meeting (default, strict): one bad ticket -> the whole meeting counts as empty
           --invalid ticket: only the bad tickets are dropped
  step 1   follow-up detection: tickets paired 1-to-1 (best pair first) by evidence overlap + text overlap + due date
           -> precision, recall, F1 (micro + per-meeting), empty-meeting accuracy, extras from declined items, count error
  step 2-3 on paired tickets: template outcome (correct / wrong / forced / missed), dueDate, assignee, priority,
           sub-tasks, title (exact for template tickets), text similarity (LLM judge or word overlap),
           key facts kept, [blanks] filled like the reference
           on ALL predicted tickets: invented facts (amounts, %, dates) and invented names (not in summary/org/templates)
  NORTH STAR  field-level F1 per meeting, averaged over meetings (each meeting counts equally).
           Each ticket = 6 items: the task itself + template, dueDate, assignee, priority, sub-tasks (equal weights).
           Same-task gate: with --judge, a pair the judge scores < 3/5 is a DIFFERENT task -> missed + extra (its fields earn nothing).
  ranges   bootstrap 95% ranges over meetings
  slices   every metric per meeting slice and per reference-ticket slice
  run      latency and cost per meeting, if meta.json files are present
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from nevis.data import dates_in_text, is_near_copy, load_dataset, meeting_date, meeting_order
from nevis.runmeta import read_meta

REQUIRED = {"templateId", "title", "text", "priority", "dueDate", "assigneeId", "subTasks", "evidence"}
SUB_REQUIRED = {"title", "text", "priority", "dueDate", "assigneeId"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DECLINED_RE = re.compile(r"\bdeclined\b|\bpostponed\b|\bdeferred\b(?! comp)|without a (?:current )?commitment", re.I)
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
BLANK_RE = re.compile(r"\[[^\]]+\]|_{3,}")   # form blanks: [LIKE THIS] (140 templates) or ______ (25)
MIN_PAIR_SCORE = 0.30          # below this, two tickets are not considered the same follow-up
SAME_TASK_GATE = 3             # judge score (0-5) needed for a pair to count as the SAME task; below -> missed + extra
GOOD_TEXT_JUDGE = 4            # text counts as GOOD (north-star "task" item) only if judge >= 4 ...
TEXT_LEN_RATIO = (0.5, 2.0)    # ... and our text is 0.5x-2x the reference length ...
#                                ... and title + text start with an action verb (the reference's style) ...
#                                ... and, on a template ticket, the template is used like the reference uses it:
TEMPLATE_KEPT_MIN = 0.8        #   we keep >= 80% of the template's wording (or at least as much as the reference kept)
BLANKS_F1_MIN = 0.8            #   form templates: blank slots filled / left like the reference (slot F1 >= 0.8)
NS_FIELDS = ["template_ok", "dueDate_ok", "assignee_ok", "priority_ok", "subtasks_ok"]   # + the task itself = 6 items per ticket
HEADLINE_KEYS = ["north_star", "north_star_by_ticket", "f1", "precision", "recall"]


# ============================================================================ step 0: validity
def ticket_problems(t, meeting):
    """Problems with one predicted ticket; empty list = valid."""
    if not isinstance(t, dict) or set(t) != REQUIRED:
        return [f"keys {sorted(set(t) ^ REQUIRED) if isinstance(t, dict) else type(t).__name__}"]
    p = []
    if t["templateId"] is not None and t["templateId"] not in {x["id"] for x in meeting["templates"]}: p.append("templateId not in firm")
    if not isinstance(t["title"], str) or not t["title"].strip() or len(t["title"]) > 120: p.append("title missing/>120")
    if not isinstance(t["text"], str) or not t["text"].strip(): p.append("text missing")
    if t["priority"] not in (None, "low", "medium", "high"): p.append("priority value")
    if t["dueDate"] is not None and not (isinstance(t["dueDate"], str) and DATE_RE.match(t["dueDate"])): p.append("dueDate format")
    if t["assigneeId"] is not None and t["assigneeId"] not in {m["id"] for m in meeting["org"]["members"]}: p.append("assignee not member")
    if not isinstance(t["evidence"], str) or t["evidence"] not in meeting["summary"]: p.append("evidence not verbatim")
    elif len(t["evidence"]) > 300: p.append("evidence >300")
    if not isinstance(t["subTasks"], list) or any(not isinstance(s, dict) or set(s) != SUB_REQUIRED for s in t["subTasks"]):
        p.append("subTasks shape")
    return p


def validate(pred, meeting):
    """Returns (list of valid tickets, list of problems 'tN: ...', file_ok)."""
    if not isinstance(pred, dict) or not isinstance(pred.get("tickets"), list):
        return [], ["file: not an object with a 'tickets' list"], False
    good, problems = [], []
    for i, t in enumerate(pred["tickets"]):
        pr = ticket_problems(t, meeting)
        problems += [f"t{i}: {x}" for x in pr]
        if not pr:
            good.append(t)
    return good, problems, True


# ============================================================================ text helpers
def words(s):
    return re.findall(r"[a-z0-9$%]+", (s or "").lower())


def word_f1(a, b):
    """Bag-of-words F1 between two strings (cheap, deterministic text similarity)."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    common = sum((Counter(wa) & Counter(wb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(wa), common / len(wb)
    return 2 * p * r / (p + r)


MONEY = re.compile(r"\$\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion))?")
PCT = re.compile(r"\d+(?:\.\d+)?%")
NAME_WORD = re.compile(r"\b[A-Z][a-z]+(?=(?:[’']s)?\b)")
NAME_STOP = {"The", "This", "These", "That", "Use", "Include", "Confirm", "Review", "Prepare", "Address", "Update", "Obtain",
             "Coordinate", "Reconcile", "Resolve", "Compare", "Collect", "Follow", "Provide", "Request", "Check", "Verify",
             "Complete", "Send", "Create", "Document", "Identify", "Assess", "Reflect", "Account", "Consider", "Note",
             "Show", "Keep", "Track", "Build", "Model", "Draft", "Schedule", "Set", "Help", "Gather", "Summarize", "Evaluate",
             "Client", "Household", "Agreed", "Roth", "IRA", "HSA", "FSA", "RMD", "CPA", "Social", "Security", "Medicare",
             "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}


def facts(s):
    """Concrete facts: amounts, percentages, dates (normalised). Any Unicode space counts as a plain space
    (models write e.g. "$16.9\u202fmillion" with a narrow no-break space; the summary has a normal one)."""
    s = re.sub(r"\s", " ", s or "")
    return {x.rstrip(",.") for x in MONEY.findall(s)} | set(PCT.findall(s)) | dates_in_text(s)


def name_words(s):
    """Capitalised words that are not the first word of a sentence and not common task/finance words."""
    out = set()
    for sent in re.split(r"(?<=[.!?:])\s+|\n+", s or ""):
        out |= {w for w in NAME_WORD.findall(sent[1:]) if w not in NAME_STOP}
    return out


def span(summary, ev):
    i = summary.find(ev) if isinstance(ev, str) and ev else -1
    return (i, i + len(ev)) if i >= 0 else None


def span_overlap(a, b):
    """IoU of two character spans in the summary: shared characters / characters covered by either quote.
    (IoU, not 'overlap relative to the shorter quote', so a long quote can't cover a short one for free.
    On our runs both give identical pairings.)"""
    if not a or not b:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


# ============================================================================ step 1: pairing
def pair_score(p, r, summary):
    """How likely two tickets describe the same follow-up (0..1)."""
    ev = span_overlap(span(summary, p.get("evidence")), span(summary, r["evidence"]))
    txt = word_f1(f"{p.get('title', '')} {p.get('text', '')}", f"{r['title']} {r['text']}")
    due = 1.0 if p.get("dueDate") and p.get("dueDate") == r["dueDate"] else 0.0
    return 0.6 * ev + 0.3 * txt + 0.1 * due


def match(pred_tickets, ref_tickets, summary):
    """1-to-1 pairing, best remaining pair first. Returns list of (pred_idx, ref_idx, score)."""
    if not pred_tickets or not ref_tickets:
        return []
    S = np.array([[pair_score(p, r, summary) for r in ref_tickets] for p in pred_tickets])
    order = sorted(((S[i, j], i, j) for i in range(S.shape[0]) for j in range(S.shape[1])), reverse=True)
    used_p, used_r, out = set(), set(), []
    for sc, i, j in order:
        if sc < MIN_PAIR_SCORE:
            break
        if i in used_p or j in used_r:
            continue
        used_p.add(i); used_r.add(j); out.append((i, j, float(sc)))
    return out


# ============================================================================ step 2-3: fields
def template_outcome(p, r):
    pt, rt = p.get("templateId"), r["templateId"]
    if pt == rt:
        return "correct_template" if rt else "correct_freeform"
    if rt is None:
        return "forced"          # we used a template, reference didn't
    if pt is None:
        return "missed"          # reference used a template, we didn't
    return "wrong_template"


def sources_of(meeting):
    return meeting["summary"] + " " + json.dumps(meeting["templates"], ensure_ascii=False) + " " + json.dumps(meeting["org"], ensure_ascii=False)


def inventions(t, meeting, src=None):
    """Facts and names in a predicted ticket's title/text that appear in none of the inputs."""
    src = src or sources_of(meeting)
    src_dates = dates_in_text(src, (meeting_date(meeting["summary"]) or "")[:4] or None)
    text = f"{t.get('title', '')}. {t.get('text', '')}"
    bad_facts = sorted(f for f in facts(text) if f not in src and f not in src_dates)
    # Names: text only. Many titles are Title Case, where capitals don't mean names.
    # A name counts as invented only if it is in no input in any capitalisation, also without a plural "s"
    # (Title-Case titles capitalise ordinary words; "the Thimbleferns" = the Thimblefern family).
    src_l = src.lower()
    bad_names = sorted(n for n in name_words(t.get("text", ""))
                       if n.lower() not in src_l and not (n.endswith("s") and n[:-1].lower() in src_l))
    return bad_facts, bad_names


_VERBS = None


def action_verbs():
    """The verbs the reference starts tickets with (its style): the first word of every reference title and text,
    frozen in nevis/action_verbs.json so the scorer works on any folder of meetings (no copy of our dataset needed)."""
    global _VERBS
    if _VERBS is None:
        _VERBS = set(json.loads((Path(__file__).parent / "action_verbs.json").read_text())["verbs"])
    return _VERBS


def style_ok(t):
    """Title and text both start with an action verb the reference uses (e.g. Prepare, Review, Obtain...)."""
    first = lambda x: (re.match(r"[A-Za-z][A-Za-z-]*", (x or "").strip()) or [""])[0].lower()
    return first(t.get("title")) in action_verbs() and first(t.get("text")) in action_verbs()


# ============================================================================ template use (mechanical)
# A template ticket keeps the template's wording and makes it specific (plain/workflow: median 98% of the template's
# words kept; forms: 69% only fill blanks, the rest also add or edit a little). So the STRUCTURE is checked here,
# in code; whether the added and filled-in CONTENT is right is the judge's job.
def template_of(ticket, meeting):
    tid = ticket.get("templateId")
    return next((t for t in meeting["templates"] if t["id"] == tid), None) if tid else None


def template_kept(template_text, text):
    """Share of the template's fixed words (blanks removed) still in `text`, in the same order (word-level diff)."""
    tw = words(BLANK_RE.sub(" ", template_text or ""))
    if not tw:
        return np.nan
    sm = difflib.SequenceMatcher(None, tw, words(text), autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(tw)


def slot_states(template_text, text):
    """For each blank of a form template: True = filled in `text`, False = left as the blank, None = can't tell.
    The fixed wording in front of each blank (its label, e.g. "Phone numbers:") is looked up in `text`, in order;
    what follows the label is either the blank itself (left) or anything else (filled). A blank kept with a note
    added after it on the same line counts as filled (the reference does this 17 times)."""
    t, x = (template_text or "").translate(_QUOTES), (text or "").translate(_QUOTES)
    states, pos, prev = [], 0, 0
    for b in BLANK_RE.finditer(t):
        before = t[prev:b.start()].strip()
        label = before.splitlines()[-1].strip()[-40:] if before else ""
        prev = b.end()
        i = x.find(label, pos) if len(label) >= 3 else -1
        if i < 0:
            states.append(None)
            continue
        pos = i + len(label)
        rest = x[pos:].lstrip()
        tok = b.group(0)
        m = re.match(re.escape(tok) if not tok.startswith("___") else r"_{3,}", rest)
        # left = the blank is still there with nothing added after it on that line ("[NAME] (known: ...)" = filled)
        states.append(not (m and not rest[m.end():].split("\n", 1)[0].strip(" .,;:")))
    return states


def blank_counts(template_text, ref_text, pred_text):
    """Blank slots, 'filled' = positive: (hit = both filled, missed = the reference filled it but we left it,
    extra = we filled it but the reference left it). Slots we can't locate in either text are skipped."""
    pairs = [(a, b) for a, b in zip(slot_states(template_text, ref_text), slot_states(template_text, pred_text))
             if a is not None and b is not None]
    return (sum(a and b for a, b in pairs), sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs))


def blanks_f1(template_text, ref_text, pred_text):
    """F1 over the template's blank slots (see blank_counts); 1.0 when neither text fills any."""
    tp, fn, fp = blank_counts(template_text, ref_text, pred_text)
    return 1.0 if tp + fn + fp == 0 else 2 * tp / (2 * tp + fn + fp)


def template_checks(template, p, r):
    """Is the template used like the reference uses it? Only for pairs with the same template (else the template
    point is lost already). Returns template_kept (our share), blanks_f1 and template_used_ok."""
    out = dict(template_kept=np.nan, blanks_f1=np.nan, blank_hit=np.nan, blank_missed=np.nan, blank_extra=np.nan,
               template_used_ok=True)
    t_text = ((template or {}).get("task", {}).get("text") or "").strip()
    if not t_text:                      # freeform or placeholder: nothing to keep
        return out
    ours, ref = template_kept(t_text, p.get("text")), template_kept(t_text, r["text"])
    out["template_kept"] = ours
    ok = ours >= min(TEMPLATE_KEPT_MIN, ref) - 1e-9
    if BLANK_RE.search(t_text):
        tp, fn, fp = blank_counts(t_text, r["text"], p.get("text"))
        out.update(blank_hit=tp, blank_missed=fn, blank_extra=fp,
                   blanks_f1=1.0 if tp + fn + fp == 0 else 2 * tp / (2 * tp + fn + fp))
        ok = ok and out["blanks_f1"] >= BLANKS_F1_MIN
    out["template_used_ok"] = bool(ok)
    return out


def compare_pair(p, r, meeting, judge=None):
    p_subs, r_subs = p.get("subTasks") or [], r["subTasks"]
    out = dict(
        template_outcome=template_outcome(p, r),
        template_ok=p.get("templateId") == r["templateId"],
        dueDate_ok=p.get("dueDate") == r["dueDate"],
        assignee_ok=p.get("assigneeId") == r["assigneeId"],
        priority_ok=p.get("priority") == r["priority"],
        subtasks_ok=(len(p_subs) == len(r_subs)
                     and [s.get("title") for s in p_subs] == [s["title"] for s in r_subs]
                     and [s.get("assigneeId") for s in p_subs] == [s["assigneeId"] for s in r_subs]),
    )
    template = template_of(r, meeting) if p.get("templateId") == r["templateId"] else None
    out.update(template_checks(template, p, r))
    out["text_len_ratio"] = len(p.get("text") or "") / max(1, len(r["text"]))
    out["text_size_ok"] = TEXT_LEN_RATIO[0] <= out["text_len_ratio"] <= TEXT_LEN_RATIO[1]
    out["text_style_ok"] = style_ok(p)
    if judge is not None:
        out["judge_0_5"] = judge(p, r, meeting["summary"], template_of(r, meeting))
        meaning_ok = out["judge_0_5"] >= GOOD_TEXT_JUDGE
    else:   # no judge (offline tests only): word overlap as a rough stand-in for "same meaning"
        overlap = (word_f1(p.get("title"), r["title"]) + word_f1(p.get("text"), r["text"])) / 2
        meaning_ok = overlap >= 0.6
    out["text_good"] = bool(meaning_ok and out["text_size_ok"] and out["text_style_ok"] and out["template_used_ok"])
    return out


# ============================================================================ LLM judge (optional)
JUDGE_PROMPT = """You compare two task tickets created from the same client-meeting notes.
REFERENCE is the expected ticket; CANDIDATE is the one to grade. SUMMARY_EXCERPT is the part of the meeting notes
both tickets come from: use it to check facts.
TEMPLATE_TEXT, when given, is the firm template both tickets start from. Both are expected to keep its wording,
fill its blanks and add this meeting's specifics. Keeping the template wording and filling blanks are checked
separately: judge the CONTENT, i.e. what each ticket adds or fills in (people, institutions, accounts, amounts,
dates, actions). SUBTASKS, when given, are part of the ticket: include their text in your judgement.
Score 0-5 how well CANDIDATE matches REFERENCE in meaning:
5 = same task; contains the reference's key facts (people, institutions, amounts, dates); wording may differ
4 = same task; one minor fact missing
3 = same task; several key facts missing
2 = related task, but noticeably different scope
1 = barely related
0 = different task
Rules: extra details in CANDIDATE that the SUMMARY_EXCERPT supports are FINE (do not lower the score for them).
A detail that CONTRADICTS the excerpt (wrong person, wrong institution, wrong amount) counts as a wrong key fact:
score at most 3. Ignore style, length and the due-date field.
Return only JSON: {"score": <0-5 integer>, "reason": "<one short sentence>"}"""


def judge_payload(p, r, summary, template=None):
    """What the judge sees: both tickets (title, text, sub-tasks if any), the summary around both evidence quotes,
    and the template text when the reference uses a template."""
    def view(t):
        v = {"title": t.get("title"), "text": t.get("text")}
        if t.get("subTasks"):
            v["subtasks"] = [{"title": x.get("title"), "text": x.get("text")} for x in t["subTasks"]]
        return v
    body = {"reference": view(r), "candidate": view(p),
            "summary_excerpt": summary_excerpt(summary, r.get("evidence"), p.get("evidence"))}
    t_text = ((template or {}).get("task", {}).get("text") or "").strip()
    if t_text:
        body["template_text"] = t_text
    return json.dumps(body, ensure_ascii=False, sort_keys=True)


def summary_excerpt(summary, *quotes, pad=700):
    """The part of the summary around the evidence quotes (both tickets), padded, max ~2,500 characters."""
    spans = [(i, i + len(q)) for q in quotes if q and (i := summary.find(q)) >= 0]
    if not spans:
        return summary[:2500]
    lo, hi = max(0, min(a for a, _ in spans) - pad), min(len(summary), max(b for _, b in spans) + pad)
    return summary[lo:hi][:2500]


def parse_judge_score(text):
    """Read the 0-5 score from the judge's reply. Tolerates code fences, text around the JSON,
    or several JSON objects (the first one with a 'score' wins). Returns None if no score is found."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "score" in obj:
            try:
                return max(0, min(5, int(obj["score"])))
            except (TypeError, ValueError):
                continue
    m = re.search(r'"?score"?\s*[:=]\s*([0-5])\b', text)
    return int(m.group(1)) if m else None


JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"     # open-weight judge on Fireworks (FIREWORKS_API_KEY)
_JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["score", "reason"],
                 "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 5}, "reason": {"type": "string"}}}


def fireworks_judge_call(model, api_key=None, reasoning_effort="low", max_tokens=2000):
    """call(system, user) -> (reply text, input tokens, output tokens), via Fireworks' OpenAI-compatible API.
    The client is created on the first call, so re-scoring a fully cached run needs no key."""
    state = {"client": None}

    def call(system, user):
        if state["client"] is None:
            import openai
            key = api_key or os.environ.get("FIREWORKS_API_KEY")
            if not key:
                raise SystemExit("FIREWORKS_API_KEY is not set (needed for --judge)")
            state["client"] = openai.OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1", timeout=120)
        r = state["client"].chat.completions.create(
            model=model, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {"name": "judgement", "schema": _JUDGE_SCHEMA}},
            extra_body={"reasoning_effort": reasoning_effort} if reasoning_effort else None)
        return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens

    return call


def make_judge(cache_dir, model=JUDGE_MODEL, call=None, retries=3):
    """judge(pred ticket, ref ticket, summary, template) -> 0-5, with an on-disk cache keyed on (model, prompt, input).
    `call(system, user) -> (text, in_tokens, out_tokens)` talks to the model; default = Fireworks."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    call = call or fireworks_judge_call(model)
    stats = {"model": model, "calls": 0, "cached": 0, "unparsed": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
    prompt = JUDGE_PROMPT

    def judge(p, r, summary, template=None):
        payload = judge_payload(p, r, summary, template)
        key = hashlib.sha256((model + prompt + payload).encode()).hexdigest()[:24]
        f = cache_dir / f"{key}.json"
        if f.exists():
            stats["cached"] += 1
            cached = json.loads(f.read_text())
            score = cached.get("score")
            if score is None:
                score = parse_judge_score(cached.get("raw", ""))
            return score if score is not None else 0
        for attempt in range(retries + 1):
            try:
                t0 = time.time()
                text, n_in, n_out = call(prompt, payload)
                break
            except SystemExit:
                raise
            except Exception:                       # network / rate limit -> retry
                if attempt == retries:
                    raise
                time.sleep(min(2 ** attempt, 30))
        stats["seconds"] += time.time() - t0
        stats["calls"] += 1
        stats["input_tokens"] += n_in
        stats["output_tokens"] += n_out
        score = parse_judge_score(text)
        f.write_text(json.dumps({"score": score, "raw": text}))
        if score is None:
            stats["unparsed"] += 1
            return 0
        return score

    judge.stats = stats
    return judge


# ============================================================================ slices
def ticket_kind(t, meeting):
    if t["templateId"] is None:
        return "freeform"
    tm = next((x for x in meeting["templates"] if x["id"] == t["templateId"]), None)
    if tm is None:
        return "unknown"
    if tm.get("subTasks"):
        return "workflow"
    if not (tm["task"]["text"] or "").strip():
        return "placeholder"
    return "form" if BLANK_RE.search(tm["task"]["text"]) else "plain"


def evidence_wording(ev):
    if re.search(r"agreed follow-up", ev, re.I): return "'agreed follow-up'"
    if re.search(r"\bagreed\b", ev, re.I): return "other 'agreed'"
    if re.search(r"\bneeded\b", ev, re.I): return "'needed (by)'"
    if re.search(r"\basked\b|\brequested\b", ev, re.I): return "'asked/requested'"
    return "other"


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def due_date_format(t, summary):
    """How the reference ticket's due date is written in the summary."""
    d = t["dueDate"]
    if not d:
        return "no due date"
    if d in summary:
        return "ISO 2024-04-24"
    y, mth, day = d.split("-")
    mname = MONTHS.split("|")[int(mth) - 1]
    if re.search(rf"\b{mname} {int(day)}, {y}\b", summary): return "long 'June 10, 2026'"
    if re.search(rf"\b{int(mth)}/{int(day)}/{y}\b", summary): return "US 11/18/2024"
    if re.search(rf"\b{mname} {int(day)}\b", summary): return "no year 'June 14'"
    return "other"


def collision_index(data):
    """Member full names per firm, to flag meetings where another firm's member name (or a member's first name
    with a different surname) appears in the summary (person names are reused across firms in this dataset)."""
    all_names = {m["name"] for d in data.values() for m in d["org"]["members"]}

    def flag(meeting):
        own = {m["name"] for m in meeting["org"]["members"]}
        s = meeting["summary"]
        if any(n in s for n in all_names - own):
            return "other firm's member name"
        for n in own:
            first, last = n.split()[0], n.split()[-1]
            if any(l != last for l in re.findall(rf"\b{re.escape(first)} ([A-Z][a-z]+)\b", s)):
                return "same first name as member"
        return "none"
    return flag


def meeting_slices(meeting, split_info, collision_flag, tune_twin=False):
    words_n = len(meeting["summary"].split())
    n = len(meeting["reference"]["tickets"])
    n_members = len(meeting["org"]["members"])
    n_tmpl = len(meeting["templates"])
    return dict(
        length=("short (<400 words)" if words_n < 400 else "medium" if words_n <= 900 else "long (>900)"),
        ref_tickets=("0" if n == 0 else "1-3" if n <= 3 else "4-7" if n <= 7 else "8+"),
        empty_meeting=(n == 0),
        declined_in_summary=bool(DECLINED_RE.search(meeting["summary"])),
        firm=split_info.get(meeting["meetingId"], "dev / not in hold-out" if split_info else "n/a (not our dataset)"),
        firm_size=("1 member" if n_members == 1 else "2-3 members" if n_members <= 3 else "4+ members"),
        firm_templates=("4-6" if n_tmpl <= 6 else "7-9" if n_tmpl <= 9 else "10-12"),
        name_collision=collision_flag(meeting),
        twin_in_tune=bool(tune_twin),          # a reworded variant of this meeting is in the tune split
    )


# ============================================================================ per-meeting scoring
def score_meeting(meeting, pred, judge=None, invalid="meeting"):
    summary, ref = meeting["summary"], meeting["reference"]["tickets"]
    if pred is None:
        good, problems, file_ok = [], ["file: missing"], False
    else:
        good, problems, file_ok = validate(pred, meeting)
    n_given = len(pred["tickets"]) if file_ok else 0
    fully_valid = file_ok and not problems
    if invalid == "meeting":
        pred_t = good if fully_valid else []
    else:
        pred_t = good
    src = sources_of(meeting)

    candidates = match(pred_t, ref, summary)
    pair_rows, errors, inv_rows = [], [], []
    for i, j, s in candidates:
        row = dict(pred_idx=i, ref_idx=j, pair_score=s, ref_kind=ticket_kind(ref[j], meeting),
                   ref_wording=evidence_wording(ref[j]["evidence"]), ref_date_format=due_date_format(ref[j], summary),
                   **compare_pair(pred_t[i], ref[j], meeting, judge))
        # Same-task gate: if the judge says it's a different task, the other fields don't matter -> missed + extra
        if judge is not None and row["judge_0_5"] < SAME_TASK_GATE:
            errors.append(dict(type="different_task", detail=f"judge {row['judge_0_5']}/5 -> counted as missed + extra",
                               ref_title=ref[j]["title"], pred_title=pred_t[i]["title"], evidence=ref[j]["evidence"][:200]))
            continue
        pair_rows.append(row)
    pairs = [(r["pred_idx"], r["ref_idx"], r["pair_score"]) for r in pair_rows]
    paired_p, paired_r = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
    for row in pair_rows:
        i, j = row["pred_idx"], row["ref_idx"]
        wrong = [f for f in NS_FIELDS if not row[f]]
        if wrong:
            errors.append(dict(type="wrong_fields", detail=",".join(wrong) + f" | template: {row['template_outcome']}",
                               ref_title=ref[j]["title"], pred_title=pred_t[i]["title"], evidence=ref[j]["evidence"][:200]))
    for j, r in enumerate(ref):
        if j not in paired_r:
            errors.append(dict(type="missed", detail=f"{ticket_kind(r, meeting)} | {evidence_wording(r['evidence'])}",
                               ref_title=r["title"], pred_title="", evidence=r["evidence"][:200]))
    for i, p in enumerate(pred_t):
        if i not in paired_p:
            errors.append(dict(type="extra", detail="declined/postponed item" if DECLINED_RE.search(p.get("evidence", "")) else "",
                               ref_title="", pred_title=p.get("title", ""), evidence=(p.get("evidence") or "")[:200]))
    for i, p in enumerate(pred_t):                    # inventions are checked on ALL predicted tickets
        bf, bn = inventions(p, meeting, src)
        if bf or bn:
            inv_rows.append(dict(pred_idx=i, title=p.get("title", ""), invented_facts=", ".join(bf), invented_names=", ".join(bn)))
    for prob in problems:
        errors.append(dict(type="invalid", detail=prob, ref_title="", pred_title="", evidence=""))

    # North star for this meeting: field-level F1. Each ticket = 6 items (the task text + 5 fields).
    # A paired (same-task) ticket earns 1 for GOOD text (judge >= 4, similar size, same style) + 1 per correct field;
    # missed/extra tickets earn nothing.
    n_items = 1 + len(NS_FIELDS)
    correct_items = sum(int(r["text_good"]) + sum(bool(r[f]) for f in NS_FIELDS) for r in pair_rows)
    if not ref and not pred_t:
        ns = 1.0
    elif not ref or not pred_t:
        ns = 0.0
    else:
        ns_p, ns_r = correct_items / (len(pred_t) * n_items), correct_items / (len(ref) * n_items)
        ns = 2 * ns_p * ns_r / (ns_p + ns_r) if ns_p + ns_r else 0.0

    ref_rows = [dict(kind=ticket_kind(r, meeting), wording=evidence_wording(r["evidence"]),
                     date_format=due_date_format(r, summary), found=j in paired_r) for j, r in enumerate(ref)]
    m_row = dict(file_ok=file_ok, valid=fully_valid, n_given=n_given, n_invalid_tickets=n_given - len(good) if file_ok else 0,
                 problems="; ".join(problems), n_pred=len(pred_t), n_ref=len(ref), matched=len(pairs),
                 north_star=ns, correct_items=correct_items, different_task=sum(1 for e in errors if e["type"] == "different_task"),
                 extra_declined=sum(1 for e in errors if e["type"] == "extra" and e["detail"]),
                 tickets_with_invention=len(inv_rows),
                 invented_facts=sum(len(x["invented_facts"].split(", ")) for x in inv_rows if x["invented_facts"]),
                 invented_names=sum(len(x["invented_names"].split(", ")) for x in inv_rows if x["invented_names"]))
    return m_row, pair_rows, errors, inv_rows, ref_rows


# ============================================================================ aggregation
def prf(matched, n_pred, n_ref):
    p = matched / n_pred if n_pred else (1.0 if n_ref == 0 else 0.0)
    r = matched / n_ref if n_ref else (1.0 if n_pred == 0 else 0.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def detection_of(pm):
    """Ticket-level detection over all meetings: did we find the right follow-ups (fields ignored)?"""
    P, R, F1 = prf(pm.matched.sum(), pm.n_pred.sum(), pm.n_ref.sum())
    return dict(f1=F1, precision=P, recall=R)


def north_star_pooled(pm, n_items=6):
    """Same field-level F1, but every TICKET counts equally: pool correct items over all meetings, then one F1.
    (The north star averages per meeting instead, so every MEETING counts equally.)"""
    pred_items, ref_items = pm.n_pred.sum() * n_items, pm.n_ref.sum() * n_items
    if pred_items == 0 or ref_items == 0:
        return float(pred_items == ref_items)
    p, r = pm.correct_items.sum() / pred_items, pm.correct_items.sum() / ref_items
    return float(2 * p * r / (p + r)) if p + r else 0.0


def summarise(pm, pairs):
    """Aggregate metrics for a set of meetings (pm) and their paired tickets."""
    out = dict(meetings=len(pm), ref_tickets=int(pm.n_ref.sum()), pred_tickets=int(pm.n_pred.sum()),
               north_star=float(pm.north_star.mean()) if len(pm) else np.nan,
               north_star_by_ticket=north_star_pooled(pm),
               meetings_without_miss=float((pm.matched == pm.n_ref).mean()) if len(pm) else np.nan,
               north_star_worst10pct=float(pm.north_star.quantile(0.10)) if len(pm) else np.nan,
               different_task_pairs=int(pm.different_task.sum()))
    out.update(detection_of(pm))
    out.update(
        valid_rate=float(pm.valid.mean()) if len(pm) else np.nan,
        invalid_tickets=int(pm.n_invalid_tickets.sum()),
        f1_macro=float(np.mean([prf(a, b, c)[2] for a, b, c in zip(pm.matched, pm.n_pred, pm.n_ref)])) if len(pm) else np.nan,
        empty_meeting_acc=float((pm[pm.n_ref == 0].n_pred == 0).mean()) if (pm.n_ref == 0).any() else np.nan,
        extra_from_declined=int(pm.extra_declined.sum()),
        count_abs_error=float((pm.n_pred - pm.n_ref).abs().mean()) if len(pm) else np.nan,
        tickets_with_invention=int(pm.tickets_with_invention.sum()),
        invented_facts=int(pm.invented_facts.sum()),
        invented_names=int(pm.invented_names.sum()),
    )
    if len(pairs):
        for f in ["template_ok", "dueDate_ok", "assignee_ok", "priority_ok", "subtasks_ok", "text_good",
                  "text_size_ok", "text_style_ok", "template_used_ok", "template_kept", "blanks_f1"] + (["judge_0_5"] if "judge_0_5" in pairs else []):
            out[f] = float(pairs[f].astype(float).mean())
        for f in ["blank_hit", "blank_missed", "blank_extra"]:     # form blanks: which way do we go wrong?
            out[f] = int(pairs[f].sum())
        for k, v in pairs.template_outcome.value_counts().items():
            out[f"template_{k}"] = int(v)
    return out


def bootstrap(pm, n_boot=1000, seed=0):
    """95% ranges for the main numbers, resampling MEETINGS with replacement."""
    if n_boot <= 0 or len(pm) < 2:
        return {}
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in HEADLINE_KEYS}
    for _ in range(n_boot):
        s = pm.iloc[rng.choice(len(pm), len(pm), replace=True)]
        h = detection_of(s)
        h["north_star"] = float(s.north_star.mean())
        h["north_star_by_ticket"] = north_star_pooled(s)
        for k in HEADLINE_KEYS:
            vals[k].append(h[k])
    return {f"{k}_95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def run_metrics(pred_dir, ids):
    """Latency and cost from <meeting>/meta.json, if the pipeline wrote them."""
    metas = [m for m in (read_meta(Path(pred_dir) / mid) for mid in ids) if m]
    if not metas:
        return {}
    df = pd.DataFrame(metas)
    out = {"meetings_with_meta": len(df), "models": df.model.value_counts().to_dict() if "model" in df else {}}
    if "latency_s" in df and df.latency_s.notna().any():
        out.update(latency_median_s=float(df.latency_s.median()), latency_p90_s=float(df.latency_s.quantile(0.9)))
    if "cost_usd" in df and df.cost_usd.notna().any():
        out.update(cost_mean_usd=float(df.cost_usd.mean()), cost_median_usd=float(df.cost_usd.median()), cost_total_usd=float(df.cost_usd.sum()))
    return out


def score_folder(pred_dir, ref_dir="dataset", meeting_ids=None, judge=None, split_file="splits.json", data=None,
                 invalid="meeting", n_boot=1000):
    data = data or load_dataset(ref_dir)
    ids = meeting_ids or sorted(data, key=meeting_order)
    no_ref = [m for m in ids if not data[m].get("reference")]
    if no_ref:
        print(f"WARNING: {len(no_ref)} meeting(s) have no reference tickets.json and are not scored: {no_ref[:10]}")
        ids = [m for m in ids if m not in no_ref]
    split_info, sp = {}, {}
    if split_file and Path(split_file).exists():         # splits.json describes OUR dataset; None for any other folder
        sp = json.loads(Path(split_file).read_text())
        split_info = ({m: "hold-out, unseen firm" for m in sp.get("holdout_unseen_firm", [])}
                      | {m: "hold-out, seen firm" for m in sp.get("holdout_seen_firm", [])})
    collision_flag = collision_index(data)
    tune_ids = [m for m in sp.get("tune", []) if m in data]
    pm_rows, pair_rows, err_rows, inv_rows, ref_rows = [], [], [], [], []
    for mid in ids:
        f = Path(pred_dir) / mid / "tickets.json"
        try:
            pred = json.loads(f.read_text()) if f.exists() else None
        except json.JSONDecodeError:
            pred = {"tickets": "invalid json"}
        m_row, pairs, errors, invs, refs = score_meeting(data[mid], pred, judge, invalid)
        twin = any(t != mid and is_near_copy(data[mid], data[t]) for t in tune_ids)
        sl = meeting_slices(data[mid], split_info, collision_flag, twin)
        pm_rows.append(dict(meetingId=mid, org_id=data[mid]["org"]["id"], **sl, **m_row))
        pair_rows += [dict(meetingId=mid, **sl, **p) for p in pairs]
        err_rows += [dict(meetingId=mid, **e) for e in errors]
        inv_rows += [dict(meetingId=mid, **x) for x in invs]
        ref_rows += [dict(meetingId=mid, **r) for r in refs]
    pm, pairs, errs = pd.DataFrame(pm_rows), pd.DataFrame(pair_rows), pd.DataFrame(err_rows)
    invs, refs = pd.DataFrame(inv_rows), pd.DataFrame(ref_rows)

    overall = summarise(pm, pairs)
    overall.update(bootstrap(pm, n_boot))
    clean = pm[~pm.twin_in_tune]
    overall["north_star_no_tune_twin"] = float(clean.north_star.mean()) if len(clean) else np.nan
    overall["meetings_no_tune_twin"] = int(len(clean))
    overall["own_dataset"] = bool(sp)
    overall["run"] = run_metrics(pred_dir, ids)
    overall["settings"] = dict(invalid_policy=invalid, judge=(judge.stats["model"] if judge is not None else "word overlap"),
                               min_pair_score=MIN_PAIR_SCORE)

    slice_rows = []
    # meeting-level slices: everything recomputed on that subset of meetings
    for col in ["length", "ref_tickets", "empty_meeting", "declined_in_summary", "firm", "firm_size", "firm_templates", "name_collision", "twin_in_tune"]:
        for val, g in pm.groupby(col):
            gp = pairs[pairs.meetingId.isin(g.meetingId)] if len(pairs) else pairs
            slice_rows.append(dict(level="meeting", slice=col, value=str(val), **summarise(g, gp)))
    # reference-ticket slices: recall (was it found?) and field accuracy on the found ones
    for col, pcol in [("kind", "ref_kind"), ("wording", "ref_wording"), ("date_format", "ref_date_format")]:
        if not len(refs):
            break
        for val, g in refs.groupby(col):
            gp = pairs[pairs[pcol] == val] if len(pairs) else pairs
            row = dict(level="ref ticket", slice={"kind": "ref_ticket_type"}.get(col, col), value=str(val),
                       ref_tickets=len(g), recall=float(g.found.mean()))
            for f in ["template_ok", "dueDate_ok", "assignee_ok", "priority_ok", "subtasks_ok", "text_good"]:
                row[f] = float(gp[f].astype(float).mean()) if len(gp) else np.nan
            slice_rows.append(row)
    return overall, pm, pairs, errs, pd.DataFrame(slice_rows), invs


# ============================================================================ report
def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x)


def write_report(out_dir, overall, pm, pairs, errs, slices, invs=None, judge=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if judge is not None:
        overall["judge_stats"] = judge.stats
    (out / "summary.json").write_text(json.dumps(overall, indent=1, default=float))
    pm.to_csv(out / "per_meeting.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    errs.to_csv(out / "errors.csv", index=False)
    slices.to_csv(out / "slices.csv", index=False)
    (invs if invs is not None else pd.DataFrame()).to_csv(out / "inventions.csv", index=False)

    def rng(k):
        r = overall.get(f"{k}_95")
        return f" ({r[0]:.3f}–{r[1]:.3f})" if r else ""
    L = ["# Score report", "",
         f"Meetings: {overall['meetings']} | reference tickets: {overall['ref_tickets']} | predicted (after validity): {overall['pred_tickets']}",
         f"Settings: {overall['settings']}", "",
         "## North star and detection (95% range by bootstrap over meetings)", "",
         "North star = field-level F1 per meeting (task + template, date, assignee, priority, sub-tasks; same-task gate), averaged over meetings.", "",
         "| metric | value |", "|---|---|"]
    L += [f"| {k} | {overall[k]:.3f}{rng(k)} |" for k in HEADLINE_KEYS]
    if overall.get("own_dataset"):          # only meaningful on our own dataset (twins of tune meetings)
        L += [f"| north star, meetings without a reworded twin in tune ({overall['meetings_no_tune_twin']}) | {overall['north_star_no_tune_twin']:.3f} |"]
    g_ok = overall["invented_facts"] == 0 and overall["invented_names"] == 0
    L += ["", f"## Guardrails (not in the north star): {'no flags' if g_ok else 'FLAGS TO REVIEW (inventions.csv)'}", "",
          "| guardrail | count |", "|---|---|",
          f"| invented facts (amounts, %, dates in no input) | {overall['invented_facts']} |",
          f"| invented names (in no input) | {overall['invented_names']} |",
          f"| tickets with an invention (listed in inventions.csv) | {overall['tickets_with_invention']} |"]
    L += ["", "## All metrics", "", "| metric | value |", "|---|---|"]
    L += [f"| {k} | {_fmt(v)} |" for k, v in overall.items() if not isinstance(v, (dict, list)) and k not in HEADLINE_KEYS]
    if overall.get("run"):
        L += ["", "## Latency and cost", "", "| metric | value |", "|---|---|"]
        L += [f"| {k} | {_fmt(v)} |" for k, v in overall["run"].items()]
    L += ["", "## Errors by type", "", errs.type.value_counts().to_frame("count").to_markdown() if len(errs) else "none",
          "", "## Slices", "", slices.round(3).to_markdown(index=False) if len(slices) else "none"]
    (out / "report.md").write_text("\n".join(L))


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="folder with <meetingId>/tickets.json (+ optional meta.json)")
    ap.add_argument("--ref", default="dataset", help="dataset folder with the reference tickets.json")
    ap.add_argument("--split", default=None, choices=["ref", "pred", "validation", "holdout", "all"],
                    help="meetings to score. Default: for a new folder (--ref is not our dataset) every meeting in --ref "
                         "with reference tickets, a meeting without a prediction counting as empty ('ref'); for our "
                         "own dataset the meetings in --pred ('pred', i.e. the split the run was made on)")
    ap.add_argument("--judge", action="store_true", help="LLM judge (open-weight on Fireworks) for title/text meaning (needs FIREWORKS_API_KEY unless cached)")
    ap.add_argument("--judge-model", default=JUDGE_MODEL, help="Fireworks model id of the judge")
    ap.add_argument("--invalid", default="meeting", choices=["meeting", "ticket"],
                    help="meeting: one invalid ticket makes the meeting count as empty (strict); ticket: drop only invalid tickets")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for 95%% ranges (0 = off)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load_dataset(a.ref)
    own = Path(a.ref).resolve() == Path("dataset").resolve()     # split-based slices only for our own dataset
    a.split = a.split or ("pred" if own else "ref")
    if a.split in ("ref", "all"):
        ids = sorted((m for m in data if data[m].get("reference")), key=meeting_order)
        missing = [m for m in ids if not (Path(a.pred) / m / "tickets.json").exists()]
        if missing:
            print(f"WARNING: {len(missing)} meeting(s) have no prediction and count as empty answers: {missing[:10]}")
    elif a.split == "pred":
        ids = sorted([p.name for p in Path(a.pred).iterdir() if p.is_dir() and p.name in data], key=meeting_order)
    else:
        ids = json.loads(Path("splits.json").read_text())[a.split]
    judge = make_judge(Path(a.out) / "judge_cache", model=a.judge_model) if a.judge else None
    overall, pm, pairs, errs, slices, invs = score_folder(a.pred, a.ref, ids, judge, data=data, invalid=a.invalid,
                                                          n_boot=a.bootstrap, split_file="splits.json" if own else None)
    write_report(a.out, overall, pm, pairs, errs, slices, invs, judge)
    print(f"NORTH STAR {overall['north_star']:.3f} | F1 {overall['f1']:.3f} | P {overall['precision']:.3f} | R {overall['recall']:.3f} "
          f"| valid {overall['valid_rate']:.2f} | GUARDRAILS invented facts {overall['invented_facts']}, "
          f"names {overall['invented_names']} -> {Path(a.out) / 'report.md'}")


if __name__ == "__main__":
    main()
