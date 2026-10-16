"""Deterministic fixes applied to model output, so the model doesn't have to get mechanical parts right.

The reference copies a workflow template's sub-tasks into the ticket 100% of the time: same count, same
order, same title / priority / assignee (true for all 873 template tickets). So the STRUCTURE is copied in code,
and so are the ticket's own title and priority (copy_template_fields).
The TEXT is the one part that varies (31% of reference sub-tasks add summary details or fill in names),
so we keep the model's text when it wrote that sub-task, and fall back to the template text otherwise.
"""
import re



def _template_subtasks(template):
    out = []
    for s in template.get("subTasks") or []:
        a = s.get("assignee")
        out.append({
            "title": s["title"],
            "text": s["text"],
            "priority": s["priority"],
            "dueDate": None,  # 98% of reference sub-tasks have no due date
            "assigneeId": a.get("id") if isinstance(a, dict) else a,
        })
    return out


def copy_template_subtasks(ticket, template):
    """Replace ticket['subTasks'] with the template's sub-tasks.

    Count, order, title, priority, assignee and dueDate come from the template.
    Text: the model's text for the sub-task with the same title if it wrote one, else the template text.
    """
    model_text = {s.get("title"): (s.get("text") or "").strip() for s in ticket.get("subTasks") or []}
    subs = _template_subtasks(template)
    for s in subs:
        if model_text.get(s["title"]):
            s["text"] = model_text[s["title"]]
    ticket["subTasks"] = subs
    return ticket


BLANK = re.compile(r"\[[^\]]+\]|_{3,}")     # form blanks: [LIKE THIS] or ______


def copy_template_fields(ticket, template):
    """Fields the reference never changes on a template ticket (all 873 template tickets):
    - title: the template's title (871/873; the other 2 fill a [blank] in the title, so a title with a blank
      is left to the model)
    - priority: the template's priority, also when it is empty (873/873)"""
    title = template["task"]["title"]
    if not BLANK.search(title):
        ticket["title"] = title
    ticket["priority"] = template["task"]["priority"]
    return ticket


def enforce_assignees(ticket, template, member_ids):
    """Assignee rules backed by the data (the reference never assigns a non-member, never changes a template's person):
    - an assignee must be a member of THIS firm (org.json); anything else (a client, an outsider,
      a made-up id) becomes null. The reference never assigns a non-member.
    - if the template pre-assigns someone, keep that person. The reference never changes it.
    """
    t_assignee = ((template or {}).get("task", {}).get("assignee") or {}).get("id")
    if t_assignee:
        ticket["assigneeId"] = t_assignee
    if ticket.get("assigneeId") not in member_ids:
        ticket["assigneeId"] = None
    for s in ticket.get("subTasks") or []:
        if s.get("assigneeId") not in member_ids:
            s["assigneeId"] = None
    return ticket


_CHAR_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def _normalise_with_map(s):
    """Normalise quotes/dashes/whitespace, keeping a map from each normalised char to its original index."""
    out, idx = [], []
    prev_space = False
    for i, ch in enumerate(s.translate(_CHAR_MAP)):
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        out.append(ch); idx.append(i)
    return "".join(out), idx


def _sentences(s):
    return [x for x in re.split(r"(?<=[.!?;])\s+", s) if x.strip()]


def snap_evidence(ev, summary, max_len=300):
    """Make `ev` an exact substring of the summary, at most max_len characters (Rule 9).
    1) already exact -> keep (cut to max_len)
    2) differs only in quotes/dashes/spaces (e.g. ' vs ’) -> take the exact original text from the summary
    3) otherwise -> the summary sentence sharing the most words with ev
    """
    ev = (ev or "").strip()
    if ev and ev in summary:
        return ev[:max_len]
    n_sum, idx = _normalise_with_map(summary)
    n_ev, _ = _normalise_with_map(ev)
    k = n_sum.find(n_ev) if n_ev else -1
    if k >= 0:
        start, end = idx[k], idx[k + len(n_ev) - 1] + 1
        return summary[start:end][:max_len]
    ev_words = set(n_ev.lower().split())
    best = max(_sentences(summary), key=lambda s_: len(ev_words & set(s_.lower().split())), default="")
    return best.strip()[:max_len]


_TICKET_DEFAULTS = {"templateId": None, "title": "", "text": "", "priority": None, "dueDate": None,
                    "assigneeId": None, "subTasks": [], "evidence": ""}
_SUB_KEYS = ["title", "text", "priority", "dueDate", "assigneeId"]


def repair_shape(ticket):
    """Make one ticket schema-shaped instead of letting a single bad field erase the whole meeting.
    Returns the repaired ticket, or None if it has neither a title nor a text (nothing to keep)."""
    if not isinstance(ticket, dict):
        return None
    t = {k: ticket.get(k, v) for k, v in _TICKET_DEFAULTS.items()}            # missing keys -> defaults; extra keys dropped
    t["title"] = (t["title"] or "").strip() if isinstance(t["title"], str) else ""
    t["text"] = (t["text"] or "").strip() if isinstance(t["text"], str) else ""
    if not t["title"] and not t["text"]:
        return None
    t["text"] = t["text"] or t["title"]                                     # empty text -> reuse the title
    t["title"] = t["title"] or t["text"][:120]
    if t["priority"] not in ("low", "medium", "high"):
        t["priority"] = None
    if t["dueDate"] is not None and not (isinstance(t["dueDate"], str) and len(t["dueDate"]) == 10 and t["dueDate"][4] == "-"):
        t["dueDate"] = None
    subs = t["subTasks"] if isinstance(t["subTasks"], list) else []
    t["subTasks"] = [{k: s.get(k) for k in _SUB_KEYS} for s in subs if isinstance(s, dict) and (s.get("title") or s.get("text"))]
    for s in t["subTasks"]:
        s["title"] = s["title"] or (s["text"] or "")[:120]
        s["text"] = s["text"] or s["title"]
        if s["priority"] not in ("low", "medium", "high"):
            s["priority"] = None
    return t


def postprocess(output, meeting):
    """Apply all deterministic fixes to one meeting's output dict ({'meetingId', 'tickets'})."""
    templates = {t["id"]: t for t in meeting["templates"]}
    member_ids = {m["id"] for m in meeting["org"]["members"]}
    output["tickets"] = [t for t in (repair_shape(x) for x in output.get("tickets") or []) if t is not None]
    for ticket in output.get("tickets", []):
        t = templates.get(ticket.get("templateId"))
        if ticket.get("templateId") is not None and t is None:
            ticket["templateId"] = None          # unknown id -> freeform (an id from another firm is never valid)
        if t:
            copy_template_fields(ticket, t)
        if t and t.get("subTasks"):
            copy_template_subtasks(ticket, t)
        elif t:
            ticket["subTasks"] = []              # template without sub-tasks: the reference never adds any (687/687)
        enforce_assignees(ticket, t, member_ids)
        ticket["evidence"] = snap_evidence(ticket.get("evidence"), meeting["summary"])
        if isinstance(ticket.get("title"), str):
            ticket["title"] = ticket["title"][:120]
    return output
