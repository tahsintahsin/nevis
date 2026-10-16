"""Prompts for meeting -> tickets, the user message builder, the JSON schema, and the JSON parser.

READING GUIDE (top to bottom):
  1. SYSTEM                   the system prompt: the task, what counts as a follow-up (with do/don't examples),
                              template rules, field rules, defaults, writing style, nothing invented
  2. build_user_prompt()      the per-meeting message, ordered for prompt caching: firm templates (no names)
                              -> firm members -> meeting summary
  3. parse_json()             reads the model's JSON; repairs broken JSON instead of losing the meeting
  4. TICKETS_SCHEMA           JSON schema for structured output (the API then guarantees valid JSON)
  5. two-step prompts         step 1 finds the follow-ups, step 2 writes the tickets (pipeline: two_step);
                              same rules as SYSTEM, cut from it by heading
A run records the prompt name + a hash of the exact text (meta.json), so results stay tied to the text used.
"""
import json
import re

SYSTEM = """# Your task
You read the notes an advisor at a wealth-management firm wrote after a client meeting. Your job is to find every
task the FIRM committed to do after the meeting (a "follow-up") and write each one as a ticket for the firm's task
system: one ticket per agreed follow-up, written the way this firm writes tickets, using the firm's own task template
when one applies, assigned to the right firm member, with the due date that was agreed in the meeting.

You get: the firm's templates, the firm's members, and the meeting summary. Summaries contain three kinds of content:
facts about the household, follow-ups the firm committed to, and things that were discussed but declined or postponed.

**Some meetings have NO follow-ups** (about 1 in 9): nothing was agreed, or everything was declined/postponed.
Then the correct answer is an empty list: {"meetingId": "<id>", "tickets": []}. Do not invent tickets to fill it.

# What counts as a follow-up
Only take what is EXPLICITLY stated as agreed, requested-and-accepted, or needed by a date. Do not infer or guess tasks.

Take it (explicit signals):
- "the agreed follow-up is to ...", "they agreed to ...", "Agreed <date>: ..."
- "<client> asked <firm/member> to ...", "<member> will ..."
- "... is needed by <date>", "follow-up is needed by <date>": a stated need with a deadline IS a follow-up,
  even without the word "agreed"
- a firm action (prepare, send, confirm, review, coordinate, follow up, reconcile, obtain, update) with a due date

Leave it (no ticket):
- items that were declined, postponed, deferred, or "without a current commitment", even if a template matches them
- problems that are only described ("remained unresolved", "was not available", "differed by ...") with no commitment to act
- facts about the household (income, balances, policies, family); use them INSIDE ticket text, never as tickets

Examples (from other meetings):
| Sentence in a summary | Ticket? |
|---|---|
| "the agreed follow-up is to prepare the pre-clearance packet by 2024-04-19." | yes |
| "They agreed by 2024-04-05 to reset portal credentials, confirm MFA methods, and temporarily restrict new external money movement pending review" | yes, ONE ticket with 3 sub-tasks |
| "...a working liquidity figure is needed by 12/6/2024 before choosing cash kept aside." | yes ("needed by" + date) |
| "...margin borrowing for legal fees was declined." | no (declined) |
| "That review was postponed, with no current commitment and no supplied date." | no (postponed) |
| "...so that topic was deferred without a current commitment." | no (deferred) |
| "The equity portal's released-share total differed from the brokerage position by 150 shares..., leaving the correct available share count unresolved." | no (problem described, nobody committed) |
| "Elena is Vice President of Product ... with a $248,000 base salary" | no (a fact) |
| "...requested an education savings account ...; the education account registration and initial funding request was declined." | no, even though the firm has a matching template |

One sentence can contain two follow-ups -> two tickets (they may quote the same evidence).
Two sentences describing the same follow-up -> one ticket.

# Templates
- If one of the firm's templates describes the SAME KIND OF WORK as the follow-up (not just the same topic), set
  templateId to its id and start from its content: keep its title and text shape, its sub-tasks, its priority and its
  pre-assigned assignee, unless the summary says otherwise. Example: a "gather refinance documents" template fits a
  "collect the missing payoff quote" follow-up, but not a "compare refinance options" analysis.
- Use the id. Template names are not shown and are not unique; titles are.
- Placeholder template (only a title, no text): still a valid template; keep the title, write the body yourself.
- Form template with [blanks]: fill each blank from the summary; if the summary does not have the value, leave the
  blank exactly as written.
- Workflow template with sub-tasks: copy all of them, in order.
- Check EVERY template before writing a freeform ticket, including placeholder templates (title only). If a
  template's title or description names the same kind of work as the follow-up (even in different words, e.g.
  "Coordinate benefits enrollment support" for "review the client's benefit elections"), use it.
- Only if no template describes that work: templateId null (freeform). A template about a different kind of work
  is wrong.

# Fields
- title: at most 120 characters. Short, starts with a verb, about 6 words ("Confirm incoming wire instructions with Teacup Comet Bank").
  For a template ticket, keep the template's title.
- text: the body an advisor reads and acts on. 1-2 sentences: the action with the client's full name and the
  institutions involved; then the concrete figures, constraints or reason, taken from anywhere in the summary.
  Do not put the due date, the word "agreed", or the advisor's name in the text.
- priority: from the template if one is used. Otherwise only if the summary makes urgency explicit ("urgent",
  "time-sensitive"); else null.
- dueDate: only when a date is stated or unambiguously derivable; write it as YYYY-MM-DD ("June 10, 2026" -> 2026-06-10,
  "11/18/2024" -> 2024-11-18, "June 14" -> use the meeting's year). Vague timing ("in the next few weeks") or
  "no due date" -> null. Never default to the meeting date.
- assigneeId: the id of a firm member, only if the summary NAMES the firm member who will do the task, or the
  template pre-assigns someone and the summary is silent. Otherwise null. The advisor who held the meeting is NOT the
  assignee by default. Never assign the client or an outside professional (CPA, attorney, lender). Firm members are
  exactly the people in the member list; anyone else is a client, family member or outsider, even with the same first name.
- subTasks: same fields as a ticket minus templateId and evidence. Empty unless the template has sub-tasks or the
  summary clearly breaks the work into steps.
- evidence: an EXACT quote copied character for character from the summary (at most 300 characters) that shows the
  follow-up was agreed. It traces the ticket to the sentence that caused it.

# Defaults: what a ticket looks like when the summary says nothing more
A "freeform" ticket is one that uses no template (templateId null): you write its title and text from the summary.
- Freeform ticket: priority null, assigneeId null, subTasks [] - unless the summary explicitly says otherwise
  (urgency for priority; a named firm member doing the task for assignee; an explicit list of steps for sub-tasks,
  e.g. "agreed to reset credentials, confirm MFA methods, and restrict money movement" -> 3 sub-tasks).
- Template ticket: copy the template's title, priority, pre-assigned person and ALL its sub-tasks (same order, same
  titles); only fill [blanks] and add summary details to the text. Templates without sub-tasks -> subTasks [].
- dueDate: almost every follow-up states one; copy it (converted to YYYY-MM-DD). Null only if no date is given.
- When unsure whether to add a value (priority, assignee, sub-tasks), leave it empty.

# Style (how the firm writes tickets)
- title: starts with an action verb (Prepare, Review, Obtain, Coordinate, Reconcile, Resolve, Compare, Update, Confirm,
  Collect, Send, Request, ...), about 6 words.
- text: also starts with an action verb; freeform text is 1-2 sentences, roughly 20-45 words; never start with "The", a name, or "This".


# Text content (important)
- Template ticket: keep the template's text (do not shorten, summarise or rewrite it), but personalise it: replace
  generic words such as "the client" or "the household" with the actual names, and fill its [blanks] (also any
  [blank] in the template TITLE, e.g. "Collect Tax Documents for [Tax Year]" -> "Collect Tax Documents for 2025").
  Then add 1-2 sentences with this follow-up's specifics. Template tickets are therefore longer than freeform ones.
- The same template may be used for two different follow-ups in one meeting (one ticket each).
- Ticket text is a rewritten instruction for the advisor. Never paste the summary sentence (that is what evidence is for).
- Sub-tasks copied from a template keep their titles; you may add this follow-up's specifics to a sub-task's text.
  Sub-task due dates stay null.
- Every ticket text names the client (full name) and every institution involved (custodian, bank, insurer,
  employer, law firm) exactly as the summary writes them, plus the concrete amounts and dates that matter.

# Nothing invented
Every date, amount, name, account, institution and percentage in a ticket must appear in the summary, the member list
or the template. Prefer omitting to guessing.

# Output
Only the JSON object: {"meetingId": "<id>", "tickets": [ ... ]}."""


def _template_view(t):
    """How a template is shown to the model.
    - NO name: names are not unique within a firm and are often misleading (22 firms have two different templates with the same name).
    - Title is always present and unique within a firm. Description/text/sub-tasks are included only
      when present (placeholders have only a title).
    """
    task = t["task"]
    view = {"id": t["id"], "title": task["title"]}
    if (t.get("description") or "").strip():
        view["description"] = t["description"]
    if (task.get("text") or "").strip():
        view["text"] = task["text"]
    if task.get("priority"):
        view["priority"] = task["priority"]
    if (task.get("assignee") or {}).get("id"):
        view["assigneeId"] = task["assignee"]["id"]
    if t.get("subTasks"):
        view["subTasks"] = [
            {"title": s["title"], "text": s["text"], "priority": s["priority"],
             "assigneeId": (s.get("assignee") or {}).get("id") if isinstance(s.get("assignee"), dict) else s.get("assignee")}
            for s in t["subTasks"]
        ]
    if not view.get("text"):
        view["note"] = "placeholder template: keep the title, write the body from the summary"
    return view


def build_user_prompt(meeting):
    org = meeting["org"]
    members = [{"id": m["id"], "name": m["name"], "role": m["role"]} for m in org["members"]]
    # Order = least-changing first, for prompt caching: templates and members are the same for
    # every meeting of a firm; the summary and meetingId change every time, so they go last.
    return (
        f"## Firm templates\n{json.dumps([_template_view(t) for t in meeting['templates']], ensure_ascii=False, indent=1)}\n\n"
        f"## Firm members (the only people who can be assigned)\n{json.dumps(members, ensure_ascii=False, indent=1)}\n\n"
        f"## Meeting\nmeetingId: {meeting['meetingId']}\n\n{meeting['summary']}\n\n"
        "Return the JSON object now."
    )


def parse_json(text, required_key="tickets"):
    """Extract the JSON object from a model reply (tolerates ```json fences / leading prose / <think> blocks).
    If the JSON is broken (e.g. a missing comma), fall back to a repairing parser instead of losing the meeting.
    required_key: the key the repaired object must contain ("tickets"; "followups" for two-step step 1)."""
    text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.S).strip()   # open-weight reasoning models
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        from json_repair import repair_json
        fixed = repair_json(candidate, return_objects=True)
        if not isinstance(fixed, dict) or required_key not in fixed:
            raise
        return fixed


# ---------------------------------------------------------------------------- structured output
# JSON schema for the API's structured-output mode (output_config.format): the answer is guaranteed to be
# valid JSON of this shape. Ids are free strings here; nevis/postprocess.py checks them against the folder.
_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_PRIORITY = {"anyOf": [{"type": "string", "enum": ["low", "medium", "high"]}, {"type": "null"}]}
_SUBTASK = {"type": "object", "additionalProperties": False,
            "required": ["title", "text", "priority", "dueDate", "assigneeId"],
            "properties": {"title": {"type": "string"}, "text": {"type": "string"}, "priority": _PRIORITY,
                           "dueDate": _NULLABLE_STR, "assigneeId": _NULLABLE_STR}}
_TICKET = {"type": "object", "additionalProperties": False,
           "required": ["templateId", "title", "text", "priority", "dueDate", "assigneeId", "subTasks", "evidence"],
           "properties": {"templateId": _NULLABLE_STR, "title": {"type": "string"}, "text": {"type": "string"},
                          "priority": _PRIORITY, "dueDate": _NULLABLE_STR, "assigneeId": _NULLABLE_STR,
                          "subTasks": {"type": "array", "items": _SUBTASK}, "evidence": {"type": "string"}}}
TICKETS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["meetingId", "tickets"],
                  "properties": {"meetingId": {"type": "string"}, "tickets": {"type": "array", "items": _TICKET}}}


# ---------------------------------------------------------------------------- two-step pipeline (pipeline: two_step)
# Step 1 only FINDS the follow-ups; step 2 WRITES one ticket per follow-up, with the full summary and templates.
# Two variants (config `prompt`), differing only in who picks the template:
#   two_step       step 1: evidence + action                -> step 2 decides template, date and everything else
#   two_step_tmpl  step 1: evidence + action + template id  -> step 2 keeps that template, decides the rest
# Both steps see the firm's templates. Both reuse the rule text of SYSTEM verbatim (cut by heading).
def _section(prompt, heading):
    """The text of one '# heading' section of a prompt, heading included."""
    m = re.search(rf"^# {re.escape(heading)}\n.*?(?=^# |\Z)", prompt, re.S | re.M)
    return m.group(0).strip()


def _find_prompt(with_template):
    example = '{"evidence": "...", "action": "..."' + (', "templateId": null' if with_template else "") + "}"
    return f"""# Your task
You read the notes an advisor at a wealth-management firm wrote after a client meeting. This is step 1 of 2: LIST every
task the FIRM committed to do after the meeting (a "follow-up"). Do not write tickets yet; a second step writes them.

You get: the firm's templates, the firm's members, and the meeting summary. Summaries contain three kinds of content:
facts about the household, follow-ups the firm committed to, and things that were discussed but declined or postponed.
The templates show the kinds of tasks this firm does; list only what THIS meeting agreed.

**Some meetings have NO follow-ups** (about 1 in 9): nothing was agreed, or everything was declined/postponed.
Then the correct answer is an empty list: {{"meetingId": "<id>", "followups": []}}. Do not invent follow-ups to fill it.

{_section(SYSTEM, "What counts as a follow-up")}
""" + (f"""
{_section(SYSTEM, "Templates")}
""" if with_template else "") + f"""
# For each follow-up give
- "evidence": an EXACT quote copied character for character from the summary (at most 300 characters) that shows the
  follow-up was agreed.
- "action": one short line saying what the firm will do (for your colleague who writes the ticket). Two follow-ups in
  one sentence -> two entries with the same evidence and different actions.""" + ("""
- "templateId": the id of the firm template that describes the same kind of work, or null (freeform).""" if with_template else "") + f"""

# Output
Only the JSON object: {{"meetingId": "<id>", "followups": [{example}, ...]}}."""


def _write_prompt(with_template):
    keep = "evidence and templateId" if with_template else "evidence"
    return f"""# Your task
You write tickets for a wealth-management firm's task system. This is step 2 of 2: the follow-ups agreed in the meeting
have already been found (list "Follow-ups found in step 1"). Write EXACTLY ONE ticket per follow-up, in the same order,
written the way this firm writes tickets. Keep each follow-up's {keep} as given; use the summary for every detail.
Do not add, merge or drop follow-ups.

{_section(SYSTEM, "Templates")}

{_section(SYSTEM, "Fields")}

{_section(SYSTEM, "Defaults: what a ticket looks like when the summary says nothing more")}

{_section(SYSTEM, "Style (how the firm writes tickets)")}

{_section(SYSTEM, "Text content (important)")}

{_section(SYSTEM, "Nothing invented")}

# Output
Only the JSON object: {{"meetingId": "<id>", "tickets": [ ... ]}}."""


PROMPTS = {"main": SYSTEM,
           "two_step_find": _find_prompt(False), "two_step_write": _write_prompt(False),
           "two_step_tmpl_find": _find_prompt(True), "two_step_tmpl_write": _write_prompt(True)}
# two-step prompt sets: config `prompt: <name>` + `pipeline: two_step`.
# fields = what step 1 returns; locked = what step 2 may not change (set back to step 1's value).
TWO_STEP = {
    "two_step": dict(find="two_step_find", write="two_step_write",
                     fields=("evidence", "action"), locked=("evidence",)),
    "two_step_tmpl": dict(find="two_step_tmpl_find", write="two_step_tmpl_write",
                          fields=("evidence", "action", "templateId"), locked=("evidence", "templateId")),
}


def followups_schema(fields):
    props = {"evidence": {"type": "string"}, "action": {"type": "string"}, "templateId": _NULLABLE_STR}
    item = {"type": "object", "additionalProperties": False, "required": list(fields),
            "properties": {f: props[f] for f in fields}}
    return {"type": "object", "additionalProperties": False, "required": ["meetingId", "followups"],
            "properties": {"meetingId": {"type": "string"}, "followups": {"type": "array", "items": item}}}


_ASK = "Return the JSON object now."


def build_step2_user_prompt(meeting, followups):
    """Step-2 input: the same firm/meeting blocks as build_user_prompt, plus the step-1 follow-up list."""
    base = build_user_prompt(meeting)
    assert base.endswith(_ASK)
    return (base[: -len(_ASK)]
            + f"## Follow-ups found in step 1 ({len(followups)})\n{json.dumps(followups, ensure_ascii=False, indent=1)}\n\n"
            + f"Write exactly {len(followups)} ticket(s), one per follow-up, in the same order. " + _ASK)


def followups_from_tickets(tickets, fields=("evidence", "action", "templateId")):
    """Reference tickets -> the step-1 answer they imply (few-shot examples of step 1); action = the ticket's title."""
    fields = fields or ("evidence", "action", "templateId")
    full = lambda t: {"evidence": t.get("evidence", ""), "action": t.get("title", ""), "templateId": t.get("templateId")}
    return [{f: full(t)[f] for f in fields} for t in tickets]
