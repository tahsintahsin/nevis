# Design note

## (a) The pipeline

**One model call per meeting.** The model gets instructions (what counts as an agreed follow-up, how to use
templates, how this firm writes tickets), then the firm's templates and members, then the summary, and must answer in
the exact ticket format (enforced by a JSON schema).

**Then code** copies a template's title, priority, assignee and sub-tasks unchanged, removes assignees who aren't in the
firm, turns the evidence into an exact quote of the summary, and repairs a malformed ticket instead of losing the
meeting. The model only does the judgement: which follow-ups exist, which template fits, how to word the ticket.

**Model and route:** GLM 5.3 (open weights) on Fireworks serverless, reasoning set to *low*.

**What we tried:** the prompt was refined on the tune meetings; then, on Fireworks, 7 open models, reasoning levels,
output budgets, worked examples in the prompt, a two-call pipeline (find, then write), output without an enforced
format, and LoRA fine-tuning (1–3 epochs). Every run's outputs and scores are in this zip (`outputs/`, `scores/`);
the fine-tuning jobs, data and models are in the Fireworks account.

## (b) Measuring closeness, and why to trust it

**How a meeting is scored.** Each of our tickets is matched to a reference ticket by the summary sentence both quote.
A judge model (gpt-oss-120b) reads each matched pair: if they are different tasks, the match is cancelled; otherwise
it rates whether our text carries the same facts. A matched ticket earns up to six points: text (judge's rating,
similar length and style, template wording kept and its blanks filled like the reference), template, due date,
assignee, priority and sub-tasks. A ticket we invented, or one we missed, earns nothing. The points become an F1
score per meeting (penalising both extra and missing tickets), and the headline, the **north star**, averages
meetings, because an advisor experiences meetings, not tickets.

**Guardrails,** reported next to the score: any amount, date or name in our tickets that appears
in none of the meeting's inputs is flagged as a possible invention; in our runs every flag was checked, and a model
with real inventions isn't shipped.

**Why trust it:** we broke the reference on purpose in 15 ways and the score moved exactly where it should and
nowhere else (`tests/test_scorer_trust.py`). A second judge from another model family (Claude Haiku) moves the north star by at most 0.006. The reported numbers aren't tuned to the test: every decision was made on
validation, and the hold-out was run once per final candidate (split in (c)).

## (c) The trade-off and what we ship

We split the 530 meetings into *tune* (370) to build the prompt and code, *validation* (80) for every decision, and
*hold-out* (80, half from firms never used), run once per final candidate.

| Hold-out | North star (95% range) | New firms only | $ / meeting | Seconds per meeting (median / p90) |
|---|---|---|---|---|
| **GLM 5.3** | **0.922** (0.90–0.94) | **0.924** | 0.0107 | 18 / 36 |
| MiniMax M3 | 0.897 (0.86–0.93) | 0.874 | 0.0082 | 27 / 60 |
| DeepSeek V4.1 Flash | 0.871 (0.82–0.92) | 0.837 | 0.0188 | 56 / 290 |
| gpt-oss-120b | 0.869 (0.83–0.90) | 0.858 | 0.0028 | 16 / 25 |
| Qwen3.5 9B fine-tuned (untuned 0.747) | 0.865 (0.83–0.90) | 0.884 | ~0.008* | 13 / 25 |
| Nemotron Lightning 30B | 0.773 (0.74–0.81) | 0.791 | **0.0010** | **3 / 4** |

\* Own GPU, billed busy or idle: cheaper than gpt-oss only at thousands of meetings an hour, around the clock.

**We ship GLM 5.3.** On validation the top three were tied on score (0.913–0.923); GLM was the fastest (median 17 s vs
34–55 s) and raised no guardrail flags; the hold-out confirmed it. gpt-oss-120b is the budget option: a quarter of the cost, 0.05
lower; in practice I'd settle this with the business (budget, volume, latency). Fine-tuning worked (the 9B model gained
0.12 and reached gpt-oss's level) but needs a dedicated GPU, which only pays off at high, steady volume.

**How close can anyone get?** Claude Haiku, same prompt: 0.925 on validation; GLM is level with it.

## (d) What's wrong in the reference, and what we did

The reference follows the format rules but has errors: invented percentages (meetings 249, 947, 950), a
stated due date left out (263), a fitting template not used (224, 980). It is also inconsistent: of 120 meetings with
an "X is needed by <date>" item, it tickets those items in 82 and ignores them all in 38, so no model can match it every
time. We listed the errors (`reference_issue_meetings.json`; without those 13 validation meetings scores rise by
0.02–0.03, same ranking), and did not tune to them.

## (e) How I used AI tools, and where I checked them

I started on my local setup (Claude Code for code and analysis, Claude Haiku as a cheap test model and first judge),
then moved to open-weight models on Fireworks. Claude Code wrote and ran the code; I directed the work and checked it.
I started with data analysis and had every claim re-checked on the data (two of the AI's early claims were wrong).
The key decisions were mine: hide template names and rely on ids (names are reused across firms), copy in code
whatever the reference never changes, score each meeting equally with "a wrong task earns nothing", a three-way split
against overfitting, and suspect the reference before the model (which surfaced its errors and inconsistency). My
checks changed the outcome: I asked whether worked examples could leak answers (they could, through reworded
meetings: fixed), whether pasting template text back kept tickets coherent (it didn't: removed), and how form blanks
were scored (they weren't: added).

## "The new model missed two follow-ups… the scorer says 94%"

Both can be true: 94% is an average over meetings, so one bad meeting barely moves it; and if the reference also
lacks those two tickets, the scorer can't see the miss at all ("needed by" items are the usual case: the reference
ignores them in 38 of 120 meetings). In the next hour, in this order:

1. **Look at that meeting:** the summary, our tickets and the model's raw answer (`raw.json`). Were both follow-ups
   really agreed, not declined or postponed? How were they phrased ("agreed to", "needed by <date>")? Was the answer
   cut off, or were they merged into another ticket?
2. **Check the scorer:** does the reference have those two tickets? Score that meeting alone, and run the previous
   model on it.
3. **Check whether it's a pattern:** recall for that kind of follow-up, old vs new model, on the last validation run
   and on recent traffic.

**Roll back** if the misses are systematic (recall on that kind of follow-up drops beyond its normal range), if any
invented fact or wrong person shows up, if answers are cut off or invalid, or if the cause isn't clear within the
hour. Otherwise fix it (a prompt rule or code), add this meeting with the correct tickets to validation, re-run
validation, and ship behind a canary.
