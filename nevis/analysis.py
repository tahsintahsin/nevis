"""Analysis of scored runs: compare two runs, see where points are lost, export low judge scores for review.

Works on the scorer's output folders (scores/<run>/: per_meeting.csv, pairs.csv, errors.csv, summary.json), so it
works for any model.

    python -m nevis.analysis compare  scores/<run_A> scores/<run_B>      # paired difference + 95% range
    python -m nevis.analysis breakdown scores/<run>                       # where north-star points are lost
    python -m nevis.analysis templates scores/<run>                       # every check by template outcome + template P/R/F1
    python -m nevis.analysis review   scores/<run> --out review.md        # tickets the judge scored < 4, side by side
Add --exclude-reference-errors to drop the meetings whose reference has confirmed errors (reference_issue_meetings.json).

Overfitting rule: read individual errors ('review') only for TUNE runs; for validation use numbers only;
never look at hold-out errors.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = ["text_good", "template_ok", "dueDate_ok", "assignee_ok", "priority_ok", "subtasks_ok"]   # the 6 north-star items
REF_ERRORS_FILE = Path("reference_issue_meetings.json")


# ----------------------------------------------------------------------------------------------- loading
def load(score_dir, exclude_reference_errors=False):
    """Return (summary dict, per_meeting DataFrame, pairs DataFrame) for one scored run."""
    d = Path(score_dir)
    summary = json.loads((d / "summary.json").read_text())
    pm = pd.read_csv(d / "per_meeting.csv", dtype={"meetingId": str})
    pairs = pd.read_csv(d / "pairs.csv", dtype={"meetingId": str}) if (d / "pairs.csv").stat().st_size > 1 else pd.DataFrame()
    if exclude_reference_errors and REF_ERRORS_FILE.exists():
        bad = set(json.loads(REF_ERRORS_FILE.read_text())["reference_error_meetings"])
        pm = pm[~pm.meetingId.isin(bad)]
        pairs = pairs[~pairs.meetingId.isin(bad)] if len(pairs) else pairs
    return summary, pm, pairs


def bootstrap_mean(values, n_boot=5000, seed=0):
    """Mean and 95% range of `values`, resampling meetings with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return v.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ----------------------------------------------------------------------------------------------- compare
def compare(score_a, score_b, exclude_reference_errors=False):
    """Paired comparison of two runs on the SAME meetings: per-meeting north-star difference A - B, with a 95% range.
    If the range includes 0 there is no clear difference."""
    _, a, _ = load(score_a, exclude_reference_errors)
    _, b, _ = load(score_b, exclude_reference_errors)
    m = a[["meetingId", "north_star"]].merge(b[["meetingId", "north_star"]], on="meetingId", suffixes=("_a", "_b"))
    if len(m) < len(a) or len(m) < len(b):
        print(f"note: only {len(m)} meetings are in both runs; comparing those")
    diff = m.north_star_a - m.north_star_b
    mean, lo, hi = bootstrap_mean(diff)
    return dict(meetings=len(m), north_star_a=m.north_star_a.mean(), north_star_b=m.north_star_b.mean(),
                diff=mean, lo=lo, hi=hi, a_better=int((diff > 0).sum()), b_better=int((diff < 0).sum()),
                verdict="A better" if lo > 0 else "B better" if hi < 0 else "no clear difference")


# ----------------------------------------------------------------------------------------------- breakdown
def breakdown(score_dir, exclude_reference_errors=False):
    """Where north-star points are lost. Every ticket has 6 items; extra and missed tickets lose all 6.
    Wrong-template tickets (forced / missed / wrong) are counted as one group because their fields cascade."""
    _, pm, p = load(score_dir, exclude_reference_errors)
    lost = {"extra tickets": 6 * int((pm.n_pred - pm.matched).sum()),
            "missed tickets": 6 * int((pm.n_ref - pm.matched).sum())}
    wrong = p.template_outcome.isin(["forced", "missed", "wrong_template"]) if len(p) else pd.Series(dtype=bool)
    lost["wrong template choice (all fields of those tickets)"] = int(sum((~p.loc[wrong, i].astype(bool)).sum() for i in ITEMS)) if len(p) else 0
    for i in ITEMS:
        lost[f"{i} (right template choice)"] = int((~p.loc[~wrong, i].astype(bool)).sum()) if len(p) else 0
    s = pd.Series(lost)
    return pd.DataFrame({"lost items": s, "share": (s / max(1, s.sum())).round(3)})


# ----------------------------------------------------------------------------------------------- templates
def template_table(score_dir, exclude_reference_errors=False):
    """Every check, split by template outcome (correct freeform / correct template / forced / missed / wrong)."""
    _, _, p = load(score_dir, exclude_reference_errors)
    cols = [c for c in ["judge_0_5", "text_good", "text_size_ok", "text_style_ok", "template_used_ok", "template_kept", "blanks_f1"] + ITEMS[1:] if c in p]
    t = p.groupby("template_outcome")[cols].mean()
    t.insert(0, "tickets", p.template_outcome.value_counts())
    t.loc["ALL"] = [len(p)] + list(p[cols].astype(float).mean())
    return t.round(3)


def template_prf(score_dir, exclude_reference_errors=False):
    """'Use a template' as a yes/no decision. Precision: when we used one, was it the reference's? Recall: when the
    reference used one, did we use the same? (forced = we used one, ref didn't; missed = ref used one, we didn't)."""
    _, _, p = load(score_dir, exclude_reference_errors)
    c = p.template_outcome.value_counts()
    tp, wrong, fp, fn = (int(c.get(k, 0)) for k in ["correct_template", "wrong_template", "forced", "missed"])
    prec = tp / (tp + fp + wrong) if tp + fp + wrong else float("nan")
    rec = tp / (tp + fn + wrong) if tp + fn + wrong else float("nan")
    return dict(correct=tp, wrong=wrong, forced=fp, missed=fn, precision=round(prec, 3), recall=round(rec, 3),
                f1=round(2 * prec * rec / (prec + rec), 3) if prec + rec else float("nan"))


# ----------------------------------------------------------------------------------------------- review
def review(score_dir, pred_dir, out_md, max_score=3, dataset="dataset"):
    """Write a Markdown list of paired tickets with judge score <= max_score: reference vs ours, side by side.
    (The judge's own reason is in scores/<run>/judge_cache.) Use on TUNE runs only."""
    from nevis.data import load_dataset
    data = load_dataset(dataset)
    _, _, p = load(score_dir)
    low = p[p.judge_0_5 <= max_score].sort_values(["judge_0_5", "meetingId"])
    lines = [f"# Judge score <= {max_score}: {score_dir}", "", f"{len(low)} of {len(p)} paired tickets.", ""]
    for n, r in enumerate(low.itertuples(), 1):
        ref = data[r.meetingId]["reference"]["tickets"][r.ref_idx]
        ours = json.loads((Path(pred_dir) / r.meetingId / "tickets.json").read_text())["tickets"][r.pred_idx]
        clean = lambda s: (s or "").replace("\n", " ").replace("|", "/")
        lines += [f"## {n}. Meeting {r.meetingId}: judge {int(r.judge_0_5)}/5 ({r.template_outcome})", "",
                  "| | Reference | Ours |", "|---|---|---|",
                  f"| title | {clean(ref['title'])} | {clean(ours['title'])} |",
                  f"| text | {clean(ref['text'])} | {clean(ours['text'])} |", "",
                  f"Evidence (reference): \"{ref['evidence']}\"", ""]
    Path(out_md).write_text("\n".join(lines))
    return len(low)


# ----------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    b = sub.add_parser("breakdown"); b.add_argument("scores")
    t = sub.add_parser("templates"); t.add_argument("scores")
    r = sub.add_parser("review"); r.add_argument("scores"); r.add_argument("--pred", required=True); r.add_argument("--out", required=True)
    for s in (c, b, t):
        s.add_argument("--exclude-reference-errors", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.cmd == "compare":
        x = compare(a.a, a.b, a.exclude_reference_errors)
        print(f"{x['meetings']} meetings | A {x['north_star_a']:.3f} vs B {x['north_star_b']:.3f} | A-B {x['diff']:+.3f} "
              f"(95%: {x['lo']:+.3f} to {x['hi']:+.3f}) -> {x['verdict']} | A better in {x['a_better']}, B in {x['b_better']}")
    elif a.cmd == "breakdown":
        print(breakdown(a.scores, a.exclude_reference_errors).to_string())
    elif a.cmd == "templates":
        print(template_table(a.scores, a.exclude_reference_errors).to_string())
        print("template decision:", template_prf(a.scores, a.exclude_reference_errors))
    else:
        print(f"{review(a.scores, a.pred, a.out)} tickets written to {a.out}")


if __name__ == "__main__":
    main()
