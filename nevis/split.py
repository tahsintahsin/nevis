"""Data split: tune / validation / hold-out Reproduces splits.json exactly.

    python -m nevis.split --check     # rebuild in memory and confirm it equals splits.json (no write)
    python -m nevis.split --write     # write splits.json (refuses to change an existing hold-out or validation)

How the split is built (all choices fixed by seeds, so it is reproducible):
  1. Every meeting gets 4 labels: empty?, number of reference tickets (0 / 1-3 / 4-7 / 8+), template use
     (none / some / many), summary length (short / medium / long).
  2. HOLD-OUT (80): ~40 meetings from WHOLE firms that never appear elsewhere ("unseen firms") + 40 meetings from
     other firms (max 1 per firm). 20,000 random candidates are tried; the one whose label mix is closest to the whole
     dataset wins. Meetings we already looked at (FORCED_DEV) can never be in the hold-out.
  3. VALIDATION (80) from the remaining 450 the same way (~40 from unseen-in-tune firms + 40 others), additionally
     matching the mix of reference tickets by template type (freeform / plain / form / workflow / placeholder).
  4. TUNE = the remaining 370. We read errors and pick prompt examples only from tune.
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from nevis.data import load_dataset

SPLIT_FILE = Path("splits.json")
SEED = 2026
N_HOLDOUT, N_UNSEEN_TARGET, N_TRIES = 80, 40, 20000
VAL_SIZE, VAL_UNSEEN_TARGET, N_TRIES_VAL = 80, 40, 20000
STRATA = ["empty", "tickets", "template_use", "length"]
KINDS = ["freeform", "plain", "form", "workflow", "placeholder"]
# Meetings we had already looked at when the split was made: the first quick tests + prompt-example meetings.
QUICK_TEST_IDS = ["98", "114", "135", "169", "277", "394", "644", "706", "758", "967"]
PROMPT_EXAMPLE_IDS = ["15", "60", "75", "99", "100", "187", "390"]
FORCED_DEV = sorted(set(QUICK_TEST_IDS + PROMPT_EXAMPLE_IDS), key=int)


def meeting_labels(data):
    """One row per meeting with the 4 stratification labels."""
    rows = []
    for mid, m in data.items():
        t = m["reference"]["tickets"]
        n, n_tmpl, words = len(t), sum(x["templateId"] is not None for x in t), len(m["summary"].split())
        rows.append(dict(meetingId=mid, org_id=m["org"]["id"], empty=n == 0,
                         tickets=("0" if n == 0 else "1-3" if n <= 3 else "4-7" if n <= 7 else "8+"),
                         template_use=("none" if n_tmpl == 0 else "some" if n_tmpl / n < 0.5 else "many"),
                         length=("short" if words < 400 else "medium" if words <= 900 else "long")))
    return pd.DataFrame(rows).set_index("meetingId")


def template_kind(t):
    if t.get("subTasks"):
        return "workflow"
    if not t["task"]["text"].strip():
        return "placeholder"
    return "form" if re.search(r"\[[^\]]+\]", t["task"]["text"]) else "plain"


def _distance(mt, ids):
    """Sum over the 4 labels of the total-variation distance to the whole dataset (0 = identical mix)."""
    h = mt.loc[ids]
    return sum(0.5 * (h[c].value_counts(normalize=True).reindex(mt[c].unique(), fill_value=0)
                      - mt[c].value_counts(normalize=True)).abs().sum() for c in STRATA)


def make_split(data):
    mt = meeting_labels(data)
    firm_of = mt.org_id.to_dict()

    # ---- hold-out
    forced_firms = set(mt.loc[FORCED_DEV, "org_id"])
    firm_size = mt.groupby("org_id").size()
    eligible = [f for f in sorted(mt.org_id.unique()) if f not in forced_firms]
    rng, best = random.Random(SEED), None
    for _ in range(N_TRIES):
        order = eligible[:]; rng.shuffle(order)
        u_firms, n = [], 0
        for f in order:
            if n + firm_size[f] > N_UNSEEN_TARGET + 3: continue
            u_firms.append(f); n += firm_size[f]
            if n >= N_UNSEEN_TARGET - 3: break
        unseen = list(mt.index[mt.org_id.isin(u_firms)])
        pool = [m for m in mt.index if firm_of[m] not in u_firms and m not in FORCED_DEV]; rng.shuffle(pool)
        seen, used = [], set()
        for m in pool:
            if len(seen) == N_HOLDOUT - len(unseen): break
            if firm_of[m] in used: continue
            seen.append(m); used.add(firm_of[m])
        hold = unseen + seen
        if len(hold) != N_HOLDOUT: continue
        d = _distance(mt, hold)
        if best is None or d < best[0]:
            best = (d, u_firms, unseen, seen)
    _, u_firms, h_unseen, h_seen = best
    holdout = sorted(h_unseen + h_seen, key=int)
    dev = sorted(set(mt.index) - set(holdout), key=int)

    # ---- validation (from dev), also matching the ticket template-type mix
    kind_counts = pd.DataFrame([Counter(template_kind(next(x for x in data[m]["templates"] if x["id"] == t["templateId"]))
                                        if t["templateId"] else "freeform" for t in data[m]["reference"]["tickets"])
                                for m in mt.index], index=mt.index).reindex(columns=KINDS).fillna(0)
    all_share = kind_counts.sum() / kind_counts.sum().sum()

    def kind_distance(ids):
        c = kind_counts.loc[ids].sum()
        return 0.5 * float((c / c.sum() - all_share).abs().sum())

    dev_mt = mt.loc[dev]
    pool_firms = [f for f in sorted({firm_of[m] for m in dev}) if f not in {firm_of[m] for m in FORCED_DEV}]
    rng, best_v = random.Random(SEED + 1), None
    for _ in range(N_TRIES_VAL):
        order = pool_firms[:]; rng.shuffle(order)
        v_firms, n = [], 0
        for f in order:
            size = int((dev_mt.org_id == f).sum())
            if n + size > VAL_UNSEEN_TARGET + 3: continue
            v_firms.append(f); n += size
            if n >= VAL_UNSEEN_TARGET - 3: break
        v_unseen = list(dev_mt.index[dev_mt.org_id.isin(v_firms)])
        pool = [m for m in dev if firm_of[m] not in v_firms and m not in FORCED_DEV]; rng.shuffle(pool)
        v_seen, used = [], set()
        for m in pool:
            if len(v_seen) == VAL_SIZE - len(v_unseen): break
            if firm_of[m] in used: continue
            v_seen.append(m); used.add(firm_of[m])
        val = v_unseen + v_seen
        if len(val) != VAL_SIZE: continue
        score = _distance(mt, val) + kind_distance(val)
        if best_v is None or score < best_v[0]:
            best_v = (score, v_firms, v_unseen, v_seen)
    _, v_firms, v_unseen, v_seen = best_v
    validation = sorted(v_unseen + v_seen, key=int)
    tune = sorted(set(dev) - set(validation), key=int)

    # ---- safety checks
    assert not set(FORCED_DEV) & (set(holdout) | set(validation)), "an already-seen meeting would be in a test set"
    assert not {firm_of[m] for m in dev} & set(u_firms), "an unseen hold-out firm appears in dev"
    assert not {firm_of[m] for m in tune} & set(v_firms), "an unseen validation firm appears in tune"
    return {"seed": SEED, "strata": STRATA, "forced_dev": FORCED_DEV,
            "dev": dev, "holdout": holdout, "holdout_unseen_firm": sorted(h_unseen, key=int),
            "holdout_seen_firm": sorted(h_seen, key=int), "unseen_firm_ids": sorted(u_firms),
            "tune": tune, "validation": validation, "validation_unseen_firm": sorted(v_unseen, key=int),
            "validation_seen_firm": sorted(v_seen, key=int), "validation_unseen_firm_ids": sorted(v_firms)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    new = make_split(load_dataset(a.dataset))
    old = json.loads(SPLIT_FILE.read_text()) if SPLIT_FILE.exists() else {}
    keys = ["holdout", "validation", "tune", "dev"]
    same = {k: old.get(k) == new[k] for k in keys}
    print(f"tune {len(new['tune'])} | validation {len(new['validation'])} | hold-out {len(new['holdout'])} | "
          f"identical to {SPLIT_FILE}: {same}")
    if a.write:
        if old and (old.get("holdout") != new["holdout"] or old.get("validation", new["validation"]) != new["validation"]):
            raise SystemExit("refusing: splits.json has a different hold-out/validation; delete it on purpose to re-split")
        SPLIT_FILE.write_text(json.dumps({**old, **new}, indent=1))
        print("written")


if __name__ == "__main__":
    main()
