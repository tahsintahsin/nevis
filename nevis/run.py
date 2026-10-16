"""Run the ticket pipeline over a folder of meetings, driven by one YAML config (configs/*.yaml).

    python -m nevis.run --config configs/fw_large_single.yaml --meetings dataset --split validation --out outputs/<run>
    python -m nevis.run --config configs/fw_mid_single.yaml --split 15,75,99 --score --judge
    python -m nevis.run --config configs/fw_mid_fewshot_samefirm.yaml --split validation --dry-run   # no API call

--split: validation | tune | holdout | all | a comma-separated list of meeting ids ("all" = every folder in --meetings).
For each meeting writes <out>/<meetingId>/tickets.json (after nevis.postprocess), meta.json (tokens, cost, latency,
config) and raw.json (raw model answers). Resumable: meetings that already have tickets.json + meta.json are skipped.
--score then runs the scorer (scores/<run>/report.md) and adds a row to experiments.csv / EXPERIMENTS.md.

Config keys (defaults in DEFAULTS below):
  provider        fireworks
  model           Fireworks model id (accounts/fireworks/models/<name>, or your fine-tuned accounts/<account>/models/<name>)
  deployment      serverless | dedicated  (Fireworks). dedicated -> deployment_id "accounts/<account>/deployments/<id>"
                  is sent as the model; cost = GPU-hours (gpu_hour_usd x gpu_count) instead of tokens
  fine_tuned      true if `model` is a fine-tuned model (bookkeeping only; Fireworks serves it from a dedicated deployment)
  prompt          single: main (nevis/prompt.py SYSTEM); two_step: two_step | two_step_tmpl (nevis/prompt.py TWO_STEP)
  pipeline        single | two_step
  few_shot        none | global | same_firm   (examples = tune meetings only; same_firm falls back to global)
  structured_output  json_schema | json_object | none
"""
import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from nevis.data import is_near_copy, load_dataset, meeting_order
from nevis.postprocess import postprocess
from nevis.prompt import (PROMPTS, TICKETS_SCHEMA, TWO_STEP, build_step2_user_prompt, build_user_prompt,
                          followups_from_tickets, followups_schema, parse_json)
from nevis.runmeta import cost_usd, write_meta

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

DEFAULTS = dict(
    name=None,                     # defaults to the YAML file name
    provider="fireworks",
    model=None,
    deployment="serverless",       # serverless | dedicated
    deployment_id=None,            # dedicated: accounts/<account>/deployments/<id>
    fine_tuned=False,
    price_key=None,                # key in nevis.runmeta.PRICES if it differs from `model`
    gpu_type=None,                 # dedicated: e.g. H100 (bookkeeping)
    gpu_hour_usd=None,             # dedicated: $ per GPU-hour (nevis.runmeta.GPU_HOUR_USD)
    gpu_count=1,
    prompt="main",                 # the system prompt (nevis/prompt.py SYSTEM)
    pipeline="single",             # single | two_step
    few_shot="none",               # none | global | same_firm
    few_shot_k=2,
    few_shot_seed=0,
    structured_output="json_schema",   # json_schema | json_object | none
    temperature=0.0,
    max_tokens=32000,              # output budget INCLUDING hidden reasoning (8k was too small for reasoning models)
    concurrency=8,
    retries=3,
    timeout_s=300,
    extra_body={},                 # passed through to the API (e.g. {"reasoning_effort": "low"})
    notes="",
)
CHOICES = dict(provider={"fireworks"}, deployment={"serverless", "dedicated"},
               pipeline={"single", "two_step"}, few_shot={"none", "global", "same_firm"},
               structured_output={"json_schema", "json_object", "none"})


# ============================================================================ config
def load_config(path_or_dict):
    """YAML file (or dict) -> validated config dict with defaults filled in."""
    if isinstance(path_or_dict, dict):
        raw, stem = dict(path_or_dict), path_or_dict.get("name") or "config"
    else:
        raw, stem = yaml.safe_load(Path(path_or_dict).read_text()) or {}, Path(path_or_dict).stem
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)} (allowed: {sorted(DEFAULTS)})")
    cfg = {**DEFAULTS, **raw}
    cfg["name"] = cfg["name"] or stem
    # fine-tuned configs point at "${FIREWORKS_DEPLOYMENT_ID}": the same deployment `nevis.finetune deploy` creates
    from nevis.finetune import DEFAULT_DEPLOYMENT_ID
    os.environ.setdefault("FIREWORKS_DEPLOYMENT_ID", DEFAULT_DEPLOYMENT_ID)
    for k in ("model", "deployment_id"):
        if isinstance(cfg[k], str):
            cfg[k] = os.path.expandvars(cfg[k])
    for k, allowed in CHOICES.items():
        if cfg[k] not in allowed:
            raise ValueError(f"config {k}={cfg[k]!r}: must be one of {sorted(allowed)}")
    if not cfg["model"]:
        raise ValueError("config needs a `model`")
    if cfg["pipeline"] == "two_step" and cfg["prompt"] not in TWO_STEP:
        raise ValueError(f"two_step pipeline needs a two-step prompt set: {sorted(TWO_STEP)}")
    single = [p for p in PROMPTS if not any(p in (t["find"], t["write"]) for t in TWO_STEP.values())]
    if cfg["pipeline"] == "single" and cfg["prompt"] not in single:
        raise ValueError(f"single pipeline needs a single-call prompt from {single}")
    if cfg["deployment"] == "dedicated" and not (cfg["deployment_id"] or "#" in cfg["model"] or "/deployments/" in cfg["model"]):
        raise ValueError("dedicated deployment needs deployment_id: accounts/<account>/deployments/<id>")
    return cfg


def config_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("notes", "concurrency", "retries", "timeout_s")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def prompt_names(cfg):
    if cfg["pipeline"] == "two_step":
        return [TWO_STEP[cfg["prompt"]]["find"], TWO_STEP[cfg["prompt"]]["write"]]
    return [cfg["prompt"]]


def prompt_version(cfg):
    """'main-e2d185a2' style: prompt name + hash of the exact text (two-step: of both texts)."""
    text = "".join(PROMPTS[n] for n in prompt_names(cfg))
    return f"{cfg['prompt']}-" + hashlib.sha256(text.encode()).hexdigest()[:8]


def api_model(cfg):
    """The `model` string sent to the API.
    Fireworks dedicated deployment: 'accounts/<account>/deployments/<id>' (docs.fireworks.ai/guides/ondemand-deployments);
    the 'accounts/fireworks/models/<model>#<deployment>' form also works if written directly in `model`."""
    if cfg["provider"] == "fireworks" and cfg["deployment"] == "dedicated":
        mid = cfg["deployment_id"] or cfg["model"]
        if "<" in mid:
            raise ValueError(f"placeholder deployment id {mid!r}: create the deployment first (see README, fine-tuned model)")
        return mid
    return cfg["model"]


# ============================================================================ meetings and few-shot examples
def load_splits(splits_file):
    p = Path(splits_file)
    return json.loads(p.read_text()) if p.exists() else {}


def resolve_split(split, data, splits):
    if split == "all":
        return sorted(data, key=meeting_order)
    if split in ("validation", "tune", "holdout"):
        if split not in splits:
            raise ValueError(f"split {split!r} needs splits.json; use --split all for a new folder of meetings")
        ids = [m for m in splits[split] if m in data]
        if split != "holdout":
            assert not set(splits.get("holdout", [])) & set(ids), "hold-out meetings must not be run here"
        return ids
    ids = [x.strip() for x in split.split(",") if x.strip()]
    missing = [m for m in ids if m not in data]
    if missing:
        raise ValueError(f"meetings not found: {missing}")
    return ids


def select_examples(meeting_id, org_id, pool_data, splits, mode, k, seed=0, meeting=None):
    """Few-shot example meeting ids. Rules (no leakage):
    - examples come ONLY from the tune split; never validation or hold-out; never the meeting being processed;
    - never a near-copy of the meeting being processed (a reworded variant of it, see nevis.data.is_near_copy);
    - same_firm: the firm's own tune meetings first (sorted by id), topped up with global examples;
    - global: a fixed seeded sample of tune meetings that have at least one reference ticket (same for every meeting,
      so the prompt prefix can be cached)."""
    if mode == "none" or k <= 0:
        return []
    forbidden = set(splits.get("validation", [])) | set(splits.get("holdout", [])) | {meeting_id}
    me = meeting if meeting is not None else pool_data.get(meeting_id)   # the meeting itself (it may be from another folder)
    tune = sorted((m for m in splits.get("tune", []) if m in pool_data and m not in forbidden
                   and pool_data[m].get("reference") is not None
                   and not (me is not None and is_near_copy(me, pool_data[m]))), key=int)
    with_tickets = [m for m in tune if pool_data[m]["reference"].get("tickets")]
    global_order = random.Random(seed).sample(with_tickets, len(with_tickets))
    chosen = []
    if mode == "same_firm":
        chosen = [m for m in tune if pool_data[m]["org"]["id"] == org_id][:k]
    for m in global_order:
        if len(chosen) >= k:
            break
        if m not in chosen:
            chosen.append(m)
    assert not set(chosen) & forbidden, "few-shot leakage"
    return chosen


def example_messages(example_ids, pool_data, step, fields=None):
    """user/assistant pairs: the example meeting's prompt and its reference answer, in the format of this step
    (fields = what step 1 returns in a two-step set)."""
    msgs = []
    for m in example_ids:
        ex = pool_data[m]
        tickets = ex["reference"]["tickets"]
        if step == "step1":
            user, answer = build_user_prompt(ex), {"meetingId": m, "followups": followups_from_tickets(tickets, fields)}
        elif step == "step2":
            user, answer = build_step2_user_prompt(ex, followups_from_tickets(tickets, fields)), {"meetingId": m, "tickets": tickets}
        else:
            user, answer = build_user_prompt(ex), {"meetingId": m, "tickets": tickets}
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]
    return msgs


# ============================================================================ API calls
def make_client(cfg):
    import openai
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        sys.exit("FIREWORKS_API_KEY is not set (see README, setup)")
    return openai.OpenAI(api_key=key, base_url=FIREWORKS_BASE_URL, timeout=cfg["timeout_s"])


def fireworks_request(cfg, system, messages, schema, schema_name, structured=None):
    """Chat-completions request for Fireworks (OpenAI-compatible).
    Structured output (docs.fireworks.ai/structured-responses/structured-response-formatting):
      {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}  or  {"type": "json_object"}.
    The schema is also described in the system prompt, as the docs recommend."""
    structured = structured or cfg["structured_output"]
    req = dict(model=api_model(cfg), messages=[{"role": "system", "content": system}] + messages,
               max_tokens=cfg["max_tokens"], temperature=cfg["temperature"])
    if structured == "json_schema":
        req["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}}
    elif structured == "json_object":
        req["response_format"] = {"type": "json_object"}
    if cfg["extra_body"]:
        req["extra_body"] = dict(cfg["extra_body"])
    return req


def _is_bad_request(e):
    return getattr(e, "status_code", None) == 400 or type(e).__name__ == "BadRequestError"


def call_model(cfg, client, system, messages, schema, schema_name, state):
    """One model call with retries -> (text, usage dict, stop reason).
    If Fireworks rejects the JSON schema (HTTP 400), fall back to JSON mode for the rest of the run;
    the answer is then parsed by the tolerant parse_json either way."""
    last = None
    for attempt in range(cfg["retries"] + 1):
        structured = state.get("structured", cfg["structured_output"])
        try:
            r = client.chat.completions.create(**fireworks_request(cfg, system, messages, schema, schema_name, structured))
            u = r.usage
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None) or 0
            usage = {"input_tokens": u.prompt_tokens - cached, "output_tokens": u.completion_tokens,
                     "cache_write_tokens": 0, "cache_read_tokens": cached}
            return r.choices[0].message.content or "", usage, r.choices[0].finish_reason
        except Exception as e:                      # network / rate limit / server errors -> retry
            last = e
            if structured == "json_schema" and _is_bad_request(e):
                state["structured"] = "json_object"
                state["structured_fallback"] = f"{type(e).__name__}: {str(e)[:200]}"
                continue
            if _is_bad_request(e) or attempt == cfg["retries"]:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"model call failed after retries: {type(last).__name__}: {last}")


# ============================================================================ one meeting
def _add(total, usage):
    for k, v in usage.items():
        total[k] = total.get(k, 0) + v


def run_meeting(cfg, client, meeting, pool_data, splits, out_dir, state):
    """Run the configured pipeline on one meeting; write tickets.json, meta.json, raw.json. Returns meta."""
    mid = meeting["meetingId"]
    examples = select_examples(mid, meeting["org"]["id"], pool_data, splits, cfg["few_shot"], cfg["few_shot_k"],
                               cfg["few_shot_seed"], meeting=meeting)
    usage, raw, stops = {}, [], []
    t0 = time.perf_counter()                         # latency = first call start -> final tickets
    if cfg["pipeline"] == "single":
        msgs = example_messages(examples, pool_data, "single") + [{"role": "user", "content": build_user_prompt(meeting)}]
        text, u, stop = call_model(cfg, client, PROMPTS[cfg["prompt"]], msgs, TICKETS_SCHEMA, "tickets", state)
        _add(usage, u); stops.append(stop); raw.append({"step": "single", "text": text, "usage": u, "stop": stop})
        answer_text = text
    else:
        spec = TWO_STEP[cfg["prompt"]]
        msgs = (example_messages(examples, pool_data, "step1", spec["fields"])
                + [{"role": "user", "content": build_user_prompt(meeting)}])
        text, u, stop = call_model(cfg, client, PROMPTS[spec["find"]], msgs, followups_schema(spec["fields"]), "followups", state)
        _add(usage, u); stops.append(stop); raw.append({"step": "step1", "text": text, "usage": u, "stop": stop})
        try:
            followups = [{k: f.get(k) for k in spec["fields"]}
                         for f in parse_json(text, "followups").get("followups") or [] if isinstance(f, dict)]
        except Exception as e:
            followups, raw[-1]["parse_error"] = [], str(e)
        if followups:
            msgs = (example_messages(examples, pool_data, "step2", spec["fields"])
                    + [{"role": "user", "content": build_step2_user_prompt(meeting, followups)}])
            text, u, stop = call_model(cfg, client, PROMPTS[spec["write"]], msgs, TICKETS_SCHEMA, "tickets", state)
            _add(usage, u); stops.append(stop); raw.append({"step": "step2", "text": text, "usage": u, "stop": stop})
            answer_text = text
        else:
            answer_text = json.dumps({"meetingId": mid, "tickets": []})
    try:
        parsed = parse_json(answer_text)
        parsed["meetingId"] = mid
        if cfg["pipeline"] == "two_step" and len(parsed.get("tickets") or []) == len(followups):
            for t, f in zip(parsed["tickets"], followups):     # step 1 decided these; step 2 may not change them
                if isinstance(t, dict):
                    t.update({k: f.get(k) for k in spec["locked"]})
        parsed = postprocess(parsed, meeting)
    except Exception as e:               # unparsable answer -> empty list (the scorer counts it as missing everything)
        parsed = {"meetingId": mid, "tickets": []}   # schema-exact; the error is kept in raw.json/meta.json
    latency = time.perf_counter() - t0

    d = Path(out_dir) / mid
    d.mkdir(parents=True, exist_ok=True)
    pver = prompt_version(cfg)
    (d / "raw.json").write_text(json.dumps({"config": cfg["name"], "prompt_version": pver, "few_shot_examples": examples,
                                            "calls": raw}, indent=1, ensure_ascii=False))
    (d / "tickets.json").write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
    cost, basis = None, "tokens"
    if cfg["deployment"] == "dedicated":
        basis = "gpu_time_estimate"      # share of GPU time while this meeting ran (concurrent meetings share the GPUs)
        if cfg["gpu_hour_usd"]:
            cost = latency * cfg["gpu_hour_usd"] * cfg["gpu_count"] / 3600 / max(1, cfg["concurrency"])
    meta_model = cfg["model"]
    return write_meta(d, meta_model, round(latency, 6), usage, price_key=cfg["price_key"], cost=cost,
                      cost_basis=basis, token_cost_usd=cost_usd(usage, cfg["price_key"] or meta_model),
                      config=cfg["name"], provider=cfg["provider"], deployment=cfg["deployment"], api_model=api_model(cfg),
                      fine_tuned=cfg["fine_tuned"], prompt_version=pver, pipeline=cfg["pipeline"], few_shot=cfg["few_shot"],
                      few_shot_examples=examples, structured_output=state.get("structured", cfg["structured_output"]),
                      calls=len(raw), stop_reasons=stops, effort=(cfg["extra_body"] or {}).get("reasoning_effort"))


# ============================================================================ a whole run
def is_done(out_dir, mid):
    d = Path(out_dir) / mid
    return (d / "tickets.json").exists() and (d / "meta.json").exists()


def run(cfg, ids, out_dir, data, splits, pool_data=None, client=None, split_label="", force=False, log=print):
    """Run cfg on meeting ids; skip finished meetings. Returns {'done': n, 'skipped': n, 'errors': {...}}."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pool_data = pool_data if pool_data is not None else data
    state_f = out / "_run.json"
    h = config_hash(cfg)
    if state_f.exists():
        old = json.loads(state_f.read_text())
        if old.get("config_hash") != h and not force:
            raise SystemExit(f"{out} was made with a different config ({old.get('config', {}).get('name')}, "
                             f"hash {old.get('config_hash')} != {h}); use another --out or --force")
    todo = [m for m in ids if not is_done(out, m)]
    todo.sort(key=lambda m: (data[m]["org"]["id"], meeting_order(m)))   # firm-grouped: prompt-cache reuse
    st = dict(config=cfg, config_hash=h, prompt_version=prompt_version(cfg), api_model=api_model(cfg), split=split_label,
              meetings=list(ids), started=time.time())
    state_f.write_text(json.dumps(st, indent=1))
    log(f"{cfg['name']}: {len(ids)} meetings, {len(ids) - len(todo)} already done, {len(todo)} to run -> {out}")
    if todo and client is None:
        client = make_client(cfg)
    state, errors, done = {}, {}, 0
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, cfg["concurrency"])) as ex:
        futs = {ex.submit(run_meeting, cfg, client, data[m], pool_data, splits, out, state): m for m in todo}
        for f in as_completed(futs):
            m = futs[f]
            with lock:
                try:
                    meta = f.result()
                    done += 1
                    log(f"  [{done}/{len(todo)}] {m}: {meta['latency_s']:.1f}s, cost {meta['cost_usd']}")
                except Exception as e:      # no tickets.json written -> re-running the command retries it
                    errors[m] = f"{type(e).__name__}: {e}"
                    log(f"  ERROR {m}: {errors[m]}")
    st.update(finished=time.time(), wall_s=round(time.time() - t0, 1), ran=done, skipped=len(ids) - len(todo),
              errors=errors, structured_fallback=state.get("structured_fallback"))
    state_f.write_text(json.dumps(st, indent=1))
    return {"done": done, "skipped": len(ids) - len(todo), "errors": errors}


def score_run(run_dir, ids, scores_dir, data, judge=False, notes="", split_label=None, splits_file="splits.json"):
    """Score a run folder (nevis.score) and log it (nevis.experiments). Judge = open-weight model on Fireworks (paid unless cached)."""
    from nevis import experiments
    from nevis.score import make_judge, score_folder, write_report
    j = make_judge(Path(scores_dir) / "judge_cache") if judge else None
    overall, pm, pairs, errs, slices, invs = score_folder(run_dir, meeting_ids=ids, judge=j, data=data, split_file=splits_file)
    write_report(scores_dir, overall, pm, pairs, errs, slices, invs, j)
    row = experiments.log(run_dir, scores_dir, notes, split_label)
    return overall, row


def dry_run(cfg, meeting, pool_data, splits):
    """Print the first request (no API call)."""
    examples = select_examples(meeting["meetingId"], meeting["org"]["id"], pool_data, splits, cfg["few_shot"],
                               cfg["few_shot_k"], cfg["few_shot_seed"], meeting=meeting)
    step = "single" if cfg["pipeline"] == "single" else "step1"
    system = PROMPTS[prompt_names(cfg)[0]]
    fields = TWO_STEP[cfg["prompt"]]["fields"] if step == "step1" else None
    schema, name = (TICKETS_SCHEMA, "tickets") if step == "single" else (followups_schema(fields), "followups")
    msgs = example_messages(examples, pool_data, step, fields) + [{"role": "user", "content": build_user_prompt(meeting)}]
    build = fireworks_request
    try:
        req = build(cfg, system, msgs, schema, name)
    except ValueError as e:              # placeholder deployment id: show the request anyway
        print(f"WARNING: {e}")
        req = build({**cfg, "deployment": "serverless", "model": cfg["deployment_id"]}, system, msgs, schema, name)
    show = {k: v for k, v in req.items() if k not in ("messages", "system")}
    if "response_format" in show and "json_schema" in show["response_format"]:
        show["response_format"] = {**show["response_format"], "json_schema": {"name": name, "schema": "<schema>"}}
    print(json.dumps({"meeting": meeting["meetingId"], "pipeline": cfg["pipeline"], "prompt_version": prompt_version(cfg),
                      "few_shot_examples": examples, "messages": len(req["messages"]) + (1 if "system" in req else 0),
                      "prompt_chars": len(system) + sum(len(m["content"]) for m in msgs), "request": show}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/<name>.yaml")
    ap.add_argument("--meetings", default="dataset", help="folder with one sub-folder per meeting")
    ap.add_argument("--split", default="all", help="validation | tune | holdout | all | id,id,...")
    ap.add_argument("--out", default=None, help="output folder (default outputs/<config>_<split>)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N meetings (smoke test)")
    ap.add_argument("--splits-file", default="splits.json")
    ap.add_argument("--examples-from", default="dataset", help="dataset whose tune meetings provide few-shot examples")
    ap.add_argument("--score", action="store_true", help="score the run and log it to experiments.csv")
    ap.add_argument("--judge", action="store_true", help="scorer uses the LLM judge on Fireworks (same-task gate and text meaning)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--force", action="store_true", help="reuse an output folder made with a different config")
    ap.add_argument("--dry-run", action="store_true", help="print the first request and exit (no API call)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    data = load_dataset(a.meetings)
    splits = load_splits(a.splits_file)
    if cfg["few_shot"] == "none":
        pool = data                                    # no examples needed (works without our dataset on disk)
    else:
        pool = data if Path(a.examples_from).resolve() == Path(a.meetings).resolve() else load_dataset(a.examples_from)
    ids = resolve_split(a.split, data, splits)
    if a.limit:
        ids = ids[: a.limit]
    split_label = a.split if a.split in ("validation", "tune", "holdout", "all") else f"list ({len(ids)})"
    if a.limit:
        split_label += f" (first {a.limit})"
    if a.dry_run:
        dry_run(cfg, data[ids[0]], pool, splits)
        return
    out = Path(a.out or Path("outputs") / f"{cfg['name']}_{a.split if ',' not in a.split else 'list'}")
    res = run(cfg, ids, out, data, splits, pool_data=pool, split_label=split_label, force=a.force)
    print(f"done {res['done']}, skipped {res['skipped']}, errors {len(res['errors'])} -> {out}")
    if a.score:
        if res["errors"]:
            print("some meetings failed; re-run the same command to retry them before scoring is final")
        own = Path(a.meetings).resolve() == Path("dataset").resolve()      # splits.json describes OUR dataset only
        overall, _ = score_run(out, ids, Path("scores") / out.name, data, a.judge, a.notes or cfg["notes"], split_label,
                               a.splits_file if own else None)
        r = overall.get("run", {})
        print(f"NORTH STAR {overall['north_star']:.3f} | P {overall['precision']:.3f} | R {overall['recall']:.3f} | "
              f"cost/meeting {r.get('cost_mean_usd')} | latency median {r.get('latency_median_s')} s "
              f"-> scores/{out.name}/report.md")


if __name__ == "__main__":
    main()
