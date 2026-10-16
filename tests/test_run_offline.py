"""Offline tests for the configuration-driven pipeline (nevis/run.py, nevis/sweep.py). No network, no API keys:
the model clients are fakes that answer with the reference tickets of the meeting they are asked about.

    .venv/bin/python -m pytest -q tests
"""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

from nevis import experiments, run as R, sweep  # noqa: E402
from nevis.data import load_dataset  # noqa: E402
from nevis.prompt import PROMPTS, TICKETS_SCHEMA, followups_from_tickets, followups_schema, parse_json  # noqa: E402
from nevis.runmeta import PRICES, cost_usd  # noqa: E402

FW_MODEL = "accounts/fireworks/models/gpt-oss-120b"


@pytest.fixture(scope="session")
def data():
    if not (ROOT / "dataset").is_dir():
        pytest.skip("needs the home-task dataset in ./dataset")
    return load_dataset(ROOT / "dataset")


@pytest.fixture(scope="session")
def splits():
    return json.loads((ROOT / "splits.json").read_text())


def cfg(**kw):
    base = dict(name="test_cfg", provider="fireworks", model=FW_MODEL, prompt="main", pipeline="single", concurrency=2,
                retries=0)
    return R.load_config({**base, **kw})


# ---------------------------------------------------------------------------- fake clients
def _meeting_id(messages):
    return re.search(r"meetingId: (\S+)", messages[-1]["content"]).group(1)


def _answer(system, messages, data):
    mid = _meeting_id(messages)
    tickets = data[mid]["reference"]["tickets"]
    if "step 1 of 2" in system:
        return json.dumps({"meetingId": mid, "followups": followups_from_tickets(tickets)})
    return json.dumps({"meetingId": mid, "tickets": tickets})


class FakeFireworks:
    """Mimics openai.OpenAI(...).chat.completions.create."""

    def __init__(self, data, reject_schema=False):
        self.data, self.calls, self.reject_schema = data, [], reject_schema
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if self.reject_schema and kw.get("response_format", {}).get("type") == "json_schema":
            e = Exception("response_format json_schema not supported")
            e.status_code = 400
            raise e
        system = kw["messages"][0]["content"]
        text = _answer(system, kw["messages"][1:], self.data)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=400))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
                               usage=usage)


def run_on(tmp_path, c, ids, data, splits, client, **kw):
    out = tmp_path / c["name"]
    res = R.run(c, ids, out, data, splits, client=client, log=lambda *_: None, **kw)
    return out, res


# ---------------------------------------------------------------------------- configs
def test_all_example_configs_load():
    files = sorted(f for f in (ROOT / "configs").glob("*.yaml") if f.name != "grid.yaml")   # grid.yaml = list of experiments
    assert len(files) >= 6
    seen = {k: set() for k in ["provider", "deployment", "pipeline", "few_shot"]}
    for f in files:
        c = R.load_config(f)
        for k in seen:
            seen[k].add(c[k])
    assert seen["provider"] == {"fireworks"}
    assert seen["deployment"] == {"serverless", "dedicated"}
    assert seen["pipeline"] == {"single", "two_step"}
    assert {"none", "same_firm"} <= seen["few_shot"]


def test_config_validation():
    with pytest.raises(ValueError):
        cfg(unknown_key=1)
    with pytest.raises(ValueError):
        cfg(pipeline="two_step", prompt="main")          # two-step needs a two-step prompt set
    with pytest.raises(ValueError):
        cfg(deployment="dedicated")                    # no deployment id
    with pytest.raises(ValueError):
        cfg(few_shot="random")


def test_prompt_unchanged():
    # the hash recorded with every run (meta.json); if the prompt text changes, results are no longer comparable
    assert R.prompt_version(cfg()) == "main-e2d185a2"


# ---------------------------------------------------------------------------- request shapes
def test_fireworks_structured_output_request(data, splits, tmp_path):
    client = FakeFireworks(data)
    run_on(tmp_path, cfg(), ["15"], data, splits, client)
    kw = client.calls[0]
    assert kw["model"] == FW_MODEL
    assert kw["response_format"] == {"type": "json_schema", "json_schema": {"name": "tickets", "schema": TICKETS_SCHEMA}}
    assert kw["messages"][0] == {"role": "system", "content": PROMPTS["main"]}
    assert kw["messages"][-1]["role"] == "user" and "meetingId: 15" in kw["messages"][-1]["content"]
    assert kw["temperature"] == 0.0 and kw["max_tokens"] == 32000


def test_fireworks_json_mode_and_extra_body(data, splits, tmp_path):
    client = FakeFireworks(data)
    run_on(tmp_path, cfg(structured_output="json_object", extra_body={"reasoning_effort": "low"}), ["15"], data, splits, client)
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["extra_body"] == {"reasoning_effort": "low"}
    client = FakeFireworks(data)
    run_on(tmp_path, cfg(name="nofmt", structured_output="none"), ["15"], data, splits, client)
    assert "response_format" not in client.calls[0]


def test_fireworks_schema_rejected_falls_back_to_json_mode(data, splits, tmp_path):
    client = FakeFireworks(data, reject_schema=True)
    out, res = run_on(tmp_path, cfg(retries=1), ["15", "18"], data, splits, client)
    assert not res["errors"]
    assert any(c.get("response_format", {}).get("type") == "json_object" for c in client.calls)
    assert json.loads((out / "15" / "meta.json").read_text())["structured_output"] == "json_object"
    assert json.loads((out / "_run.json").read_text())["structured_fallback"]


def test_dedicated_deployment_model_id(data, splits, tmp_path):
    dep = "accounts/acme/deployments/abc123"
    c = cfg(deployment="dedicated", deployment_id=dep, gpu_hour_usd=8.0, concurrency=1)
    assert R.api_model(c) == dep
    client = FakeFireworks(data)
    out, _ = run_on(tmp_path, c, ["15"], data, splits, client)
    assert client.calls[0]["model"] == dep
    meta = json.loads((out / "15" / "meta.json").read_text())
    assert meta["deployment"] == "dedicated" and meta["api_model"] == dep and meta["cost_basis"] == "gpu_time_estimate"
    assert meta["cost_usd"] == pytest.approx(meta["latency_s"] * 8.0 / 3600, rel=0.05)
    # the 'model#deployment' form written directly in `model` is passed through unchanged
    c2 = cfg(deployment="dedicated", model=FW_MODEL + "#" + dep)
    assert R.api_model(c2) == FW_MODEL + "#" + dep
    with pytest.raises(ValueError):
        R.api_model(cfg(deployment="dedicated", deployment_id="accounts/<YOUR_ACCOUNT>/deployments/<ID>"))


def test_fine_tuned_model_recorded(data, splits, tmp_path):
    c = cfg(model="accounts/acme/models/nevis-ft-1", fine_tuned=True, deployment="dedicated",
            deployment_id="accounts/acme/deployments/ft1")
    out, _ = run_on(tmp_path, c, ["15"], data, splits, FakeFireworks(data))
    meta = json.loads((out / "15" / "meta.json").read_text())
    assert meta["fine_tuned"] is True and meta["model"] == "accounts/acme/models/nevis-ft-1"


# ---------------------------------------------------------------------------- pipeline shapes
def test_single_call_pipeline(data, splits, tmp_path):
    client = FakeFireworks(data)
    out, res = run_on(tmp_path, cfg(), ["15", "18"], data, splits, client)
    assert res == {"done": 2, "skipped": 0, "errors": {}}
    assert len(client.calls) == 2                       # one call per meeting
    t = json.loads((out / "15" / "tickets.json").read_text())
    assert t["meetingId"] == "15" and len(t["tickets"]) == len(data["15"]["reference"]["tickets"])


def test_two_step_pipeline(data, splits, tmp_path):
    client = FakeFireworks(data)
    c = cfg(pipeline="two_step", prompt="two_step")
    out, res = run_on(tmp_path, c, ["15", "46"], data, splits, client)   # 46: a meeting with no reference tickets
    assert not res["errors"]
    by_meeting = {}
    for kw in client.calls:
        by_meeting.setdefault(_meeting_id(kw["messages"][1:]), []).append(kw)
    s1, s2 = by_meeting["15"]
    assert s1["response_format"]["json_schema"] == {"name": "followups", "schema": followups_schema(("evidence", "action"))}
    assert s1["messages"][0]["content"] == PROMPTS["two_step_find"]
    assert s2["response_format"]["json_schema"]["name"] == "tickets"
    assert s2["messages"][0]["content"] == PROMPTS["two_step_write"]
    assert "Follow-ups found in step 1" in s2["messages"][-1]["content"]
    assert len(by_meeting["46"]) == 1                    # no follow-ups -> no step 2
    ref = data["15"]["reference"]["tickets"]
    got = json.loads((out / "15" / "tickets.json").read_text())["tickets"]
    assert [x["templateId"] for x in got] == [x["templateId"] for x in ref]
    assert [x["dueDate"] for x in got] == [x["dueDate"] for x in ref]
    assert json.loads((out / "46" / "tickets.json").read_text())["tickets"] == []
    meta = json.loads((out / "15" / "meta.json").read_text())
    assert meta["calls"] == 2 and meta["pipeline"] == "two_step" and meta["input_tokens"] == 2 * 600
    assert meta["prompt_version"].startswith("two_step-")


def test_two_step_variants_who_picks_the_template(data, splits, tmp_path):
    # step 2 answers with every template removed: two_step keeps step 2's choice, two_step_tmpl restores step 1's
    class NoTemplateInStep2(FakeFireworks):
        def create(self, **kw):
            r = super().create(**kw)
            if "step 2 of 2" in kw["messages"][0]["content"]:
                a = json.loads(r.choices[0].message.content)
                for t in a["tickets"]:
                    t["templateId"] = None; t["subTasks"] = []
                r.choices[0].message.content = json.dumps(a)
            return r
    ref = [x["templateId"] for x in data["15"]["reference"]["tickets"]]
    assert any(ref)
    got = {}
    for name in ["two_step", "two_step_tmpl"]:
        out, _ = run_on(tmp_path, cfg(name=name, pipeline="two_step", prompt=name), ["15"], data, splits, NoTemplateInStep2(data))
        got[name] = [x["templateId"] for x in json.loads((out / "15" / "tickets.json").read_text())["tickets"]]
    assert got["two_step"] == [None] * len(ref) and got["two_step_tmpl"] == ref
    assert "templateId" not in PROMPTS["two_step_find"] and "templateId" in PROMPTS["two_step_tmpl_find"]


# ---------------------------------------------------------------------------- few-shot selection (no leakage)
def test_few_shot_never_leaks(data, splits):
    tune, val, hold = set(splits["tune"]), set(splits["validation"]), set(splits["holdout"])
    sample = splits["validation"] + splits["holdout"] + splits["tune"][:60]
    for mode in ("global", "same_firm"):
        for mid in sample:
            ex = R.select_examples(mid, data[mid]["org"]["id"], data, splits, mode, 3)
            assert len(ex) == 3
            assert mid not in ex
            assert set(ex) <= tune
            assert not set(ex) & (val | hold)


def test_few_shot_same_firm_and_fallback(data, splits):
    tune = splits["tune"]
    org_of = {m: data[m]["org"]["id"] for m in data}
    tune_orgs = {org_of[m] for m in tune}
    # a validation meeting whose firm has tune meetings -> the examples are that firm's tune meetings
    seen = next(m for m in splits["validation"] if sum(org_of[t] == org_of[m] for t in tune) >= 2)
    ex = R.select_examples(seen, org_of[seen], data, splits, "same_firm", 2)
    assert all(org_of[e] == org_of[seen] for e in ex)
    # a firm with no tune meetings -> global tune examples
    unseen = next(m for m in splits["validation"] if org_of[m] not in tune_orgs)
    ex = R.select_examples(unseen, org_of[unseen], data, splits, "same_firm", 2)
    assert ex == R.select_examples(unseen, org_of[unseen], data, splits, "global", 2)
    # a tune meeting is never its own example, even when it is the firm's first tune meeting
    t0 = sorted(tune, key=int)[0]
    assert t0 not in R.select_examples(t0, org_of[t0], data, splits, "same_firm", 5)
    assert R.select_examples(t0, org_of[t0], data, splits, "none", 5) == []


def test_few_shot_messages_sent(data, splits, tmp_path):
    client = FakeFireworks(data)
    c = cfg(few_shot="same_firm", few_shot_k=2)
    mid = splits["validation"][0]
    out, _ = run_on(tmp_path, c, [mid], data, splits, client)
    msgs = client.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    ex_ids = [re.search(r"meetingId: (\S+)", m["content"]).group(1) for m in msgs if m["role"] == "user"][:-1]
    assert set(ex_ids) <= set(splits["tune"]) and mid not in ex_ids
    assert json.loads(msgs[2]["content"])["tickets"] == data[ex_ids[0]]["reference"]["tickets"]
    assert json.loads((out / mid / "meta.json").read_text())["few_shot_examples"] == ex_ids


# ---------------------------------------------------------------------------- resumability and meta.json
def test_resumable(data, splits, tmp_path):
    c = cfg()
    client = FakeFireworks(data)
    out, _ = run_on(tmp_path, c, ["15", "18", "19"], data, splits, client)
    assert len(client.calls) == 3
    client2 = FakeFireworks(data)
    _, res = run_on(tmp_path, c, ["15", "18", "19"], data, splits, client2)
    assert res["skipped"] == 3 and client2.calls == []
    (out / "18" / "tickets.json").unlink()
    client3 = FakeFireworks(data)
    _, res = run_on(tmp_path, c, ["15", "18", "19"], data, splits, client3)
    assert res["done"] == 1 and len(client3.calls) == 1
    with pytest.raises(SystemExit):                      # same folder, different config -> refuse
        R.run(cfg(pipeline="two_step", prompt="two_step"), ["15"], out, data, splits, client=FakeFireworks(data), log=lambda *_: None)


def test_failed_meeting_is_retried_next_time(data, splits, tmp_path):
    class Broken(FakeFireworks):
        def create(self, **kw):
            raise ConnectionError("network down")
    out, res = run_on(tmp_path, cfg(), ["15"], data, splits, Broken(data))
    assert "15" in res["errors"] and not (out / "15" / "tickets.json").exists()
    _, res = run_on(tmp_path, cfg(), ["15"], data, splits, FakeFireworks(data))
    assert res["done"] == 1


def test_meta_contents(data, splits, tmp_path):
    out, _ = run_on(tmp_path, cfg(few_shot="global", few_shot_k=1), ["15"], data, splits, FakeFireworks(data))
    meta = json.loads((out / "15" / "meta.json").read_text())
    for k in ["model", "latency_s", "input_tokens", "output_tokens", "cache_read_tokens", "cost_usd", "config", "provider",
              "deployment", "api_model", "prompt_version", "pipeline", "few_shot", "few_shot_examples", "calls"]:
        assert k in meta, k
    assert meta["config"] == "test_cfg" and meta["provider"] == "fireworks" and meta["prompt_version"] == "main-e2d185a2"
    assert meta["pipeline"] == "single" and meta["few_shot"] == "global" and len(meta["few_shot_examples"]) == 1
    assert meta["input_tokens"] == 600 and meta["cache_read_tokens"] == 400 and meta["latency_s"] > 0
    p = PRICES[FW_MODEL]
    assert meta["cost_usd"] == pytest.approx((600 * p["input"] + 400 * p["cache_read"] + 200 * p["output"]) / 1e6)
    raw = json.loads((out / "15" / "raw.json").read_text())
    assert raw["calls"][0]["step"] == "single"
    st = json.loads((out / "_run.json").read_text())
    assert st["config"]["name"] == "test_cfg" and st["meetings"] == ["15"]


def test_parse_json_tolerates_reasoning_and_fences():
    t = '<think>maybe {"x": 1}</think>\n```json\n{"meetingId": "1", "tickets": []}\n```'
    assert parse_json(t) == {"meetingId": "1", "tickets": []}
    assert parse_json('{"meetingId": "1", "followups": [{"evidence": "a" "action": "b"}]}', "followups")["followups"]


# ---------------------------------------------------------------------------- scoring, experiment log, frontier
def test_score_and_log(data, splits, tmp_path, monkeypatch):
    # logging a new run adds one row and keeps the rows already in experiments.csv
    old = pd.DataFrame([{c: None for c in experiments.COLS} | {"run": f"old_run_{i}", "scores_dir": f"scores/old_{i}", "north_star": 0.9}
                        for i in range(2)])
    log_csv, log_md = tmp_path / "experiments.csv", tmp_path / "EXPERIMENTS.md"
    old.to_csv(log_csv, index=False)
    monkeypatch.setattr(experiments, "LOG_CSV", log_csv)
    monkeypatch.setattr(experiments, "LOG_MD", log_md)
    monkeypatch.chdir(ROOT)                                       # scorer reads splits.json from the project
    ids = ["15", "18", "46"]
    c = cfg(name="fake_fw", pipeline="two_step", prompt="two_step", few_shot="same_firm")
    out, _ = run_on(tmp_path, c, ids, data, splits, FakeFireworks(data), split_label="tune (test)")
    overall, row = R.score_run(out, ids, tmp_path / "scores" / "fake_fw", data, judge=False, notes="offline")
    assert overall["north_star"] > 0.9                             # answers = reference -> near-perfect
    assert overall["run"]["latency_median_s"] > 0 and overall["run"]["cost_mean_usd"] > 0
    df = pd.read_csv(log_csv)
    assert len(df) == 3 and list(df.columns) == experiments.COLS
    new = df.iloc[-1]
    assert (new.config, new.provider, new.pipeline, new.deployment) == ("fake_fw", "fireworks", "two_step", "serverless")
    assert new.few_shot == "same_firm k=2" and new.split == "tune (test)"
    assert df.iloc[0].run == old.iloc[0].run                      # existing rows kept
    assert "pipeline" in log_md.read_text()

    md, png = sweep.write_frontier(sweep.frontier_table("tune (test)", log_csv=log_csv), "tune (test)",
                                   md=tmp_path / "FRONTIER.md", png=tmp_path / "frontier.png")
    text = md.read_text()
    assert "fake_fw" in text and "latency_p90_s" in text and png.stat().st_size > 1000


def test_grid_experiments_are_valid_configs():
    """Every enabled experiment in configs/grid.yaml must be a valid run config (checked before any money is spent)."""
    from nevis.grid import expand
    exps = expand(ROOT / "configs" / "grid.yaml")
    enabled = [cfg for _, cfg, on in exps if on]
    assert len(enabled) >= 10
    for cfg in enabled:
        R.load_config(dict(cfg))
    assert len({name for name, _, _ in exps}) == len(exps), "experiment names must be unique"


# ---------------------------------------------------------------------------- judge
def test_judge_request_cache_and_parsing(data, tmp_path):
    from nevis.score import JUDGE_PROMPT, make_judge, template_of
    seen = []
    def fake_call(system, user):
        seen.append((system, json.loads(user)))
        return '{"score": 4, "reason": "one fact missing"}', 900, 30
    m = data["15"]
    r = m["reference"]["tickets"][0]
    judge = make_judge(tmp_path / "jc", model="fake-model", call=fake_call)
    assert judge(r, r, m["summary"]) == 4 and judge(r, r, m["summary"]) == 4      # second time from the cache
    assert len(seen) == 1 and seen[0][0] == JUDGE_PROMPT
    assert set(seen[0][1]) == {"reference", "candidate", "summary_excerpt"} and r["evidence"] in seen[0][1]["summary_excerpt"]
    # a workflow ticket: the judge also gets the template text and both tickets' sub-tasks
    wm = next(m for m in data.values() for t in m["reference"]["tickets"] if t["subTasks"] and t["templateId"])
    wt = next(t for t in wm["reference"]["tickets"] if t["subTasks"] and t["templateId"])
    judge(wt, wt, wm["summary"], template_of(wt, wm))
    body = seen[-1][1]
    assert body["template_text"] == template_of(wt, wm)["task"]["text"].strip()
    assert [s["title"] for s in body["candidate"]["subtasks"]] == [s["title"] for s in wt["subTasks"]]
    assert judge.stats["calls"] == 2 and judge.stats["cached"] == 1 and judge.stats["input_tokens"] == 1800
    other = make_judge(tmp_path / "jc", model="other-model", call=lambda *_: ("no score", 1, 1))
    assert other(r, r, m["summary"]) == 0 and other.stats["unparsed"] == 1       # cache is per model; unparsed -> 0


def test_few_shot_never_uses_a_reworded_twin(data, splits):
    # the dataset has reworded variants of the same meeting; one must never be the example for the other
    from nevis.data import is_near_copy
    for m in splits["validation"] + splits["holdout"]:
        for e in R.select_examples(m, data[m]["org"]["id"], data, splits, "same_firm", 2):
            assert not is_near_copy(data[m], data[e]), (m, e)



def test_score_and_run_work_on_a_foreign_folder(data, tmp_path, monkeypatch):
    # Nevis runs the pipeline and the scorer on THEIR folder: ids may collide with ours and our dataset may be absent
    import shutil
    from nevis.data import load_dataset
    from nevis.score import score_folder
    meetings, pred = tmp_path / "meetings", tmp_path / "pred"
    for ours, theirs in [("980", "15"), ("293", "11")]:            # ids that are tune / hold-out ids in OUR split
        shutil.copytree(ROOT / "dataset" / ours, meetings / theirs)
        (pred / theirs).mkdir(parents=True)
        t = json.loads((ROOT / "dataset" / ours / "tickets.json").read_text()); t["meetingId"] = theirs
        (pred / theirs / "tickets.json").write_text(json.dumps(t))
    monkeypatch.chdir(tmp_path)                                    # no dataset/ and no splits.json here
    foreign = load_dataset(meetings)
    overall, pm, *_ = score_folder(pred, meetings, judge=None, data=foreign, split_file=None, n_boot=0)
    assert overall["north_star"] == pytest.approx(1.0)              # needs the frozen verb list, not dataset/
    assert not pm.twin_in_tune.any() and set(pm.firm) == {"n/a (not our dataset)"}
    c = R.load_config(ROOT / "configs" / "fw_large_single.yaml")
    assert R.select_examples("15", foreign["15"]["org"]["id"], foreign, {}, c["few_shot"], 2) == []
