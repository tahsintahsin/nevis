# Meeting summary → tickets, on open-weight models (Fireworks)

Turns an advisor's meeting summary into the firm's tickets (`tickets.json`, in the task's exact format), and scores
any folder of tickets against reference tickets.

**Configuration to ship: `configs/glm53.yaml`** (GLM 5.3 on Fireworks serverless, reasoning low). **No deployment is
needed**: it runs on Fireworks' serverless API with the key below. Cheaper alternative: `configs/gpt_oss_120b.yaml`.

Design note, with the hold-out results and the choice: **[DESIGN.md](DESIGN.md)**

## 1. Setup

```bash
python3.12 -m venv .venv          # Python 3.11+ (tested with 3.12)
.venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
export FIREWORKS_API_KEY=...          # one key for the pipeline and the scorer's judge
```

## 2. Run the pipeline on a folder of meetings

A folder with one sub-folder per meeting (`summary.md`, `org.json`, `templates.json`; any meeting ids):

```bash
.venv/bin/python -m nevis.run --config configs/glm53.yaml --meetings <your meetings folder> --split all --out outputs/new_firms
```

(`--split all` = every meeting in the folder.)

- Writes `outputs/new_firms/<meetingId>/tickets.json` (exactly the task's schema), plus `meta.json` (tokens, cost,
  latency) and `raw.json` (the model's raw answer).
- Resumable: if a meeting fails (e.g. a rate limit), run the same command again; finished meetings are skipped.
- `--dry-run` prints the first request without calling the API.

## 3. Score against your reference tickets

```bash
.venv/bin/python -m nevis.score --pred outputs/new_firms --ref <your meetings folder with tickets.json> --out scores/new_firms --judge
```

- Writes `scores/new_firms/report.md`: the **north star** (how much of the reference's work each meeting got right,
  averaged over meetings, with a 95% range; explained in DESIGN.md (b)), tickets found / missed / extra, accuracy per
  field, the **guardrails** (values in our tickets found in none of the inputs; check each flag against its summary),
  latency and cost, the errors by type, and scores by kind of meeting. Details per meeting and per ticket are in the
  CSV files next to it (section 4).
- `--judge` uses an open-weight LLM on Fireworks (gpt-oss-120b) to decide whether two tickets are the same task and
  whether the text says the same thing; without it, a rough word-overlap stand-in is used (for quick checks only).
- The scorer works on any folder: it needs only `tickets.json` per meeting in `--pred`, and the meetings with their
  reference `tickets.json` in `--ref`.

## 4. What's in this folder

**Results**

| Path | What it is |
|---|---|
| `outputs/<run>/` | one run (= one configuration on one split). Run name = `<configuration>_<validation or holdout>` |
| `outputs/<run>/<meeting>/tickets.json` | the pipeline's answer for that meeting, exactly as produced |
| `outputs/<run>/<meeting>/raw.json` | the trace: the model's raw answer, tokens and stop reason, before our code fixes |
| `outputs/<run>/<meeting>/meta.json` | tokens, cost, latency, model and settings for that meeting |
| `outputs/<run>/_run.json` | the exact configuration the run used |
| `scores/<run>/report.md` | the score report: north star with 95% range, guardrails, accuracy per field, latency and cost, errors by type, scores by kind of meeting |
| `scores/<run>/summary.json` | all numbers of the report, machine-readable |
| `scores/<run>/per_meeting.csv` | score per meeting |
| `scores/<run>/pairs.csv` | every matched ticket pair: judge rating and each field right / wrong |
| `scores/<run>/errors.csv` | every error, by type: missed ticket, extra ticket, matched ticket with wrong fields, different task |
| `scores/<run>/inventions.csv` | guardrail flags: values in our tickets found in none of the inputs |
| `scores/<run>/slices.csv` | the scores by kind of meeting and ticket (length, firm seen or new, template type, …) |

**Which configuration made which run**

| Run name | Configuration |
|---|---|
| `fw_glm_big_*` | `configs/glm53.yaml` (**shipped**) |
| `fw_minimax_*` | `configs/minimax_m3.yaml` |
| `fw_large_*` | `configs/gpt_oss_120b.yaml` |
| `fw_qwen9b_base_*`, `fw_qwen9b_e1/e2/e3_*` | `configs/fw_qwen9b_*.yaml` (Qwen3.5 9B untuned, fine-tuned 1/2/3 epochs) |
| all other runs (variants) | the experiment of the same name in `configs/grid.yaml`; `_run.json` holds the exact settings |

**Code and documents**

| Path | What it is |
|---|---|
| `nevis/run.py` | the pipeline: runs a configuration over a folder of meetings |
| `nevis/prompt.py`, `nevis/postprocess.py` | what the model is told; the code fixes applied after the model |
| `nevis/score.py` | the scorer (`action_verbs.json`: the reference's ticket style, used by the scorer) |
| `nevis/finetune.py` | fine-tuning and dedicated deployments on Fireworks |
| `nevis/grid.py`, `sweep.py`, `experiments.py`, `analysis.py`, `runmeta.py`, `split.py`, `data.py` | experiment grid, trade-off charts, run log, run comparisons, prices, the train / validation / hold-out split, data loading |
| `configs/` | one YAML file per configuration; `grid.yaml` lists every experiment |
| `tests/` | offline tests; `test_scorer_trust.py` breaks the reference on purpose and checks the scorer reacts correctly |
| `splits.json` | which meeting is in tune / validation / hold-out |
| `reference_issue_meetings.json` | validation meetings where the reference itself is wrong |
| `DESIGN.md` | design note: the pipeline, how closeness is measured, the trade-off, the reference's problems |

## 5. Fine-tuned model (optional)

The LoRA adapters are in the Fireworks account `nevis-home-task` (best: `qwen35-9b-tickets-e3r`, Qwen3.5 9B, 3 epochs;
also `-e1`, `-e2`; training data: dataset `nevis-tune-sft`). Fine-tuned models are served from a dedicated GPU
deployment, billed while it exists (1 x H100, ~$8/hour; start-up ~15 min):

```bash
export FIREWORKS_DEPLOYMENT_ID=nevis-tickets-ft                     # any new name (a deleted deployment's name stays reserved)
.venv/bin/python -m nevis.finetune deploy                           # waits until READY (~15 min); GPU billing starts now
.venv/bin/python -m nevis.finetune load --epochs 3 --tag r          # loads qwen35-9b-tickets-e3r onto it, waits (~10 min)
.venv/bin/python -m nevis.run --config configs/fw_qwen9b_e3.yaml --meetings <your meetings folder> --split all --out outputs/new_firms_ft
.venv/bin/python -m nevis.finetune delete                           # stop GPU billing
```

## 6. Tests (offline, no API calls)

```bash
.venv/bin/python -m pytest -q tests      # tests that need the home-task dataset (./dataset) are skipped without it
```
