"""LoRA fine-tuning on Fireworks: build the training file, upload it, start the jobs, check them.

    python -m nevis.finetune build                     # -> finetune/train.jsonl (+ stats; free)
    python -m nevis.finetune upload                    # create the dataset on Fireworks and upload the file
    python -m nevis.finetune train --epochs 1 2        # one LoRA job per epoch setting
    python -m nevis.finetune status                    # state, progress and Fireworks' cost estimate per job
    python -m nevis.finetune deploy                    # one GPU deployment of the BASE model with add-ons enabled
    python -m nevis.finetune load --epochs 1 2         # load the fine-tuned adapters onto it (base + adapters together)
    python -m nevis.finetune delete                    # delete the deployment (GPU billing stops)

A deployment is billed by GPU time while it exists (H100 $8/h): run the evaluations back to back, then delete it.

Training data = TUNE meetings only, minus tune meetings that are reworded twins of a validation or hold-out meeting
(nevis.data.is_near_copy), so nothing the model is evaluated on is learned. Each example is exactly what the
pipeline sends at inference (system prompt `main` + the user message) and the reference answer as the assistant
turn, so before and after fine-tuning differ only in the weights. Loss is on the assistant turn (Fireworks default).
Needs FIREWORKS_API_KEY; the account comes from FIREWORKS_ACCOUNT (default: nevis-home-task).
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from nevis.data import is_near_copy, load_dataset
from nevis.prompt import PROMPTS, build_user_prompt

API = "https://api.fireworks.ai/v1"
OUT = Path("finetune")
TRAIN_FILE = OUT / "train.jsonl"
DATASET_ID = "nevis-tune-sft"
BASE_MODEL = "accounts/fireworks/models/qwen3p5-9b"     # Qwen3.5 9B: newest <=16B tunable model (cheapest training tier)
LORA_RANK = 8                                             # Fireworks default; small adapter = less room to memorise
DEFAULT_DEPLOYMENT_ID = "nevis-tickets-ft"                  # a deleted deployment's name stays reserved: use a new one
DEPLOYMENT_ID = os.environ.get("FIREWORKS_DEPLOYMENT_ID", DEFAULT_DEPLOYMENT_ID)   # base + adapters on one GPU (no validated shape for this
                                                          # model, so a plain 1 x H100 BF16 deployment; add-ons need BF16)


def account():
    return os.environ.get("FIREWORKS_ACCOUNT", "nevis-home-task")


def call(method, path, body=None, params=None, file=None):
    """One REST call to the Fireworks account API -> (status, parsed JSON or text). `file` = path for a multipart upload."""
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("FIREWORKS_API_KEY is not set")
    url = f"{API}/accounts/{account()}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    headers, data = {"Authorization": f"Bearer {key}"}, None
    if file is not None:
        boundary = uuid.uuid4().hex
        data = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{Path(file).name}\"\r\n"
                f"Content-Type: application/jsonl\r\n\r\n").encode() + Path(file).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            status, raw = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode()
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def training_meetings(data, splits):
    """Tune meetings minus reworded twins of any validation / hold-out meeting."""
    others = [data[m] for m in splits["validation"] + splits["holdout"] if m in data]
    tune = [m for m in splits["tune"] if m in data]
    keep = [m for m in tune if not any(is_near_copy(data[m], o) for o in others)]
    return keep, len(tune) - len(keep)


def example(meeting, prompt="main"):
    answer = {"meetingId": meeting["meetingId"], "tickets": meeting["reference"]["tickets"]}
    return {"messages": [{"role": "system", "content": PROMPTS[prompt]},
                         {"role": "user", "content": build_user_prompt(meeting)},
                         {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]}


def build(data_dir="dataset", splits_file="splits.json"):
    data, splits = load_dataset(data_dir), json.loads(Path(splits_file).read_text())
    keep, dropped = training_meetings(data, splits)
    OUT.mkdir(exist_ok=True)
    rows = [example(data[m]) for m in keep]
    TRAIN_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    chars = [sum(len(x["content"]) for x in r["messages"]) for r in rows]
    stats = dict(meetings=len(keep), twins_dropped=dropped, tickets=sum(len(data[m]["reference"]["tickets"]) for m in keep),
                 empty_meetings=sum(not data[m]["reference"]["tickets"] for m in keep), chars=sum(chars),
                 est_tokens_per_epoch=[round(sum(chars) / 4.2), round(sum(chars) / 3.3)],
                 longest_example_est_tokens=round(max(chars) / 3.3))
    (OUT / "train_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), f"\n-> {TRAIN_FILE}")
    return stats


def upload():
    n = sum(1 for _ in TRAIN_FILE.open())
    st, r = call("POST", "/datasets", {"datasetId": DATASET_ID, "dataset": {"exampleCount": str(n), "format": "CHAT",
                                                                          "displayName": "Nevis tune meetings (SFT)"}})
    if st >= 400 and "already exists" not in str(r).lower():
        raise SystemExit(f"create dataset failed: {st} {str(r)[:300]}")
    st, r = call("POST", f"/datasets/{DATASET_ID}:upload", file=TRAIN_FILE)
    if st >= 400:
        raise SystemExit(f"upload failed: {st} {str(r)[:300]}")
    print(call("GET", f"/datasets/{DATASET_ID}")[1])


def train(epochs, tag=""):
    jobs = []
    for e in epochs:
        job_id = f"qwen35-9b-tickets-e{e}{tag}"
        body = {"dataset": f"accounts/{account()}/datasets/{DATASET_ID}", "baseModel": BASE_MODEL,
                "outputModel": f"accounts/{account()}/models/{job_id}", "displayName": f"Qwen3.5 9B tickets, {e} epoch(s)",
                "epochs": e, "loraRank": LORA_RANK, "evalAutoCarveout": True}
        st, r = call("POST", "/supervisedFineTuningJobs", body, params={"supervisedFineTuningJobId": job_id})
        if st >= 400:
            raise SystemExit(f"create job {job_id} failed: {st} {str(r)[:400]}")
        jobs.append(r)
        print(job_id, r.get("state"), "estimated cost:", r.get("estimatedCost"))
    return jobs


def status():
    for j in call("GET", "/supervisedFineTuningJobs")[1].get("supervisedFineTuningJobs", []):
        print(j["name"].split("/")[-1], j.get("state"), "| progress", j.get("jobProgress"), "| estimated cost",
              j.get("estimatedCost"), "| output", j.get("outputModel"))


def deployment_name():
    return f"accounts/{account()}/deployments/{DEPLOYMENT_ID}"


def deploy():
    body = {"baseModel": BASE_MODEL, "displayName": "qwen35-9b eval (base + LoRA)", "enableAddons": True,
            "acceleratorType": "NVIDIA_H100_80GB", "acceleratorCount": 1, "precision": "BF16",
            "minReplicaCount": 1, "maxReplicaCount": 1,
            "autoscalingPolicy": {"scaleToZeroWindow": "300s", "scaleDownWindow": "300s"}}
    st, r = call("POST", "/deployments", body, params={"deploymentId": DEPLOYMENT_ID, "acceptShapelessRisk": "true"})
    if st >= 400:
        raise SystemExit(f"create deployment {DEPLOYMENT_ID} failed: {st} {str(r)[:300]}"
                         + (" (this name is taken or reserved: set another FIREWORKS_DEPLOYMENT_ID)" if st == 409 else ""))
    print(f"deployment {DEPLOYMENT_ID} created; GPU billing has started. Waiting until it is READY (~15 min)...")
    wait_for(lambda: call("GET", f"/deployments/{DEPLOYMENT_ID}")[1].get("state"), "READY", ("FAILED", "DELETED"))


def wait_for(get_state, ready, failed=(), every=20, limit_s=3600):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        s = get_state()
        print(f"  {time.strftime('%H:%M:%S')} {s}", flush=True)
        if s == ready:
            return
        if s in failed:
            raise SystemExit(f"state {s}")
        time.sleep(every)
    raise SystemExit(f"not {ready} after {limit_s} s; check the Fireworks UI (and delete the deployment if you stop)")


def load(epochs, tag=""):
    for e in epochs:
        model = f"accounts/{account()}/models/qwen35-9b-tickets-e{e}{tag}"
        st, r = call("POST", "/deployedModels", {"model": model, "deployment": deployment_name(), "default": False})
        if st >= 400:
            raise SystemExit(f"load {model} failed: {st} {str(r)[:300]}")
        print(f"loading {model.split('/')[-1]} onto {DEPLOYMENT_ID} (~10 min)...")
        def state(model=model):
            dms = call("GET", "/deployedModels")[1].get("deployedModels", [])
            return next((d.get("state") for d in dms if d.get("model") == model and d.get("deployment") == deployment_name()), None)
        wait_for(state, "DEPLOYED")


def delete():
    # ignoreChecks: Fireworks refuses to delete a deployment that served requests in the last hour otherwise
    st, r = call("DELETE", f"/deployments/{DEPLOYMENT_ID}", params={"ignoreChecks": "true"})
    print("delete:", st, str(r)[:200])
    print("deployments left:", [(d["name"], d.get("state")) for d in call("GET", "/deployments")[1].get("deployments", [])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "upload", "train", "status", "deploy", "load", "delete"])
    ap.add_argument("--epochs", type=int, nargs="*", default=[1])
    ap.add_argument("--tag", default="", help="suffix for job/model names (e.g. a retry after a failed job)")
    a = ap.parse_args()
    {"build": build, "upload": upload, "status": status, "deploy": deploy, "delete": delete,
     "load": lambda: load(a.epochs, a.tag)}.get(a.command, lambda: train(a.epochs, a.tag))()


if __name__ == "__main__":
    main()
