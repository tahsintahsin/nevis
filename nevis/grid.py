"""Expand configs/grid.yaml (base + experiments) into runnable configs, and optionally run them.

    python -m nevis.grid --list
    python -m nevis.grid --write                                   # -> configs/generated/<name>.yaml
    python -m nevis.grid --run --split validation --judge          # write + run + score + log (costs money)
    python -m nevis.grid --run --split validation --judge --only fw_mid fw_large

Each experiment = `base` + its own keys. `model:` may name an alias from the `models:` block. Experiments with
`enabled: false` are listed but never run (e.g. they need a deployment id first).
"""
import argparse
from pathlib import Path

import yaml

from nevis.run import load_config

GRID = Path("configs/grid.yaml")
OUT_DIR = Path("configs/generated")


def expand(grid_path=GRID):
    """Return [(name, config dict, enabled)] with base + overrides merged and model aliases resolved."""
    g = yaml.safe_load(Path(grid_path).read_text())
    base, aliases = g.get("base", {}), g.get("models", {})
    out = []
    for e in g["experiments"]:
        e = dict(e)
        enabled = e.pop("enabled", True)
        cfg = {**base, **e}
        cfg["model"] = aliases.get(cfg.get("model"), cfg.get("model"))
        out.append((cfg["name"], cfg, enabled))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=str(GRID))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", nargs="*", help="experiment names to keep")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()

    exps = [x for x in expand(a.grid) if not a.only or x[0] in a.only]
    for name, cfg, enabled in exps:
        load_config(dict(cfg)) if enabled else None       # validate enabled configs now, before any money is spent
        print(f"{'  ' if enabled else '- '}{name:28s} {cfg['model'][-45:]:45s} prompt={cfg.get('prompt')} "
              f"pipeline={cfg.get('pipeline')} few_shot={cfg.get('few_shot')}{'' if enabled else '  (disabled)'}")
    if a.write or a.run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, cfg, enabled in exps:
            p = OUT_DIR / f"{name}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            if enabled:
                paths.append(str(p))
        print(f"wrote {len(exps)} configs to {OUT_DIR}/ ({len(paths)} enabled)")
        if a.run:
            from nevis import sweep
            sweep.main(["--configs", *paths, "--split", a.split] + (["--judge"] if a.judge else []))


if __name__ == "__main__":
    main()
