"""Per-meeting run metadata, the same for every model/provider.

Any pipeline writes <run>/<meetingId>/meta.json next to tickets.json:
    {"model": "...", "latency_s": 12.3, "input_tokens": 4100, "output_tokens": 2200,
     "cache_read_tokens": 1000, "cache_write_tokens": 0, "cost_usd": 0.061}
The scorer reads these to report latency and cost next to closeness. Missing files are simply skipped.
"""
import json
from pathlib import Path

# $ per 1M tokens: input, output, cache write, cache read. ONE table for every provider.
# Keys are the Fireworks base-model id (configs may override with `price_key`).
PRICES = {
    # ---- Fireworks serverless (pay per token). Source: https://docs.fireworks.ai/serverless/pricing, checked 2026-10-12
    #      ("Standard" tier). Fireworks has no cache-write charge, so cache_write = input. Batch inference = 50% of these.
    #      Ids verified on the model pages fireworks.ai/models/fireworks/<name> unless marked otherwise.
    # large MoE: OpenAI gpt-oss-120b (~117B total / 5.1B active)
    "accounts/fireworks/models/gpt-oss-120b": {"input": 0.15, "output": 0.60, "cache_write": 0.15, "cache_read": 0.015},
    # mid MoE: NVIDIA Nemotron 3.5 Lightning 30B A3B (id from Fireworks model page via search; price from the pricing doc)
    "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b": {"input": 0.05, "output": 0.20, "cache_write": 0.05, "cache_read": 0.01},
    # GLM 5.3 Flash (id = Fireworks quickstart example model)
    "accounts/fireworks/models/glm-5p3-flash": {"input": 0.15, "output": 0.50, "cache_write": 0.15, "cache_read": 0.03},
    # large MoE: GLM 5.3
    "accounts/fireworks/models/glm-5p3": {"input": 1.40, "output": 4.40, "cache_write": 1.40, "cache_read": 0.26},
    # large MoE: DeepSeek V4.1 Flash (552B MoE)
    "accounts/fireworks/models/deepseek-v4p1-flash": {"input": 0.30, "output": 1.20, "cache_write": 0.30, "cache_read": 0.006},
    # MiniMax M3 (id from the fireworks.ai/models list slug)
    "accounts/fireworks/models/minimax-m3": {"input": 0.30, "output": 1.20, "cache_write": 0.30, "cache_read": 0.06},
    # Added 2026-10-14 (pricing doc + /v1/models list of this account; open weights checked via each model's HF link)
    # NVIDIA Nemotron 3 Ultra 550B-A55B NVFP4 (preview)
    "accounts/fireworks/models/nemotron-3-ultra-nvfp4": {"input": 0.60, "output": 2.40, "cache_write": 0.60, "cache_read": 0.12},
    # Qwen 3.8 Max = Qwen3.8-2.4T-A95B
    "accounts/fireworks/models/qwen3p8-max": {"input": 2.00, "output": 6.00, "cache_write": 2.00, "cache_read": 0.25},
    # Kimi K3 (~2.8T MoE)
    "accounts/fireworks/models/kimi-k3": {"input": 3.00, "output": 15.00, "cache_write": 3.00, "cache_read": 0.30},
    # "Other base models" by size (serverless, same price in and out). Use via `price_key` for a model not listed above.
    "fireworks-size:<4B": {"input": 0.10, "output": 0.10},
    "fireworks-size:4B-16B": {"input": 0.20, "output": 0.20},
    "fireworks-size:>16B": {"input": 0.90, "output": 0.90},
    "fireworks-size:MoE<=56B": {"input": 0.50, "output": 0.50},
    "fireworks-size:MoE56-176B": {"input": 1.20, "output": 1.20},
}

# Fireworks dedicated (on-demand) deployments are billed per GPU-hour, not per token.
# Source: https://fireworks.ai/pricing ("On-demand"), checked 2026-10-12. Region-restricted deployments cost 1.5x.
GPU_HOUR_USD = {"H100": 8.00, "H200": 8.00, "B200": 13.00, "B300": 15.00, "GB300": 20.00}

PRICE_SOURCES = {
    "fireworks_serverless": "https://docs.fireworks.ai/serverless/pricing (checked 2026-10-12)",
    "fireworks_gpu_hour": "https://fireworks.ai/pricing (checked 2026-10-12)",
}


def cost_usd(usage, model):
    p = PRICES.get(model)
    if p is None:
        return None
    return (usage.get("input_tokens", 0) * p["input"] + usage.get("output_tokens", 0) * p["output"]
            + usage.get("cache_write_tokens", 0) * p.get("cache_write", p["input"])
            + usage.get("cache_read_tokens", 0) * p.get("cache_read", p["input"])) / 1e6


def write_meta(folder, model, latency_s, usage, price_key=None, cost=None, **extra):
    """Write <folder>/meta.json. Cost: `cost` if given (e.g. GPU-time for a dedicated deployment),
    else tokens x PRICES[price_key or model]. Extra keyword fields (config name, pipeline, ...) are stored as-is."""
    meta = {"model": model, "latency_s": latency_s, **{k: v for k, v in usage.items() if isinstance(v, (int, float))}}
    meta["cost_usd"] = cost if cost is not None else cost_usd(usage, price_key or model)
    meta.update(extra)
    Path(folder, "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def read_meta(folder):
    """meta.json of one meeting, or None if the pipeline didn't write one."""
    f = Path(folder, "meta.json")
    return json.loads(f.read_text()) if f.exists() else None
