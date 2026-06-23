"""Shared utilities for the dVLM autolabeling speed/quality benchmark.

All four model runners (run_fast_dvlm.py, run_nemotron.py, run_diffusiongemma.py,
run_qwen3vl.py) import from here so the task, prompt, token budget and output
schema are IDENTICAL across models -> a fair speed + quality comparison.

Task: single front-camera driving frame -> structured scene-analysis "reasoning
trace" (perception -> prediction -> planning + region inference), the kind of
label an autonomous-driving reasoning dataset needs.
"""
from __future__ import annotations

import json
import os
import time

# Repo root = parent of utils/ (this file lives in utils/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# All overridable via env (the demo/ uses this to point at its bundled frames).
FRAMES_DIR = os.environ.get("FRAMES_DIR", os.path.join(ROOT, "frames"))
MANIFEST = os.environ.get("MANIFEST", os.path.join(FRAMES_DIR, "manifest.json"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))

# Common decoding budget shared by every model (fair comparison).
MAX_NEW_TOKENS = 512
BLOCK_LENGTH = 32          # diffusion block size
THRESHOLD = 0.9            # diffusion confidence-commit threshold

# Single common annotation prompt. Free-form so each model generates in its
# native decoding mode; asks for a long, structured reasoning trace + JSON.
ANNOTATION_PROMPT = (
    "You are an expert autonomous-driving scene annotator building a reasoning "
    "dataset. Analyse this front-camera frame and produce a detailed, structured "
    "annotation. Reason step by step in three stages.\n\n"
    "1. PERCEPTION: Describe the road type, lane layout, lane markings, traffic "
    "signals/signs, crosswalks, lighting and weather, and every important traffic "
    "participant (vehicles, pedestrians, cyclists) with its position.\n"
    "2. PREDICTION: For the key participants, state their likely short-term "
    "behaviour over the next few seconds and why.\n"
    "3. PLANNING: State what the ego vehicle should do next and the rule/reason "
    "behind it.\n"
    "Also infer the country/region from the road infrastructure, signage and "
    "driving side, and justify it.\n\n"
    "Return ONLY a JSON object with these keys: \"scene_description\", "
    "\"traffic_elements\", \"dynamic_agents\", \"prediction\", \"planning\", "
    "\"region_inference\", \"reasoning\"."
)


def load_manifest():
    with open(MANIFEST, "r") as f:
        items = json.load(f)
    # Resolve relative frame paths against FRAMES_DIR (lets the demo ship portable
    # manifests with bare filenames).
    for it in items:
        if not os.path.isabs(it["frame_path"]):
            it["frame_path"] = os.path.join(FRAMES_DIR, it["frame_path"])
    return items


def count_tokens(text, tokenizer):
    try:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    except Exception:
        return len(text.split())


def new_record(model_name, n_frames):
    return {
        "model": model_name,
        "prompt": ANNOTATION_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_frames": n_frames,
        "samples": [],
        "summary": {},
    }


def add_sample(record, frame_id, frame_path, text, latency_s, out_tokens,
               tokens_per_forward=None, extra=None):
    rec = {
        "frame_id": frame_id,
        "frame_path": frame_path,
        "latency_s": round(latency_s, 3),
        "out_tokens": int(out_tokens),
        "tok_per_s": round(out_tokens / latency_s, 2) if latency_s > 0 else 0.0,
        "tokens_per_forward": (round(float(tokens_per_forward), 3)
                               if tokens_per_forward is not None else None),
        "output": text,
    }
    try:
        json.loads(text)
        rec["valid_json"] = True
    except Exception:
        rec["valid_json"] = False
    if extra:
        rec.update(extra)
    record["samples"].append(rec)
    return rec


def finalize(record, load_time_s):
    s = record["samples"]
    # Skip the first sample in averages (warm-up / CUDA graph capture).
    warm = s[1:] if len(s) > 1 else s
    n = len(warm)
    if n:
        tot_lat = sum(x["latency_s"] for x in warm)
        tot_tok = sum(x["out_tokens"] for x in warm)
        tpfs = [x["tokens_per_forward"] for x in warm if x["tokens_per_forward"]]
        record["summary"] = {
            "load_time_s": round(load_time_s, 1),
            "n_samples": len(s),
            "n_in_avg": n,
            "avg_latency_s": round(tot_lat / n, 3),
            "avg_out_tokens": round(tot_tok / n, 1),
            "avg_tok_per_s": round(tot_tok / tot_lat, 2) if tot_lat else 0.0,
            "avg_tokens_per_forward": round(sum(tpfs) / len(tpfs), 3) if tpfs else None,
            "valid_json_rate": round(sum(x["valid_json"] for x in s) / len(s), 3),
            "total_wall_s": round(sum(x["latency_s"] for x in s), 1),
        }
    return record


def save(record, model_key):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{model_key}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {path}")
    print(json.dumps(record["summary"], indent=2))
    return path


class Timer:
    def __enter__(self):
        import torch
        torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        import torch
        torch.cuda.synchronize()
        self.dt = time.time() - self.t0
