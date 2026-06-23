"""Autolabel 50 frames against an OpenAI-compatible vLLM server (image+text chat
completions). Used for DiffusionGemma-26B and Qwen3-VL-8B, both served by vLLM.

  python run_vllm_client.py --model google/diffusiongemma-26B-A4B-it \
      --key diffusiongemma --name "DiffusionGemma-26B (vLLM)" [--limit N]

Assumes a server at --base-url (default http://localhost:8000/v1).
"""
import argparse
import base64
import mimetypes
import time

import requests

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "utils"))
import common


def data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--key", required=True, help="results/<key>.json")
    ap.add_argument("--name", required=True, help="display name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/chat/completions"
    manifest = common.load_manifest()
    if args.limit:
        manifest = manifest[: args.limit]
    record = common.new_record(args.name, len(manifest))
    record["note"] = args.note or f"OpenAI endpoint {args.base_url}, model={args.model}"

    t_load0 = time.time()
    # Wait for server health.
    health = args.base_url.rstrip("/").rsplit("/", 1)[0] + "/health"
    for _ in range(600):
        try:
            if requests.get(health, timeout=5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    load_time = time.time() - t_load0

    for i, item in enumerate(manifest):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(item["frame_path"])}},
                {"type": "text", "text": common.ANNOTATION_PROMPT},
            ]}],
            "max_tokens": common.MAX_NEW_TOKENS,
            "temperature": args.temperature,
        }
        t = time.time()
        r = requests.post(url, json=payload, timeout=600)
        dt = time.time() - t
        r.raise_for_status()
        j = r.json()
        msg = j["choices"][0]["message"]
        # DiffusionGemma w/ a reasoning parser may put text in reasoning_content
        # and leave content None; fall back across both.
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        usage = j.get("usage", {})
        n_tok = usage.get("completion_tokens") or common.count_tokens(text, None)
        # diffusion servers may expose forward-pass counts in usage; derive tok/fwd.
        tpf = usage.get("tokens_per_forward") or usage.get("nfe_tokens_per_forward")
        nfe = usage.get("nfe") or usage.get("num_forward_passes")
        if tpf is None and nfe:
            tpf = n_tok / nfe
        common.add_sample(record, item["id"], item["frame_path"], text, dt,
                          n_tok, tokens_per_forward=tpf, extra={"usage": usage})
        print(f"[{i+1}/{len(manifest)}] {dt:.2f}s {n_tok}tok "
              f"{n_tok/dt:.1f}tok/s", flush=True)

    common.finalize(record, load_time)
    common.save(record, args.key)


if __name__ == "__main__":
    main()
