"""Autolabel 50 frames with Nemotron-Labs-Diffusion-VLM-8B (diffusion VLM).
Run in the `dllm` conda env. Free-form generation (no template), common budget.

  python run_nemotron.py [--limit N]
"""
import argparse
import sys
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

# --- transformers>=5.9 shim (model remote code targets <5.9) ---
import transformers.masking_utils as _mu
import inspect as _ins
if not hasattr(_mu, "sdpa_mask_older_torch"):
    _mu.sdpa_mask_older_torch = _mu.sdpa_mask
def _alias(fn):
    acc = set(_ins.signature(fn).parameters)
    def w(*a, **k):
        if "input_embeds" in k and "inputs_embeds" not in k:
            k["inputs_embeds"] = k.pop("input_embeds")
        return fn(*a, **{kk: vv for kk, vv in k.items() if kk in acc})
    return w
_mu.create_causal_mask = _alias(_mu.create_causal_mask)
_mu.create_sliding_window_causal_mask = _alias(_mu.create_sliding_window_causal_mask)

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "utils"))
import common

REPO = "nvidia/Nemotron-Labs-Diffusion-VLM-8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--enable_thinking", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading Nemotron-Labs-Diffusion-VLM-8B ...", flush=True)
    local_dir = snapshot_download(REPO)
    sys.path.insert(0, local_dir)
    from image_processing import process_messages

    tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True).cuda().to(torch.bfloat16).eval()
    load_time = time.time() - t0
    print(f"loaded in {load_time:.1f}s  mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    manifest = common.load_manifest()
    if args.limit:
        manifest = manifest[: args.limit]
    name = "Nemotron-Diffusion-VLM-8B" + (" (thinking)" if args.enable_thinking else "")
    record = common.new_record(name, len(manifest))

    for i, item in enumerate(manifest):
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": item["frame_path"]}},
            {"type": "text", "text": common.ANNOTATION_PROMPT},
        ]}]
        batch = process_messages(tokenizer, messages, add_generation_prompt=True,
                                 enable_thinking=args.enable_thinking)
        prompt_ids = batch["input_ids"].to("cuda")
        pixel_values = batch["pixel_values"].to("cuda", dtype=torch.bfloat16)

        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            out_ids, nfe = model.generate(
                prompt_ids, pixel_values=pixel_values,
                image_sizes=batch["image_sizes"],
                max_new_tokens=common.MAX_NEW_TOKENS, steps=common.MAX_NEW_TOKENS,
                block_length=common.BLOCK_LENGTH, shift_logits=False,
                threshold=common.THRESHOLD, eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize(); dt = time.time() - t

        gen = out_ids[:, prompt_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        n_tok = int(gen.shape[1])
        nfe = int(nfe)
        tpf = n_tok / nfe if nfe else None
        common.add_sample(record, item["id"], item["frame_path"], decoded, dt,
                          n_tok, tokens_per_forward=tpf, extra={"nfe": nfe})
        print(f"[{i+1}/{len(manifest)}] {dt:.2f}s {n_tok}tok nfe={nfe} "
              f"tpf={tpf:.2f}" if tpf else f"[{i+1}] {dt:.2f}s", flush=True)

    common.finalize(record, load_time)
    common.save(record, "nemotron")


if __name__ == "__main__":
    main()
