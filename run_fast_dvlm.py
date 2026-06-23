"""Autolabel 50 frames with Fast-dVLM-3B (block-diffusion VLM).
Run in the `dllm` conda env. Free-form generation, common prompt/budget.

  python run_fast_dvlm.py [--limit N]
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

import common

import os
MODEL_PATH = os.path.join(os.environ.get("MODEL_ROOT",
    "/weka/scratch/aszalay1_ssci/yy/huggingface"), "Fast_dVLM_3B")
MASK_TOKEN = "|<MASK>|"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model_path", default=MODEL_PATH)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading Fast-dVLM-3B ...", flush=True)
    # transformers>=5.9 no longer auto-populates these on the sub-config; the
    # repo's modeling.py (written for tf 4.57) reads them directly.
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    for sub in ("text_config", "vision_config"):
        sc = getattr(cfg, sub, None)
        if sc is not None and not hasattr(sc, "pad_token_id"):
            sc.pad_token_id = getattr(sc, "eos_token_id", None) or 0
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=cfg, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False,
                                              trust_remote_code=True)
    processor.tokenizer = tokenizer
    load_time = time.time() - t0
    mask_id = tokenizer.encode(MASK_TOKEN)[0]
    print(f"loaded in {load_time:.1f}s  mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Count forward() calls to derive tokens-per-forward (diffusion efficiency).
    fwd = [0]
    orig_forward = model.forward
    def counting_forward(*a, **k):
        fwd[0] += 1
        return orig_forward(*a, **k)
    model.forward = counting_forward

    manifest = common.load_manifest()
    if args.limit:
        manifest = manifest[: args.limit]
    record = common.new_record("Fast-dVLM-3B", len(manifest))

    for i, item in enumerate(manifest):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": item["frame_path"]},
            {"type": "text", "text": common.ANNOTATION_PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "tokenizer": tokenizer,
            "block_size": common.BLOCK_LENGTH,
            "max_tokens": common.MAX_NEW_TOKENS,
            "mask_id": mask_id,
        }
        if hasattr(inputs, "pixel_values"):
            gen_kwargs["pixel_values"] = inputs.pixel_values
        if hasattr(inputs, "image_grid_thw"):
            gen_kwargs["image_grid_thw"] = inputs.image_grid_thw

        fwd[0] = 0
        torch.cuda.synchronize(); t = time.time()
        out = model.generate(**gen_kwargs)
        torch.cuda.synchronize(); dt = time.time() - t

        trimmed = out[0][inputs.input_ids.shape[1]:]
        decoded = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
        n_tok = int((trimmed != mask_id).sum().item())
        tpf = n_tok / fwd[0] if fwd[0] else None
        common.add_sample(record, item["id"], item["frame_path"], decoded, dt,
                          n_tok, tokens_per_forward=tpf, extra={"forwards": fwd[0]})
        print(f"[{i+1}/{len(manifest)}] {dt:.2f}s {n_tok}tok {fwd[0]}fwd "
              f"tpf={tpf:.2f}" if tpf else f"[{i+1}] {dt:.2f}s", flush=True)

    common.finalize(record, load_time)
    common.save(record, "fast_dvlm")


if __name__ == "__main__":
    main()
