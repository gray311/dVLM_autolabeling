"""Autolabel 50 frames with Qwen3-VL-8B-Instruct (autoregressive CONTROL model).
Run in the `dllm` conda env. Same prompt/budget as the dVLMs.

  python run_qwen3vl.py [--limit N]
"""
import argparse
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "utils"))
import common

import os
MODEL_PATH = os.path.join(os.environ.get("MODEL_ROOT",
    "/weka/scratch/aszalay1_ssci/yy/huggingface"), "Qwen3-VL-8B-Instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model_path", default=MODEL_PATH)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading Qwen3-VL-8B-Instruct ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    load_time = time.time() - t0
    print(f"loaded in {load_time:.1f}s  mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    manifest = common.load_manifest()
    if args.limit:
        manifest = manifest[: args.limit]
    record = common.new_record("Qwen3-VL-8B (AR control)", len(manifest))

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

        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=common.MAX_NEW_TOKENS,
                                 do_sample=False)
        torch.cuda.synchronize(); dt = time.time() - t

        trimmed = out[0][inputs.input_ids.shape[1]:]
        decoded = processor.decode(trimmed, skip_special_tokens=True).strip()
        n_tok = int(trimmed.shape[0])
        # AR = 1 token per forward by construction.
        common.add_sample(record, item["id"], item["frame_path"], decoded, dt,
                          n_tok, tokens_per_forward=1.0)
        print(f"[{i+1}/{len(manifest)}] {dt:.2f}s {n_tok}tok "
              f"{n_tok/dt:.1f}tok/s", flush=True)

    common.finalize(record, load_time)
    common.save(record, "qwen3vl")


if __name__ == "__main__":
    main()
