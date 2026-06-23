"""Autolabel 50 frames with DiffusionGemma-26B-A4B-it (diffusion MoE VLM, vLLM-class).
Run in the `dgemma` conda env.

NOTE: 26B bf16 weights are ~49GB > the 46GB L40S, so device_map='auto' offloads
some layers to CPU. Latency therefore includes a CPU<->GPU offload penalty and is
an UPPER BOUND, not a fair-hardware number -- flagged in the report.

  python run_diffusiongemma.py [--limit N]
"""
import argparse
import os
import time

import torch
from PIL import Image
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor

# 26B bf16 (~49GB) > 46GB L40S and bitsandbytes can't quantize its fused MoE
# experts, so the ONLY way to run on one card is accelerate CPU offload. But the
# gemma4 vision tower builds its mask eagerly and calls .item() on a meta tensor
# during offload -> crash. Patch the skip-check to bail out on meta tensors.
import transformers.masking_utils as _mu
_orig_ignore = _mu._ignore_bidirectional_mask_sdpa
def _ignore_patched(padding_mask, *a, **k):
    if padding_mask is not None and getattr(padding_mask, "is_meta", False):
        return False
    return _orig_ignore(padding_mask, *a, **k)
_mu._ignore_bidirectional_mask_sdpa = _ignore_patched

import common

MODEL_PATH = os.path.join(os.environ.get("MODEL_ROOT",
    "/weka/scratch/aszalay1_ssci/yy/huggingface"), "diffusiongemma-26B-A4B-it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model_path", default=MODEL_PATH)
    ap.add_argument("--cpu", action="store_true", help="CPU-only (slow) to harvest quality samples")
    ap.add_argument("--full-gpu", dest="full_gpu", action="store_true",
                    help="put the whole model on one GPU (needs >=80GB, e.g. H100/A100-80GB)")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading DiffusionGemma-26B-A4B-it ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    import os as _os
    from accelerate import init_empty_weights, infer_auto_device_map
    from transformers import AutoConfig
    offdir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_offload")
    _os.makedirs(offdir, exist_ok=True)
    # The vision/encoder stack has data-dependent ops (nonzero/.item()) that can't
    # run on offloaded/meta tensors -> pin ALL of model.encoder to GPU; offload
    # only deep model.decoder layers to CPU.
    if args.cpu:
        # CPU-only: no meta tensors, no OOM, but very slow. Quality samples only.
        device_map = {"": "cpu"}
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map=device_map,
        ).eval()
    elif args.full_gpu:
        # Whole 26B (~49GB) on one GPU -> needs >=80GB. No offload, real speed.
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map={"": 0},
        ).eval()
    else:
        # The vision/encoder stack has data-dependent ops (nonzero/.item()) that
        # can't run on offloaded/meta tensors -> pin ALL of model.encoder to GPU;
        # offload only deep model.decoder layers to CPU. (NOTE: on a 46GB L40S the
        # encoder alone exceeds the card, so this still OOMs -- see report.)
        cfg = AutoConfig.from_pretrained(args.model_path)
        with init_empty_weights():
            skel = DiffusionGemmaForBlockDiffusion._from_config(cfg)
        dmap = infer_auto_device_map(
            skel, max_memory={0: "38GiB", "cpu": "300GiB"}, dtype=torch.bfloat16,
            no_split_module_classes=list(getattr(skel, "_no_split_modules", []) or []))
        for k in list(dmap):
            if k.startswith("model.encoder") or k == "lm_head":
                dmap[k] = 0
        del skel
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map=dmap,
            offload_folder=offdir,
        ).eval()
    load_time = time.time() - t0
    # Detect CPU offload.
    devs = {str(p.device) for p in model.parameters()}
    offloaded = any("cpu" in d for d in devs)
    print(f"loaded in {load_time:.1f}s  devices={devs}  offloaded={offloaded}  "
          f"gpu_mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    manifest = common.load_manifest()
    if args.limit:
        manifest = manifest[: args.limit]
    record = common.new_record("DiffusionGemma-26B-A4B-it", len(manifest))
    if args.full_gpu:
        record["note"] = ("bf16, whole 26B on one >=80GB GPU (no offload) -> real "
                          "GPU latency, comparable to the other models.")
    elif args.cpu:
        record["note"] = ("CPU-only bf16: tokens/forward is hardware-independent but "
                          "wall latency is CPU-bound (NOT comparable). Quality samples.")
    else:
        record["note"] = (f"bf16 accelerate offload (offloaded={offloaded}); 26B does NOT "
                          "fit on 46GB L40S. LATENCY NOT HARDWARE-COMPARABLE (offload-bound).")

    for i, item in enumerate(manifest):
        img = Image.open(item["frame_path"]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": common.ANNOTATION_PROMPT},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to("cpu" if args.cpu else "cuda:0")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        n_in = inputs["input_ids"].shape[1]

        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=common.MAX_NEW_TOKENS)
        torch.cuda.synchronize(); dt = time.time() - t

        seq = out.sequences
        n_tok = int(seq.shape[1] - n_in)
        decoded = processor.decode(seq[0][n_in:], skip_special_tokens=True).strip()
        tpf = getattr(out, "tokens_per_forward", None)
        if hasattr(tpf, "float"):
            tpf = float(tpf.float().mean())
        common.add_sample(record, item["id"], item["frame_path"], decoded, dt,
                          n_tok, tokens_per_forward=tpf)
        print(f"[{i+1}/{len(manifest)}] {dt:.2f}s {n_tok}tok "
              f"tpf={tpf:.2f}" if tpf else f"[{i+1}] {dt:.2f}s", flush=True)

    common.finalize(record, load_time)
    common.save(record, "diffusiongemma")


if __name__ == "__main__":
    main()
