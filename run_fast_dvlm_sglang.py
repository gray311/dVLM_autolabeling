"""Autolabel 50 frames with Fast-dVLM-3B via the vendored SGLang fork
(HierarchyBlock block-diffusion). Run INSIDE the sglang container with the
Fast-dLLM fork PYTHONPATH-shadowed. Mirrors run_chatbot_sglang.py.

  python run_fast_dvlm_sglang.py [--algorithm mdm|spec] [--limit N]
"""
import argparse
import os
import sys
import time

# Drop our own dir so `import sglang` resolves to the fork, not a local file.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _HERE]
sys.path.insert(0, _HERE)  # but keep it for `import common`

import common

ALGO_MAP = {"mdm": "HierarchyBlock", "spec": "SpeculativeBlock"}
_ROOT = os.environ.get("MODEL_ROOT", "/weka/scratch/aszalay1_ssci/yy/huggingface")
MODEL_PATH = os.path.join(_ROOT, "Fast_dVLM_3B")
PROCESSOR = os.path.join(_ROOT, "Qwen2.5-VL-3B-Instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=list(ALGO_MAP), default="mdm")
    ap.add_argument("--model_path", default=MODEL_PATH)
    ap.add_argument("--processor_path", default=PROCESSOR)
    ap.add_argument("--mem-fraction-static", type=float, default=0.8)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    ap.add_argument("--attention-backend", default=None,
                    help="e.g. triton / torch_native to avoid flashinfer JIT")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    import sglang as sgl
    from transformers import AutoProcessor, AutoTokenizer
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(args.processor_path, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer = tokenizer

    def build_inputs(image, prompt):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inp = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt")
        return inp.input_ids[0].tolist()

    t0 = time.time()
    print(f"Launching sglang Engine dllm_algorithm={ALGO_MAP[args.algorithm]} ...", flush=True)
    ekw = dict(
        model_path=args.model_path, trust_remote_code=True, dtype="bfloat16",
        mem_fraction_static=args.mem_fraction_static, max_running_requests=1,
        chunked_prefill_size=16384, dllm_algorithm=ALGO_MAP[args.algorithm],
        disable_cuda_graph=args.disable_cuda_graph, log_level="info",
        enable_metrics=True, mm_attention_backend="triton_attn",
    )
    if args.attention_backend:
        ekw["attention_backend"] = args.attention_backend
    engine = sgl.Engine(**ekw)
    load_time = time.time() - t0
    sampling = {"max_new_tokens": common.MAX_NEW_TOKENS, "temperature": 0.0}

    manifest = common.load_manifest()
    if args.limit:
        manifest = manifest[: args.limit]
    record = common.new_record(f"Fast-dVLM-3B (SGLang {args.algorithm})", len(manifest))
    record["note"] = f"SGLang fork, dllm_algorithm={ALGO_MAP[args.algorithm]}, bf16"

    try:
        for i, item in enumerate(manifest):
            ids = build_inputs(item["frame_path"], common.ANNOTATION_PROMPT)
            t = time.time()
            out = engine.generate(input_ids=ids, image_data=[item["frame_path"]],
                                  sampling_params=sampling)
            dt = time.time() - t
            if isinstance(out, list):
                out = out[0]
            text = out["text"].strip()
            meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
            n_tok = meta.get("completion_tokens") or common.count_tokens(text, tokenizer)
            common.add_sample(record, item["id"], item["frame_path"], text, dt,
                              n_tok, tokens_per_forward=None, extra={"meta": meta})
            print(f"[{i+1}/{len(manifest)}] {dt:.2f}s {n_tok}tok {n_tok/dt:.1f}tok/s", flush=True)
    finally:
        engine.shutdown()

    common.finalize(record, load_time)
    common.save(record, "fast_dvlm")


if __name__ == "__main__":
    main()
