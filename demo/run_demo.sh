#!/bin/bash
# ============================================================================
# ONE-CLICK DEMO — autolabel the 6 bundled frames (demo/frames/) with one model
# and print the structured annotations.
#
#   cd demo && bash run_demo.sh [model]
#       model = fast_dvlm (default) | diffusiongemma | qwen3vl | nemotron
#
# Prereqs (one-time, see ../README.md):
#   1) edit ../paths.sh   (MODEL_ROOT, container .sif paths, conda pythons)
#   2) have the model downloaded + the matching container pulled
#   3) run on a GPU node (DiffusionGemma-26B needs >=80GB; the others fit small)
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$REPO/paths.sh"

MODEL="${1:-fast_dvlm}"

# Point the harness at the bundled demo frames + a separate results dir.
export FRAMES_DIR="$HERE/frames"
export MANIFEST="$HERE/frames/manifest.json"
export RESULTS_DIR="$HERE/results"
mkdir -p "$RESULTS_DIR"

echo "=================================================================="
echo " dVLM autolabeling demo | model=$MODEL | $(ls "$FRAMES_DIR"/*.jpg | wc -l) frames"
echo "=================================================================="
cd "$REPO"

case "$MODEL" in
  fast_dvlm)
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ENABLE_CG=1 bash inference/fast_dvlm_sglang_run.sh 0
    ;;
  diffusiongemma)
    bash inference/vllm_serve_and_label.sh "$MODEL_ROOT/diffusiongemma-26B-A4B-it" \
      diffusiongemma diffusiongemma "DiffusionGemma-26B (vLLM)" 0 \
      --max-model-len 8192 --mm-processor-kwargs '{"max_soft_tokens":1120}' --limit-mm-per-prompt '{"image":1}'
    ;;
  qwen3vl)
    bash inference/vllm_serve_and_label.sh "$MODEL_ROOT/Qwen3-VL-8B-Instruct" \
      qwen3vl qwen3vl "Qwen3-VL-8B (vLLM)" 0 \
      --max-model-len 8192 --limit-mm-per-prompt '{"image":1}'
    ;;
  nemotron)
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} bash inference/nemotron_serve_and_label.sh 0
    ;;
  *) echo "unknown model '$MODEL' (use: fast_dvlm | diffusiongemma | qwen3vl | nemotron)"; exit 1 ;;
esac

echo
echo "=================================================================="
echo " RESULT -> $RESULTS_DIR/${MODEL/diffusiongemma/diffusiongemma}.json"
echo "=================================================================="
RES="$RESULTS_DIR/${MODEL}.json"
[ "$MODEL" = "fast_dvlm" ] && RES="$RESULTS_DIR/fast_dvlm.json"
"$PY_CLIENT" - "$RES" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("model :", r["model"])
print("speed :", r["summary"])
print("\n--- sample annotation (frame 0) ---")
print(r["samples"][0]["output"][:1200])
PY
