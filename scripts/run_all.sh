#!/bin/bash
# HF-baseline runner (NO containers): runs the 3 small models on a single GPU via
# plain HuggingFace transformers, each in its own conda env. Useful as an
# engine-controlled baseline (same engine for all) or when you don't have the
# vLLM/SGLang containers. DiffusionGemma-26B is NOT here (needs vLLM + 80GB).
# For the optimized engine-based run use sbatch_all4_h100.sh instead.
set -u
cd "$(dirname "$0")/.."; source ./paths.sh   # repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
mkdir -p logs

echo "===== [1/3] Fast-dVLM-3B (HF block-diffusion) ====="
$PY_FASTDVLM inference/run_fast_dvlm.py 2>&1 | tee logs/fast_dvlm.log | grep -E "^\[|saved|load_time|avg_|valid_json"

echo "===== [2/3] Nemotron-Diffusion-VLM-8B (HF diffusion) ====="
$PY_NEMO inference/run_nemotron.py 2>&1 | tee logs/nemotron.log | grep -E "^\[|saved|load_time|avg_|valid_json"

echo "===== [3/3] Qwen3-VL-8B (HF, AR control) ====="
$PY_NEMO inference/run_qwen3vl.py 2>&1 | tee logs/qwen3vl.log | grep -E "^\[|saved|load_time|avg_|valid_json"

echo "===== DONE. results/ has the json outputs. ====="
