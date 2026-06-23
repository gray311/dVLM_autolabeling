#!/bin/bash
#SBATCH -A aszalay1_ssci
#SBATCH -p h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --job-name=autolabel4
#SBATCH --output=logs/all4_h100_%j.out
# ^ EDIT -A / -p for your cluster. Submit from the repo ROOT: `sbatch scripts/sbatch_all4_h100.sh`

# All FOUR models on ONE H100 (80GB) -> directly comparable latency.
# Serving engine per model:
#   DiffusionGemma-26B -> vLLM (gemma container, fits in bf16 on 80GB)
#   Qwen3-VL-8B (AR)   -> vLLM (gemma container)
#   Nemotron-VLM-8B    -> native FastAPI diffusion server (SGLang can't serve this VLM arch)
#   Fast-dVLM-3B       -> SGLang (Fast-dLLM fork, HierarchyBlock + CUDA graph)
set -u
cd "$(dirname "$0")/.."; source ./paths.sh   # repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 LC_ALL=C.UTF-8 LANG=C.UTF-8

echo "node=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv

echo "########## [1/4] DiffusionGemma-26B (vLLM diffusion) ##########"
bash inference/vllm_serve_and_label.sh "$MODEL_ROOT/diffusiongemma-26B-A4B-it" \
  diffusiongemma diffusiongemma "DiffusionGemma-26B (vLLM)" 0 \
  --max-model-len 8192 --mm-processor-kwargs '{"max_soft_tokens":1120}' \
  --limit-mm-per-prompt '{"image":1}'

echo "########## [2/4] Qwen3-VL-8B (vLLM, AR control) ##########"
bash inference/vllm_serve_and_label.sh "$MODEL_ROOT/Qwen3-VL-8B-Instruct" \
  qwen3vl qwen3vl "Qwen3-VL-8B (vLLM, AR control)" 0 \
  --max-model-len 8192 --limit-mm-per-prompt '{"image":1}'

echo "########## [3/4] Nemotron-VLM-8B (FastAPI diffusion) ##########"
CUDA_VISIBLE_DEVICES=0 bash inference/nemotron_serve_and_label.sh 0

echo "########## [4/4] Fast-dVLM-3B (SGLang HierarchyBlock + CUDA graph) ##########"
CUDA_VISIBLE_DEVICES=0 ENABLE_CG=1 bash inference/fast_dvlm_sglang_run.sh 0

echo "########## ALL 4 DONE on $(hostname). regenerating comparison ##########"
$PY_CLIENT utils/compare.py
