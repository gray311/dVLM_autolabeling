#!/bin/bash
# ============================================================================
# Central config — EDIT THESE PATHS for your machine, then everything else works.
# Each is overridable from the environment (export VAR=... before launching).
# ============================================================================

# This repo's directory (auto-detected; usually no need to change).
AUTOLABEL_DIR="${AUTOLABEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Root that holds the downloaded HF model folders (Fast_dVLM_3B, Qwen2.5-VL-3B-Instruct,
# Qwen3-VL-8B-Instruct, diffusiongemma-26B-A4B-it). HF_HOME is pointed here too.
export MODEL_ROOT="${MODEL_ROOT:-/weka/scratch/aszalay1_ssci/yy/huggingface}"

# Apptainer/Singularity images (see README §Setup for the two `apptainer pull`s).
export VLLM_SIF="${VLLM_SIF:-/weka/scratch/aszalay1_ssci/yy/containers/vllm-gemma.sif}"     # docker://vllm/vllm-openai:gemma  (vLLM >=0.24, supports DiffusionGemma)
export SGLANG_SIF="${SGLANG_SIF:-/weka/scratch/aszalay1_ssci/yy/containers/sglang-0.5.6.sif}" # docker://lmsysorg/sglang:v0.5.6.post2

# Vendored Fast-dLLM SGLang fork (the `python/` dir, PYTHONPATH-shadowed in-container).
export FASTDLLM_FORK="${FASTDLLM_FORK:-/weka/home/ext-yingzima/Fast-dLLM/third_party/sglang/python}"

# Nemotron-Labs-Diffusion repo (provides vlm_serve/serve_vlm.py).
export NEMOTRON_REPO="${NEMOTRON_REPO:-/weka/home/ext-yingzima/Nemotron-Labs-Diffusion}"

# A writable dir for pip --user installs inside the read-only sglang container.
export SGLANG_USERBASE="${SGLANG_USERBASE:-/weka/scratch/aszalay1_ssci/yy/sglang_userbase}"

# Python interpreters per conda env (see README §Environments).
#   PY_CLIENT : has `requests` (for the OpenAI/vLLM client) -> env `dllm`
#   PY_NEMO   : transformers>=5.9 + torch (Nemotron + Qwen3-VL HF) -> env `dllm`
#   PY_FASTDVLM: transformers==4.57 + qwen_vl_utils (Fast-dVLM HF) -> env `fastdvlm`
#   PY_DGEMMA : transformers 5.13-dev with DiffusionGemmaForBlockDiffusion -> env `dgemma`
export PY_CLIENT="${PY_CLIENT:-/home/ext-yingzima/miniconda3/envs/dllm/bin/python}"
export PY_NEMO="${PY_NEMO:-/home/ext-yingzima/miniconda3/envs/dllm/bin/python}"
export PY_FASTDVLM="${PY_FASTDVLM:-/weka/scratch/aszalay1_ssci/yy/conda_envs/fastdvlm/bin/python}"
export PY_DGEMMA="${PY_DGEMMA:-/home/ext-yingzima/miniconda3/envs/dgemma/bin/python}"

# weka mount points to bind into containers (space-separated host paths).
export BIND_DIRS="${BIND_DIRS:-/weka/home /weka/scratch}"
