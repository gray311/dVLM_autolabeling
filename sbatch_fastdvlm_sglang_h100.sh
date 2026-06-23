#!/bin/bash
#SBATCH -A aszalay1_ssci
#SBATCH -p h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=fastdvlm_sgl
#SBATCH --output=logs/fastdvlm_sglang_h100_%j.out
# ^ EDIT -A / -p for your cluster. Submit from the repo dir so logs/ resolves.

# Fast-dVLM-3B via the vendored SGLang fork (HierarchyBlock + CUDA graph) only.
# Overwrites results/fast_dvlm.json, then regenerates the comparison.
set -u
cd "$(dirname "$0")"; source ./paths.sh
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
echo "node=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv

CUDA_VISIBLE_DEVICES=0 ENABLE_CG=1 bash fast_dvlm_sglang_run.sh 0

echo "########## Fast-dVLM SGLang done; regenerating comparison ##########"
$PY_CLIENT compare.py
