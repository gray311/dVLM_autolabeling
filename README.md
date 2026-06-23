# dVLM Autolabeling

A unified harness to **autolabel driving frames into structured scene-analysis
reasoning traces** (perception → prediction → planning + region inference, JSON)
with **diffusion VLMs (dVLMs)** served on their intended inference engines, and to
benchmark them against an **autoregressive (AR) control** — on identical hardware,
with one common prompt + token budget (`common.py`) so the comparison is fair.

**TL;DR result** (50 frames, one H100, see [`REPORT.md`](REPORT.md)):

| model | engine | latency/frame | tok/s |
|---|---|--:|--:|
| **DiffusionGemma-26B-A4B** | **vLLM** | **1.18 s** | **431** |
| **Fast-dVLM-3B** | **SGLang** (HierarchyBlock + CUDA graph) | **2.20 s** | **196** |
| Qwen3-VL-8B (AR control) | vLLM | 3.63 s | 141 |
| Nemotron-Labs-Diffusion-VLM-8B | FastAPI diffusion | 5.70 s | 70 |

On the right engine the diffusion VLMs are the two fastest — **both beat the 8B AR
model** — via parallel multi-canvas decoding (~16 tok/forward vs AR's 1).

### Quick start (one-click demo)
After editing `paths.sh` and pulling a container (below), label 6 bundled frames:
```bash
cd demo && bash run_demo.sh           # Fast-dVLM-3B via SGLang (small, ~2 s/frame)
#         bash run_demo.sh diffusiongemma|qwen3vl|nemotron   # other models/engines
```
See [`demo/README.md`](demo/README.md) and the committed `demo/example_output_fast_dvlm.json`.

---

## Models & engines

| model | type | serving engine | why |
|---|---|---|---|
| `google/diffusiongemma-26B-A4B-it` | diffusion MoE | **vLLM ≥0.24** | first dLLM natively supported in vLLM; multi-canvas diffusion decode |
| `Efficient-Large-Model/Fast_dVLM_3B` | diffusion | **SGLang** (vendored Fast-dLLM fork, `HierarchyBlock`) | block-diffusion parallel decode |
| `nvidia/Nemotron-Labs-Diffusion-VLM-8B` | diffusion | **FastAPI** (`vlm_serve/serve_vlm.py`) | SGLang has **no** class for this VLM arch — only its *text* sibling |
| `Qwen/Qwen3-VL-8B-Instruct` | autoregressive (control) | **vLLM** | baseline |

---

## Repo layout

```
paths.sh                     # <-- EDIT THIS: all machine-specific paths (models, containers, conda pythons)
README.md  REPORT.md         # this guide / full speed+quality writeup + conclusions

utils/                       # shared helpers
  common.py                  #   shared prompt + token budget + metrics + result schema
  prepare_frames.py          #   extract 50 frames from the dataset -> frames/manifest.json
  compare.py                 #   aggregate results/*.json -> comparison table

inference/                   # serve a model + label all frames -> results/<key>.json
  vllm_serve_and_label.sh    #   vLLM container: DiffusionGemma & Qwen3-VL
  fast_dvlm_sglang_run.sh    #   SGLang container: Fast-dVLM
  nemotron_serve_and_label.sh#   Nemotron FastAPI diffusion server
  run_vllm_client.py         #   OpenAI-compatible client (vLLM + Nemotron launchers)
  run_fast_dvlm_sglang.py    #   SGLang Engine batch runner (runs INSIDE the sglang container)
  run_fast_dvlm.py  run_nemotron.py  run_qwen3vl.py  run_diffusiongemma.py  # HF fallbacks (no containers)

scripts/                     # orchestration
  sbatch_all4_h100.sh        #   all 4 on one H100 (vLLM + FastAPI + SGLang)  <-- main entrypoint
  sbatch_all4_a100.sh        #   same, a100 partition
  sbatch_fastdvlm_sglang_h100.sh  # just Fast-dVLM SGLang
  run_all.sh                 #   HF baseline for the 3 small models (no containers)

demo/                        # one-click demo: 6 bundled frames + run_demo.sh
results/                     # the autolabel outputs + comparison.json (engine-based, H100)
results_l40s/                # earlier HF-on-L40S baseline (engine-controlled)
```
All scripts are run from the **repo root** and locate their siblings automatically.

---

## Setup

### 1. Conda environments
| env | needs | used for |
|---|---|---|
| `dllm` | torch 2.11, transformers ≥5.9, `requests` | Nemotron + Qwen3-VL (HF), the OpenAI client, `compare.py` |
| `fastdvlm` | torch 2.6, **transformers 4.57.1**, `qwen_vl_utils` | Fast-dVLM HF (its custom modeling is incompatible with tf≥5.9) |
| `dgemma` | torch 2.12, transformers 5.13-dev w/ `DiffusionGemmaForBlockDiffusion` | DiffusionGemma HF (fallback only) |

```bash
conda create -n fastdvlm python=3.10 -y
conda run -n fastdvlm pip install torch==2.6.0 torchvision==0.21.0 transformers==4.57.1 qwen-vl-utils==0.0.14 accelerate pillow
```

### 2. Containers (apptainer / singularity)
The optimized engines run from containers (no fragile local builds):

```bash
# vLLM ≥0.24 with DiffusionGemma support (the cluster's pip vLLM 0.22 does NOT register it)
apptainer pull $CONTAINER_DIR/vllm-gemma.sif  docker://vllm/vllm-openai:gemma

# SGLang matching the vendored Fast-dLLM fork (sgl-kernel 0.3.x, flashinfer 0.5.3)
apptainer pull $CONTAINER_DIR/sglang-0.5.6.sif docker://lmsysorg/sglang:v0.5.6.post2
```

### 3. Models
Download the four HF repos into `$MODEL_ROOT` (folder names must match):
`Fast_dVLM_3B`, `Qwen2.5-VL-3B-Instruct` (Fast-dVLM's processor), `Qwen3-VL-8B-Instruct`,
`diffusiongemma-26B-A4B-it`. (Nemotron is auto-downloaded by its server on first run.)

### 4. Edit `paths.sh`
Set `MODEL_ROOT`, `VLLM_SIF`, `SGLANG_SIF`, `FASTDLLM_FORK`, `NEMOTRON_REPO`,
`PY_*` (conda-env python paths), and `BIND_DIRS` (host mounts to expose in containers).
Everything else reads from there.

### 5. Frames
```bash
DATASET_DIR=/path/to/dataset/with/mp4s_and_images  $PY_CLIENT utils/prepare_frames.py
# -> frames/frame_###.jpg + frames/manifest.json (50 frames)
```

---

## Launch vLLM inference (DiffusionGemma, Qwen3-VL)

One helper serves a model with the vLLM container, waits for `/health`, labels all
frames via the OpenAI client, then stops the server. Run on a GPU node (DiffusionGemma-26B
needs ≥80 GB; Qwen3-VL-8B fits in ~24 GB).

```bash
source ./paths.sh

# DiffusionGemma-26B  (args: <model_path> <served_name> <results_key> "<display>" <limit|0> [vllm flags...])
bash inference/vllm_serve_and_label.sh "$MODEL_ROOT/diffusiongemma-26B-A4B-it" \
     diffusiongemma diffusiongemma "DiffusionGemma-26B (vLLM)" 0 \
     --max-model-len 8192 --mm-processor-kwargs '{"max_soft_tokens":1120}' \
     --limit-mm-per-prompt '{"image":1}'

# Qwen3-VL-8B (AR control)
bash inference/vllm_serve_and_label.sh "$MODEL_ROOT/Qwen3-VL-8B-Instruct" \
     qwen3vl qwen3vl "Qwen3-VL-8B (vLLM, AR control)" 0 \
     --max-model-len 8192 --limit-mm-per-prompt '{"image":1}'
```

Under the hood it runs (you can run this directly to just start a server):
```bash
apptainer exec --nv -B /weka/home -B /weka/scratch --env HF_HOME=$MODEL_ROOT $VLLM_SIF \
  vllm serve $MODEL_ROOT/diffusiongemma-26B-A4B-it --served-model-name diffusiongemma \
  --gpu-memory-utilization 0.90 --max-num-seqs 4 --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 --mm-processor-kwargs '{"max_soft_tokens":1120}' --limit-mm-per-prompt '{"image":1}'
```
Then `inference/run_vllm_client.py --base-url http://localhost:8000/v1 --model diffusiongemma ...`.

> **Gotcha:** do **not** pass `--reasoning-parser gemma4` — it routes the answer to
> `reasoning_content` and leaves `content: null`. (The client falls back to
> `reasoning_content`, but dropping the flag keeps the full trace in `content`.)

Outputs → `results/<key>.json` (per-frame latency, tokens, output) + summary.

---

## Launch SGLang inference (Fast-dVLM)

Fast-dVLM uses the **vendored Fast-dLLM SGLang fork** (`$FASTDLLM_FORK`, which adds the
`fast_dvlm` model + `HierarchyBlock` dLLM algorithm) PYTHONPATH-shadowed inside the
`lmsysorg/sglang` container. `qwen_vl_utils` is pip-installed to a writable userbase on
first run. One command labels all frames via the offline `sgl.Engine`:

```bash
source ./paths.sh
CUDA_VISIBLE_DEVICES=0 ENABLE_CG=1 bash inference/fast_dvlm_sglang_run.sh 0     # 0 = all frames; or a small N to smoke-test
```

Toggles:
- `ENABLE_CG=1` (default) → **CUDA graph ON** (~1.9× faster: 4.2 → 2.2 s/frame). Forces
  the flashinfer attention backend.
- `ENABLE_CG=0` → CUDA graph OFF, uses `ATTN_BACKEND=triton` (more portable, slower).

> **Two gotchas that cost real time — both handled by the script, but know them:**
> 1. **flashinfer JIT build fails** on the no-graph path → we use `attention_backend=triton`.
> 2. **CUDA-graph capture fails** with a flashinfer *"missing source file"* ninja error when
>    `~/.cache/flashinfer` is polluted with a `build.ninja` pointing at a host conda's
>    flashinfer source (hidden by the container's narrow binds). **Fix: `rm -rf ~/.cache/flashinfer`**
>    so the container's own flashinfer rebuilds, then re-run. The launcher uses `--cleanenv`
>    + narrow binds to avoid host-package shadowing in the first place.

Why not SGLang for Nemotron-VLM too? Its arch (`NemotronLabsDiffusionVLMModel`) has **no**
SGLang implementation — the dLLM fork only added the *text* decoder. Use its FastAPI server.

---

## Launch Nemotron (FastAPI diffusion server)

```bash
source ./paths.sh
CUDA_VISIBLE_DEVICES=0 bash inference/nemotron_serve_and_label.sh 0
```
Starts `$NEMOTRON_REPO/vlm_serve/serve_vlm.py` (OpenAI-compatible, runs the model's native
diffusion `model.generate`), labels all frames, reports `nfe` → tokens/forward.

---

## HF-transformers fallback (no containers)

For the 3 small models without any container (one GPU each, plain HF):
```bash
bash scripts/run_all.sh        # Fast-dVLM (fastdvlm env) + Nemotron + Qwen3-VL (dllm env)
```
Individual: `$PY_FASTDVLM inference/run_fast_dvlm.py [--limit N]`, `$PY_NEMO inference/run_qwen3vl.py`, etc.
DiffusionGemma-26B has an HF runner too (`inference/run_diffusiongemma.py`) but only fits with offload
on <80 GB cards — use vLLM instead.

---

## Full benchmark + comparison

```bash
# all four on ONE H100 (vLLM + FastAPI + SGLang), then compare.py
sbatch scripts/sbatch_all4_h100.sh           # edit -A / -p for your cluster; submit from the repo dir

# or interactively on a GPU node:
source ./paths.sh
bash scripts/sbatch_all4_h100.sh             # the body runs fine outside Slurm too

# regenerate the table any time:
$PY_CLIENT utils/compare.py
```

`compare.py` prints load time, latency/frame, tok/s, tokens/forward, output length,
valid-JSON rate, and structural completeness (expected keys present), and writes
`results/comparison.json`. Full analysis and conclusions: [`REPORT.md`](REPORT.md).
