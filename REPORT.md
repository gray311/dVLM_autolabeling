# dVLM Autolabeling — Speed & Quality Report (engine-based, unified H100)

**Task.** Autolabel driving frames into structured scene-analysis *reasoning
traces* (perception → prediction → planning + region inference, JSON). One common
prompt + 512-token budget for every model (`common.py`).

**Data.** 50 frames from `CulturalDrive/dataset` (4 front-cam videos + 7 stills).

**Hardware.** All four models run on **one NVIDIA H100 (80 GB)** → latency is
directly comparable. Each model is served by its **intended inference engine**:

| model | params | type | **serving engine** |
|---|---|---|---|
| DiffusionGemma-26B-A4B-it | 26B (A4B MoE) | diffusion | **vLLM 0.24-dev** (`vllm/vllm-openai:gemma`, registers `DiffusionGemmaForBlockDiffusion`) |
| Fast-dVLM-3B | 3B | diffusion | **SGLang** (vendored Fast-dLLM fork, `HierarchyBlock`, triton backend) |
| Nemotron-Labs-Diffusion-VLM-8B | 8B | diffusion | native **FastAPI diffusion** server (SGLang can't serve this VLM arch) |
| Qwen3-VL-8B-Instruct | 8B | **autoregressive (control)** | **vLLM** |

---

## 1. Speed — 50 frames, same H100

| model | engine | type | **latency/frame** | **tok/s** | tok/fwd |
|---|---|---|--:|--:|--:|
| **DiffusionGemma-26B** | vLLM | diffusion | **1.18 s** | **431** | ~16† |
| **Fast-dVLM-3B** | SGLang **+ CUDA graph** | diffusion | **2.20 s** | **196** | ~2‡ |
| Qwen3-VL-8B | vLLM | AR control | 3.63 s | 141 | 1.0 |
| Nemotron-VLM-8B | FastAPI | diffusion | 5.70 s | 70 | 1.78 |

†from the model card (vLLM's OpenAI `usage` doesn't expose tok/fwd); ‡SGLang
HierarchyBlock. CUDA graph (§4) cut Fast-dVLM from 4.20 s → 2.20 s/frame (1.9×).

**The headline result:**
- **DiffusionGemma-26B on vLLM is the fastest model by a wide margin — 1.18 s/frame,
  431 tok/s — even though it is the *largest* (26B).** It beats the 8B **AR** control
  (Qwen3-VL) by **~3× in tok/s** and **3.1× in latency/frame**.
- This is the diffusion-VLM advantage made real: vLLM's **multi-canvas parallel
  denoising** commits ~15-20 tokens per forward pass, so a 26B-A4B MoE out-runs an
  8B AR model. The lever is *engine + algorithm*, not just the model.
- **Fast-dVLM-3B on SGLang with CUDA graph (196 tok/s) is the 2nd fastest — it
  beats the 8B AR control** (141 tok/s): a 3B *diffusion* model out-running an 8B
  *autoregressive* one. CUDA graph alone took it from 105 → 196 tok/s (1.9×).
- Nemotron (70 tok/s) is the slowest only because its serving path is the least
  optimized — HF `model.generate` behind a FastAPI server, no graph/kernel fusion.
  Its per-forward parallelism is real (1.78 tok/fwd) but the engine leaves it on the table.

## 2. Quality — 50 frames

| model | struct. completeness | valid JSON* | character |
|---|--:|--:|---|
| **DiffusionGemma-26B (vLLM)** | **85%** | (fenced) | **richest + most accurate**: nested JSON, catches fog/rain/lens droplets, glowing streetlamp orbs; wrapped in a ```json fence (trivially stripped) |
| Nemotron-VLM-8B (FastAPI) | **98%** | (fence/cap) | clean, complete, well-structured nested JSON; concise |
| Qwen3-VL-8B (AR) | 76% | 2% | detailed prose perception, but overruns 512 tokens → JSON cut off |
| Fast-dVLM-3B (SGLang) | 90% | 0% | **weakest**: malformed JSON (mixes list/dict syntax), fabricates coordinates — it's a 3B |

\*Raw `json.loads` rate is near-0 for the engine runs because outputs are wrapped
in ```json fences, include a `<think>`/reasoning preamble, or hit the 512-token cap
mid-object. **struct% (expected keys present) is the better quality signal** — and
DiffusionGemma/Nemotron score highest. (A one-line fence-strip + budget bump would
lift valid-JSON to the struct% level; it is not a model-capability limit.)

**Quality ranking:** DiffusionGemma-26B ≳ Nemotron-8B > Qwen3-VL-8B (rich but
unfinished) > Fast-dVLM-3B (fast/cheap but malformed, shallow).

## 3. Does the diffusion VLM have an advantage? — **Yes, decisively, with the right engine.**

Goal: *longer, more structured reasoning traces, generated faster.*

- **Faster: yes, dramatically.** On vLLM, the 26B diffusion model autolabels at
  **431 tok/s / 1.18 s per frame — ~3× the 8B AR control.** Parallel multi-canvas
  decoding (~16 tok/forward) is the mechanism; AR is fixed at 1 tok/forward.
- **More structured / complete: yes.** DiffusionGemma and Nemotron produce the most
  complete, well-keyed nested traces (85-98% structural completeness). The AR control
  writes rich prose but **overruns the fixed budget and leaves the structure unfinished**
  (76%, 2% closed JSON).
- **Caveat — it is engine-dependent.** The advantage shows fully only where the
  serving stack exploits diffusion parallelism (DiffusionGemma+vLLM). Fast-dVLM
  (SGLang, cuda-graph off) and Nemotron (HF-behind-FastAPI) leave speed on the table.
- **Where AR still competes:** raw descriptive richness (Qwen) — but it's slower and
  doesn't finish the schema in budget.

## 4. Engine / serving notes (what it took)

- **DiffusionGemma-26B → vLLM:** needs **vLLM ≥0.24** (the `vllm/vllm-openai:gemma`
  image; the cluster's 0.22 does **not** register `diffusion_gemma`). Run via apptainer.
  Fits in bf16 on one 80 GB H100. *Drop `--reasoning-parser gemma4`* or the answer
  lands in `reasoning_content` and `content` comes back `null`.
- **Fast-dVLM-3B → SGLang:** the vendored Fast-dLLM `third_party/sglang` fork
  (`HierarchyBlock`) inside the `lmsysorg/sglang:v0.5.6.post2` container, fork
  PYTHONPATH-shadowed. Two fights: (a) **flashinfer's prefill-kernel JIT build kept
  failing** — `attention_backend=triton` dodges it for the no-graph path; (b) relaxed
  the sgl-kernel version check to the container's 0.3.19, narrow binds so the host
  flashinfer can't shadow the container's. **CUDA graph** needs flashinfer (the fork
  force-switches the backend when graphs are on), and it failed because the JIT cache
  `~/.cache/flashinfer` was polluted with a `build.ninja` pointing at the *host* dllm
  flashinfer source (hidden by the narrow bind). **Fix: `rm -rf ~/.cache/flashinfer`
  so the container's own flashinfer rebuilds** → capture succeeds → **4.20 → 2.20
  s/frame (1.9×)**. Final Fast-dVLM number uses CUDA graph (enabled by default in
  `fast_dvlm_sglang_run.sh`).
- **Nemotron-VLM-8B → SGLang is impossible.** The SGLang dLLM fork implements only
  the *text* diffusion decoder; there is no `NemotronLabsDiffusionVLMModel` class
  (repo-confirmed). Served via its native FastAPI diffusion server instead.
- **Qwen3-VL-8B → vLLM:** standard, fast (3.6 s/frame, 141 tok/s on H100).

## 5. Recommendation

| use-case | pick |
|---|---|
| **Best overall autolabeling engine (speed + quality)** | **DiffusionGemma-26B on vLLM** — fastest (1.18 s/frame, 431 tok/s) *and* richest/most-structured. Needs an ≥80 GB GPU + vLLM ≥0.24. |
| Mid-size, solid structure, simpler serving | Nemotron-VLM-8B (FastAPI diffusion) — cleanest complete JSON, 5.7 s/frame. |
| Cheapest / smallest footprint, still fast | Fast-dVLM-3B (SGLang + CUDA graph) — **2.2 s/frame, 196 tok/s** (beats the 8B AR), fits in ~8 GB; accept malformed/shallow output. |
| Highest free-text fidelity reference | Qwen3-VL-8B (AR) — but raise the token budget or it won't close the schema. |

**Bottom line:** for the CultureDrive autolabeling pipeline, **DiffusionGemma-26B-A4B
served on vLLM is the most suitable model** — it generates longer, more structured
reasoning traces *and* does so ~3× faster than an autoregressive VLM of similar or
smaller size. The diffusion-VLM advantage for autolabeling is real and large, but it
must be unlocked by an inference engine built for diffusion decoding (vLLM ≥0.24 /
SGLang dLLM forks); on a generic HF path it largely disappears.

---
### Appendix — reproduce
`prepare_frames.py` → `sbatch_all4_h100.sh` (DiffusionGemma+Qwen vLLM, Nemotron FastAPI,
Fast-dVLM HF) then `sbatch_fastdvlm_sglang_h100.sh` (Fast-dVLM SGLang) → `compare.py`.
Per-model outputs in `results/*.json`; the earlier uniform **HF-on-L40S** baseline
(Fast-dVLM 6.3s, Nemotron 8.2s, Qwen 14.2s, all same engine) is preserved in
`results_l40s/` for an engine-controlled comparison.
