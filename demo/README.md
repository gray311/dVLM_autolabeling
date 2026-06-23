# Demo — one-click autolabeling

6 bundled driving frames (`demo/frames/`) you can label out-of-the-box to see the
pipeline produce structured reasoning traces.

## Run

```bash
# from the repo root: edit ../paths.sh once (MODEL_ROOT, container .sif paths, conda pythons)
cd demo
bash run_demo.sh                 # default: Fast-dVLM-3B via SGLang (small, ~2 s/frame)
# or pick another model/engine:
bash run_demo.sh diffusiongemma  # DiffusionGemma-26B via vLLM   (needs >=80GB GPU)
bash run_demo.sh qwen3vl         # Qwen3-VL-8B via vLLM          (AR control)
bash run_demo.sh nemotron        # Nemotron-VLM-8B FastAPI diffusion
```

Run on a GPU node. It serves the model, labels all 6 frames, writes
`demo/results/<model>.json`, and prints the speed summary + a sample annotation.

## What you should see

`example_output_fast_dvlm.json` is a committed sample (Fast-dVLM, SGLang+CUDA-graph,
H100): **6 frames, ~2.1 s/frame, ~207 tok/s**, each a structured JSON scene analysis,
e.g.:

```json
{
  "scene_description": "The image shows a snowy street at night with a car driving on...",
  "traffic_elements": { ... },
  "dynamic_agents": [ ... ],
  "prediction": { ... },
  "planning": { ... },
  "region_inference": "...",
  "reasoning": "..."
}
```

Prereqs and engine details: see [`../README.md`](../README.md).
