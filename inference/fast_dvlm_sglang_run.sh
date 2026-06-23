#!/bin/bash
# Run Fast-dVLM-3B autolabeling via the vendored Fast-dLLM SGLang fork
# (HierarchyBlock block-diffusion) inside the lmsysorg/sglang container, with the
# fork PYTHONPATH-shadowed and qwen_vl_utils installed to a writable userbase.
#   fast_dvlm_sglang_run.sh <limit|0>
#
# Env toggles:
#   ENABLE_CG=1 (default)  -> CUDA graph ON  (~1.9x faster; forces flashinfer backend)
#   ENABLE_CG=0            -> CUDA graph OFF (uses --attention-backend triton)
#   ATTN_BACKEND=triton    -> attention backend when CUDA graph is OFF
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/../paths.sh"
cd "$AUTOLABEL_DIR"; mkdir -p logs
LIMIT="${1:-0}"
LIMARG=""; [ "$LIMIT" != "0" ] && LIMARG="--limit $LIMIT"
# CUDA graph on by default (the triton backend captures fine once the flashinfer
# JIT cache is clean — see README); ENABLE_CG=0 to disable.
CG_FLAG="--disable-cuda-graph"; [ "${ENABLE_CG:-1}" = 1 ] && CG_FLAG=""

HFREAL="$MODEL_ROOT"
mkdir -p "$SGLANG_USERBASE" "$HFREAL/modules"
BINDS=""; for d in $BIND_DIRS; do BINDS="$BINDS -B $d:$d"; done
# Pass through frames/results overrides (used by demo/run_demo.sh) into the container.
XENV=""
for v in FRAMES_DIR MANIFEST RESULTS_DIR; do
  [ -n "${!v:-}" ] && XENV="$XENV --env $v=${!v}"
done

# IMPORTANT: if CUDA graph capture fails with a flashinfer "missing source file"
# ninja error, the JIT cache is stale -> `rm -rf ~/.cache/flashinfer` and re-run.

# --cleanenv + NARROW binds keep the host conda's flashinfer from shadowing the
# container's (which breaks the JIT build).
apptainer exec --nv --cleanenv $BINDS $XENV \
  --env PYTHONPATH=$FASTDLLM_FORK \
  --env PYTHONUSERBASE=$SGLANG_USERBASE \
  --env MODEL_ROOT=$HFREAL \
  --env HF_HOME=$HFREAL \
  --env HF_MODULES_CACHE=$HFREAL/modules \
  --env TRANSFORMERS_CACHE=$HFREAL/transformers \
  --env HF_HUB_OFFLINE=1 \
  --env SGLANG_DISABLE_CUDNN_CHECK=1 \
  --env PATH=/usr/local/bin:/usr/bin:/bin \
  "$SGLANG_SIF" bash -lc "
    python -c 'import qwen_vl_utils' 2>/dev/null || pip install --user -q --break-system-packages --no-deps qwen_vl_utils packaging
    python $HERE/run_fast_dvlm_sglang.py --algorithm mdm ${CG_FLAG} --mem-fraction-static 0.7 --attention-backend ${ATTN_BACKEND:-triton} $LIMARG
  "
