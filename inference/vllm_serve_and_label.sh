#!/bin/bash
# Serve one model with the vLLM container (OpenAI-compatible) and label frames
# against it. The server runs in the background of THIS script's lifetime, then
# is killed.  Works for DiffusionGemma-26B and Qwen3-VL-8B.
#   vllm_serve_and_label.sh <model_path> <served_name> <key> "<display>" <limit|0> [extra vllm flags...]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/../paths.sh"
cd "$AUTOLABEL_DIR"; mkdir -p logs
PORT=${PORT:-8000}
BINDS=""; for d in $BIND_DIRS; do BINDS="$BINDS -B $d:$d"; done

MPATH="$1"; SNAME="$2"; KEY="$3"; DISP="$4"; LIMIT="$5"; shift 5
LIMARG=""; [ "$LIMIT" != "0" ] && LIMARG="--limit $LIMIT"

echo "===== serving $SNAME ($MPATH) via vLLM on port $PORT ====="
apptainer exec --nv $BINDS --env HF_HOME=$MODEL_ROOT "$VLLM_SIF" \
  vllm serve "$MPATH" --served-model-name "$SNAME" \
    --gpu-memory-utilization 0.90 --max-num-seqs 4 \
    --host 0.0.0.0 --port $PORT "$@" \
    > logs/vllm_server_${KEY}.log 2>&1 &
SVPID=$!

echo "server pid $SVPID; waiting for health (max 1200s)..."
ok=0
for i in $(seq 1 1200); do
  if curl -fsS http://localhost:$PORT/health >/dev/null 2>&1; then echo "healthy after ${i}s"; ok=1; break; fi
  if ! kill -0 $SVPID 2>/dev/null; then echo "SERVER DIED — tail:"; tail -40 logs/vllm_server_${KEY}.log; exit 1; fi
  sleep 1
done
[ $ok -eq 1 ] || { echo "TIMEOUT waiting for health"; kill $SVPID 2>/dev/null; exit 1; }

$PY_CLIENT "$HERE/run_vllm_client.py" --base-url http://localhost:$PORT/v1 \
  --model "$SNAME" --key "$KEY" --name "$DISP" $LIMARG
RC=$?

echo "stopping server $SVPID"; kill $SVPID 2>/dev/null; sleep 8; kill -9 $SVPID 2>/dev/null; sleep 3
exit $RC
