#!/bin/bash
# Start the Nemotron-VLM FastAPI diffusion server (OpenAI-compatible) and label
# frames against it. SGLang can't serve this VLM arch (repo-confirmed), so this
# is the model's native diffusion serving path.
#   nemotron_serve_and_label.sh <limit|0>
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/paths.sh"
cd "$AUTOLABEL_DIR"; mkdir -p logs
PORT=${PORT:-30000}
LIMIT="${1:-0}"
LIMARG=""; [ "$LIMIT" != "0" ] && LIMARG="--limit $LIMIT"

echo "===== starting Nemotron-VLM FastAPI server on port $PORT ====="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY_NEMO \
  "$NEMOTRON_REPO/vlm_serve/serve_vlm.py" --port $PORT \
  > logs/nemotron_server.log 2>&1 &
SVPID=$!
echo "server pid $SVPID; waiting for health (max 600s)..."
ok=0
for i in $(seq 1 600); do
  if curl -fsS http://localhost:$PORT/health >/dev/null 2>&1; then echo "healthy after ${i}s"; ok=1; break; fi
  if ! kill -0 $SVPID 2>/dev/null; then echo "SERVER DIED — tail:"; tail -40 logs/nemotron_server.log; exit 1; fi
  sleep 1
done
[ $ok -eq 1 ] || { echo "TIMEOUT"; kill $SVPID 2>/dev/null; exit 1; }

$PY_CLIENT run_vllm_client.py --base-url http://localhost:$PORT/v1 \
  --model nemotron --key nemotron --name "Nemotron-Diffusion-VLM-8B (FastAPI diffusion)" \
  --note "native FastAPI diffusion server (model.generate); SGLang can't serve this VLM arch" $LIMARG
RC=$?
echo "stopping server $SVPID"; kill $SVPID 2>/dev/null; sleep 8; kill -9 $SVPID 2>/dev/null; sleep 3
exit $RC
