#!/usr/bin/env bash
# Benchmark IPC overhead: TCP vs untrusted_shm.
# Runs 5 cases to decompose end-to-end cost:
#   tcp 256KB gpu   ;  tcp 256KB noop
#   shm 256KB gpu   ;  shm 256KB noop   ;  shm tiny noop (16B)
# noop  = worker skips GPU compute (returns zeros), so RTT measures pure IPC.
# tiny  = strips data-transfer cost too, leaving the sync/poll floor.
set -euo pipefail

IMAGE="${IMAGE:-tee-gramine:stage1}"
TEE="${TEE:-/home/user/tee}"
ITERS=${ITERS:-200}

SHM_FILE=/dev/shm/tee_ipc.bin
SHM_SIZE=$((64 * 1024 * 1024))

cleanup() { docker rm -f tee-worker >/dev/null 2>&1 || true; rm -f "$SHM_FILE"; }
trap cleanup EXIT
cleanup

bench_one() {
  local LABEL=$1 MODE=$2 NOOP=$3 LN=$4 LD=$5 LK=$6
  echo
  echo "================ $LABEL (iters=$ITERS, payload=$(awk -v n=$LN -v d=$LD 'BEGIN{printf "%.3fKB", n*d*4/1024}')) ================"
  docker rm -f tee-worker >/dev/null 2>&1 || true
  rm -f "$SHM_FILE"
  if [ "$MODE" = "shm" ]; then
    fallocate -l "$SHM_SIZE" "$SHM_FILE"
    chmod 666 "$SHM_FILE"
  fi

  docker run -d --name tee-worker --gpus all --network host --ipc=host \
    -v "$TEE":/workspace -w /workspace "$IMAGE" \
    python3 proto/bench_worker.py "$MODE" --N $LN --D $LD --K $LK $NOOP >/dev/null

  for _ in $(seq 1 30); do
    if [ "$MODE" = "tcp" ]; then
      docker logs tee-worker 2>&1 | grep -q "listening on" && break
    else
      docker logs tee-worker 2>&1 | grep -q "shm ready" && break
    fi
    sleep 1
  done

  docker run --rm --network host --ipc=host \
    --device=/dev/sgx_enclave --device=/dev/sgx_provision \
    -v "$TEE":/workspace -w /workspace "$IMAGE" \
    bash -lc "
      PYBIN=\$(readlink -f \"\$(command -v python3)\")
      gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
          -Dra_type=none -Dentrypoint=\"\$PYBIN\" \
          python.manifest.template python.manifest >/dev/null 2>&1
      gramine-sgx-sign --manifest python.manifest --output python.manifest.sgx >/dev/null 2>&1
      gramine-sgx python /workspace/app/bench_client.py $MODE --iters $ITERS --N $LN --D $LD --K $LK $NOOP
    " 2>&1 | grep -E "^\[client\]"
}

bench_one "TCP   256KB  +GPU"  tcp  ""       64 1024 1024
bench_one "TCP   256KB   noop" tcp  --noop   64 1024 1024
bench_one "SHM   256KB  +GPU"  shm  ""       64 1024 1024
bench_one "SHM   256KB   noop" shm  --noop   64 1024 1024
bench_one "SHM   16B     noop" shm  --noop   1  4    4

echo
echo "================ interpretation ================"
echo "  (whole256KB) - (noop256KB) = GPU matmul + cudaMemcpy + sync"
echo "  (noop256KB)  - (tiny noop) = data-transfer cost (2x 256KB memcpy)"
echo "  (tiny noop)               ~= pure sync floor (poll + atomic flags + python overhead)"
