#!/usr/bin/env bash
# Step 4: multi-lane sweep. For each N_LANES in {1, 2, 4}, run the same
# total handoff count (ITERS rounds × N_LANES lanes per round). Compare
# per-round latency: ideal parallelism → constant per-round time as lanes
# increase. PCIe / driver contention shows up as growing per-round time.
set -euo pipefail

IMAGE="${IMAGE:-tee-gramine:stage1}"
TEE="${TEE:-/home/user/tee}"
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-20}
BATCH=${BATCH:-4}

SHM_FILE=/dev/shm/loro_smoke_ml.bin
SHM_SIZE=$((256 * 1024 * 1024))
HERE=/workspace/proto/smoke_data
LOG_DIR=/home/user/tee/proto/smoke_data/logs_ml

cleanup() {
  docker rm -f loro-smoke-ml-enclave >/dev/null 2>&1 || true
  rm -f "$SHM_FILE"
}
trap cleanup EXIT
cleanup
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

echo "== build =="
docker run --rm --gpus all -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
  bash -lc "nvcc -O2 -o worker_ml worker_ml.cu -lcuda 2>&1; gcc -O2 -pthread -o enclave_ml enclave_ml.c 2>&1" \
  >"$LOG_DIR/build.log" 2>&1
ls -l "$TEE/proto/smoke_data/worker_ml" "$TEE/proto/smoke_data/enclave_ml"

run_lanes() {
  local LANES=$1
  echo
  echo "== N_LANES=$LANES =="

  fallocate -l "$SHM_SIZE" "$SHM_FILE"
  chmod 666 "$SHM_FILE"
  dd if=/dev/zero of="$SHM_FILE" bs=4096 count=1 conv=notrunc status=none

  docker rm -f loro-smoke-ml-enclave >/dev/null 2>&1 || true
  docker run -d --name loro-smoke-ml-enclave --ipc=host \
    --device=/dev/sgx_enclave --device=/dev/sgx_provision \
    -v "$TEE":/workspace -w /workspace "$IMAGE" \
    bash -lc "
      gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
          -Dra_type=none -Dentrypoint=/workspace/proto/smoke_data/enclave_ml \
          python.manifest.template enclave_ml.manifest >/dev/null 2>&1
      gramine-sgx-sign --manifest enclave_ml.manifest --output enclave_ml.manifest.sgx >/dev/null 2>&1
      exec gramine-sgx enclave_ml $LANES
    " >/dev/null

  # Wait for all lanes to be ready (each at its own LANE_HDR offset, READY at +8).
  local ready_ok=0
  for _ in $(seq 1 60); do
    ready_ok=1
    for L in $(seq 0 $((LANES-1))); do
      local offset=$((L * 64 + 8))
      local v=$(od -An -tu4 -N4 -j $offset "$SHM_FILE" 2>/dev/null | tr -d ' ')
      if [ "$v" != "1" ]; then ready_ok=0; break; fi
    done
    [ "$ready_ok" = "1" ] && break
    sleep 0.5
  done
  if [ "$ready_ok" != "1" ]; then
    echo "  enclave lanes never ready"; docker logs loro-smoke-ml-enclave 2>&1 | tail -10; rm -f "$SHM_FILE"; return 1
  fi

  docker run --rm --gpus all --ipc=host \
    -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
    bash -c "./worker_ml $LANES $ITERS $WARMUP $BATCH 16384 65536 262144 1048576 4194304 > $HERE/logs_ml/lanes${LANES}.log 2>&1; echo exit=\$? >> $HERE/logs_ml/lanes${LANES}.log"

  cat "$LOG_DIR/lanes${LANES}.log"
  docker logs loro-smoke-ml-enclave 2>&1 | grep -E "^\[enclave\]"

  rm -f "$SHM_FILE"
}

run_lanes 1
run_lanes 2
run_lanes 4
