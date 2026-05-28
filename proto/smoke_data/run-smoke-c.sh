#!/usr/bin/env bash
# Step 3: same payload sweep as Step 2 but with the C-based enclave responder.
# Compare per-handoff numbers to enclave.py (Python) baseline.
set -euo pipefail

IMAGE="${IMAGE:-tee-gramine:stage1}"
TEE="${TEE:-/home/user/tee}"
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-20}

SHM_FILE=/dev/shm/loro_smoke_data.bin
SHM_SIZE=$((256 * 1024 * 1024))
HERE=/workspace/proto/smoke_data
LOG_DIR=/home/user/tee/proto/smoke_data/logs_c

cleanup() {
  docker rm -f loro-smoke-c-enclave loro-smoke-c-worker >/dev/null 2>&1 || true
  rm -f "$SHM_FILE"
}
trap cleanup EXIT
cleanup
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

echo "== build worker + enclave_c =="
docker run --rm --gpus all -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
  bash -lc "nvcc -O2 -o worker worker.cu -lcuda 2>&1; gcc -O2 -o enclave_c enclave_c.c 2>&1" \
  >"$LOG_DIR/build.log" 2>&1
ls -l "$TEE/proto/smoke_data/worker" "$TEE/proto/smoke_data/enclave_c"

echo "== shared file =="
fallocate -l "$SHM_SIZE" "$SHM_FILE"
chmod 666 "$SHM_FILE"
dd if=/dev/zero of="$SHM_FILE" bs=4096 count=1 conv=notrunc status=none

echo "== start enclave (C binary under Gramine) =="
docker run -d --name loro-smoke-c-enclave --ipc=host \
  --device=/dev/sgx_enclave --device=/dev/sgx_provision \
  -v "$TEE":/workspace -w /workspace "$IMAGE" \
  bash -lc "
    gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
        -Dra_type=none -Dentrypoint=/workspace/proto/smoke_data/enclave_c \
        python.manifest.template enclave_c.manifest >/dev/null 2>&1
    gramine-sgx-sign --manifest enclave_c.manifest --output enclave_c.manifest.sgx >/dev/null 2>&1
    exec gramine-sgx enclave_c
  " >/dev/null

for _ in $(seq 1 60); do
  v=$(od -An -tu4 -N4 -j 12 "$SHM_FILE" 2>/dev/null | tr -d ' ')
  [ "$v" = "1" ] && break
  sleep 0.5
done
[ "$v" = "1" ] || { docker logs loro-smoke-c-enclave 2>&1 | tail -20; exit 1; }
echo "   enclave ready"

echo "== sweep (iters=$ITERS warmup=$WARMUP) — C enclave responder =="
docker run --rm --name loro-smoke-c-worker --gpus all --ipc=host \
  -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
  bash -c "./worker $ITERS $WARMUP 16384 65536 262144 1048576 4194304 16777216 67108864 > $HERE/logs_c/sweep.log 2>&1; echo exit=\$? >> $HERE/logs_c/sweep.log"

echo
echo "== results =="
cat "$LOG_DIR/sweep.log"
echo
printf '\xff\xff\xff\xff' | dd of="$SHM_FILE" bs=1 seek=8 count=4 conv=notrunc status=none
sleep 0.5
docker logs loro-smoke-c-enclave 2>&1 | grep -E "^\[enclave\]"
