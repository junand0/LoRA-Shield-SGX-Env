#!/usr/bin/env bash
# Step 7a: sweep BATCH={1, 4, 16, 64} across payload sizes.
# BATCH=1 = baseline (Step 2 numbers). Larger BATCH amortizes the
# cudaStreamSynchronize overhead so we can see how much of the floor it is.
set -euo pipefail

IMAGE="${IMAGE:-tee-gramine:stage1}"
TEE="${TEE:-/home/user/tee}"
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-20}

SHM_FILE=/dev/shm/loro_smoke_data.bin
SHM_SIZE=$((256 * 1024 * 1024))
HERE=/workspace/proto/smoke_data
LOG_DIR=/home/user/tee/proto/smoke_data/logs_batch

cleanup() {
  docker rm -f loro-smoke-batch-enclave >/dev/null 2>&1 || true
  rm -f "$SHM_FILE"
}
trap cleanup EXIT
cleanup
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

echo "== build =="
docker run --rm --gpus all -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
  bash -lc "nvcc -O2 -o worker worker.cu -lcuda 2>&1; gcc -O2 -o enclave_c enclave_c.c 2>&1" \
  >"$LOG_DIR/build.log" 2>&1
ls -l "$TEE/proto/smoke_data/worker" "$TEE/proto/smoke_data/enclave_c"

echo "== shared file =="
fallocate -l "$SHM_SIZE" "$SHM_FILE"
chmod 666 "$SHM_FILE"
dd if=/dev/zero of="$SHM_FILE" bs=4096 count=1 conv=notrunc status=none

echo "== start C enclave =="
docker run -d --name loro-smoke-batch-enclave --ipc=host \
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
[ "$v" = "1" ] || { docker logs loro-smoke-batch-enclave 2>&1 | tail -20; exit 1; }
echo "   enclave ready"

for BATCH in 1 4 16 64; do
  echo
  echo "== BATCH=$BATCH =="
  docker run --rm --gpus all --ipc=host \
    -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
    bash -c "./worker $ITERS $WARMUP $BATCH 16384 65536 262144 1048576 4194304 16777216 > $HERE/logs_batch/b${BATCH}.log 2>&1; echo exit=\$? >> $HERE/logs_batch/b${BATCH}.log"
  cat "$LOG_DIR/b${BATCH}.log"
done

echo
printf '\xff\xff\xff\xff' | dd of="$SHM_FILE" bs=1 seek=8 count=4 conv=notrunc status=none
sleep 0.5
docker logs loro-smoke-batch-enclave 2>&1 | grep -E "^\[enclave\]"
