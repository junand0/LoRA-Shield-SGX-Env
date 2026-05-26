#!/usr/bin/env bash
# Single-process sweep: one worker container runs all payload sizes.
set -euo pipefail

IMAGE="${IMAGE:-tee-gramine:stage1}"
TEE="${TEE:-/home/user/tee}"
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-20}

SHM_FILE=/dev/shm/loro_smoke_data.bin
SHM_SIZE=$((256 * 1024 * 1024))
HERE=/workspace/proto/smoke_data
LOG_DIR=/home/user/tee/proto/smoke_data/logs

cleanup() {
  docker rm -f loro-smoke-data-enclave loro-smoke-data-worker >/dev/null 2>&1 || true
  rm -f "$SHM_FILE"
}
trap cleanup EXIT
cleanup
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

echo "== build =="
docker run --rm --gpus all -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
  bash -lc "nvcc -O2 -o worker worker.cu -lcuda 2>&1 | tail -5" >"$LOG_DIR/build.log" 2>&1
ls -l "$TEE/proto/smoke_data/worker"

echo "== shared file =="
fallocate -l "$SHM_SIZE" "$SHM_FILE"
chmod 666 "$SHM_FILE"
dd if=/dev/zero of="$SHM_FILE" bs=4096 count=1 conv=notrunc status=none

echo "== start enclave =="
docker run -d --name loro-smoke-data-enclave --ipc=host \
  --device=/dev/sgx_enclave --device=/dev/sgx_provision \
  -v "$TEE":/workspace -w /workspace "$IMAGE" \
  bash -lc "
    PYBIN=\$(readlink -f \"\$(command -v python3)\")
    gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
        -Dra_type=none -Dentrypoint=\"\$PYBIN\" \
        python.manifest.template python.manifest >/dev/null 2>&1
    gramine-sgx-sign --manifest python.manifest --output python.manifest.sgx >/dev/null 2>&1
    exec gramine-sgx python /workspace/proto/smoke_data/enclave.py
  " >/dev/null

for _ in $(seq 1 60); do
  v=$(od -An -tu4 -N4 -j 12 "$SHM_FILE" 2>/dev/null | tr -d ' ')
  [ "$v" = "1" ] && break
  sleep 0.5
done
[ "$v" = "1" ] || { docker logs loro-smoke-data-enclave 2>&1 | tail -10; exit 1; }
echo "   enclave ready"

echo "== sweep (iters=$ITERS warmup=$WARMUP) =="
docker run --rm --name loro-smoke-data-worker --gpus all --ipc=host \
  -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
  bash -c "./worker $ITERS $WARMUP 16384 65536 262144 1048576 4194304 16777216 67108864 > $HERE/logs/sweep.log 2>&1; echo exit=\$? >> $HERE/logs/sweep.log"

echo
echo "== results =="
cat "$LOG_DIR/sweep.log"
echo
printf '\xff\xff\xff\xff' | dd of="$SHM_FILE" bs=1 seek=8 count=4 conv=notrunc status=none
sleep 0.5
docker logs loro-smoke-data-enclave 2>&1 | grep -E "^\[enclave\]"
