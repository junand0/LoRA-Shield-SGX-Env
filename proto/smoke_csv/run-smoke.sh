#!/usr/bin/env bash
# Step-1 smoke test for the LoRO-on-SGX design:
#   - enclave (Gramine SGX) mmaps /dev/shm/loro_smoke.bin via untrusted_shm
#   - worker (--gpus all) mmaps the same file and cudaHostRegisters it
#   - worker GPU stream issues cuStreamWriteValue32 / cuStreamWaitValue32
#   - enclave (Python) tight-spins on the worker->enclave flag and replies
# Measures per-handoff round-trip latency.
set -euo pipefail

IMAGE="${IMAGE:-tee-gramine:stage1}"
TEE="${TEE:-/home/user/tee}"
ITERS=${ITERS:-10000}
WARMUP=${WARMUP:-100}

SHM_FILE=/dev/shm/loro_smoke.bin
HERE=/workspace/proto/smoke_csv

cleanup() {
  docker rm -f loro-smoke-enclave >/dev/null 2>&1 || true
  rm -f "$SHM_FILE"
}
trap cleanup EXIT
cleanup

echo "== building worker.cu =="
docker run --rm --gpus all \
  -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
  bash -lc "nvcc -O2 -o worker worker.cu -lcuda 2>&1 | tail -20"
ls -l /home/user/tee/proto/smoke_csv/worker

echo "== creating shared file =="
fallocate -l 4096 "$SHM_FILE"
chmod 666 "$SHM_FILE"
# Zero the ready flag so we know enclave sets it.
printf '\x00\x00\x00\x00' | dd of="$SHM_FILE" bs=1 seek=12 count=4 conv=notrunc status=none

echo "== starting enclave (Gramine SGX) =="
docker run -d --name loro-smoke-enclave --ipc=host \
  --device=/dev/sgx_enclave --device=/dev/sgx_provision \
  -v "$TEE":/workspace -w /workspace "$IMAGE" \
  bash -lc "
    PYBIN=\$(readlink -f \"\$(command -v python3)\")
    gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
        -Dra_type=none -Dentrypoint=\"\$PYBIN\" \
        python.manifest.template python.manifest >/dev/null 2>&1
    gramine-sgx-sign --manifest python.manifest --output python.manifest.sgx >/dev/null 2>&1
    exec gramine-sgx python /workspace/proto/smoke_csv/enclave.py
  " >/dev/null

echo "== waiting for enclave ready flag =="
ready=0
for i in $(seq 1 60); do
  val=$(od -An -tu4 -N4 -j 12 "$SHM_FILE" 2>/dev/null | tr -d ' ' || true)
  if [ "${val:-0}" = "1" ]; then ready=1; break; fi
  sleep 0.5
done
if [ "$ready" != "1" ]; then
  echo "!! enclave never signaled ready; logs:"
  docker logs loro-smoke-enclave 2>&1 | tail -30
  exit 1
fi
echo "   enclave ready"

echo "== running worker =="
docker run --rm --gpus all --ipc=host \
  -v "$TEE":/workspace -w "$HERE" "$IMAGE" \
  ./worker "$ITERS" "$WARMUP" 2>&1 \
  | grep -vE "Copyright|reserved|GOVERNING|found at|Idiap|NEC|NYU|Deepmind|Google|Caffe|Facebook|Yangqing|CORPORATION|Various files|PyTorch Version|NVIDIA Release|^==|SHMEM|insufficient|NVIDIA recommends|docker run --gpus all --ipc|NVIDIA Driver was|Container Toolkit|cloud-native|Product-Specific|governed|agreements|^\s*$"

sleep 0.5
echo "== enclave logs =="
docker logs loro-smoke-enclave 2>&1 | grep -E "^\[enclave\]" | tail -5
