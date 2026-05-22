#!/usr/bin/env bash
# Split-architecture validation:
#   - untrusted GPU worker (normal CUDA) in one container
#   - trusted OTP client in a Gramine SGX enclave in another container
#   - they talk over TCP (host network)
set -euo pipefail

IMAGE="${IMAGE:-tee-gramine:stage1}"
TEE="${TEE:-/home/user/tee}"

cleanup() { docker rm -f tee-worker >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

echo "== starting untrusted GPU worker =="
docker run -d --name tee-worker --gpus all --network host \
  -v "$TEE":/workspace -w /workspace "$IMAGE" \
  python3 proto/worker.py >/dev/null

# wait until the worker is listening
for _ in $(seq 1 30); do
  docker logs tee-worker 2>&1 | grep -q "listening on" && break
  sleep 1
done
docker logs tee-worker 2>&1 | grep -E "worker]" || true

echo "== starting trusted OTP client in SGX enclave =="
docker run --rm --network host \
  --device=/dev/sgx_enclave --device=/dev/sgx_provision \
  -v "$TEE":/workspace -w /workspace "$IMAGE" \
  bash -lc '
    PYBIN=$(readlink -f "$(command -v python3)")
    gramine-manifest -Dlog_level=error -Darch_libdir=/lib/x86_64-linux-gnu \
        -Dra_type=none -Dentrypoint="$PYBIN" \
        python.manifest.template python.manifest >/dev/null 2>&1
    gramine-sgx-sign --manifest python.manifest --output python.manifest.sgx >/dev/null 2>&1
    gramine-sgx python /workspace/app/client.py
  '

echo "== worker final logs =="
docker logs tee-worker 2>&1 | grep -E "worker]" || true
