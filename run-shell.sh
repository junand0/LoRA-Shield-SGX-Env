#!/usr/bin/env bash
# Launch the Stage 1 image with full SGX + DCAP wiring for THIS host, drop to a shell.
# Encodes the host facts established in Phase 0:
#   - /dev/sgx_enclave + /dev/sgx_provision : enclave run + DCAP quote generation
#   - aesm socket                            : architectural enclaves / out-of-proc quote
#   - /etc/sgx_default_qcnl.conf (ro)        : QPL -> local PCCS (use_secure_cert=false)
#   - --network host                         : PCCS binds 127.0.0.1 only, container must reach it
set -euo pipefail

IMAGE="${IMAGE:-tee-gramine:stage1}"

exec docker run --rm -it \
  --device=/dev/sgx_enclave \
  --device=/dev/sgx_provision \
  -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
  -v /etc/sgx_default_qcnl.conf:/etc/sgx_default_qcnl.conf:ro \
  --network host \
  -v "$PWD":/workspace \
  -w /workspace \
  "$IMAGE" /bin/bash
