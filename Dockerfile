# syntax=docker/dockerfile:1
# Stage 1: NVIDIA PyTorch base + Gramine + SGX/DCAP runtime (CPU validation only).
# GPU passthrough (/dev/nvidia*, sys.allowed_ioctls/ioctl_structs) is layered on later.
ARG BASE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Pin Gramine version (latest stable line is v1.9). Override with --build-arg if needed.
ARG GRAMINE_PKG=gramine

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release && \
    install -m0755 -d /etc/apt/keyrings && \
    CODENAME="$(lsb_release -sc)" && \
    # --- Gramine apt repo (per-distro signing key) ---
    curl -fsSL "https://packages.gramineproject.io/gramine-keyring-${CODENAME}.gpg" \
        -o "/etc/apt/keyrings/gramine-keyring-${CODENAME}.gpg" && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/gramine-keyring-${CODENAME}.gpg] https://packages.gramineproject.io/ ${CODENAME} main" \
        > /etc/apt/sources.list.d/gramine.list && \
    # --- Intel SGX apt repo (same line as host packages) ---
    curl -fsSL https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key \
        -o /etc/apt/keyrings/intel-sgx-deb.asc && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/intel-sgx-deb.asc] https://download.01.org/intel-sgx/sgx_repo/ubuntu ${CODENAME} main" \
        > /etc/apt/sources.list.d/intel-sgx.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        "${GRAMINE_PKG}" \
        libsgx-urts libsgx-enclave-common \
        libsgx-dcap-ql libsgx-dcap-default-qpl libsgx-quote-ex && \
    rm -rf /var/lib/apt/lists/*

# DEV-ONLY enclave signing key. Production signing must use a managed/HSM key.
RUN gramine-sgx-gen-private-key

WORKDIR /workspace
CMD ["/bin/bash"]
