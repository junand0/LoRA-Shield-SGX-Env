"""Enclave side of the cuStreamWaitValue smoke test.

Runs INSIDE the Gramine SGX enclave (no CUDA here). Tight-spins on the
worker_to_enclave flag via untrusted_shm mmap and writes the response flag.
Python loop overhead is the ~us-class polling cadence on this side; the GPU
spin on the other side is faster (hardware). Combined RTT is what worker.cu
reports.
"""
import mmap
import os
import struct
import sys
import time

SHM_PATH = "/dev/shm/loro_smoke.bin"
SHM_SIZE = 4096
OFF_W2E = 0
OFF_E2W = 4
OFF_SHUTDOWN = 8
OFF_READY = 12


def main():
    fd = os.open(SHM_PATH, os.O_RDWR)
    mm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    # Zero header (whole region).
    mm[0:SHM_SIZE] = b"\x00" * SHM_SIZE

    # Signal ready.
    struct.pack_into("<I", mm, OFF_READY, 1)
    print("[enclave] ready; tight-spinning on /dev/shm/loro_smoke.bin", flush=True)

    expected = 1
    served = 0
    last = time.perf_counter()
    while True:
        v = struct.unpack_from("<I", mm, OFF_W2E)[0]
        if v == 0xFFFFFFFF:
            break
        if v == expected:
            struct.pack_into("<I", mm, OFF_E2W, expected)
            expected += 1
            served += 1
        # No sleep: tight spin.

    elapsed = time.perf_counter() - last
    print(f"[enclave] shutdown; served={served} elapsed={elapsed:.2f}s "
          f"avg_iter={(elapsed * 1e6 / max(served, 1)):.2f}us", flush=True)

    mm.close()
    os.close(fd)


if __name__ == "__main__":
    main()
