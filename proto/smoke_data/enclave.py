"""Step 2 enclave responder with session-reset detection.

A new worker session is detected by watching the worker-to-enclave flag: if
it goes back DOWN (from a previously seen high value to a low value), that
signals a new worker has connected and we reset our expected counter to 1.
"""
import mmap
import os
import struct

SHM_PATH = "/dev/shm/loro_smoke_data.bin"
SHM_SIZE = 4 * 1024 * 1024
OFF_W2E = 0
OFF_E2W = 4
OFF_SHUTDOWN = 8
OFF_READY = 12


def main():
    fd = os.open(SHM_PATH, os.O_RDWR)
    mm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    mm[0:4096] = b"\x00" * 4096
    struct.pack_into("<I", mm, OFF_READY, 1)
    print("[enclave] ready; session-aware tight spin", flush=True)

    expected = 1
    prev_w2e = 0
    sessions = 0
    served = 0
    try:
        while True:
            v = struct.unpack_from("<I", mm, OFF_W2E)[0]
            if v == 0xFFFFFFFF:
                break
            # Detect new session: W2E went down (new worker, seq restart).
            if v != 0 and v < prev_w2e:
                expected = 1
                struct.pack_into("<I", mm, OFF_E2W, 0)
                sessions += 1
                print(f"[enclave] session reset #{sessions}: prev_w2e={prev_w2e} new_w2e={v}", flush=True)
            prev_w2e = v
            if v == expected:
                struct.pack_into("<I", mm, OFF_E2W, expected)
                expected += 1
                served += 1
    finally:
        print(f"[enclave] shutdown; sessions={sessions} total_served={served}", flush=True)
        mm.close()
        os.close(fd)


if __name__ == "__main__":
    main()
