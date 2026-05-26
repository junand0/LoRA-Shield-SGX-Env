"""Benchmark client (TRUSTED, runs INSIDE the Gramine SGX enclave).
Same workload as worker.py; selects IPC mode (tcp/shm) and reports latency stats.
With --noop the worker skips GPU compute -> measures pure IPC overhead.
"""
import argparse
import mmap
import os
import socket
import struct
import time

import numpy as np

SHM_PATH = "/dev/shm/tee_ipc.bin"
SHM_SIZE = 64 * 1024 * 1024
HDR_SIZE = 64
H_MAGIC, H_W_SIZE, H_REQ_SEQ, H_RESP_SEQ, H_REQ_SIZE, H_RESP_SIZE, H_SHUTDOWN = 0, 4, 8, 12, 16, 20, 24
W_OFF = HDR_SIZE
W_MAX = 16 * 1024 * 1024
REQ_OFF = W_OFF + W_MAX
REQ_MAX = 16 * 1024 * 1024
RESP_OFF = REQ_OFF + REQ_MAX

HOST, PORT = "127.0.0.1", 5555


def u32(mm, off): return struct.unpack_from("<I", mm, off)[0]
def w32(mm, off, v): struct.pack_into("<I", mm, off, v)


def recv_all(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError
        b += c
    return bytes(b)


def connect_tcp():
    for _ in range(30):
        try:
            return socket.create_connection((HOST, PORT), timeout=3)
        except OSError:
            time.sleep(1)
    raise SystemExit("[client] cannot reach worker (tcp)")


def bench_tcp(iters, N, D, K, W, X, noop):
    s = connect_tcp()
    s.sendall(W.tobytes())
    nb_y = N * K * 4
    rng = np.random.default_rng(1)
    R = rng.standard_normal((N, D), dtype=np.float32)
    s.sendall((X + R).tobytes())
    _ = recv_all(s, nb_y)

    samples = []
    first_err = None
    for i in range(iters):
        R = rng.standard_normal((N, D), dtype=np.float32)
        Xp = X + R
        t0 = time.perf_counter()
        s.sendall(Xp.tobytes())
        raw = recv_all(s, nb_y)
        samples.append((time.perf_counter() - t0) * 1e6)
        if first_err is None and not noop:
            Yp = np.frombuffer(raw, dtype=np.float32).reshape(N, K)
            Y = Yp - R @ W
            first_err = float(np.max(np.abs(Y - X @ W)))
    s.close()
    return samples, first_err


def bench_shm(iters, N, D, K, W, X, noop):
    fd = os.open(SHM_PATH, os.O_RDWR)
    mm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    for _ in range(300):
        if u32(mm, H_MAGIC) == 0xCAFEBABE:
            break
        time.sleep(0.05)
    else:
        raise SystemExit("[client] worker not ready (shm)")

    w_bytes = W.tobytes()
    mm[W_OFF:W_OFF + len(w_bytes)] = w_bytes
    w32(mm, H_W_SIZE, len(w_bytes))

    rng = np.random.default_rng(1)
    seq = 0
    nb_x = N * D * 4
    nb_y = N * K * 4

    R = rng.standard_normal((N, D), dtype=np.float32)
    Xp = (X + R)
    mm[REQ_OFF:REQ_OFF + nb_x] = Xp.tobytes()
    w32(mm, H_REQ_SIZE, nb_x)
    seq += 1
    w32(mm, H_REQ_SEQ, seq)
    while u32(mm, H_RESP_SEQ) != seq:
        pass

    samples = []
    first_err = None
    for i in range(iters):
        R = rng.standard_normal((N, D), dtype=np.float32)
        Xp = X + R
        xp_bytes = Xp.tobytes()
        seq += 1
        t0 = time.perf_counter()
        mm[REQ_OFF:REQ_OFF + nb_x] = xp_bytes
        w32(mm, H_REQ_SIZE, nb_x)
        w32(mm, H_REQ_SEQ, seq)
        while u32(mm, H_RESP_SEQ) != seq:
            pass
        raw = bytes(mm[RESP_OFF:RESP_OFF + nb_y])
        samples.append((time.perf_counter() - t0) * 1e6)
        if first_err is None and not noop:
            Yp = np.frombuffer(raw, dtype=np.float32).reshape(N, K)
            Y = Yp - R @ W
            first_err = float(np.max(np.abs(Y - X @ W)))

    w32(mm, H_SHUTDOWN, 1)
    mm.close()
    os.close(fd)
    return samples, first_err


def stats(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "p50": s[n // 2],
        "p95": s[min(n - 1, int(n * 0.95))],
        "p99": s[min(n - 1, int(n * 0.99))],
        "min": s[0],
        "max": s[-1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["tcp", "shm"])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--D", type=int, default=1024)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--noop", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    W = rng.standard_normal((args.D, args.K), dtype=np.float32)
    X = rng.standard_normal((args.N, args.D), dtype=np.float32)

    fn = bench_tcp if args.mode == "tcp" else bench_shm
    samples, err = fn(args.iters, args.N, args.D, args.K, W, X, args.noop)
    s = stats(samples)
    payload_kb = (args.N * args.D * 4) / 1024
    suffix = "-noop" if args.noop else ""
    print(f"[client] mode={args.mode}{suffix} iters={s['n']} "
          f"shape=({args.N}x{args.D})@({args.D}x{args.K}) "
          f"payload_one_way={payload_kb:.3f}KB", flush=True)
    print(f"[client]   mean={s['mean']:8.1f}us  p50={s['p50']:8.1f}us  "
          f"p95={s['p95']:8.1f}us  p99={s['p99']:8.1f}us  "
          f"min={s['min']:8.1f}us  max={s['max']:8.1f}us", flush=True)
    if err is not None:
        print(f"[client]   first_iter_max_abs_err={err:.3e}", flush=True)


if __name__ == "__main__":
    main()
