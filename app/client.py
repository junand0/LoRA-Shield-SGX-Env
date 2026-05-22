"""Trusted client (runs INSIDE the Gramine SGX enclave).

Holds the secrets (data X, and in the real system the LoRA weights), generates a
one-time pad R, sends only masked data X' = X + R to the untrusted GPU worker,
then unmasks the returned result and verifies correctness. No CUDA here.

This is an ENVIRONMENT-validation prototype: it exercises enclave execution +
RNG + TCP IPC + GPU offload round-trip. The real masking/LoRA logic is the
user's; this just proves the plumbing.
"""
import io
import socket
import struct
import time

import numpy as np

HOST, PORT = "127.0.0.1", 5555


def recv_all(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def recv_array(conn):
    (length,) = struct.unpack("!Q", recv_all(conn, 8))
    return np.load(io.BytesIO(recv_all(conn, length)), allow_pickle=False)


def send_array(conn, arr):
    bio = io.BytesIO()
    np.save(bio, np.ascontiguousarray(arr), allow_pickle=False)
    data = bio.getvalue()
    conn.sendall(struct.pack("!Q", len(data)) + data)


def connect_with_retry(host, port, attempts=30):
    for _ in range(attempts):
        try:
            return socket.create_connection((host, port), timeout=3)
        except OSError:
            time.sleep(1)
    raise SystemExit("[client] could not reach GPU worker")


def main():
    rng = np.random.default_rng()
    n, d, k = 64, 1024, 1024
    X = rng.standard_normal((n, d), dtype=np.float32)   # secret activations
    W = rng.standard_normal((d, k), dtype=np.float32)   # public weight (lives on GPU)
    R = rng.standard_normal((n, d), dtype=np.float32)   # one-time pad

    conn = connect_with_retry(HOST, PORT)
    print("[client] connected to GPU worker", flush=True)
    send_array(conn, W)

    Xp = X + R                       # mask: only this leaves the enclave
    t0 = time.time()
    send_array(conn, Xp)
    Yp = recv_array(conn)            # = (X+R) @ W  (still masked)
    dt = (time.time() - t0) * 1000

    Y = Yp - (R @ W)                 # unmask (prototype: CPU correction; prod: precompute)
    Y_ref = X @ W
    err = float(np.max(np.abs(Y - Y_ref)))

    print(f"[client] gpu round-trip={dt:.1f}ms  max_abs_err={err:.3e}", flush=True)
    print("[client] PASS: OTP mask -> GPU -> unmask verified" if err < 1e-1
          else f"[client] FAIL: error too high ({err:.3e})", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
