"""Untrusted GPU worker (runs OUTSIDE the enclave, normal CUDA).

Receives a public weight W once, then masked activations X', computes X'@W on
the GPU, and returns the (still-masked) result. It never sees unmasked data.
This file is intentionally outside the enclave trust boundary.
"""
import io
import socket
import struct

import numpy as np
import torch

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


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[worker] torch={torch.__version__} device={dev} gpus={torch.cuda.device_count()}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[worker] listening on {HOST}:{PORT}", flush=True)

    conn, addr = srv.accept()
    print(f"[worker] client connected from {addr}", flush=True)
    with conn:
        W = torch.from_numpy(recv_array(conn)).to(dev)
        print(f"[worker] received W shape={tuple(W.shape)}", flush=True)
        n = 0
        while True:
            try:
                Xp = recv_array(conn)
            except ConnectionError:
                break
            Xt = torch.from_numpy(Xp).to(dev)
            Yt = Xt @ W
            if dev == "cuda":
                torch.cuda.synchronize()
            send_array(conn, Yt.cpu().numpy())
            n += 1
        print(f"[worker] served {n} request(s), exiting", flush=True)


if __name__ == "__main__":
    main()
