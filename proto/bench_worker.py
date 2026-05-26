"""Benchmark worker (untrusted, runs OUTSIDE the enclave). Two IPC modes:
   tcp: framed-bytes TCP server
   shm: untrusted shared memory at /dev/shm/tee_ipc.bin
With --noop the GPU compute is skipped (response is zeros of the same size) so
the round-trip measures pure IPC + memcpy + Python overhead.
"""
import argparse
import mmap
import os
import socket
import struct

import numpy as np
import torch

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


def run_tcp(N, D, K, dev, noop):
    label = "TCP-noop" if noop else "TCP"
    print(f"[worker] {label} mode, device={dev}", flush=True)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[worker] listening on {HOST}:{PORT}", flush=True)
    conn, _ = srv.accept()
    nb_x = N * D * 4
    nb_y = N * K * 4
    zeros = b"\x00" * nb_y
    with conn:
        w_bytes = recv_all(conn, D * K * 4)
        W = None if noop else torch.from_numpy(np.frombuffer(w_bytes, dtype=np.float32).reshape(D, K).copy()).to(dev)
        served = 0
        try:
            while True:
                xb = recv_all(conn, nb_x)
                if noop:
                    conn.sendall(zeros)
                else:
                    Xt = torch.from_numpy(np.frombuffer(xb, dtype=np.float32).reshape(N, D).copy()).to(dev)
                    Yt = Xt @ W
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    conn.sendall(Yt.cpu().numpy().tobytes())
                served += 1
        except ConnectionError:
            pass
    print(f"[worker] {label} done, served {served}", flush=True)


def run_shm(N, D, K, dev, noop):
    label = "SHM-noop" if noop else "SHM"
    print(f"[worker] {label} mode, device={dev}", flush=True)
    fd = os.open(SHM_PATH, os.O_RDWR)
    mm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    mm[0:HDR_SIZE] = b"\x00" * HDR_SIZE
    w32(mm, H_MAGIC, 0xCAFEBABE)
    print(f"[worker] shm ready at {SHM_PATH}", flush=True)

    nb_x = N * D * 4
    nb_y = N * K * 4
    zeros = b"\x00" * nb_y
    W_tensor = None
    last_seq = 0
    served = 0
    try:
        while True:
            if u32(mm, H_SHUTDOWN):
                break
            cur = u32(mm, H_REQ_SEQ)
            if cur == last_seq:
                continue
            last_seq = cur
            if noop:
                mm[RESP_OFF:RESP_OFF + nb_y] = zeros
                w32(mm, H_RESP_SIZE, nb_y)
                w32(mm, H_RESP_SEQ, cur)
            else:
                if W_tensor is None:
                    assert u32(mm, H_W_SIZE) == D * K * 4
                    W_np = np.frombuffer(bytes(mm[W_OFF:W_OFF + D * K * 4]), dtype=np.float32).reshape(D, K)
                    W_tensor = torch.from_numpy(W_np.copy()).to(dev)
                assert u32(mm, H_REQ_SIZE) == nb_x
                Xp = np.frombuffer(bytes(mm[REQ_OFF:REQ_OFF + nb_x]), dtype=np.float32).reshape(N, D)
                Xt = torch.from_numpy(Xp.copy()).to(dev)
                Yt = Xt @ W_tensor
                if dev == "cuda":
                    torch.cuda.synchronize()
                yb = Yt.cpu().numpy().tobytes()
                mm[RESP_OFF:RESP_OFF + nb_y] = yb
                w32(mm, H_RESP_SIZE, nb_y)
                w32(mm, H_RESP_SEQ, cur)
            served += 1
    finally:
        mm.close()
        os.close(fd)
    print(f"[worker] {label} done, served {served}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["tcp", "shm"])
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--D", type=int, default=1024)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--noop", action="store_true", help="skip GPU compute (measure pure IPC)")
    args = p.parse_args()
    dev = "cuda" if (torch.cuda.is_available() and not args.noop) else "cpu"
    (run_tcp if args.mode == "tcp" else run_shm)(args.N, args.D, args.K, dev, args.noop)


if __name__ == "__main__":
    main()
