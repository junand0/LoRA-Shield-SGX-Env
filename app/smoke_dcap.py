import os
import sys

print("[smoke] hello from inside the enclave; python", sys.version.split()[0], flush=True)

BASE = "/dev/attestation"


def read_text(name):
    with open(os.path.join(BASE, name)) as f:
        return f.read().strip()


try:
    print("[smoke] attestation_type:", read_text("attestation_type"), flush=True)
    with open(os.path.join(BASE, "user_report_data"), "wb") as f:
        f.write(b"\x11" * 64)
    with open(os.path.join(BASE, "quote"), "rb") as f:
        quote = f.read()
except FileNotFoundError as e:
    print("[smoke] attestation pseudo-file missing:", e, flush=True)
    sys.exit(2)

print("[smoke] DCAP quote generated:", len(quote), "bytes", flush=True)
print("[smoke] quote header[:8]:", quote[:8].hex(), flush=True)
print("[smoke] OK", flush=True)
