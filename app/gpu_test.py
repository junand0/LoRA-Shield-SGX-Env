import sys

import torch

print("[gpu] torch", torch.__version__, flush=True)
print("[gpu] cuda.is_available:", torch.cuda.is_available(), flush=True)
print("[gpu] device_count:", torch.cuda.device_count(), flush=True)

x = torch.tensor([1.0, 2.0, 3.0]).cuda()
y = x * 2
torch.cuda.synchronize()
print("[gpu] result:", y.detach().cpu().tolist(), flush=True)
print("[gpu] OK", flush=True)
