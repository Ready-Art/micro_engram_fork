"""Pull the micro_engram weights out of merged safetensors into micro_engram.pt.

usage: python extract_engram.py /path/to/merged_model_dir [micro_engram.pt]
"""
import glob
import os
import sys

import torch
from safetensors import safe_open

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "micro_engram.pt"
sd = {}
for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
    with safe_open(f, "pt") as fh:
        for k in fh.keys():
            if ".micro_engram." in k:
                sd[k.split(".micro_engram.", 1)[1]] = fh.get_tensor(k)
if not sd:
    sys.exit("no micro_engram keys found")
torch.save(sd, out)
print(f"saved {len(sd)} tensors to {out}: " + ", ".join(sorted(sd)[:4]) + ", ...")
