#!/usr/bin/env python3
"""Emit and compare the gated decode plans; no model download or GPU required."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from ruda_ptx import compile_exported


class GatedDecode(torch.nn.Module):
    def __init__(self, width, hidden, dtype):
        super().__init__()
        self.gate = torch.nn.Linear(width, hidden, dtype=dtype)
        self.up = torch.nn.Linear(width, hidden, bias=False, dtype=dtype)

    def forward(self, x):
        return F.silu(self.gate(x))*self.up(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--emit-dir',type=Path,required=True)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--rows',type=int,default=1)
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float16')
    a = p.parse_args()
    if not 1 <= a.rows <= 4 or min(a.width,a.hidden) <= 0:
        p.error('rows must be 1..4 and dimensions positive')
    if a.emit_dir.exists() and any(a.emit_dir.iterdir()):
        p.error('Refusing a nonempty emit directory')
    dtype = getattr(torch,a.dtype)
    torch.manual_seed(17)
    model = GatedDecode(a.width,a.hidden,dtype).eval()
    ep = torch.export.export(model,(torch.randn(a.rows,a.width,dtype=dtype),))
    old = compile_exported(ep,fuse_gated_decode=False)
    new = compile_exported(ep)
    old.write(a.emit_dir/'unfused')
    new.write(a.emit_dir/'fused')
    report = {label:{'kernels':len(plan.steps),'workspace_bytes':plan.report()['workspace_bytes'],
                     'constant_bytes':plan.report()['constant_buffer_bytes']}
              for label,plan in [('unfused',old),('fused',new)]}
    report.update({'gpu_executed':False,'measured_speedup':None,'peak_vram_measured':False})
    (a.emit_dir/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
