"""Small real RUDA training example; no model downloads or CPU execution fallback."""
import argparse
from pathlib import Path
import torch
import ruda_torch as r

class GatedBlock(torch.nn.Module):
    def __init__(self,width=64):
        super().__init__();self.norm=r.RMSNorm(width)
        self.gate=torch.nn.Linear(width,2*width);self.up=torch.nn.Linear(width,2*width)
        self.down=torch.nn.Linear(2*width,width)
    def forward(self,x):
        h=self.norm(x)
        return x+self.down(r.silu_mul(self.gate(h),self.up(h)))

def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=10);p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float16')
    p.add_argument('--hierarchical-stats',action='store_true');p.add_argument('--fused-step',action='store_true');p.add_argument('--max-grad-norm',type=float)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if args.steps<=0:p.error('--steps must be positive')
    if not r._training_available:raise RuntimeError('build training API 4 first')
    torch.manual_seed(43)
    model=GatedBlock().to(device='ruda',dtype=getattr(torch,args.dtype))
    optimizer=r.AdamW(model.parameters(),lr=1e-3,fused_step=args.fused_step,max_grad_norm=args.max_grad_norm,hierarchical_stats=args.hierarchical_stats)
    scaler=r.GradScaler(init_scale=128)
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
    # Fixed synthetic batch demonstrates mechanics, not convergence of a real LLM.
    x=torch.randn(8,64,dtype=getattr(torch,args.dtype)).to('ruda')
    target=torch.zeros(8,64).to('ruda')
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss=(model(x).float()-target).square().mean()
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        print({'step':step,'loss':loss.detach().cpu().item(),'scale':scaler.get_scale(),'skipped':optimizer.last_step_skipped,'grad_norm':optimizer.last_grad_norm,'clip_coef':optimizer.last_clip_coef},flush=True)
    if args.checkpoint:
        r.synchronize()
        torch.save(cpu_tree({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict()}),args.checkpoint)

if __name__=='__main__':main()
