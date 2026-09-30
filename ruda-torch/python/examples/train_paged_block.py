"""Real single-block training acceptance, not a pretrained LLM or speed claim.

RUDA runs normalization/projections/paged attention/gated FFN/backward/AdamW.
An independently executed CPU block checks outputs, EVERY parameter gradient,
updates and the next step after model+optimizer checkpoint restoration.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import torch

class Block(torch.nn.Module):
    def __init__(self,r=None,*,frozen_history=False,backward_strategy="atomic"):
        super().__init__();self.r=r;self.frozen_history=frozen_history;self.backward_strategy=backward_strategy
        self.norm=(torch.nn.RMSNorm if r is None else r.RMSNorm)(16,eps=1e-5)
        self.q=torch.nn.Linear(16,16,bias=False)
        self.k=torch.nn.Linear(16,8,bias=False);self.v=torch.nn.Linear(16,8,bias=False)
        self.proj=torch.nn.Linear(16,16,bias=False)
        self.gate=torch.nn.Linear(16,24,bias=False);self.up=torch.nn.Linear(16,24,bias=False)
        self.down=torch.nn.Linear(24,16,bias=False);self.plan=None
        if frozen_history:
            self.k.weight.requires_grad_(False);self.v.weight.requires_grad_(False)
    def forward(self,x):
        z=self.norm(x);q=self.q(z).reshape(6,4,4)
        # In adapter-style tests the entire historical K/V branch is fixed.
        # Frozen weights alone would STILL require K/V gradients into z.
        source=z.detach() if self.frozen_history else z
        k=self.k(source).reshape(2,3,2,4);v=self.v(source).reshape(2,3,2,4)
        if self.r is None:
            kk=k.reshape(6,2,4).repeat_interleave(2,dim=1).transpose(0,1)
            vv=v.reshape(6,2,4).repeat_interleave(2,dim=1).transpose(0,1)
            scores=(q.transpose(0,1)@kk.transpose(1,2))*.5
            mask=torch.ones(6,6,dtype=torch.bool).tril()
            probs=torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=-1)
            context=(probs@vv).transpose(0,1).contiguous()
        else:
            if self.plan is None:
                self.plan=self.r.PagedAttentionPlan(page_size=3,num_pages=2,block_tables=[[0,1]],
                    kv_lengths=[6],sequence_ids=[0]*6,positions=list(range(6)),backward_strategy=self.backward_strategy)
            context=self.plan.attention(q,k,v,scale=.5,causal=True)
        residual=x+self.proj(context.reshape(6,16))
        gate=self.gate(residual);up=self.up(residual)
        activated=torch.nn.functional.silu(gate)*up if self.r is None else self.r.silu_mul(gate,up)
        return residual+self.down(activated)

def cpu_tree(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_tree(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_tree(x) for x in v)
    return v

def compare_training(r,*,steps=5,frozen_history=False,checkpoint=None,backward_strategy="atomic"):
    if steps<1:raise ValueError('steps must be positive')
    if not r._paged_backward_available or not r._training_available:
        raise RuntimeError('rebuild native paged backward and training APIs')
    torch.manual_seed(834)
    ref=Block(frozen_history=frozen_history)
    model=Block(r,frozen_history=frozen_history,backward_strategy=backward_strategy).to('ruda')
    model.load_state_dict(ref.state_dict())
    ropt=torch.optim.AdamW(ref.parameters(),lr=1e-4,eps=1e-6,weight_decay=.01,foreach=False)
    opt=r.AdamW(model.parameters(),lr=1e-4,eps=1e-6,weight_decay=.01,fused_step=True)
    inputs=torch.randn(6,16)*.1;target=torch.randn(6,16)*.1
    xd=inputs.to('ruda');td=target.to('ruda');losses=[]
    for step in range(steps):
        ropt.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
        yr=ref(inputs);yd=model(xd)
        torch.testing.assert_close(yd.cpu(),yr,rtol=3e-3,atol=3e-4)
        lr=(yr-target).square().mean();ld=(yd-td).square().mean()
        lr.backward();ld.backward()
        for (name,p),(_,pr) in zip(model.named_parameters(),ref.named_parameters()):
            if pr.grad is None:assert p.grad is None,name
            else:
                assert p.grad is not None,name
                torch.testing.assert_close(p.grad.cpu(),pr.grad,rtol=7e-3,atol=5e-4,msg=lambda s: name+': '+s)
        ropt.step();opt.step();r.synchronize()
        assert not opt.last_step_skipped
        for (name,p),(_,pr) in zip(model.named_parameters(),ref.named_parameters()):
            torch.testing.assert_close(p.cpu(),pr,rtol=7e-3,atol=5e-4,msg=lambda s:name+': '+s)
        losses.append({'step':step,'reference':float(lr.detach()),'ruda':float(ld.detach().cpu())})
    # Resume must reproduce the next update, not merely load without error.
    state=cpu_tree({'model':model.state_dict(),'optimizer':opt.state_dict()})
    if checkpoint is not None:
        checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save(state,checkpoint);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    resumed=Block(r,frozen_history=frozen_history,backward_strategy=backward_strategy).to('ruda');resumed.load_state_dict(state['model'])
    resumed_opt=r.AdamW(resumed.parameters(),lr=1e-4,eps=1e-6,weight_decay=.01,fused_step=True)
    resumed_opt.load_state_dict(state['optimizer'])
    for block,optim in [(model,opt),(resumed,resumed_opt)]:
        optim.zero_grad(set_to_none=True);loss=(block(xd)-td).square().mean();loss.backward();optim.step()
    r.synchronize()
    for a,b in zip(model.parameters(),resumed.parameters()):
        torch.testing.assert_close(a.cpu(),b.cpu(),rtol=2e-5,atol=2e-6)
    return {'scope':'FP32 single-block numerical training acceptance, NOT full-model convergence or performance',
            'steps':steps,'backward_strategy':backward_strategy,'frozen_history':frozen_history,'losses':losses,'checkpoint_next_step_verified':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,default=5)
    p.add_argument('--backward-strategy',choices=('atomic','ordered'),default='atomic')
    p.add_argument('--frozen-history',action='store_true');p.add_argument('--checkpoint',type=Path,default=Path('v32-block-state.pt'))
    p.add_argument('--output',type=Path,default=Path('v32-block-result.json'));args=p.parse_args()
    import ruda_torch as r # Missing native backend fails; never fall back to CPU.
    result=compare_training(r,steps=args.steps,frozen_history=args.frozen_history,checkpoint=args.checkpoint,backward_strategy=args.backward_strategy)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
