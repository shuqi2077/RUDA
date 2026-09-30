"""Tiny native router-projection training example; NOT full DeepSeek/Kimi training.

Requires rebuilt optional router API 1. No model weights are downloaded.
"""
import torch
import ruda_torch as r

torch.manual_seed(30)
projection=torch.nn.Linear(16,8,bias=False).to('ruda')
optimizer=r.AdamW(projection.parameters(),lr=1e-3)
x=torch.randn(4,16).to('ruda')
# Coefficients stand for the derivative supplied by expert-output combination.
coefficient=torch.randn(4,2).to('ruda')
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    logits=projection(x)
    with torch.no_grad():
        # A model may use group masks/correction bias HERE, for selection only.
        indices=logits.topk(2,dim=-1).indices
    weights=r.selected_router_weights(logits,indices,scoring='sigmoid',renormalize=True,scale=2.5)
    loss=(weights*coefficient).sum()
    loss.backward()
    optimizer.step()
    r.synchronize()
    print({'step':step,'loss':float(loss.detach().cpu())})
