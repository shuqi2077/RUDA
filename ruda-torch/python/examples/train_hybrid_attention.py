#!/usr/bin/env python3
"""Train a small mHC + CSA/HCA + DSA model with Muon/AdamW groups.

Default: require the real RUDA extension/device. --cpu-reference explicitly
loads the same Python components on CPU, without native-runtime impersonation.
No pretrained model download, fused-attention claim or implicit CPU fallback.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import types
import torch


def components(cpu_reference):
    if cpu_reference:
        name='ruda_hybrid_cpu_reference'
        package=types.ModuleType(name)
        package.__path__=[str(Path(__file__).resolve().parents[1]/'ruda_torch')]
        package._graph_available=False
        sys.modules[name]=package
        model=importlib.import_module(name+'.hybrid_model')
        optim=importlib.import_module(name+'.optim')
        compiler=importlib.import_module(name+'.compiler')
        return model,optim,compiler,None
    # Fails explicitly when native dependencies are unavailable.
    import ruda_torch
    from ruda_torch import hybrid_model,optim,compiler
    return hybrid_model,optim,compiler,ruda_torch


def run(args):
    if args.steps<1 or args.warmup_steps<0:raise ValueError('--steps must be positive and --warmup-steps nonnegative')
    torch.manual_seed(51)
    hm,optim,compiler,device_module=components(args.cpu_reference)
    device=torch.device('cpu' if args.cpu_reference else 'ruda:0')
    # CPU initialization followed by an explicit device move; RUDA does not
    # implement every initializer. This is not CPU fallback during execution.
    model=hm.HybridAttentionLanguageModel(31,16,2,2,streams=2,csa_ratio=2,hca_ratio=4,
        query_rank=8,index_heads=2,index_dim=4,rope_dim=4,topk=3,window_size=4,
        query_chunk_size=4,key_chunk_size=4,sinkhorn_iterations=5).to(device)
    optimizer=optim.MuonAdamW.from_model(model,muon_modules=list(model.layers),
        adamw_modules=[model.embedding,model.head],lr=.003,adamw_lr=.001,ns_steps=3,max_grad_norm=1.)
    scaler=None
    if args.loss_scale:
        if device_module is None:raise ValueError('--loss-scale requires the real RUDA GradScaler device path')
        scaler=device_module.GradScaler(init_scale=128)
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        if scaler is not None and state.get('scaler') is not None:scaler.load_state_dict(state['scaler'])
    tokens=torch.randint(0,31,(2,12)).to(device)
    valid=torch.tensor([[True]*12,[True]*10+[False]*2],device=device)
    compiled=None
    if args.compile:
        compiled=compiler.compile(model,device_type=device.type,native='off' if args.cpu_reference else 'auto',fullgraph=True)
    executable=compiled if compiled is not None else model
    losses=[];indexer_losses=[];warmup_losses=[];indexer_gradient=None
    initial=model.layers[0].attention.indexer.query.weight.detach().clone()
    try:
        for _ in range(args.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            result=executable(tokens,valid_mask=valid,return_aux=True,indexer_warmup=True)
            if scaler is None:
                result.indexer_loss.backward();optimizer.step()
            else:
                scaler.scale(result.indexer_loss.float()).backward();scaler.step(optimizer);scaler.update()
            if optimizer.last_step_skipped:raise RuntimeError('non-finite indexer warm-up update')
            warmup_losses.append(float(result.indexer_loss.detach().cpu()))
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            result=executable(tokens,valid_mask=valid,return_aux=True)
            main_loss=hm.next_token_loss(result.output.float(),tokens,valid_mask=valid)
            # Discrete selection alone has no useful gradient: explicitly train
            # the indexer against detached main-attention probabilities.
            loss=main_loss+args.indexer_loss_weight*result.indexer_loss
            if scaler is None:
                loss.backward();optimizer.step()
            else:
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            if optimizer.last_step_skipped:raise RuntimeError('non-finite training update was skipped')
            # Explicit diagnostic readbacks, outside the compiled model.
            losses.append(float(loss.detach().cpu()))
            indexer_losses.append(float(result.indexer_loss.detach().cpu()))
            grad=model.layers[0].attention.indexer.query.weight.grad
            indexer_gradient=0. if grad is None else float(grad.detach().abs().sum().cpu())
        if device_module is not None:device_module.synchronize()
        changed=bool((model.layers[0].attention.indexer.query.weight.detach()!=initial).any().cpu())
        if not changed or not indexer_gradient:raise RuntimeError('indexer did not receive a meaningful training update')
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scaler':None if scaler is None else scaler.state_dict()},args.checkpoint)
        return {'status':'passed','execution':'cpu-reference' if args.cpu_reference else 'real-ruda',
            'gpu_executed':not args.cpu_reference,'steps':args.steps,'torch':torch.__version__,
            'losses':losses,'warmup_losses':warmup_losses,'indexer_losses':indexer_losses,'indexer_gradient_l1':indexer_gradient,
            'indexer_updated':changed,'compiled':args.compile,
            'compiler_info':None if compiled is None else compiled.info,
            'fused_mhc_csa_hca_muon':False,
            'ruda_stats':None if device_module is None else device_module.execution_stats()}
    finally:
        if compiled is not None:compiled.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-reference',action='store_true')
    parser.add_argument('--compile',action='store_true')
    parser.add_argument('--loss-scale',action='store_true')
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--warmup-steps',type=int,default=0)
    parser.add_argument('--indexer-loss-weight',type=float,default=.1)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();code=0
    try:
        if not 0<args.indexer_loss_weight<100:raise ValueError('indexer loss weight must be positive and finite')
        report=run(args)
    except Exception as error:
        code=2;report={'status':'failed','gpu_executed':False,'execution':'cpu-reference' if args.cpu_reference else 'real-ruda',
            'error':f'{type(error).__name__}: {error}'}
    text=json.dumps(report,indent=2,ensure_ascii=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text);return code

if __name__=='__main__':raise SystemExit(main())
