"""Production Python control flow; explicitly substituted CPU test numerics."""
import copy
import importlib.util
import sys
import types
import pytest
import torch
from test_optimizer_fused_host import fused
from test_training_host import mod
from gradient_stats_reference import layout

@pytest.fixture
def hierarchical(fused,monkeypatch):
    m,ref=fused
    package=types.ModuleType('_ruda_hierarchy_test');package.__path__=[]
    monkeypatch.setitem(sys.modules,'_ruda_hierarchy_test',package)
    monkeypatch.setitem(sys.modules,'_ruda_hierarchy_test._gradient_stats',layout)
    monkeypatch.setattr(m,'__package__','_ruda_hierarchy_test')
    monkeypatch.setattr(m,'__spec__',importlib.util.spec_from_file_location('_ruda_hierarchy_test.training',m.__file__))
    ref.hierarchy_calls=[]
    def analyze(gs,inv,w,s,report,with_norm):
        rows=sum(max(1,min(1024,(g.numel()+31)//32)) for g in gs)
        assert s.numel()==layout.statistics_plan(rows).scratch_elements
        ref.hierarchy_calls.append((rows,s.data_ptr(),s.numel()))
        # Independent float64 numerical callback for Python controls only.
        ref.training_analyze(gs,inv,w,report,with_norm)
    ref.training_analyze_hierarchical=analyze
    return m,ref

@pytest.mark.parametrize('kwargs',[{'hierarchical_stats':1}, {'hierarchical_stats':True},
    {'fused_step':True,'hierarchical_stats':'yes'}])
def test_explicit_options_rejected(hierarchical,kwargs):
    m,ref=hierarchical
    with pytest.raises((TypeError,ValueError)):m.AdamW([torch.nn.Parameter(torch.ones(1))],**kwargs)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('limit',[None,0.,1.])
def test_matches_existing_control_and_keeps_grads(hierarchical,dtype,limit):
    m,ref=hierarchical
    torch.manual_seed(27)
    ps=[torch.nn.Parameter(torch.ones(n,dtype=dtype)) for n in (32768,33)]
    qs=[torch.nn.Parameter(p.detach().clone()) for p in ps]
    a=m.AdamW(ps,fused_step=True,max_grad_norm=limit,hierarchical_stats=True)
    b=m.AdamW(qs,fused_step=True,max_grad_norm=limit)
    for _ in range(3):
        for p,q in zip(ps,qs):p.grad=torch.randn_like(p)*8;q.grad=p.grad.clone()
        versions=[p.grad._version for p in ps]
        a.step(loss_scale=8);b.step(loss_scale=8)
        for p,q,v in zip(ps,qs,versions):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            assert p.grad._version==v
        assert a.last_grad_norm==b.last_grad_norm
    assert len(ref.hierarchy_calls)==3


def test_scratch_grows_and_reuses_then_restores(hierarchical):
    m,ref=hierarchical
    ps=[torch.nn.Parameter(torch.ones(32768)) for _ in range(5)]
    opt=m.AdamW(ps,fused_step=True,hierarchical_stats=True)
    ps[0].grad=torch.ones_like(ps[0]);opt.step();assert opt._reduction_scratch.numel()==0
    for p in ps:p.grad=torch.ones_like(p)
    opt.step();scratch=opt._reduction_scratch;assert scratch.numel()==15
    for p in ps[1:]:p.grad=None
    opt.step();assert opt._reduction_scratch is scratch
    saved=copy.deepcopy(opt.state_dict());assert saved['ruda_step_options']['version']==2
    opt.load_state_dict(saved);assert opt.hierarchical_stats and opt._reduction_scratch is None
    legacy=copy.deepcopy(saved);legacy['ruda_step_options'].pop('hierarchical_stats');legacy['ruda_step_options']['version']=1
    opt.load_state_dict(legacy);assert not opt.hierarchical_stats

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_tail_overflow_skips_every_update(hierarchical,bad):
    m,ref=hierarchical
    ps=[torch.nn.Parameter(torch.ones(n)) for n in (32768,33)]
    for p in ps:p.grad=torch.ones_like(p)
    ps[-1].grad[-1]=bad
    opt=m.AdamW(ps,fused_step=True,hierarchical_stats=True,max_grad_norm=1.)
    opt.step();assert opt.last_step_skipped and not opt.state and ref.batch_calls==0
    for p in ps:assert torch.all(p==1)


def test_corrupt_checkpoint_is_rejected_before_mutation(hierarchical):
    m,ref=hierarchical;p=torch.nn.Parameter(torch.ones(7));opt=m.AdamW([p],fused_step=True,hierarchical_stats=True)
    saved=opt.state_dict();saved['ruda_step_options']['hierarchical_stats']=1
    with pytest.raises(ValueError,match='hierarchical'):opt.load_state_dict(saved)
    assert opt.hierarchical_stats and not opt.state
