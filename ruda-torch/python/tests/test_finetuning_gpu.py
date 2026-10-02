"""Real RUDA NF4/LoRA acceptance. Missing hardware or native API is a failure."""
import copy
import os

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@pytest.fixture(scope='module')
def r():
    assert os.environ.get('RUDA_CUDA_COMPILER') == 'ptx'
    import ruda_torch
    assert ruda_torch._nf4_available and ruda_torch._nf4_matmul_available and ruda_torch._training_available
    return ruda_torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_native_nf4_forward_input_gradient_and_zero_host_transfers(r, dtype):
    from ruda_torch.finetuning import NF4_CODEBOOK
    torch.manual_seed(92)
    cpu = r.NF4Linear.from_linear(nn.Linear(33, 17), block_size=64, tile_rows=5)
    native = copy.deepcopy(cpu).to('ruda')
    data = torch.randn(2, 3, 33, dtype=dtype)
    x = data.to('ruda').requires_grad_()
    xr = data.clone().requires_grad_()
    grad = torch.randn(2, 3, 17, dtype=dtype)
    dg = grad.to('ruda')
    values = []
    for i in range(33*17):
        byte = int(cpu.packed[i//2])
        code = byte // 16 if i%2 == 0 else byte%16
        values.append(NF4_CODEBOOK[code]*float(cpu.scales[i//64]))
    weight = torch.tensor(values, dtype=dtype).view(17,33)
    expected = torch.nn.functional.linear(xr, weight) + cpu.bias.to(dtype)
    expected.backward(grad)
    before = r.execution_stats()
    actual = native(x)
    actual.backward(dg)
    r.synchronize()
    after = r.execution_stats()
    assert after['kernel_launches'] > before['kernel_launches']
    for key in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert after[key] == before[key]
    tol = {torch.float32: 3e-5, torch.float16: .006, torch.bfloat16: .04}[dtype]
    torch.testing.assert_close(actual.cpu(), expected, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.cpu(), xr.grad, rtol=tol, atol=tol)


def test_native_qlora_adamw_checkpoint_recompute_and_resume(r):
    torch.manual_seed(93)
    base = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32))
    r.quantize_nf4(base, target_modules='all-linear', block_size=64, tile_rows=16)
    r.inject_lora(base, target_modules='all-linear', rank=4, alpha=8.)
    model = copy.deepcopy(base).to('ruda')
    opt = r.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    refopt = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=.001, foreach=False)
    x = torch.randn(3, 32)
    xd = x.to('ruda')
    for _ in range(3):
        opt.zero_grad(set_to_none=True); refopt.zero_grad(set_to_none=True)
        reference = base(x)
        actual = checkpoint(model, xd, use_reentrant=False, preserve_rng_state=False)
        reference.square().mean().backward(); actual.square().mean().backward()
        torch.testing.assert_close(actual.cpu(), reference, rtol=5e-4, atol=5e-5)
        for (name, p), (_, q) in zip(model.named_parameters(), base.named_parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad.cpu(), q.grad, rtol=2e-3, atol=1e-4, msg=name)
        opt.step(); refopt.step()
    opt.zero_grad(set_to_none=True)
    state = r.finetune_state_dict(model, opt, base_id='native-unit-nf4-v1', step=3, data_state={'next': 3})
    restored = copy.deepcopy(model)
    restored_opt = r.AdamW([p for p in restored.parameters() if p.requires_grad])
    assert r.load_finetune_state_dict(restored, restored_opt, state, base_id='native-unit-nf4-v1') == (3, {'next': 3})
    for m, o in [(model, opt), (restored, restored_opt)]:
        m(xd).square().mean().backward(); o.step()
    r.synchronize()
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p.cpu(), q.cpu(), rtol=1e-5, atol=1e-6)
