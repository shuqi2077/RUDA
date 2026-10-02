"""Production LoRA/NF4 CPU reference tests; not native GPU acceptance."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

spec = importlib.util.spec_from_file_location('ruda_finetuning_cpu', Path(__file__).resolve().parents[1] / 'ruda_torch/finetuning.py')
ft = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ft
spec.loader.exec_module(ft)
torch.set_num_threads(1)


def dense_weight(layer, dtype=torch.float32):
    # Independent scalar unpack oracle, not the production tile decoder.
    codebook = ft.NF4_CODEBOOK
    values = []
    for i in range(layer.in_features * layer.out_features):
        byte = int(layer.packed[i // 2])
        code = byte // 16 if i % 2 == 0 else byte % 16
        values.append(codebook[code] * float(layer.scales[i // layer.block_size]))
    return torch.tensor(values, dtype=dtype).view(layer.out_features, layer.in_features)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(5,), (2, 5), (2, 3, 5), (0, 5)])
@pytest.mark.parametrize('tile', [1, 2, 8])
def test_nf4_forward_and_input_gradient(dtype, shape, tile):
    torch.manual_seed(19)
    base = nn.Linear(5, 7).to(dtype)
    layer = ft.NF4Linear.from_linear(base, block_size=4, tile_rows=tile)
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    weight = dense_weight(layer, dtype)
    expected = torch.nn.functional.linear(reference, weight, layer.bias)
    actual = layer(x)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    tol = {torch.float32: 2e-6, torch.float16: .004, torch.bfloat16: .025}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, reference.grad, rtol=tol, atol=tol)
    assert list(layer.parameters()) == []
    assert layer.packed.numel() == 18
    assert layer.packed.dtype == torch.uint8


def test_nf4_codes_partial_zero_and_exact_boundaries():
    x = torch.tensor([ft.NF4_CODEBOOK], dtype=torch.float32)
    packed, scales = ft.pack_nf4(x, block_size=16)
    assert packed.tolist() == [1, 35, 69, 103, 137, 171, 205, 239]
    assert scales.tolist() == [1.]
    packed, scales = ft.pack_nf4(torch.zeros(1, 5), block_size=4)
    assert packed.tolist() == [119, 119, 119]
    assert scales.tolist() == [0., 0.]


def test_nf4_metadata_preserved_across_dtype_conversion_and_checkpoint():
    layer = ft.NF4Linear.from_linear(nn.Linear(13, 9), block_size=8)
    scales = layer.scales.clone()
    layer.bfloat16()
    assert layer.scales.dtype == layer.codebook.dtype == torch.float32
    torch.testing.assert_close(layer.scales, scales, rtol=0, atol=0)
    clone = ft.NF4Linear.from_linear(nn.Linear(13, 9), block_size=8)
    clone.load_state_dict(layer.state_dict())
    assert torch.equal(clone.packed, layer.packed)
    with pytest.raises(ValueError, match='geometry'):
        layer.set_extra_state({'version': 2})


@pytest.mark.parametrize('packed', [False, True])
def test_training_freezes_base_and_retains_upstream_gradients(packed):
    torch.manual_seed(51)
    model = nn.Sequential(nn.Linear(7, 9), nn.SiLU(), nn.Linear(9, 4))
    if packed:
        ft.quantize_nf4(model, target_modules='all-linear', block_size=8, tile_rows=3)
    original = copy.deepcopy(model)
    ft.inject_lora(model, target_modules='all-linear', rank=3, alpha=5.)
    x = torch.randn(3, 7, requires_grad=True)
    torch.testing.assert_close(model(x), original(x))
    base = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        checkpoint(model, x, use_reentrant=False).square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        optimizer.step()
    assert model[0].lora_A.grad.abs().sum() > 0
    assert model[0].lora_B.grad.abs().sum() > 0
    for n, p in model.named_parameters():
        if n in base:
            assert p.grad is None
            torch.testing.assert_close(p, base[n], rtol=0, atol=0)
    state = ft.adapter_state_dict(model)
    reference = copy.deepcopy(original)
    ft.inject_lora(reference, target_modules='all-linear', rank=3, alpha=5.)
    ft.load_adapter_state_dict(reference, state)
    torch.testing.assert_close(model(x), reference(x))


def test_lora_independent_formula_and_gradient():
    layer = ft.LoRALinear(nn.Linear(5, 4), rank=2, alpha=3.)
    with torch.no_grad():
        layer.lora_B.normal_()
    x = torch.randn(3, 5, requires_grad=True)
    a = layer.lora_A.detach().clone().requires_grad_()
    b = layer.lora_B.detach().clone().requires_grad_()
    xr = x.detach().clone().requires_grad_()
    actual = layer(x)
    expected = xr @ layer.base.weight.T + layer.base.bias + ((xr @ a.T) @ b.T) * 1.5
    actual.square().sum().backward()
    expected.square().sum().backward()
    for p, q in [(actual, expected), (x.grad, xr.grad), (layer.lora_A.grad, a.grad), (layer.lora_B.grad, b.grad)]:
        torch.testing.assert_close(p, q)
    assert layer.base.weight.grad is None


def test_merge_is_permanent_dense_equivalent_and_rejects_packed():
    model = nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 2))
    ft.inject_lora(model, target_modules=['0'], rank=2)
    with torch.no_grad():
        model[0].lora_B.normal_()
    x = torch.randn(3, 5)
    with pytest.raises(ValueError, match='eval'):
        ft.merge_lora(model)
    model.eval()
    expected = model(x)
    ft.merge_lora(model)
    assert not any(isinstance(m, ft.LoRALinear) for m in model.modules())
    torch.testing.assert_close(model(x), expected)
    ft.quantize_nf4(model, target_modules=['0'])
    ft.inject_lora(model, target_modules=['0'])
    with pytest.raises(ValueError, match='dense'):
        ft.merge_lora(model.eval())


def test_shared_module_aliases_and_tied_weight_policies():
    shared = nn.Linear(5, 5)
    model = nn.ModuleDict({'left': shared, 'right': shared})
    ft.quantize_nf4(model, target_modules=['left'])
    assert model['left'] is model['right']
    ft.inject_lora(model, target_modules=['left'], rank=2)
    assert model['left'] is model['right']
    assert set(ft.adapter_state_dict(model)['layers']) == {'left'}
    tied = nn.ModuleDict({'embedding': nn.Embedding(5, 5), 'head': nn.Linear(5, 5, bias=False)})
    tied['head'].weight = tied['embedding'].weight
    with pytest.raises(ValueError, match='tied'):
        ft.quantize_nf4(tied, target_modules=['head'])
    ft.inject_lora(tied, target_modules=['head'])
    assert tied['head'].base.weight is tied['embedding'].weight
    with pytest.raises(ValueError, match='shared'):
        ft.merge_lora(tied.eval())


def test_checkpoint_rejects_mismatch_before_writing_any_adapter():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    ft.inject_lora(model, target_modules='all-linear', rank=2)
    state = ft.adapter_state_dict(model)
    state['layers']['0']['lora_A'].fill_(12)
    state['layers']['1']['rank'] = 9
    before = model[0].lora_A.clone()
    with pytest.raises(ValueError, match='configuration'):
        ft.load_adapter_state_dict(model, state)
    torch.testing.assert_close(before, model[0].lora_A, rtol=0, atol=0)


@pytest.mark.parametrize('targets', [[], ['missing'], ['0', '0'], '0', [1]])
def test_invalid_targets_leave_model_unchanged(targets):
    model = nn.Sequential(nn.Linear(4, 4))
    original = model[0]
    with pytest.raises(ValueError):
        ft.inject_lora(model, target_modules=targets)
    assert model[0] is original and model[0].weight.requires_grad


def test_nf4_noncontiguous_input_and_no_activation_saved():
    layer = ft.NF4Linear.from_linear(nn.Linear(5, 7), tile_rows=2)
    x = torch.randn(5, 3).T.requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t), t)[1], lambda t: t):
        y = layer(x)
    assert len(saved) == 3
    assert all(t is not x for t in saved)
    y.sum().backward()
    torch.testing.assert_close(x.grad, dense_weight(layer).sum(0).expand_as(x))


def test_nf4_autocast_and_adapter_fp32_master():
    model = nn.Sequential(ft.NF4Linear.from_linear(nn.Linear(5, 7)))
    ft.inject_lora(model, target_modules=['0'], rank=2)
    x = torch.randn(3, 5, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = model(x)
    assert y.dtype == torch.bfloat16
    y.float().square().mean().backward()
    assert x.grad.dtype == torch.float32
    assert model[0].lora_B.grad.dtype == torch.float32


@pytest.mark.parametrize('packed', [False, True])
def test_resume_reproduces_next_adamw_update(tmp_path, packed):
    torch.manual_seed(52)
    base = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3))
    if packed:
        ft.quantize_nf4(base, target_modules='all-linear', block_size=8, tile_rows=2)
    model, restored = copy.deepcopy(base), copy.deepcopy(base)
    for m in (model, restored):
        ft.inject_lora(m, target_modules='all-linear', rank=2, alpha=4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002)
    other = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.9)
    def step(m, optimizer):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(3, 5)
        checkpoint(m, x, use_reentrant=False).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    step(model, opt)
    state = ft.finetune_state_dict(model, opt, base_id='unit-base-v1', step=1, data_state={'cursor': 3})
    path = tmp_path / 'checkpoint.pt'
    torch.save(state, path)
    step(model, opt)
    state = torch.load(path, weights_only=True)
    count, cursor = ft.load_finetune_state_dict(restored, other, state, base_id='unit-base-v1')
    assert count == 1 and cursor == {'cursor': 3}
    step(restored, other)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='base'):
        ft.load_finetune_state_dict(restored, other, state, base_id='wrong-base')


def test_streaming_sharded_safetensors_into_meta_model(tmp_path):
    import json
    from safetensors.torch import save_file
    torch.manual_seed(92)
    original = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3))
    state = original.state_dict()
    save_file({k: v for k, v in state.items() if k.startswith('0.')}, str(tmp_path / 'first.safetensors'))
    save_file({k: v for k, v in state.items() if k.startswith('2.')}, str(tmp_path / 'second.safetensors'))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        k: 'first.safetensors' if k.startswith('0.') else 'second.safetensors' for k in state}}))
    model = nn.Sequential(nn.Linear(5, 7, device='meta'), nn.SiLU(), nn.Linear(7, 3, device='meta'))
    ft.load_nf4_safetensors(model, tmp_path, target_modules=['0'], device='cpu', dtype=torch.float32, block_size=8)
    ft.quantize_nf4(original, target_modules=['0'], block_size=8)
    x = torch.randn(3, 5)
    torch.testing.assert_close(model(x), original(x), rtol=0, atol=0)
    assert isinstance(model[0], ft.NF4Linear) and isinstance(model[2], nn.Linear)
    assert not any(p.is_meta for p in model.parameters())


def test_streaming_loader_preserves_unquantized_ties(tmp_path):
    from safetensors.torch import save_file
    original = nn.ModuleDict({'embed': nn.Embedding(7, 5), 'head': nn.Linear(5, 7, bias=False), 'hidden': nn.Linear(5, 5)})
    original['head'].weight = original['embed'].weight
    state = {k: v for k, v in original.state_dict().items() if k != 'head.weight'}
    save_file(state, str(tmp_path / 'model.safetensors'))
    model = copy.deepcopy(original).to('meta')
    model['head'].weight = model['embed'].weight
    ft.load_nf4_safetensors(model, tmp_path, target_modules=['hidden'], device='cpu', dtype=torch.float32)
    assert model['head'].weight is model['embed'].weight
    torch.testing.assert_close(model['head'].weight, original['head'].weight)


@pytest.mark.parametrize('packed', [False, True])
def test_aot_training_matches_eager_reference(packed):
    torch._dynamo.reset()
    torch.manual_seed(104)
    eager = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3))
    if packed:
        ft.quantize_nf4(eager, target_modules='all-linear', block_size=8, tile_rows=3)
    ft.inject_lora(eager, target_modules='all-linear', rank=2)
    with torch.no_grad():
        for m in eager.modules():
            if isinstance(m, ft.LoRALinear):
                m.lora_B.normal_(std=.02)
    model = copy.deepcopy(eager)
    compiled = torch.compile(model, backend='aot_eager', fullgraph=True)
    x = torch.randn(3, 5, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual, expected = compiled(x), eager(xr)
    actual.square().mean().backward(); expected.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, xr.grad)
    for p, q in zip(model.parameters(), eager.parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad)
