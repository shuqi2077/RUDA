"""Host-only interface and mathematical-reference tests, NOT GPU kernel tests.

Load selected real frontend functions without importing/registering the native
backend. The explicit mock only lives here and is never a production fallback.
"""
import ast
import hashlib
import itertools
from pathlib import Path
import re
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
DTYPES = (torch.float32, torch.float16, torch.bfloat16)

class MockNative:
    def __init__(self):
        self.calls = []
    def addmm(self, bias, a, b, out, alpha, beta):
        self.calls.append((bias, a, b, out, alpha, beta))
        # CPU reference mock, explicitly not the Rust implementation.
        out.copy_(torch.addmm(bias, a, b, alpha=alpha, beta=beta))
    def execute(self, op, a, b, out, scalar):
        self.calls.append((op, a, b, out, scalar))
        if op == 7:
            out.copy_(a @ b)
        elif op == 30:
            out.copy_(torch.bmm(a, b))
        elif op in (31, 33):
            value = a.float()
            out.copy_((torch.log_softmax if op == 33 else torch.softmax)(value, int(scalar)))
        else:
            raise AssertionError(f"unexpected mock operation {op}")

@pytest.fixture
def frontend():
    path = ROOT / 'ruda-torch/python/ruda_torch/_ops.py'
    parsed = ast.parse(path.read_text())
    names = {'_out', '_scalar', 'mm', 'bmm', 'addmm', '_dimension', 'softmax', 'softmax_backward'}
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    native = MockNative()
    scope = {'torch': torch, '_C': native, '_dtypes': DTYPES}
    exec(compile(module, str(path), 'exec'), scope)
    return types.SimpleNamespace(**scope), native

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('bias_shape', [(), (5,), (1, 5), (3, 1), (3, 5)])
@pytest.mark.parametrize('alpha,beta', [(1., 1.), (.25, -.5), (1., 0.), (0., 1.), (0., 0.)])
def test_real_addmm_frontend_keeps_original_storage(frontend, dtype, bias_shape, alpha, beta):
    front, native = frontend
    gen = torch.Generator().manual_seed(23)
    a = torch.randn(7, 3, generator=gen).to(dtype).t()
    b = torch.randn(5, 7, generator=gen).to(dtype).t()
    bias = torch.randn(bias_shape, generator=gen).to(dtype)
    expected = torch.addmm(bias, a, b, alpha=alpha, beta=beta)
    actual = front.addmm(bias, a, b, alpha=alpha, beta=beta)
    assert len(native.calls) == 1
    c, x, y, out, al, be = native.calls[0]
    assert x is a and y is b and out is actual
    assert c.untyped_storage().data_ptr() == bias.untyped_storage().data_ptr()
    assert x.dtype == y.dtype == c.dtype == out.dtype == dtype
    assert c.shape == (3, 5)
    torch.testing.assert_close(actual, expected)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(0, 4, 5), (3, 0, 5), (3, 4, 0)])
def test_addmm_empty_frontend(frontend, dtype, shape):
    front, native = frontend
    m, k, n = shape
    a, b, c = torch.ones(m, k, dtype=dtype), torch.ones(k, n, dtype=dtype), torch.ones(n, dtype=dtype)
    torch.testing.assert_close(front.addmm(c, a, b), torch.addmm(c, a, b))
    assert len(native.calls) == 1

@pytest.mark.parametrize('bad', ['rank', 'inner', 'bias_shape', 'dtype'])
def test_addmm_validation_precedes_native_call(frontend, bad):
    front, native = frontend
    a, b, c = torch.ones(3, 7), torch.ones(7, 5), torch.ones(5)
    if bad == 'rank': a = a.unsqueeze(0)
    if bad == 'inner': b = torch.ones(8, 5)
    if bad == 'bias_shape': c = torch.ones(2, 5)
    if bad == 'dtype': b = b.half()
    with pytest.raises((RuntimeError, NotImplementedError)):
        front.addmm(c, a, b, beta=0)
    assert not native.calls

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('logarithmic', [False, True])
def test_softmax_frontend_does_not_promote_input(frontend, dtype, logarithmic):
    front, native = frontend
    x = torch.linspace(-4, 4, 99).reshape(3, 33).to(dtype)
    half_to_float = dtype == torch.float16
    y = front.softmax(x, -1, half_to_float, logarithmic=logarithmic)
    op, a, b, out, axis = native.calls[0]
    assert a is x and b is x
    assert out.dtype == (torch.float32 if half_to_float else dtype)
    assert axis == 1
    assert op == (33 if logarithmic else 31)
    expected = (torch.log_softmax if logarithmic else torch.softmax)(x.float(), -1).to(out.dtype)
    torch.testing.assert_close(y, expected)

def remove_casts(text):
    for token in ['O::cast_from(', 'f32::cast_from(']:
        while token in text:
            start = text.index(token)
            end, depth = start + len(token), 1
            while depth:
                depth += (text[end] == '(') - (text[end] == ')')
                end += 1
            text = text[:start] + text[start+len(token):end-1] + text[end:]
    return text

@pytest.mark.parametrize('name,digest', [('pointwise','c8d0f2da5555f7b34813cdc2bf658e138185ef6892467f6b8ddfc31a14e47f42'), ('softmax','1b0b58c61d703ef7a8eaa9968f27dd016b852e96b4b12377ad2b4647a2a33276')])
def test_original_scalar_formulas_are_preserved(name, digest):
    text = (ROOT / 'ruda-torch/src/kernels.rs').read_text()
    start = text.index('#[ruda(launch)]\npub fn '+name+'<')
    end = text.index('\n#[ruda', start+25)
    body = remove_casts(text[start:end])
    body = re.sub('pub fn '+name+r'<[^>]+>\(', 'pub fn '+name+'(', body)
    for ty in ['A','B','O','F']:
        body = body.replace('Tensor<'+ty+'>', 'Tensor<f32>')
    assert hashlib.sha256(re.sub(r'\s+', '', body).encode()).hexdigest() == digest

def test_matrix_and_pointwise_dispatch_precede_staging():
    source = (ROOT / 'ruda-torch/src/lib.rs').read_text()
    stage = source.index('if a.dtype != 0 || b.dtype != 0 || out.dtype != 0')
    for call in ['matmul::launch(op, a, b, out)', 'pointwise::launch(op, a, b, out, scalar)', 'softmax::launch(op, a, b, out, scalar as usize)']:
        assert source.index(call) < stage

def test_all_pointwise_dtype_triples_exist():
    source = (ROOT / 'ruda-torch/src/pointwise.rs').read_text()
    triples = {tuple(map(int, t)) for t in re.findall(r'\((\d), (\d), (\d)\) => run!', source)}
    assert triples == set(itertools.product(range(3), repeat=3))

def test_abi_nine_is_consistent():
    rust = (ROOT / 'ruda-torch/src/lib.rs').read_text()
    cpp = (ROOT / 'ruda-torch/python/ruda_torch/csrc/backend.cpp').read_text()
    py = (ROOT / 'ruda-torch/python/ruda_torch/__init__.py').read_text()
    assert 'fn ruda_torch_abi_version() -> u32 { 9 }' in rust
    assert 'm.attr("abi_version") = 9' in cpp and 'addresses.size() == 13' in cpp
    assert 'addmm_native = reinterpret_cast<Addmm>(addresses[8])' in cpp
    assert 'rms_norm_native = reinterpret_cast<RMSNorm>(addresses[10])' in cpp
    assert 'ruda_torch_abi_version() != 9' in py and 'getattr(_C, "abi_version", None) != 9' in py
    assert '"layer_norm", "rms_norm", "stream", "paged")' in py

def test_v13_rms_norm_async_and_paged_decode_sources_present():
    rust = (ROOT / 'ruda-torch/src/lib.rs').read_text()
    kernels = (ROOT / 'ruda-torch/src/kernels.rs').read_text()
    ops = (ROOT / 'ruda-torch/python/ruda_torch/_ops.py').read_text()
    batching = (ROOT / 'ruLLM/src/qwen35/batching.rs').read_text()
    assert 'pub fn rms_norm_warp' in kernels and 'plane_sum(local_square)' in kernels
    assert 'def rms_norm(' in ops and '_C.rms_norm(a, weight, result, epsilon)' in ops
    assert 'RUDA_TORCH_ASYNC' in rust and 'fn finish_dispatch' in rust
    assert 'fn decode_pages(' in batching and 'online softmax' in batching
    assert 'if s == 1' in batching

def test_blas_uses_existing_output_strict_precision_and_no_store_copy():
    source = (ROOT / 'ruda-torch/src/matmul.rs').read_text()
    assert 'Some(output)' in source
    assert 'MatmulStrategy::Ruda, dtype, F32MathMode::Strict' in source
    assert 'primitives::store(' not in source.replace('primitives::store (extra copy)', '')
    assert 'catch_unwind' not in source
    assert 'MatmulSetupError::Launch(_)' not in source
    assert 'selected == Policy::Auto && setup_only' in source

def test_public_blas_auto_propagates_failure_and_restores_precision():
    source = (ROOT / 'ruBLAS/src/kernel_ir/launch/strategy.rs').read_text().split('fn auto<R: Runtime>')[1]
    assert '.unwrap()' not in source and 'panic!' not in source
    assert 'let requested = dtypes.clone();' in source and '*dtypes = requested;' in source
    assert 'Err(error) => Err(error)' in source

# Numerical reference for the new lane partition, NOT execution of Rust/PTX.
def warp_reference(x, logarithmic=False):
    data = x.float().reshape(-1, x.shape[-1])
    rows, width = data.shape
    maximum = torch.full((rows,), -float('inf'))
    nan = torch.zeros(rows, dtype=torch.bool)
    for lane in range(32):
        values = data[:, lane::32]
        if values.shape[1]:
            nan |= torch.isnan(values).any(1)
            maximum = torch.maximum(maximum, values.amax(1))
    maximum[nan] = float('nan')
    partial = torch.zeros(rows, 32)
    for lane in range(32):
        for i in range(lane, width, 32):
            partial[:,lane] += (data[:,i] - maximum).exp()
    # XOR-butterfly reduction with all 32 lanes, including inactive tail lanes.
    for delta in (16,8,4,2,1):
        partial = partial + partial[:, torch.arange(32) ^ delta]
    total = partial[:,0:1]
    shifted = data - maximum[:,None]
    out = shifted-total.log() if logarithmic else shifted.exp()/total
    return out.reshape(x.shape).to(x.dtype)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('width', [1,7,31,32,33,127,128,257,1025])
@pytest.mark.parametrize('logarithmic', [False,True])
def test_warp_math_reference(dtype, width, logarithmic):
    torch.set_num_threads(1)
    x = (torch.randn(3,width,generator=torch.Generator().manual_seed(width))*3).to(dtype)
    expected = (torch.log_softmax if logarithmic else torch.softmax)(x.float(),-1).to(dtype)
    actual = warp_reference(x,logarithmic)
    tol = {torch.float32:2e-5, torch.float16:2e-3, torch.bfloat16:2e-2}[dtype]
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol,equal_nan=True)

@pytest.mark.parametrize('special', ['nan','positive_inf','all_negative_inf','masked'])
@pytest.mark.parametrize('logarithmic', [False,True])
def test_warp_nonfinite_reference(special, logarithmic):
    x = torch.zeros(3,33)
    if special == 'nan': x[:,31] = float('nan')
    if special == 'positive_inf': x[:,32] = float('inf')
    if special == 'all_negative_inf': x.fill_(-float('inf'))
    if special == 'masked': x[:,1:] = -float('inf')
    expected = (torch.log_softmax if logarithmic else torch.softmax)(x,-1)
    torch.testing.assert_close(warp_reference(x,logarithmic), expected, equal_nan=True)

@pytest.mark.parametrize('width', [1,31,32,33,128,129,1025])
@pytest.mark.parametrize('rows', [1,3,4,5,17])
def test_warp_launch_has_exact_write_coverage(width, rows):
    writes = []
    for thread in range(((rows+3)//4)*128):
        row,lane = divmod(thread,32)
        if row<rows:
            writes += [(row,i) for i in range(lane,width,32)]
    assert len(writes) == rows*width
    assert len(set(writes)) == len(writes)
    assert set(writes) == set(itertools.product(range(rows),range(width)))

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('alpha,beta,k', [(0.,1.,7),(1.,0.,7),(0.,0.,7),(float('nan'),1.,0),(float('inf'),0.,0)])
def test_addmm_ignored_operands_fp32_reference(dtype, alpha,beta,k):
    a = torch.full((3,k),float('nan'),dtype=dtype)
    b = torch.ones(k,5,dtype=dtype)
    bias = torch.full((5,),float('nan') if beta==0 else 2.,dtype=dtype)
    # This is the explicit FP32-accumulation reference. CPU F16/BF16 addmm
    # differs from CPU F32 for alpha=0 with NaN operands on PyTorch 2.10;
    # CUDA/RUDA behavior must be verified separately, not inferred from it.
    expected = torch.addmm(bias.float(),a.float(),b.float(),alpha=alpha,beta=beta).to(dtype)
    if alpha==0 or k==0:
        actual = (bias.float().expand(3,5)*beta if beta!=0 else torch.zeros(3,5)).to(dtype)
    else:
        acc = a.float() @ b.float()
        actual = (acc*alpha if beta==0 else acc*alpha+bias.float()*beta).to(dtype)
    torch.testing.assert_close(actual,expected,equal_nan=True)
