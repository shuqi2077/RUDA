"""CPU algorithm checks, NOT execution of the Rust/tch bindings.

The tensor expressions below mirror ops/deform.rs and ops/qtensor.rs. Deformable
convolution is checked against torchvision's compiled CPU operator and its own
backward; quantization is checked against independent scalar/block indexing.
"""
import itertools
import math
import pytest
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def deform_expression(x, offset, weight, mask=None, bias=None, *, stride=(1, 1),
                      padding=(1, 1), dilation=(1, 1), offset_groups=1, padding_end=None):
    n, ci, h, w = x.shape
    co, wci, kh, kw = weight.shape
    wg = ci // wci
    og = offset_groups
    pe = padding if padding_end is None else padding_end
    oh = (h+padding[0]+pe[0]-dilation[0]*(kh-1)-1)//stride[0]+1
    ow = (w+padding[1]+pe[1]-dilation[1]*(kw-1)-1)//stride[1]+1
    kind = torch.float64 if x.dtype == torch.float64 else torch.float32
    xa, offsets, weights = x.to(kind), offset.to(kind), weight.to(kind)
    masks = None if mask is None else mask.to(kind)
    by = (torch.arange(oh, dtype=kind)*stride[0]-padding[0]).reshape(1, oh, 1)
    bx = (torch.arange(ow, dtype=kind)*stride[1]-padding[1]).reshape(1, 1, ow)
    samples = []
    for ky, kx in itertools.product(range(kh), range(kw)):
        channels = []
        for group in range(og):
            c = group*kh*kw+ky*kw+kx
            y = by+ky*dilation[0]+offsets[:, 2*c]
            xp = bx+kx*dilation[1]+offsets[:, 2*c+1]
            y0, x0 = y.floor(), xp.floor()
            dy, dx = y-y0, xp-x0
            inp = xa[:, group*(ci//og):(group+1)*(ci//og)]
            def corner(yy, xx):
                valid = ((yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)).to(kind).unsqueeze(1)
                index = yy.clamp(0, h-1).long()*w+xx.clamp(0, w-1).long()
                index = index.reshape(n, 1, oh*ow).expand(n, ci//og, oh*ow)
                return inp.reshape(n, ci//og, h*w).gather(2, index).reshape(n, ci//og, oh, ow)*valid
            ly, lx = dy.unsqueeze(1), dx.unsqueeze(1)
            value = corner(y0, x0)*(1-ly)*(1-lx)+corner(y0, x0+1)*(1-ly)*lx \
                    +corner(y0+1, x0)*ly*(1-lx)+corner(y0+1, x0+1)*ly*lx
            if masks is not None:
                value = value*masks[:, c].unsqueeze(1)
            channels.append(value)
        samples.append(torch.cat(channels, 1))
    columns = torch.stack(samples, 2).reshape(n, ci*kh*kw, oh*ow)
    groups = []
    for group in range(wg):
        width = (ci//wg)*kh*kw
        patches = columns[:, group*width:(group+1)*width]
        matrix = weights[group*(co//wg):(group+1)*(co//wg)].reshape(co//wg, width)
        groups.append(matrix @ patches)
    out = torch.cat(groups, 1).reshape(n, co, oh, ow)
    if bias is not None:
        out = out+bias.to(kind).reshape(1, co, 1, 1)
    return out.to(x.dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('ci,co,wg,og', [(2, 3, 1, 1), (4, 6, 2, 4), (6, 4, 2, 3)])
@pytest.mark.parametrize('use_mask,use_bias', [(False, False), (True, True)])
@pytest.mark.parametrize('zero', [False, True])
def test_deform_forward_and_all_operand_gradients_against_torchvision(dtype, ci, co, wg, og, use_mask, use_bias, zero):
    torch.manual_seed(307)
    n, h, w, kh, kw = 1, 4, 5, 2, 3
    stride, padding, dilation = (2, 1), (1, 1), (1, 2)
    oh, ow = 3, 3
    x = torch.randn(n, ci, h, w, dtype=dtype)
    weights = torch.randn(co, ci//wg, kh, kw, dtype=dtype)
    offsets = torch.zeros(n, 2*og*kh*kw, oh, ow, dtype=dtype)
    if not zero:
        offsets.uniform_(-1.3, 1.3)
    masks = torch.rand(n, og*kh*kw, oh, ow, dtype=dtype) if use_mask else None
    bias = torch.randn(co, dtype=dtype) if use_bias else None
    vals = [x, offsets, weights, masks, bias]
    actual_inputs = [None if v is None else v.clone().requires_grad_() for v in vals]
    ref_inputs = [None if v is None else v.clone().requires_grad_() for v in vals]
    actual = deform_expression(*actual_inputs, stride=stride, padding=padding,
                              dilation=dilation, offset_groups=og)
    rx, ro, rw, rm, rb = ref_inputs
    reference = deform_conv2d(rx, ro, rw, rb, stride=stride, padding=padding, dilation=dilation, mask=rm)
    seed = torch.randn_like(actual)
    got = torch.autograd.grad(actual, [v for v in actual_inputs if v is not None], seed)
    want = torch.autograd.grad(reference, [v for v in ref_inputs if v is not None], seed)
    tol = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, reference, rtol=tol, atol=tol)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize('position', [-1.01, -1., -.999, -.5, 0., .5, .999, 1., 1.01])
def test_singleton_and_exact_support_boundary_gradient(position):
    x = torch.tensor([[[[2.]]]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[[[3.]]]], dtype=torch.float64, requires_grad=True)
    offset = torch.tensor([[[[position]], [[0.]]]], dtype=torch.float64, requires_grad=True)
    actual = deform_expression(x, offset, weight, padding=(0, 0))
    reference = deform_conv2d(x, offset, weight)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    got = torch.autograd.grad(actual.sum(), (x, offset, weight), retain_graph=True)
    want = torch.autograd.grad(reference.sum(), (x, offset, weight))
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_asymmetric_padding_matches_regular_convolution_at_zero_offsets():
    torch.manual_seed(11)
    x = torch.randn(2, 4, 5, 6, dtype=torch.float64)
    weight = torch.randn(6, 2, 3, 2, dtype=torch.float64)
    bias = torch.randn(6, dtype=torch.float64)
    expected = F.conv2d(F.pad(x, (1, 2, 1, 0)), weight, bias, stride=(2, 1), dilation=(1, 2), groups=2)
    oh, ow = expected.shape[-2:]
    actual = deform_expression(x, torch.zeros(2, 48, oh, ow, dtype=x.dtype), weight,
        torch.ones(2, 24, oh, ow, dtype=x.dtype), bias, stride=(2, 1),
        dilation=(1, 2), offset_groups=4, padding_end=(0, 2))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('shape,block', [((3, 5), (2, 3)), ((2, 3, 7), (1, 2, 4)), ((5,), (3,)), ((1, 1), (4, 4))])
@pytest.mark.parametrize('bits', [2, 4, 8])
def test_ragged_quantization_matches_independent_block_indexing(shape, block, bits):
    torch.manual_seed(114)
    x = torch.randn(shape, dtype=torch.float32)
    params = tuple(math.ceil(d/b) for d, b in zip(shape, block))
    padding = tuple(p for d, b, n in reversed(list(zip(shape, block, params))) for p in (0, n*b-d))
    interleaved = tuple(v for pair in zip(params, block) for v in pair)
    axes = tuple(range(1, 2*len(shape), 2))
    alpha = F.pad(x.abs(), padding).reshape(interleaved).amax(axes)
    hi = 2**(bits-1)-1
    scales = (alpha/hi).clamp_min(torch.finfo(torch.float32).tiny)
    expanded = scales.reshape(tuple(v for n in params for v in (n, 1)))
    expanded = expanded.expand(interleaved).reshape(tuple(n*b for n, b in zip(params, block)))
    expanded = expanded[tuple(slice(0, d) for d in shape)]
    actual = ((x/expanded).sign()*((x/expanded).abs()+.5).floor()).clamp(-hi, hi)
    for coord in itertools.product(*(range(d) for d in shape)):
        key = tuple(c//b for c, b in zip(coord, block))
        slices = tuple(slice(k*b, min((k+1)*b, d)) for k, b, d in zip(key, block, shape))
        scalar_scale = x[slices].abs().max().item()/hi
        torch.testing.assert_close(scales[key], torch.tensor(scalar_scale), rtol=1e-6, atol=1e-7)
        value = x[coord].item()/scalar_scale
        expected = min(hi, max(-hi, math.copysign(math.floor(abs(value)+.5), value)))
        assert actual[coord].item() == expected
