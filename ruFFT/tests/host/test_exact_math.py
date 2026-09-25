"""Host mathematical references, not Rust compilation or GPU validation."""
from __future__ import annotations
import numpy as np
import pytest
from exact_reference import BluesteinReference, convolution_length, radix2, ruda_window_offset

LENGTHS = [1, 2, 3, 5, 6, 7, 11, 16, 31, 64, 127, 257, 1009, 2049, 4097]

@pytest.mark.parametrize("n", LENGTHS)
def test_forward_matches_actual_n_point_dft(n):
    x = np.random.default_rng(n).normal(size=(3, n)).astype(np.float32)
    got = BluesteinReference(n, -1).forward(x)
    np.testing.assert_allclose(got, np.fft.rfft(x), rtol=8e-5, atol=3e-4)
    assert got.shape == (3, n // 2 + 1)

@pytest.mark.parametrize("n", LENGTHS)
def test_inverse_from_independent_spectrum(n):
    x = np.random.default_rng(n + 123).normal(size=(2, n)).astype(np.float32)
    spectrum = np.fft.rfft(x)
    got = BluesteinReference(n, 1).inverse(spectrum.real, spectrum.imag)
    np.testing.assert_allclose(got, x, rtol=4e-5, atol=5e-6)

@pytest.mark.parametrize("n", [1, 6, 7, 16, 1009])
@pytest.mark.parametrize("input_len", [0, 1, 4, 31])
def test_virtual_padding_and_truncation(n, input_len):
    x = np.random.default_rng(input_len).normal(size=(2, input_len)).astype(np.float32)
    used = min(input_len, n)
    expected = np.zeros((2, n), dtype=np.float32)
    expected[:, :used] = x[:, :used]
    np.testing.assert_allclose(BluesteinReference(n, -1).forward(x, used),
                               np.fft.rfft(expected), rtol=8e-5, atol=3e-4)

@pytest.mark.parametrize("n", [1, 2, 6, 7, 8, 31, 1009])
def test_irfft_discards_only_real_only_imaginary_bins(n):
    bins = n // 2 + 1
    re, im = np.ones(bins, np.float32), np.zeros(bins, np.float32)
    im[0] = np.nan
    if n % 2 == 0:
        im[-1] = np.nan
    got = BluesteinReference(n, 1).inverse(re, im)
    expected = np.zeros(n); expected[0] = 1
    np.testing.assert_allclose(got, expected, atol=4e-6)

@pytest.mark.parametrize("n", [3, 6, 7, 11, 32])
@pytest.mark.parametrize("used", [1, 2])
def test_inverse_virtual_spectrum_padding(n, used):
    bins = n // 2 + 1
    x = np.random.default_rng(n).normal(size=n).astype(np.float32)
    spectrum = np.fft.rfft(x)
    kept = spectrum[:used]
    expected = np.fft.irfft(kept, n=n)
    got = BluesteinReference(n, 1).inverse(kept.real, kept.imag)
    np.testing.assert_allclose(got, expected, rtol=5e-5, atol=6e-6)

@pytest.mark.parametrize("n", [6, 7, 1009, 4097])
def test_chirp_cache_does_not_cache_signal_data(n):
    plan = BluesteinReference(n, -1)
    spectrum_id = id(plan.b_spectrum)
    for scale in [1, 0, -2, 0.25]:
        x = (np.arange(n, dtype=np.float32) % 7) * scale
        np.testing.assert_allclose(plan.forward(x), np.fft.rfft(x), rtol=1e-4, atol=0.01)
        assert id(plan.b_spectrum) == spectrum_id

def test_six_point_is_not_padded_eight_point():
    x = np.arange(1, 7, dtype=np.float32)
    got = BluesteinReference(6, -1).forward(x)
    np.testing.assert_allclose(got, np.array([21, -3+5.196152423j, -3+1.732050808j, -3]), atol=6e-6)
    assert got.shape != np.fft.rfft(x, n=8).shape

@pytest.mark.parametrize("shape,dim", [((2, 6, 3),1), ((6, 2, 3),0), ((2,3,6),2)])
def test_batch_layout_uses_identical_window_order_for_input_and_output(shape, dim):
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    in_steps = tuple(v // 4 for v in x.strides)
    out_shape = list(shape);out_shape[dim] = shape[dim] // 2 + 1
    output = np.empty(out_shape, dtype=np.complex64)
    out_steps = tuple(v // 8 for v in output.strides)
    rows = np.prod(shape) // shape[dim]
    plan = BluesteinReference(shape[dim], -1)
    for row in range(rows):
        start = ruda_window_offset(row, shape, in_steps, dim)
        signal = x.ravel()[start + np.arange(shape[dim]) * in_steps[dim]]
        result = plan.forward(signal)
        dst = ruda_window_offset(row, tuple(out_shape), out_steps, dim)
        output.ravel()[dst + np.arange(len(result)) * out_steps[dim]] = result
    np.testing.assert_allclose(output, np.fft.rfft(x, axis=dim), rtol=4e-5, atol=8e-5)

@pytest.mark.parametrize("n", [0, -1, 2**63-1, 4096*4096-1])
def test_transform_limit_rejection(n):
    with pytest.raises(ValueError):convolution_length(n)

@pytest.mark.parametrize("n,m", [(1,None),(2,None),(6,16),(1009,2048),(4097,16384)])
def test_convolution_limits(n,m):
    assert convolution_length(n) == m

def test_chirp_integer_reduction_avoids_u32_square_overflow():
    n = 100_003
    j = np.array([65535,65536,99999,100002],dtype=np.uint64)
    got = j*j%(2*n)
    assert got.tolist() == [(int(v)*int(v))%(2*n) for v in j]
    assert not np.array_equal(got, ((j*j).astype(np.uint32)%(2*n)).astype(np.uint64))

@pytest.mark.parametrize("sign", [-1,1])
def test_radix2_float32_reference_normalization(sign):
    x = np.random.default_rng(7).normal(size=32).astype(np.float32).astype(np.complex64)
    expected = np.fft.fft(x) if sign==-1 else np.fft.ifft(x)*32
    np.testing.assert_allclose(radix2(x,sign),expected,rtol=2e-6,atol=5e-6)
