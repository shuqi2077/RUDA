"""Independent host model of the new FFT mathematics, NOT RUDA execution.

This intentionally never imports a RUDA runtime. NumPy supplies the reference
DFT. A separate float32 radix-2 implementation models the device butterflies.
"""
from __future__ import annotations
import numpy as np

MAX_COMPLEX = 4096 * 4096

def convolution_length(n: int) -> int | None:
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("positive integer transform length required")
    if n & (n - 1) == 0:
        if n > 2 * MAX_COMPLEX:
            raise ValueError("radix-2 limit")
        return None
    m = 1 << (2 * n - 2).bit_length()
    if m > MAX_COMPLEX:
        raise ValueError("four-step limit")
    return m

def radix2(values: np.ndarray, sign: int) -> np.ndarray:
    """Unnormalized radix-2 with float32 trigonometry and arithmetic."""
    x = np.asarray(values, dtype=np.complex64)
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("radix-2 input required")
    index = np.arange(n, dtype=np.uint32)
    reverse = np.zeros(n, dtype=np.uint32)
    for _ in range(n.bit_length() - 1):
        reverse = (reverse << 1) | (index & 1)
        index >>= 1
    out = x[..., reverse].copy()
    for stage in range(1, n.bit_length()):
        size = 1 << stage
        half = size // 2
        theta = (np.float32(sign) * np.float32(2 * np.pi) * np.arange(half, dtype=np.float32)
                 / np.float32(size))
        wr, wi = np.cos(theta), np.sin(theta)
        z = out.reshape(*out.shape[:-1], -1, size)
        a = z[..., :half].copy()
        b = z[..., half:].copy()
        tr = wr * b.real - wi * b.imag
        ti = wr * b.imag + wi * b.real
        z[..., :half].real = a.real + tr
        z[..., :half].imag = a.imag + ti
        z[..., half:].real = a.real - tr
        z[..., half:].imag = a.imag - ti
    return out

class BluesteinReference:
    """Caches only data-independent tables; includes odd/even Hermitian rules."""
    def __init__(self, n: int, sign: int):
        self.n = n
        self.m = convolution_length(n)
        self.sign = sign
        if sign not in (-1, 1):
            raise ValueError("sign")
        if self.m is None:
            self.chirp = self.b_spectrum = None
            return
        j = np.arange(n, dtype=np.uint64)
        phase = (np.float32(sign) * np.float32(np.pi)
                 * ((j * j % (2 * n)).astype(np.float32) / np.float32(n)))
        self.chirp = (np.cos(phase) + 1j * np.sin(phase)).astype(np.complex64)
        b = np.zeros(self.m, dtype=np.complex64)
        b[:n] = self.chirp.conj()
        b[self.m - n + 1:] = self.chirp[1:][::-1].conj()
        self.b_spectrum = radix2(b, -1)

    def complex(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.complex64)
        if x.shape[-1] != self.n:
            raise ValueError("length")
        if self.m is None:
            y = radix2(x, self.sign)
        else:
            a = np.zeros((*x.shape[:-1], self.m), dtype=np.complex64)
            a[..., :self.n] = x * self.chirp
            spectrum = radix2(a, -1) * self.b_spectrum
            y = radix2(spectrum, 1)[..., :self.n] * self.chirp / np.float32(self.m)
        return y if self.sign == -1 else y / np.float32(self.n)

    def forward(self, values: np.ndarray, used: int | None = None) -> np.ndarray:
        if self.sign != -1:
            raise ValueError("direction")
        x = np.asarray(values, dtype=np.float32)
        used = min(x.shape[-1], self.n) if used is None else used
        if not 0 <= used <= min(x.shape[-1], self.n):
            raise ValueError("used")
        padded = np.zeros((*x.shape[:-1], self.n), dtype=np.float32)
        padded[..., :used] = x[..., :used]
        out = self.complex(padded)[..., :self.n // 2 + 1].copy()
        out[..., 0].imag = 0
        if self.n % 2 == 0:
            out[..., -1].imag = 0
        return out

    def inverse(self, real: np.ndarray, imag: np.ndarray, used: int | None = None) -> np.ndarray:
        if self.sign != 1:
            raise ValueError("direction")
        real, imag = np.asarray(real, dtype=np.float32), np.asarray(imag, dtype=np.float32)
        if real.shape != imag.shape:
            raise ValueError("shape")
        bins = self.n // 2 + 1
        used = min(real.shape[-1], bins) if used is None else used
        if not 1 <= used <= min(real.shape[-1], bins):
            raise ValueError("used")
        out = np.zeros((*real.shape[:-1], self.n), dtype=np.complex64)
        for k in range(self.n):
            src = k if k <= self.n // 2 else self.n - k
            if src < used:
                out[..., k].real = real[..., src]
                if src != 0 and not (self.n % 2 == 0 and src == self.n // 2):
                    out[..., k].imag = imag[..., src] if k <= self.n // 2 else -imag[..., src]
        return self.complex(out).real

def ruda_window_offset(row: int, shape: tuple[int, ...], strides: tuple[int, ...], dim: int) -> int:
    offset = 0
    for axis, (size, step) in enumerate(zip(shape, strides)):
        if axis != dim:
            offset += (row % size) * step
            row //= size
    return offset
