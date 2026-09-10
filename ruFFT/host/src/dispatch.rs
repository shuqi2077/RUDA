use ruda_core::tensor::{DType, host::HostTensor};

pub fn rfft(
    signal: HostTensor,
    dim: usize,
    n: Option<usize>,
) -> (HostTensor, HostTensor) {
    match signal.dtype() {
        DType::F32 => crate::rfft_f32(signal, dim, n),
        DType::F64 => crate::rfft_f64(signal, dim, n),
        DType::F16 => crate::rfft_f16(signal, dim, n),
        DType::BF16 => crate::rfft_bf16(signal, dim, n),
        dtype => panic!("rfft: unsupported dtype {:?}", dtype),
    }
}

pub fn irfft(
    spectrum_re: HostTensor,
    spectrum_im: HostTensor,
    dim: usize,
    n: Option<usize>,
) -> HostTensor {
    match spectrum_re.dtype() {
        DType::F32 => crate::irfft_f32(spectrum_re, spectrum_im, dim, n),
        DType::F64 => crate::irfft_f64(spectrum_re, spectrum_im, dim, n),
        DType::F16 => crate::irfft_f16(spectrum_re, spectrum_im, dim, n),
        DType::BF16 => crate::irfft_bf16(spectrum_re, spectrum_im, dim, n),
        dtype => panic!("irfft: unsupported dtype {:?}", dtype),
    }
}

