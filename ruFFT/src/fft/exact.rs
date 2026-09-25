//! Exact-length F32 real transforms. Non-radix-2 lengths use Bluestein's
//! convolution, never an N-point transform silently replaced by a larger DFT.
//!
//! This plan owns chirps and reusable device scratch. It must be used on the
//! execution queue on which it was created. It does not perform CPU arithmetic
//! on signal data or implicitly synchronize the device.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl::backtrace::BackTrace;
use ruda_kernel::library::tensor::TensorHandle;
use super::{FftMode, rfft_launch_padded, irfft_launch_padded};
use super::cfft::{CfftBindings, MAX_SHARED_N_FFT, cfft_launch_any_size,
    cfft_launch_with_scratch};

mod kernels;
use kernels::*;

fn invalid(reason: impl Into<String>) -> LaunchError {
    LaunchError::Unknown { reason: reason.into(), backtrace: BackTrace::capture() }
}

/// Bounded implementation limit, not a device memory availability guarantee.
const MAX_CONVOLUTION: usize = MAX_SHARED_N_FFT * MAX_SHARED_N_FFT;

/// Bluestein workspace length. Power-of-two real FFTs use the existing packed
/// real path, which can support up to twice this complex FFT limit.
pub fn exact_convolution_len(n: usize) -> Result<Option<usize>, LaunchError> {
    if n == 0 { return Err(invalid("FFT length must be positive")); }
    if n.is_power_of_two() {
        if n > MAX_CONVOLUTION * 2 { return Err(invalid("radix-2 FFT exceeds the implemented four-step limit")); }
        return Ok(None);
    }
    let m = n.checked_mul(2).and_then(|v| v.checked_sub(1))
        .and_then(usize::checked_next_power_of_two)
        .filter(|&v| v <= MAX_CONVOLUTION)
        .ok_or_else(|| invalid("Bluestein convolution exceeds the implemented four-step limit"))?;
    Ok(Some(m))
}

fn alloc<R: Runtime>(client: &ComputeClient<R>, rows: usize, cols: usize)
    -> Result<TensorHandle<R>, LaunchError>
{
    let elements = rows.checked_mul(cols).filter(|&v| v <= u32::MAX as usize)
        .ok_or_else(|| invalid("FFT workspace exceeds 32-bit indexing"))?;
    let bytes = elements.checked_mul(4).ok_or_else(|| invalid("FFT workspace byte-size overflow"))?;
    Ok(TensorHandle::new_contiguous(vec![rows, cols], client.empty(bytes),
        f32::as_type_native_unchecked().storage_type()))
}

struct Tables<R: Runtime> {
    m: usize,
    chirp_re: TensorHandle<R>,
    chirp_im: TensorHandle<R>,
    spectrum_re: TensorHandle<R>,
    spectrum_im: TensorHandle<R>,
}
struct Workspace<R: Runtime> {
    rows: usize,
    re: TensorHandle<R>,
    im: TensorHandle<R>,
    four_step: Option<(TensorHandle<R>, TensorHandle<R>)>,
}

/// Reusable exact-length F32 FFT plan. The direction and N are fixed; batches
/// may change, in which case the batched workspace is replaced, not grown
/// without bound. Caller-owned output bindings must be non-overlapping.
/// Raw bindings carry no dtype/device tag: pass F32 storage from this client.
pub struct RealFftPlan<R: Runtime> {
    client: ComputeClient<R>,
    n: usize,
    mode: FftMode,
    tables: Option<Tables<R>>,
    workspace: Option<Workspace<R>>,
}
impl<R: Runtime> RealFftPlan<R> {
    pub fn new(client: ComputeClient<R>, n: usize, mode: FftMode) -> Result<Self, LaunchError> {
        let m = exact_convolution_len(n)?;
        // Pin an implicit client to its current queue without introducing a
        // new stream or changing any allocation's ownership.
        let client = client.fixed_execution_queue();
        let tables = if let Some(m) = m {
            let chirp_re = alloc(&client, 1, n)?;
            let chirp_im = alloc(&client, 1, n)?;
            let spectrum_re = alloc(&client, 1, m)?;
            let spectrum_im = alloc(&client, 1, m)?;
            let block = RudaDim::new_1d(256);
            let grid = ruda_kernel::dsl::calculate_ruda_count_elemwise(&client, m, block);
            make_chirp::launch::<R>(&client, grid, block,
                chirp_re.clone().into_arg(), chirp_im.clone().into_arg(),
                spectrum_re.clone().into_arg(), spectrum_im.clone().into_arg(), n, m, mode);
            cfft_launch_any_size(&client, CfftBindings {
                input_re: spectrum_re.clone().binding(), input_im: spectrum_im.clone().binding(),
                output_re: spectrum_re.clone().binding(), output_im: spectrum_im.clone().binding(),
            }, 1, f32::as_type_native_unchecked().storage_type(), FftMode::Forward)?;
            Some(Tables { m, chirp_re, chirp_im, spectrum_re, spectrum_im })
        } else { None };
        Ok(Self { client, n, mode, tables, workspace: None })
    }

    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { false }
    pub fn convolution_len(&self) -> Option<usize> { self.tables.as_ref().map(|t| t.m) }
    /// Bytes retained by this plan, excluding output tensors, driver metadata,
    /// compiled kernels and temporary initialization scratch.
    pub fn retained_bytes(&self) -> usize {
        let tables = self.tables.as_ref().map_or(0, |t| 8 * (self.n + t.m));
        tables + self.workspace.as_ref().map_or(0, |w| {
            let m = self.tables.as_ref().unwrap().m;
            w.rows * m * if w.four_step.is_some() { 16 } else { 8 }
        })
    }
    /// Release scratch; chirp tables remain reusable. Runtime stream ordering
    /// handles deferred release. This is not a device-wide synchronization.
    pub fn clear_workspace(&mut self) { self.workspace = None; }

    fn prepare_workspace(&mut self, rows: usize) -> Result<(), LaunchError> {
        if self.workspace.as_ref().is_some_and(|w| w.rows == rows) { return Ok(()); }
        let m = self.tables.as_ref().unwrap().m;
        // Drop the old workspace before reserving a differently sized one.
        // The allocator still honors pending users on its execution queue.
        self.workspace = None;
        let re = alloc(&self.client, rows, m)?;
        let im = alloc(&self.client, rows, m)?;
        let four_step = if m > MAX_SHARED_N_FFT {
            Some((alloc(&self.client, rows, m)?, alloc(&self.client, rows, m)?))
        } else { None };
        self.workspace = Some(Workspace { rows, re, im, four_step });
        Ok(())
    }

    fn convolution(&self) -> Result<(), LaunchError> {
        let t = self.tables.as_ref().unwrap();
        let w = self.workspace.as_ref().unwrap();
        let dtype = f32::as_type_native_unchecked().storage_type();
        let bindings = || CfftBindings {
            input_re: w.re.clone().binding(), input_im: w.im.clone().binding(),
            output_re: w.re.clone().binding(), output_im: w.im.clone().binding(),
        };
        cfft_launch_with_scratch(&self.client, bindings(), 1, dtype, FftMode::Forward,
            w.four_step.clone())?;
        let total = w.rows * t.m;
        let block = RudaDim::new_1d(256);
        let grid = ruda_kernel::dsl::calculate_ruda_count_elemwise(&self.client, total, block);
        multiply_spectrum::launch::<R>(&self.client, grid, block,
            w.re.clone().into_arg(), w.im.clone().into_arg(),
            t.spectrum_re.clone().into_arg(), t.spectrum_im.clone().into_arg(),
            total as u32, t.m);
        cfft_launch_with_scratch(&self.client, bindings(), 1, dtype, FftMode::Inverse,
            w.four_step.clone())
    }

    /// Unnormalized N-point RFFT. Samples at `used..N` are zero, without
    /// constructing a padded input tensor. `used` may be zero.
    pub fn forward(&mut self, signal: TensorBinding<R>, real: TensorBinding<R>,
        imag: TensorBinding<R>, dim: usize, used: usize) -> Result<(), LaunchError>
    {
        if self.mode != FftMode::Forward { return Err(invalid("forward called on inverse FFT plan")); }
        let rows = validate(&signal, &real, &imag, dim, self.n, used, false)?;
        if rows == 0 { return Ok(()); }
        let block = RudaDim::new_1d(256);
        if self.n == 1 {
            scalar_forward::launch::<R>(&self.client,
                ruda_kernel::dsl::calculate_ruda_count_elemwise(&self.client, rows, block), block,
                signal.into_tensor_arg(), real.into_tensor_arg(), imag.into_tensor_arg(),
                rows as u32, used as u32, dim);
            return Ok(());
        }
        if self.tables.is_none() {
            return rfft_launch_padded(&self.client, signal, real, imag, dim, used,
                f32::as_type_native_unchecked().storage_type());
        }
        self.prepare_workspace(rows)?;
        let t = self.tables.as_ref().unwrap();
        let w = self.workspace.as_ref().unwrap();
        let total = rows * t.m;
        prepare_forward::launch::<R>(&self.client,
            ruda_kernel::dsl::calculate_ruda_count_elemwise(&self.client, total, block), block,
            signal.into_tensor_arg(), t.chirp_re.clone().into_arg(), t.chirp_im.clone().into_arg(),
            w.re.clone().into_arg(), w.im.clone().into_arg(), total as u32, used as u32,
            self.n, t.m, dim);
        self.convolution()?;
        let bins = self.n / 2 + 1;
        finish_forward::launch::<R>(&self.client,
            ruda_kernel::dsl::calculate_ruda_count_elemwise(&self.client, rows * bins, block), block,
            w.re.clone().into_arg(), w.im.clone().into_arg(),
            t.chirp_re.clone().into_arg(), t.chirp_im.clone().into_arg(),
            real.into_tensor_arg(), imag.into_tensor_arg(), (rows * bins) as u32,
            self.n, t.m, dim);
        Ok(())
    }

    /// N-point IRFFT normalized by 1/N. DC, and Nyquist for even N, are
    /// real-only. Bins at `used..N/2+1` are treated as zero.
    pub fn inverse(&mut self, real: TensorBinding<R>, imag: TensorBinding<R>,
        signal: TensorBinding<R>, dim: usize, used: usize) -> Result<(), LaunchError>
    {
        if self.mode != FftMode::Inverse { return Err(invalid("inverse called on forward FFT plan")); }
        let rows = validate(&signal, &real, &imag, dim, self.n, used, true)?;
        if rows == 0 { return Ok(()); }
        let block = RudaDim::new_1d(256);
        if self.n == 1 {
            scalar_inverse::launch::<R>(&self.client,
                ruda_kernel::dsl::calculate_ruda_count_elemwise(&self.client, rows, block), block,
                real.into_tensor_arg(), signal.into_tensor_arg(), rows as u32, used as u32, dim);
            return Ok(());
        }
        if self.tables.is_none() {
            return irfft_launch_padded(&self.client, real, imag, signal, dim, used,
                f32::as_type_native_unchecked().storage_type());
        }
        self.prepare_workspace(rows)?;
        let t = self.tables.as_ref().unwrap();
        let w = self.workspace.as_ref().unwrap();
        let total = rows * t.m;
        prepare_inverse::launch::<R>(&self.client,
            ruda_kernel::dsl::calculate_ruda_count_elemwise(&self.client, total, block), block,
            real.into_tensor_arg(), imag.into_tensor_arg(),
            t.chirp_re.clone().into_arg(), t.chirp_im.clone().into_arg(),
            w.re.clone().into_arg(), w.im.clone().into_arg(), total as u32, used as u32,
            self.n, t.m, dim);
        self.convolution()?;
        finish_inverse::launch::<R>(&self.client,
            ruda_kernel::dsl::calculate_ruda_count_elemwise(&self.client, rows * self.n, block), block,
            w.re.clone().into_arg(), w.im.clone().into_arg(),
            t.chirp_re.clone().into_arg(), t.chirp_im.clone().into_arg(),
            signal.into_tensor_arg(), (rows * self.n) as u32, self.n, t.m, dim);
        Ok(())
    }
}

fn check_binding<R: Runtime>(binding: &TensorBinding<R>) -> Result<(), LaunchError> {
    if binding.shape.len() != binding.strides.len() || binding.shape.is_empty() {
        return Err(invalid("FFT needs non-scalar, well-formed tensor metadata"));
    }
    if binding.shape.contains(&0) { return Ok(()); }
    let last = binding.shape.iter().zip(binding.strides.iter()).try_fold(0usize,
        |sum, (&size, &stride)| (size - 1).checked_mul(stride).and_then(|v| sum.checked_add(v)))
        .filter(|&v| v < u32::MAX as usize)
        .and_then(|v| v.checked_add(1)).and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("FFT tensor span overflows F32/U32 addressing"))?;
    let available = binding.handle.size.checked_sub(binding.handle.offset_start.unwrap_or(0))
        .and_then(|v| v.checked_sub(binding.handle.offset_end.unwrap_or(0)))
        .ok_or_else(|| invalid("invalid FFT buffer offsets"))?;
    if last as u64 > available { return Err(invalid("FFT tensor strides exceed buffer size")); }
    Ok(())
}

fn validate<R: Runtime>(signal: &TensorBinding<R>, real: &TensorBinding<R>,
    imag: &TensorBinding<R>, dim: usize, n: usize, used: usize, inverse: bool)
    -> Result<usize, LaunchError>
{
    for binding in [signal, real, imag] { check_binding(binding)?; }
    if dim >= signal.shape.len() || signal.shape.len() != real.shape.len()
        || real.shape != imag.shape {
        return Err(invalid("invalid FFT dimension, rank or real/imag shapes"));
    }
    let bins = n / 2 + 1;
    if (!inverse && real.shape[dim] != bins) || (inverse && signal.shape[dim] != n)
        || (!inverse && (used > signal.shape[dim] || used > n))
        || (inverse && (used == 0 || used > real.shape[dim] || used > bins)) {
        return Err(invalid("FFT input/output length does not match plan"));
    }
    let mut rows = 1usize;
    for (axis, (&a, &b)) in signal.shape.iter().zip(real.shape.iter()).enumerate() {
        if axis != dim {
            if a != b { return Err(invalid("FFT batch shapes must match")); }
            rows = rows.checked_mul(a).ok_or_else(|| invalid("FFT batch size overflow"))?;
        }
    }
    rows.checked_mul(n.max(bins)).filter(|&v| v <= u32::MAX as usize)
        .ok_or_else(|| invalid("FFT output exceeds 32-bit indexing"))?;
    // Outputs may be pitched or permuted, but must not have overlapping
    // elements. This sufficient test accepts dense permutations and padding.
    for out in if inverse { vec![signal] } else { vec![real, imag] } {
        let mut axes: Vec<_> = out.shape.iter().zip(out.strides.iter())
            .filter(|(size, _)| **size > 1).map(|(&size, &stride)| (stride, size)).collect();
        axes.sort_unstable();
        let mut span = 1usize;
        for (stride, size) in axes {
            if stride < span { return Err(invalid("FFT output layout has overlapping elements")); }
            span = (size - 1).checked_mul(stride).and_then(|v| v.checked_add(span))
                .ok_or_else(|| invalid("FFT output layout overflow"))?;
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_plan_length_limits() {
        for n in [1, 2, 4, 1024, 8192] { assert_eq!(exact_convolution_len(n).unwrap(), None); }
        for (n, m) in [(3, 8), (6, 16), (7, 16), (1009, 2048), (4097, 16384)] {
            assert_eq!(exact_convolution_len(n).unwrap(), Some(m));
        }
        assert!(exact_convolution_len(0).is_err());
        assert!(exact_convolution_len(usize::MAX).is_err());
        assert!(exact_convolution_len(MAX_CONVOLUTION).is_ok());
        assert!(exact_convolution_len(MAX_CONVOLUTION - 1).is_err());
    }
}
