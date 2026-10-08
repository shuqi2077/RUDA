// SPDX-License-Identifier: Apache-2.0
//! Opt-in native device kernels. No numerical operation falls back to the host.
//! Contiguous FP32 batched input only. LU/Cholesky/eigen n<=32, QR n<=32
//! and m<=64, dense CG n<=128. Consult each explicit API for its shape contract.
//! Default baselines use one GPU thread per system; explicit `*_warp` paths
//! cooperate within a full 32-thread block and reuse shared scratch.
//! These are experimental small-matrix kernels, not high-throughput blocked
//! large-matrix solver. Benchmarks must not compare its scope to full cuSOLVER.
use std::{
    error::Error, fmt
};
use ruda_core::{
    device::Device, tensor::DType
};
use ruda_kernel::{
    dsl::{
        Runtime, calculate_ruda_count_elemwise, prelude::RudaDim
    },
    tensor::{
        RudaTensor, allocation::empty_device_contiguous_dtype, readback::into_data
    }
};
#[allow(unsafe_code)] // Only the existing kernel DSL's generated launch boundary.
mod kernel;
#[derive(Clone, Copy, Debug)]
pub struct BatchedCholeskyOptions {
    /// Explicit regularization: solve (A + diagonal_shift*I) X = B.
    /// Defaults to zero; failure NEVER adds hidden jitter.
    pub diagonal_shift: f32,
    pub symmetry_absolute_tolerance: f32,
    pub symmetry_relative_tolerance: f32,
}
impl Default for BatchedCholeskyOptions{
    fn default()->Self{
        Self{
            diagonal_shift: 0.0, symmetry_absolute_tolerance: 0.0, symmetry_relative_tolerance: 1e-5
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum DeviceSolverError {
    InvalidInput(&'static str), DifferentQueue, SizeOverflow,
    MatrixFailure{
        batch: usize, info: i32
    }, Readback(String), Autotune(String)
}
impl fmt::Display for DeviceSolverError{
    fn fmt(&self, f: &mut fmt::Formatter<'_>)->fmt::Result{
        match self{
            Self::InvalidInput(s)=>write!(f, "invalid device solver input: {s}"),
            Self::DifferentQueue=>write!(f, "both tensors must share a device and execution queue"),
            Self::SizeOverflow=>write!(f, "device index/byte count overflows"),
            Self::MatrixFailure{
                batch, info
            }=>write!(f, "batch {batch} failed with info={info}"),
            Self::Readback(s)=>write!(f, "status readback failed: {s}"),
            Self::Autotune(s)=>write!(f, "solver autotune failed without replay: {s}"),
        }
    }
}
impl Error for DeviceSolverError{
}
pub struct BatchedCholeskyResult<R: Runtime>{
    pub lower: RudaTensor<R>, pub solution: RudaTensor<R>,
    /// 0 success; >0 first failed positive-definite pivot (one-based);
    /// -1 nonfinite input; -2 nonsymmetric; -3 nonfinite intermediate.
    /// Failed systems have zeroed factor/solution buffers, NOT valid answers.
    pub info: RudaTensor<R>, pub submitted_kernels: usize,
}
impl<R: Runtime>BatchedCholeskyResult<R>{
    /// Explicit synchronization and status-only readback. Device tensors remain
    /// on the device. Readback execution errors are mapped to DeviceSolverError::Readback.
    /// Allocation/launch still use the existing runtime's submission contract.
    pub fn check_status_sync(&self)->Result<(), DeviceSolverError>{
        let data=ruda_core::future::block_on(into_data(self.info.clone()))
        .map_err(|e|DeviceSolverError::Readback(format!("{e:?}")))?;
        let codes=data.to_vec::<i32>().map_err(|e|DeviceSolverError::Readback(format!("{e:?}")))?;
        for(batch, &info)in codes.iter().enumerate(){
            if info!=0{
                return Err(DeviceSolverError::MatrixFailure{
                    batch, info
                });
            }
        }
        Ok(())
    }
}
fn elements(shape: &[usize])->Result<usize, DeviceSolverError>{
    let n=shape.iter().try_fold(1usize, |a, &b|a.checked_mul(b)).ok_or(DeviceSolverError::SizeOverflow)?;
    if n>u32::MAX as usize{
        return Err(DeviceSolverError::SizeOverflow);
    }
    Ok(n)
}
fn validate_tensor<R: Runtime>(t: &RudaTensor<R>)->Result<(), DeviceSolverError>{
    if t.dtype!=DType::F32||t.qparams.is_some(){
        return Err(DeviceSolverError::InvalidInput("unquantized FP32 tensors required"));
    }
    if t.meta.shape().len()!=3||t.meta.strides().len()!=3{
        return Err(DeviceSolverError::InvalidInput("three axes required"));
    }
    let n=elements(t.meta.shape())?;
    if n>0&&!t.is_contiguous(){
        return Err(DeviceSolverError::InvalidInput("explicit row-major contiguous storage required"));
    }
    let start=t.handle.offset_start.unwrap_or(0);
    let end=t.handle.offset_end.unwrap_or(0);
    let usable=t.handle.size().checked_sub(start).and_then(|s|s.checked_sub(end)).ok_or(DeviceSolverError::SizeOverflow)?;
    if start%4!=0||usable<(n as u64)*4{
        return Err(DeviceSolverError::InvalidInput("misaligned/undersized backing allocation"));
    }
    Ok(())
}
/// Submit ONE native kernel for a nonempty batch: validation of numeric data,
/// Cholesky factorization and all RHS solves. No host numerical fallback, no data
/// readback, no autograd graph, no in-place modification, no automatic cast.
/// Call check_status_sync() before consuming or reporting any result as successful.
pub fn cholesky_solve_batched<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>,
options: BatchedCholeskyOptions)->Result<BatchedCholeskyResult<R>, DeviceSolverError>{
    let (batch, _, _) = validate_cholesky(a, b, options)?;
    if batch > 0 {
        let mut candidates = ruda_kernel::tensor::tuning::elementwise_candidates(a);
        #[cfg(feature = "warp-solvers")]
        if warp::plan(a, b, crate::kernel_plan::WarpDirectKind::Cholesky).is_ok() {
            candidates.push(("warp_cooperative_32", u32::MAX));
        }
        let output = ruda_kernel::tensor::tuning::execute_variants(vec![a.clone(), b.clone()],
            "solver_cholesky_solve", format!("shift={:08x};atol={:08x};rtol={:08x}",
                options.diagonal_shift.to_bits(), options.symmetry_absolute_tolerance.to_bits(),
                options.symmetry_relative_tolerance.to_bits()), candidates, move |inputs, units| {
                #[cfg(feature = "warp-solvers")]
                let result = if units == u32::MAX { warp::cholesky_solve_batched_warp(&inputs[0], &inputs[1], options) }
                    else { cholesky_solve_batched_inner(&inputs[0], &inputs[1], options, units) };
                #[cfg(not(feature = "warp-solvers"))]
                let result = cholesky_solve_batched_inner(&inputs[0], &inputs[1], options, units);
                let result = result.map_err(|error| error.to_string())?;
                Ok(vec![result.lower, result.solution, result.info])
            }).map_err(DeviceSolverError::Autotune)?;
        if let Some(outputs) = output {
            let mut outputs = outputs.into_iter();
            return Ok(BatchedCholeskyResult { lower: outputs.next().expect("factor output"),
                solution: outputs.next().expect("solution output"), info: outputs.next().expect("status output"),
                submitted_kernels: 1 });
        }
    }
    cholesky_solve_batched_inner(a, b, options, 0)
}

fn validate_cholesky<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>,
    options: BatchedCholeskyOptions) -> Result<(usize, usize, usize), DeviceSolverError> {
    validate_tensor(a)?;
    validate_tensor(b)?;
    let shape=a.meta.shape();
    let rhs=b.meta.shape();
    let(batch, n, n2)=(shape[0], shape[1], shape[2]);
    let nrhs=rhs[2];
    if n!=n2||!(1..=32).contains(&n)||!(1..=8).contains(&nrhs)||rhs[0]!=batch||rhs[1]!=n{
        return Err(DeviceSolverError::InvalidInput("A=[batch,n,n], B=[batch,n,nrhs], n=1..32, nrhs=1..8"));
    }
    if a.device.to_id()!=b.device.to_id()||!a.client.same_execution_queue(&b.client){
        return Err(DeviceSolverError::DifferentQueue);
    }
    for value in [options.diagonal_shift, options.symmetry_absolute_tolerance, options.symmetry_relative_tolerance]{
        if !value.is_finite()||value<0.0{
            return Err(DeviceSolverError::InvalidInput("finite nonnegative shift/tolerances required"));
        }
    }
    Ok((batch, n, nrhs))
}

fn cholesky_solve_batched_inner<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>,
    options: BatchedCholeskyOptions, units: u32) -> Result<BatchedCholeskyResult<R>, DeviceSolverError> {
    let (batch, n, nrhs) = validate_cholesky(a, b, options)?;
    let shape = a.meta.shape();
    let rhs = b.meta.shape();
    let client=a.client.clone();
    let lower=empty_device_contiguous_dtype(client.clone(), a.device.clone(), shape.clone(), DType::F32);
    let solution=empty_device_contiguous_dtype(client.clone(), a.device.clone(), rhs.clone(), DType::F32);
    let info=empty_device_contiguous_dtype(client.clone(), a.device.clone(), [batch].into(), DType::I32);
    if batch==0{
        return Ok(BatchedCholeskyResult{
            lower, solution, info, submitted_kernels: 0
        });
    }
    let dim=if units == 0 { RudaDim::new(client.properties(), batch) } else { RudaDim::new_1d(units) };
    kernel::cholesky_solve::launch::<R>(&client, calculate_ruda_count_elemwise(&client, batch, dim), dim,
    a.clone().into_array_arg(), b.clone().into_array_arg(), lower.clone().into_array_arg(),
    solution.clone().into_array_arg(), info.clone().into_array_arg(), n as u32, nrhs as u32,
    options.diagonal_shift, options.symmetry_absolute_tolerance, options.symmetry_relative_tolerance,
    concat!(include_str!("kernel.rs"), include_str!("mod.rs")).to_owned());
    Ok(BatchedCholeskyResult{
        lower, solution, info, submitted_kernels: 1
    })
}

#[allow(unsafe_code)] // Generated checked kernel launch boundary only.
mod advanced_kernel;
mod advanced;
pub use advanced::*;

#[cfg(feature = "warp-solvers")]
#[allow(unsafe_code)] // Generated checked launch boundary; no manual unsafe code.
mod warp_kernel;
#[cfg(feature = "warp-solvers")]
mod warp;
#[cfg(feature = "warp-solvers")]
pub use warp::{cholesky_solve_batched_warp, lu_solve_batched_warp};
