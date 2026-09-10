// SPDX-License-Identifier: Apache-2.0
//! Explicit experimental fast paths; no silent algorithm/backend substitution.
use super::{BatchedCholeskyOptions, BatchedCholeskyResult, BatchedLuOptions,
    BatchedLuResult, DeviceSolverError, validate_tensor, warp_kernel};
use crate::kernel_plan::{WarpDirectKind, WarpDirectLimits, WarpDirectPlan, WarpPlanError};
use ruda_core::{device::Device, ir::features::Plane, tensor::DType};
use ruda_kernel::{dsl::{Runtime, prelude::{RudaCount,RudaDim}},
    tensor::{RudaTensor,allocation::empty_device_contiguous_dtype}};
const SOURCE: &str = concat!(include_str!("warp_kernel.rs"),include_str!("warp.rs"),include_str!("../kernel_plan.rs"));

fn map_plan(e: WarpPlanError) -> DeviceSolverError {
    match e {
        WarpPlanError::SizeOverflow => DeviceSolverError::SizeOverflow,
        WarpPlanError::InvalidShape => DeviceSolverError::InvalidInput("A=[batch,n,n], B=[batch,n,nrhs], n=1..32, nrhs=1..8"),
        WarpPlanError::Unsupported(s) => DeviceSolverError::InvalidInput(s),
    }
}
fn plan<R:Runtime>(a:&RudaTensor<R>,b:&RudaTensor<R>,kind:WarpDirectKind)->Result<WarpDirectPlan,DeviceSolverError> {
    validate_tensor(a)?; validate_tensor(b)?;
    let s=a.meta.shape();let t=b.meta.shape();
    if s[1]!=s[2] || s[0]!=t[0] || s[1]!=t[1] {
        return Err(DeviceSolverError::InvalidInput("A=[batch,n,n], B=[batch,n,nrhs]"));
    }
    if a.device.to_id()!=b.device.to_id() || !a.client.same_execution_queue(&b.client) {
        return Err(DeviceSolverError::DifferentQueue);
    }
    let p=WarpDirectPlan::new(s[0],s[1],t[2],kind).map_err(map_plan)?;
    let props=a.client.properties();let h=&props.hardware;
    p.check_device(WarpDirectLimits{plane_min:h.plane_size_min,plane_max:h.plane_size_max,
        plane_ops:props.features.plane.contains(Plane::Ops),max_threads:h.max_units_per_ruda,
        max_block_x:h.max_ruda_dim.0,max_grid_x:h.max_ruda_count.0,shared_bytes:h.max_shared_memory_size})
        .map_err(map_plan)?;
    Ok(p)
}
fn finite_nonnegative(values:&[f32])->Result<(),DeviceSolverError> {
    if values.iter().any(|v|!v.is_finite() || *v<0.0) {
        Err(DeviceSolverError::InvalidInput("finite nonnegative shift/tolerances required"))
    } else { Ok(()) }
}
fn alloc<R:Runtime>(a:&RudaTensor<R>,shape:impl Into<ruda_core::tensor::Shape>,dtype:DType)->RudaTensor<R> {
    empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),shape.into(),dtype)
}

/// Opt-in Cholesky+solve using one full warp and shared storage per matrix.
/// Same input/output/status contract as `cholesky_solve_batched`. Inputs remain
/// read-only, success is asynchronous, and no host computation/cast is performed.
/// n=1..32, nrhs=1..8. Rejects devices without fixed 32-lane plane operations.
/// One kernel per nonempty batch. Padding/scratch is confined to shared memory.
/// Kept separate until measurements establish useful shape/batch thresholds.
pub fn cholesky_solve_batched_warp<R:Runtime>(a:&RudaTensor<R>,b:&RudaTensor<R>,o:BatchedCholeskyOptions)
    ->Result<BatchedCholeskyResult<R>,DeviceSolverError> {
    finite_nonnegative(&[o.diagonal_shift,o.symmetry_absolute_tolerance,o.symmetry_relative_tolerance])?;
    let p=plan(a,b,WarpDirectKind::Cholesky)?;
    let lower=alloc(a,[p.batch(),p.order(),p.order()],DType::F32);
    let solution=alloc(a,[p.batch(),p.order(),p.rhs()],DType::F32);
    let info=alloc(a,[p.batch()],DType::I32);
    if p.batch()>0 {
        warp_kernel::cholesky_solve_warp::launch::<R>(&a.client,
            RudaCount::new_1d(p.batch() as u32),RudaDim::new_1d(WarpDirectPlan::THREADS),
            a.clone().into_array_arg(),b.clone().into_array_arg(),lower.clone().into_array_arg(),
            solution.clone().into_array_arg(),info.clone().into_array_arg(),
            o.diagonal_shift,o.symmetry_absolute_tolerance,o.symmetry_relative_tolerance,
            p.order(),p.rhs(),p.pitch(),SOURCE.to_owned());
    }
    Ok(BatchedCholeskyResult{lower,solution,info,submitted_kernels:usize::from(p.batch()>0)})
}

/// Opt-in scaled, partial-row-pivoted LU+solve using one warp per matrix.
/// Same packed-LU, zero-based sequential pivots and failure zeroing as
/// `lu_solve_batched`. Lowest row wins pivot ties. No tensor-core/TF32 path or
/// precision change. New output buffers are allocated; this is not in-place.
pub fn lu_solve_batched_warp<R:Runtime>(a:&RudaTensor<R>,b:&RudaTensor<R>,o:BatchedLuOptions)
    ->Result<BatchedLuResult<R>,DeviceSolverError> {
    finite_nonnegative(&[o.pivot_absolute_tolerance,o.pivot_relative_tolerance])?;
    let p=plan(a,b,WarpDirectKind::Lu)?;
    let packed_lu=alloc(a,[p.batch(),p.order(),p.order()],DType::F32);
    let solution=alloc(a,[p.batch(),p.order(),p.rhs()],DType::F32);
    let pivots=alloc(a,[p.batch(),p.order()],DType::I32);
    let info=alloc(a,[p.batch()],DType::I32);
    if p.batch()>0 {
        warp_kernel::lu_solve_warp::launch::<R>(&a.client,
            RudaCount::new_1d(p.batch() as u32),RudaDim::new_1d(WarpDirectPlan::THREADS),
            a.clone().into_array_arg(),b.clone().into_array_arg(),packed_lu.clone().into_array_arg(),
            solution.clone().into_array_arg(),pivots.clone().into_array_arg(),info.clone().into_array_arg(),
            o.pivot_absolute_tolerance,o.pivot_relative_tolerance,p.order(),p.rhs(),p.pitch(),SOURCE.to_owned());
    }
    Ok(BatchedLuResult{packed_lu,solution,pivots,info,submitted_kernels:usize::from(p.batch()>0)})
}
