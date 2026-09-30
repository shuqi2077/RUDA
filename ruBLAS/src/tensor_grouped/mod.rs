mod backward_plan;
mod backward_tensorcore;
#[cfg(test)]
mod tests_backward;
mod tensorcore;
mod segmented;
pub use segmented::{GroupedStrategy, grouped_matmul_nt_segmented};
mod kernel;

use ruda_core::{
    device::Device,
    tensor::{DType, Shape},
};
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous},
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedMatmulError(pub &'static str);

impl fmt::Display for GroupedMatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for GroupedMatmulError {}

/// Device-resident `[M,K] @ [E,N,K].mT`, selecting one expert per row.
/// Group IDs outside `[0,E)` denote padding and produce zero rows.
/// Scalar kernel: FP32 accumulation, one output rounding to the input dtype.
pub fn grouped_matmul_nt<R: Runtime>(
    input: RudaTensor<R>,
    weights: RudaTensor<R>,
    row_experts: RudaTensor<R>,
) -> Result<RudaTensor<R>, GroupedMatmulError> {
    if input.meta.num_dims() != 2 || weights.meta.num_dims() != 3 {
        return Err(GroupedMatmulError(
            "grouped matmul requires [M,K] input and [E,N,K] weights",
        ));
    }
    let m = input.meta.shape()[0];
    let k = input.meta.shape()[1];
    let e = weights.meta.shape()[0];
    let n = weights.meta.shape()[1];
    if k == 0
        || n == 0
        || e == 0
        || weights.meta.shape()[2] != k
        || row_experts.meta.shape() != &Shape::from([m])
        || row_experts.dtype != DType::U32
        || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        || weights.dtype != input.dtype
        || input.qparams.is_some()
        || weights.qparams.is_some()
        || row_experts.qparams.is_some()
    {
        return Err(GroupedMatmulError(
            "grouped matmul requires matching non-quantized float operands and U32 row groups",
        ));
    }
    for tensor in [&weights, &row_experts] {
        if tensor.device.to_id() != input.device.to_id() {
            return Err(GroupedMatmulError(
                "grouped matmul operands must share a device",
            ));
        }
    }
    for dimensions in [&[m, k][..], &[m, n][..], &[e, n, k][..]] {
        if dimensions
            .iter()
            .try_fold(1usize, |size, &dim| size.checked_mul(dim))
            .is_none_or(|size| size > u32::MAX as usize)
        {
            return Err(GroupedMatmulError("grouped matmul exceeds U32 indexing"));
        }
    }
    let output = empty_device_contiguous_dtype(
        input.client.clone(),
        input.device.clone(),
        Shape::from([m, n]),
        input.dtype,
    );
    if m != 0 {
        let input = into_contiguous(input);
        let weights = into_contiguous(weights);
        let row_experts = into_contiguous(row_experts);
        let dim = RudaDim::new(input.client.properties(), m * n);
        kernel::grouped_nt::launch::<R>(
            &input.client,
            calculate_ruda_count_elemwise(&input.client, m * n, dim),
            dim,
            input.clone().into_array_arg(),
            weights.into_array_arg(),
            row_experts.into_array_arg(),
            output.clone().into_array_arg(),
            e as u32,
            n as u32,
            k as u32,
            input.dtype.into(),
        );
    }
    Ok(output)
}

/// Gradients for segmented grouped matmul. `dweights` is always FP32 so
/// low-precision expert weights do not force low-precision accumulation.
#[derive(Debug)]
pub struct GroupedBackward<R: Runtime> {
    pub dinput: RudaTensor<R>,
    pub dweights: RudaTensor<R>,
}

/// First-order backward for
/// `Y[m,n] = sum_k input[m,k] * weights[expert(m),n,k]`.
///
/// # Safety
/// `offsets` must describe the same row grouping as `row_experts`, exactly as
/// required by `grouped_matmul_nt_segmented`.
#[allow(unsafe_code)]
pub unsafe fn grouped_matmul_nt_backward_segmented<R: Runtime>(
    input: RudaTensor<R>, weights: RudaTensor<R>, grad_output: RudaTensor<R>,
    row_experts: RudaTensor<R>, offsets: RudaTensor<R>,
) -> Result<GroupedBackward<R>, GroupedMatmulError> {
    // SAFETY: this compatibility wrapper forwards the same documented invariants.
    unsafe { grouped_matmul_nt_backward_segmented_with_strategy(
        input, weights, grad_output, row_experts, offsets, GroupedStrategy::Scalar,
    ) }
}

/// Opt-in 16x16x16 cooperative backward. Scalar remains the default.
/// Auto selects cooperative kernels only for supported FP16/BF16 devices/grids;
/// compilation or launch errors are propagated, never treated as a fallback.
/// Input/gradient layouts may need contiguous copies, as in the scalar path.
/// The cooperative path uses FP32 accumulation and FP32 dweights; its reduction
/// order may differ from the scalar baseline, so bitwise equality is not promised.
///
/// # Safety
/// The same immutable offsets/row_experts prefix invariants as the scalar
/// segmented backward must hold. GPU values are not copied to the CPU to validate
/// grouping. Metadata producers must preserve these invariants and buffer lifetime.
#[allow(unsafe_code)]
pub unsafe fn grouped_matmul_nt_backward_segmented_with_strategy<R: Runtime>(
    input: RudaTensor<R>, weights: RudaTensor<R>, grad_output: RudaTensor<R>,
    row_experts: RudaTensor<R>, offsets: RudaTensor<R>, strategy: GroupedStrategy,
) -> Result<GroupedBackward<R>, GroupedMatmulError> {
    if input.meta.num_dims()!=2 || weights.meta.num_dims()!=3 || grad_output.meta.num_dims()!=2 {
        return Err(GroupedMatmulError("grouped backward requires [M,K], [E,N,K] and [M,N]"));
    }
    let (m,k)=(input.meta.shape()[0],input.meta.shape()[1]);
    let (e,n)=(weights.meta.shape()[0],weights.meta.shape()[1]);
    if e==0 || n==0 || k==0 || weights.meta.shape()[2]!=k
        || grad_output.meta.shape()[..]!=[m,n]
        || grad_output.dtype!=input.dtype || weights.dtype!=input.dtype
        || row_experts.meta.shape()[..]!=[m] || row_experts.dtype!=DType::U32
        || offsets.meta.shape()[..]!=[e+1] || offsets.dtype!=DType::U32
        || !matches!(input.dtype,DType::F16|DType::BF16|DType::F32)
    { return Err(GroupedMatmulError("invalid grouped backward shape/dtype")); }
    for t in [&input,&weights,&grad_output,&row_experts,&offsets] {
        if t.qparams.is_some() || t.device.to_id()!=input.device.to_id()
            || !t.client.same_execution_queue(&input.client)
            || t.meta.shape().iter().try_fold(1usize,|a,&b|a.checked_mul(b)).is_none_or(|x|x>u32::MAX as usize)
        { return Err(GroupedMatmulError("grouped backward device/queue/size mismatch")); }
    }
    use ruda_core::ir::{ElemType, FloatKind, features::MmaConfig};
    use ruda_kernel::dsl::prelude::RudaCount;
    let props = &input.client.properties().hardware;
    let cfg = MmaConfig { a_type: input.dtype.into(), b_type: input.dtype.into(),
        cd_type: ElemType::Float(FloatKind::F32).into(), m:16, n:16, k:16 };
    let tile_plan = backward_plan::BackwardTilePlan::new(e,n,k,
        [props.max_ruda_count.0, props.max_ruda_count.1, props.max_ruda_count.2]).ok();
    let cooperative = strategy != GroupedStrategy::Scalar
        && matches!(input.dtype, DType::F16 | DType::BF16)
        && props.plane_size_min == 32 && props.plane_size_max == 32
        && props.max_shared_memory_size as usize >= backward_plan::BackwardTilePlan::SHARED_BYTES
        && input.client.features().matmul.cmma.contains(&cfg)
        && tile_plan.is_some();
    if strategy == GroupedStrategy::TensorCore && !cooperative {
        return Err(GroupedMatmulError("requested grouped backward 16x16x16 Tensor Core configuration is unavailable"));
    }
    let input=into_contiguous(input); let weights=into_contiguous(weights);
    let grad_output=into_contiguous(grad_output); let row_experts=into_contiguous(row_experts);
    let offsets=into_contiguous(offsets);
    let dinput=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([m,k]),input.dtype);
    let dweights=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([e,n,k]),DType::F32);
    if cooperative {
        let plan = tile_plan.ok_or(GroupedMatmulError("missing grouped backward launch plan"))?;
        let [gx,gy,gz] = plan.dinput_grid;
        if m != 0 {
            backward_tensorcore::dinput::launch::<R>(&input.client,
                RudaCount::Static(gx,gy,gz),RudaDim::new_1d(32),
                grad_output.clone().into_array_arg(),weights.into_array_arg(),offsets.clone().into_array_arg(),
                dinput.clone().into_array_arg(),n as u32,k as u32,input.dtype.into());
        }
        let [gx,gy,gz] = plan.dweight_grid;
        backward_tensorcore::dweight::launch::<R>(&input.client,
            RudaCount::Static(gx,gy,gz),RudaDim::new_1d(32),
            input.clone().into_array_arg(),grad_output.into_array_arg(),offsets.into_array_arg(),
            dweights.clone().into_array_arg(),n as u32,k as u32,input.dtype.into());
        return Ok(GroupedBackward { dinput, dweights });
    }
    if m!=0 {
        let size=m*k; let dim=RudaDim::new(input.client.properties(),size);
        kernel::grouped_nt_dinput::launch::<R>(&input.client,
            calculate_ruda_count_elemwise(&input.client,size,dim),dim,
            grad_output.clone().into_array_arg(),weights.clone().into_array_arg(),row_experts.into_array_arg(),
            dinput.clone().into_array_arg(),e as u32,n as u32,k as u32,input.dtype.into());
    }
    let weight_size=e*n*k;
    if weight_size!=0 {
        let dim=RudaDim::new(input.client.properties(),weight_size);
        kernel::grouped_nt_dweight::launch::<R>(&input.client,
            calculate_ruda_count_elemwise(&input.client,weight_size,dim),dim,
            input.clone().into_array_arg(),grad_output.into_array_arg(),offsets.into_array_arg(),dweights.clone().into_array_arg(),
            e as u32,n as u32,k as u32,input.dtype.into());
    }
    Ok(GroupedBackward{dinput,dweights})
}
