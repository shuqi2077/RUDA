use super::{GroupedMatmulError, grouped_matmul_nt, tensorcore};
use ruda_core::{device::Device, tensor::{DType, Shape}, ir::{ElemType,FloatKind,features::MmaConfig}};
use ruda_kernel::{dsl::{Runtime,prelude::{RudaCount,RudaDim}},tensor::{RudaTensor,allocation::empty_device_contiguous_dtype,contiguous::into_contiguous}};

/// Explicit choice. Auto only falls back for unsupported setup, never a failed
/// compilation, GPU launch or a numerical error. Scalar remains the old baseline.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum GroupedStrategy { Scalar, Auto, TensorCore }

/// Segmented grouped GEMM using offsets already produced by device dispatch.
///
/// # Safety
/// `offsets` must be a nondecreasing exclusive prefix [E+1], starting at 0 and
/// ending at M; all rows [offsets[e],offsets[e+1]) must belong to expert e.
/// The same immutable dispatch metadata must be used by the scalar fallback.
/// These are device-value invariants: metadata shape checks do not validate them.
#[allow(unsafe_code)]
pub unsafe fn grouped_matmul_nt_segmented<R: Runtime>(input: RudaTensor<R>, weights: RudaTensor<R>,
    row_experts: RudaTensor<R>, offsets: RudaTensor<R>, strategy: GroupedStrategy)
    -> Result<RudaTensor<R>,GroupedMatmulError>
{
    if strategy==GroupedStrategy::Scalar { return grouped_matmul_nt(input,weights,row_experts); }
    if input.meta.num_dims()!=2 || weights.meta.num_dims()!=3 {
        return Err(GroupedMatmulError("segmented GEMM requires [M,K] and [E,N,K]"));
    }
    let (m,k)=(input.meta.shape()[0],input.meta.shape()[1]);
    let (e,n)=(weights.meta.shape()[0],weights.meta.shape()[1]);
    if e==0 || n==0 || k==0 || weights.meta.shape()[2]!=k || weights.dtype!=input.dtype
        || offsets.meta.shape()[..]!=[e+1] || offsets.dtype!=DType::U32
        || row_experts.meta.shape()[..]!=[m] || row_experts.dtype!=DType::U32
        || !matches!(input.dtype,DType::F16|DType::BF16|DType::F32)
    { return Err(GroupedMatmulError("invalid segmented GEMM shape/dtype")); }
    for t in [&input,&weights,&row_experts,&offsets] {
        if t.qparams.is_some() || t.device.to_id()!=input.device.to_id()
            || !t.client.same_execution_queue(&input.client)
            || t.meta.shape().iter().try_fold(1usize,|a,&b|a.checked_mul(b)).is_none_or(|s|s>u32::MAX as usize)
        { return Err(GroupedMatmulError("segmented GEMM device/queue/size mismatch")); }
    }
    if m.checked_mul(n).is_none_or(|s|s>u32::MAX as usize) {
        return Err(GroupedMatmulError("segmented GEMM output exceeds U32 indexing"));
    }
    let cfg=MmaConfig { a_type:input.dtype.into(),b_type:input.dtype.into(),
        cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16 };
    let props=&input.client.properties().hardware;
    let supported=matches!(input.dtype,DType::F16|DType::BF16)
        && props.plane_size_min==32 && props.plane_size_max==32
        && props.max_shared_memory_size>=2048
        && input.client.features().matmul.cmma.contains(&cfg)
        && n.div_ceil(16)<=props.max_ruda_count.0 as usize && e<=props.max_ruda_count.1 as usize;
    if !supported {
        if strategy==GroupedStrategy::TensorCore { return Err(GroupedMatmulError("requested 16x16x16 Tensor Core configuration is unavailable")); }
        return grouped_matmul_nt(input,weights,row_experts);
    }
    let out=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([m,n]),input.dtype);
    if m!=0 {
        let input=into_contiguous(input); let weights=into_contiguous(weights); let offsets=into_contiguous(offsets);
        tensorcore::segmented::launch::<R>(&input.client,
            RudaCount::Static(n.div_ceil(16) as u32,e as u32,1),RudaDim::new_1d(32),
            input.clone().into_array_arg(),weights.into_array_arg(),offsets.into_array_arg(),out.clone().into_array_arg(),
            n as u32,k as u32,input.dtype.into());
    }
    Ok(out)
}
