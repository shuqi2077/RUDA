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
