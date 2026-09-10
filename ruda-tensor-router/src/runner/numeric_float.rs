use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_numeric_float(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &NumericOperationIr,
    ) {
        match op {
            NumericOperationIr::Add(desc) => {
                binary_float_ops!(handles, desc, B::float_add)
            }
            NumericOperationIr::AddScalar(desc) => {
                scalar_float_ops!(handles, desc, B::float_add_scalar)
            }
            NumericOperationIr::Sub(desc) => {
                binary_float_ops!(handles, desc, B::float_sub)
            }
            NumericOperationIr::SubScalar(desc) => {
                scalar_float_ops!(handles, desc, B::float_sub_scalar)
            }
            NumericOperationIr::Div(desc) => {
                binary_float_ops!(handles, desc, B::float_div)
            }
            NumericOperationIr::DivScalar(desc) => {
                scalar_float_ops!(handles, desc, B::float_div_scalar)
            }
            NumericOperationIr::Rem(desc) => {
                binary_float_ops!(handles, desc, B::float_remainder)
            }
            NumericOperationIr::RemScalar(desc) => {
                scalar_float_ops!(handles, desc, B::float_remainder_scalar)
            }
            NumericOperationIr::Mul(desc) => {
                binary_float_ops!(handles, desc, B::float_mul)
            }
            NumericOperationIr::MulScalar(desc) => {
                scalar_float_ops!(handles, desc, B::float_mul_scalar)
            }
            NumericOperationIr::Abs(desc) => {
                unary_float_ops!(handles, desc, B::float_abs)
            }
            NumericOperationIr::Full(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::float_full(
                    shape,
                    desc.value.into(),
                    &self.device,
                    desc.out.dtype.into(),
                );
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::MeanDim(desc) => {
                reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_mean_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::Mean(desc) => {
                unary_float_ops!(handles, desc, B::float_mean)
            }
            NumericOperationIr::Sum(desc) => {
                unary_float_ops!(handles, desc, B::float_sum)
            }
            NumericOperationIr::SumDim(desc) => {
                reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_sum_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::Prod(desc) => {
                unary_float_ops!(handles, desc, B::float_prod)
            }
            NumericOperationIr::ProdDim(desc) => {
                reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_prod_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::Greater(desc) => {
                binary_float_cmp_ops!(handles, desc, B::float_greater)
            }
            NumericOperationIr::GreaterElem(desc) => {
                scalar_float_cmp_ops!(handles, desc, B::float_greater_elem)
            }
            NumericOperationIr::GreaterEqual(desc) => {
                binary_float_cmp_ops!(handles, desc, B::float_greater_equal)
            }
            NumericOperationIr::GreaterEqualElem(desc) => {
                scalar_float_cmp_ops!(handles, desc, B::float_greater_equal_elem)
            }
            NumericOperationIr::Lower(desc) => {
                binary_float_cmp_ops!(handles, desc, B::float_lower)
            }
            NumericOperationIr::LowerElem(desc) => {
                scalar_float_cmp_ops!(handles, desc, B::float_lower_elem)
            }
            NumericOperationIr::LowerEqual(desc) => {
                binary_float_cmp_ops!(handles, desc, B::float_lower_equal)
            }
            NumericOperationIr::LowerEqualElem(desc) => {
                scalar_float_cmp_ops!(handles, desc, B::float_lower_equal_elem)
            }
            NumericOperationIr::ArgMax(desc) => {
                reduce_float2int_dim_ops!(handles, desc, |tensor, axis, _, dtype| {
                    B::float_argmax(tensor, axis, dtype)
                })
            }
            NumericOperationIr::ArgTopK(desc) => {
                reduce_float2int_dim_ops!(handles, desc, B::float_argtopk)
            }
            NumericOperationIr::ArgMin(desc) => {
                reduce_float2int_dim_ops!(handles, desc, |tensor, axis, _, dtype| {
                    B::float_argmin(tensor, axis, dtype)
                })
            }
            NumericOperationIr::Max(desc) => {
                unary_float_ops!(handles, desc, B::float_max)
            }
            NumericOperationIr::MaxDimWithIndices(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                let (output, output_idx) =
                    B::float_max_dim_with_indices(tensor, desc.dim, desc.out_indices.dtype.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
                handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
            }
            NumericOperationIr::MinDimWithIndices(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                let (output, output_idx) =
                    B::float_min_dim_with_indices(tensor, desc.dim, desc.out_indices.dtype.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
                handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
            }
            NumericOperationIr::Min(desc) => {
                unary_float_ops!(handles, desc, B::float_min)
            }
            NumericOperationIr::MaxDim(desc) => {
                reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_max_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::TopK(desc) => {
                reduce_float_dim_ops!(handles, desc, B::float_topk)
            }
            NumericOperationIr::MinDim(desc) => {
                reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_min_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::MaxAbs(desc) => {
                unary_float_ops!(handles, desc, B::float_max_abs)
            }
            NumericOperationIr::MaxAbsDim(desc) => {
                reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_max_abs_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::Clamp(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                let output = B::float_clamp(tensor, desc.min.into(), desc.max.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::IntRandom(_) => unreachable!(),
            NumericOperationIr::Powi(desc) => {
                let lhs = handles.get_float_tensor::<B>(&desc.lhs);
                let rhs = handles.get_int_tensor::<B>(&desc.rhs);
                let output = (B::float_powi)(lhs, rhs);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::PowiScalar(desc) => {
                scalar_float_ops!(handles, desc, B::float_powi_scalar)
            }
            NumericOperationIr::CumSum(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);
                let output = B::float_cumsum(tensor, desc.axis);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::CumProd(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);
                let output = B::float_cumprod(tensor, desc.axis);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::CumMin(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);
                let output = B::float_cummin(tensor, desc.axis);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::CumMax(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);
                let output = B::float_cummax(tensor, desc.axis);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
        }
    }
}
