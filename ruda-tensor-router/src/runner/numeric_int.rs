use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_numeric_int(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &NumericOperationIr,
    ) {
        match op {
            NumericOperationIr::Add(desc) => {
                binary_int_ops!(handles, desc, B::int_add)
            }
            NumericOperationIr::AddScalar(desc) => {
                scalar_int_ops!(handles, desc, B::int_add_scalar)
            }
            NumericOperationIr::Sub(desc) => {
                binary_int_ops!(handles, desc, B::int_sub)
            }
            NumericOperationIr::SubScalar(desc) => {
                scalar_int_ops!(handles, desc, B::int_sub_scalar)
            }
            NumericOperationIr::Div(desc) => {
                binary_int_ops!(handles, desc, B::int_div)
            }
            NumericOperationIr::DivScalar(desc) => {
                scalar_int_ops!(handles, desc, B::int_div_scalar)
            }
            NumericOperationIr::Rem(desc) => {
                binary_int_ops!(handles, desc, B::int_remainder)
            }
            NumericOperationIr::RemScalar(desc) => {
                scalar_int_ops!(handles, desc, B::int_remainder_scalar)
            }
            NumericOperationIr::Mul(desc) => {
                binary_int_ops!(handles, desc, B::int_mul)
            }
            NumericOperationIr::MulScalar(desc) => {
                scalar_int_ops!(handles, desc, B::int_mul_scalar)
            }
            NumericOperationIr::Abs(desc) => {
                unary_int_ops!(handles, desc, B::int_abs)
            }
            NumericOperationIr::Full(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::int_full(
                    shape,
                    desc.value.into(),
                    &self.device,
                    desc.out.dtype.into(),
                );
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::MeanDim(desc) => {
                reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_mean_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::Mean(desc) => {
                unary_int_ops!(handles, desc, B::int_mean)
            }
            NumericOperationIr::Sum(desc) => {
                unary_int_ops!(handles, desc, B::int_sum)
            }
            NumericOperationIr::SumDim(desc) => {
                reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_sum_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::Prod(desc) => {
                unary_int_ops!(handles, desc, B::int_prod)
            }
            NumericOperationIr::ProdDim(desc) => {
                reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_prod_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::Greater(desc) => {
                binary_int_cmp_ops!(handles, desc, B::int_greater)
            }
            NumericOperationIr::GreaterElem(desc) => {
                scalar_int_cmp_ops!(handles, desc, B::int_greater_elem)
            }
            NumericOperationIr::GreaterEqual(desc) => {
                binary_int_cmp_ops!(handles, desc, B::int_greater_equal)
            }
            NumericOperationIr::GreaterEqualElem(desc) => {
                scalar_int_cmp_ops!(handles, desc, B::int_greater_equal_elem)
            }
            NumericOperationIr::Lower(desc) => {
                binary_int_cmp_ops!(handles, desc, B::int_lower)
            }
            NumericOperationIr::LowerElem(desc) => {
                scalar_int_cmp_ops!(handles, desc, B::int_lower_elem)
            }
            NumericOperationIr::LowerEqual(desc) => {
                binary_int_cmp_ops!(handles, desc, B::int_lower_equal)
            }
            NumericOperationIr::LowerEqualElem(desc) => {
                scalar_int_cmp_ops!(handles, desc, B::int_lower_equal_elem)
            }
            NumericOperationIr::ArgMax(desc) => {
                reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_argmax(tensor, axis))
            }
            NumericOperationIr::ArgTopK(desc) => {
                reduce_int_dim_ops!(handles, desc, B::int_argtopk)
            }
            NumericOperationIr::ArgMin(desc) => {
                reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_argmin(tensor, axis))
            }
            NumericOperationIr::Max(desc) => {
                unary_int_ops!(handles, desc, B::int_max)
            }
            NumericOperationIr::MaxDimWithIndices(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                let (output, output_idx) = B::int_max_dim_with_indices(tensor, desc.dim);
                handles.register_int_tensor::<B>(&desc.out.id, output);
                handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
            }
            NumericOperationIr::MinDimWithIndices(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                let (output, output_idx) = B::int_min_dim_with_indices(tensor, desc.dim);
                handles.register_int_tensor::<B>(&desc.out.id, output);
                handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
            }
            NumericOperationIr::Min(desc) => {
                unary_int_ops!(handles, desc, B::int_min)
            }
            NumericOperationIr::MaxDim(desc) => {
                reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_max_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::TopK(desc) => {
                reduce_int_dim_ops!(handles, desc, B::int_topk)
            }
            NumericOperationIr::MinDim(desc) => {
                reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_min_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::MaxAbs(desc) => {
                unary_int_ops!(handles, desc, B::int_max_abs)
            }
            NumericOperationIr::MaxAbsDim(desc) => {
                reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_max_abs_dim(
                    tensor, axis
                ))
            }
            NumericOperationIr::Clamp(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                let output = B::int_clamp(tensor, desc.min.into(), desc.max.into());
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::IntRandom(desc) => {
                let shape = desc.out.shape.clone();

                let output = B::int_random(
                    shape,
                    desc.distribution,
                    &self.device,
                    desc.out.dtype.into(),
                );
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::Powi(desc) => {
                let lhs = handles.get_int_tensor::<B>(&desc.lhs);
                let rhs = handles.get_int_tensor::<B>(&desc.rhs);

                let output = B::int_powi(lhs, rhs);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::PowiScalar(desc) => {
                scalar_int_ops!(handles, desc, B::int_powi_scalar)
            }
            NumericOperationIr::CumSum(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);
                let output = B::int_cumsum(tensor, desc.axis);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::CumProd(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);
                let output = B::int_cumprod(tensor, desc.axis);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::CumMin(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);
                let output = B::int_cummin(tensor, desc.axis);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            NumericOperationIr::CumMax(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);
                let output = B::int_cummax(tensor, desc.axis);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
        }
    }
}
