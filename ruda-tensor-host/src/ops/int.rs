//! Int tensor operations for the Host backend.

use alloc::vec::Vec;
use ruda_tensor::{
    Distribution, ExecutionError, Scalar, TensorData,
    ops::IntTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use ruda_tensor::{IntDType, Shape, Slice};

use crate::{Host, HostTensor, ops::matmul};

impl IntTensorOps<Host> for Host {
    fn int_from_data(data: TensorData, _device: &Device<Host>) -> IntTensor<Host> {
        HostTensor::from_data(data)
    }

    async fn int_into_data(tensor: IntTensor<Host>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn int_device(_tensor: &IntTensor<Host>) -> Device<Host> {
        Default::default()
    }

    fn int_to_device(tensor: IntTensor<Host>, _device: &Device<Host>) -> IntTensor<Host> {
        tensor
    }

    fn int_cat(tensors: Vec<IntTensor<Host>>, dim: usize) -> IntTensor<Host> {
        crate::ops::cat::cat(tensors, dim)
    }

    fn int_reshape(tensor: IntTensor<Host>, shape: Shape) -> IntTensor<Host> {
        tensor.reshape(shape)
    }

    fn int_slice(tensor: IntTensor<Host>, slices: &[Slice]) -> IntTensor<Host> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn int_empty(shape: Shape, _device: &Device<Host>, dtype: IntDType) -> IntTensor<Host> {
        HostTensor::empty(shape, dtype.into())
    }

    fn int_mask_where(
        tensor: IntTensor<Host>,
        mask: BoolTensor<Host>,
        value: IntTensor<Host>,
    ) -> IntTensor<Host> {
        ruprim_host::mask::dispatch::int_mask_where(tensor, mask, value)
    }

    fn int_mask_fill(
        tensor: IntTensor<Host>,
        mask: BoolTensor<Host>,
        value: Scalar,
    ) -> IntTensor<Host> {
        ruprim_host::mask::dispatch::int_mask_fill(tensor, mask, value)
    }

    fn int_slice_assign(
        tensor: IntTensor<Host>,
        slices: &[Slice],
        value: IntTensor<Host>,
    ) -> IntTensor<Host> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    /// Gather ints along `dim` at the given indices.
    ///
    /// The `tensor` dispatches on its own int dtype (I8/I16/I32/I64 signed or
    /// U8/U16/U32/U64 unsigned). The `indices` tensor may be any of those
    /// widths too - it's normalised to `isize` by the shared `read_indices`
    /// helper in `ops::gather_scatter` before the kernel runs, so callers are
    /// not required to pre-convert to I64.
    fn int_gather(
        dim: usize,
        tensor: IntTensor<Host>,
        indices: IntTensor<Host>,
    ) -> IntTensor<Host> {
        ruprim_host::gather_scatter::dispatch_int::int_gather(dim, tensor, indices)
    }

    /// Scatter-add int values at the given indices along `dim`.
    ///
    /// `tensor` and `value` must share the same int dtype; `indices` may be
    /// any supported int width. See [`int_gather`](Self::int_gather) for the
    /// full index-width policy.
    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Host>,
        indices: IntTensor<Host>,
        value: IntTensor<Host>,
    ) -> IntTensor<Host> {
        ruprim_host::gather_scatter::dispatch_int::int_scatter_add(dim, tensor, indices, value)
    }

    fn int_scatter_nd(
        data: IntTensor<Host>,
        indices: IntTensor<Host>,
        values: IntTensor<Host>,
        reduction: ruda_tensor::tensor::IndexingUpdateOp,
    ) -> IntTensor<Host> {
        ruprim_host::gather_scatter::dispatch_int::int_scatter_nd(data, indices, values, reduction)
    }

    fn int_gather_nd(data: IntTensor<Host>, indices: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::gather_scatter::dispatch_int::int_gather_nd(data, indices)
    }

    /// Select ints along `dim` by a 1D index tensor.
    ///
    /// The `indices` tensor may be any supported int width. See
    /// [`int_gather`](Self::int_gather) for the full index-width policy.
    fn int_select(
        tensor: IntTensor<Host>,
        dim: usize,
        indices: IntTensor<Host>,
    ) -> IntTensor<Host> {
        ruprim_host::gather_scatter::dispatch_int::int_select(tensor, dim, indices)
    }

    /// Select-add int values at a 1D index tensor along `dim`.
    ///
    /// `tensor` and `value` must share the same int dtype; `indices` may be
    /// any supported int width. See [`int_gather`](Self::int_gather) for the
    /// full index-width policy.
    fn int_select_add(
        tensor: IntTensor<Host>,
        dim: usize,
        indices: IntTensor<Host>,
        value: IntTensor<Host>,
    ) -> IntTensor<Host> {
        ruprim_host::gather_scatter::dispatch_int::int_select_add(tensor, dim, indices, value)
    }

    fn int_equal(
        lhs: IntTensor<Host>,
        rhs: IntTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::int_equal(lhs, rhs, out_dtype)
    }

    fn int_equal_elem(
        lhs: IntTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::int_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_greater(
        lhs: IntTensor<Host>,
        rhs: IntTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::int_greater(lhs, rhs, out_dtype)
    }

    fn int_greater_elem(
        lhs: IntTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::int_greater_elem(lhs, rhs, out_dtype)
    }

    fn int_greater_equal(
        lhs: IntTensor<Host>,
        rhs: IntTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::int_greater_equal(lhs, rhs, out_dtype)
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::int_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_lower(
        lhs: IntTensor<Host>,
        rhs: IntTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::int_lower(lhs, rhs, out_dtype)
    }

    fn int_lower_elem(
        lhs: IntTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::int_lower_elem(lhs, rhs, out_dtype)
    }

    fn int_lower_equal(
        lhs: IntTensor<Host>,
        rhs: IntTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::int_lower_equal(lhs, rhs, out_dtype)
    }

    fn int_lower_equal_elem(
        lhs: IntTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::int_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_add(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_add(lhs, rhs)
    }

    fn int_add_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_add_scalar(lhs, rhs)
    }

    fn int_sub(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_sub_scalar(lhs, rhs)
    }

    fn int_mul(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_mul_scalar(lhs, rhs)
    }

    fn int_div(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_div(lhs, rhs)
    }

    fn int_div_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_div_scalar(lhs, rhs)
    }

    fn int_remainder(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_remainder(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_remainder_scalar(lhs, rhs)
    }

    // Precision limits: i64/u64 > 2^24 for f32/f16/bf16, > 2^53 for f64.
    fn int_into_float(
        tensor: IntTensor<Host>,
        out_dtype: ruda_tensor::FloatDType,
    ) -> FloatTensor<Host> {
        ruprim_host::cast::int_into_float(tensor, out_dtype)
    }

    fn int_swap_dims(tensor: IntTensor<Host>, dim1: usize, dim2: usize) -> IntTensor<Host> {
        tensor.transpose(dim1, dim2)
    }

    fn int_permute(tensor: IntTensor<Host>, axes: &[usize]) -> IntTensor<Host> {
        tensor.permute(axes)
    }

    fn int_flip(tensor: IntTensor<Host>, axes: &[usize]) -> IntTensor<Host> {
        crate::ops::flip::flip(tensor, axes)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        _device: &Device<Host>,
        dtype: IntDType,
    ) -> IntTensor<Host> {
        rurand_host::int_random(shape, distribution, dtype)
    }

    fn int_expand(tensor: IntTensor<Host>, shape: Shape) -> IntTensor<Host> {
        crate::ops::expand::expand(tensor, shape)
    }

    fn int_matmul(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        matmul::int_matmul(lhs, rhs)
    }

    fn int_sum(tensor: IntTensor<Host>) -> IntTensor<Host> {
        crate::ops::reduce::sum(tensor)
    }

    fn int_sum_dim(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        crate::ops::reduce::sum_dim(tensor, dim)
    }

    fn int_prod(tensor: IntTensor<Host>) -> IntTensor<Host> {
        crate::ops::reduce::prod(tensor)
    }

    fn int_prod_dim(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        crate::ops::reduce::prod_dim(tensor, dim)
    }

    fn int_mean_dim(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        crate::ops::reduce::mean_dim(tensor, dim)
    }

    fn int_cumsum(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        ruprim_host::cumulative::dispatch::int_cumsum(tensor, dim)
    }

    fn int_cumprod(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        ruprim_host::cumulative::dispatch::int_cumprod(tensor, dim)
    }

    fn int_cummin(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        ruprim_host::cumulative::dispatch::int_cummin(tensor, dim)
    }

    fn int_cummax(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        ruprim_host::cumulative::dispatch::int_cummax(tensor, dim)
    }

    fn int_argmax(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        crate::ops::reduce::argmax(tensor, dim)
    }

    fn int_argtopk(tensor: IntTensor<Host>, dim: usize, k: usize) -> IntTensor<Host> {
        ruprim_host::sort::dispatch::int_argtopk(tensor, dim, k)
    }

    fn int_argmin(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        crate::ops::reduce::argmin(tensor, dim)
    }

    fn int_abs(tensor: IntTensor<Host>) -> IntTensor<Host> {
        crate::ops::unary::int_abs(tensor)
    }

    fn bitwise_and(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_and(lhs, rhs)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_and_scalar(lhs, rhs)
    }

    fn bitwise_or(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_or(lhs, rhs)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_or_scalar(lhs, rhs)
    }

    fn bitwise_xor(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_xor(lhs, rhs)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_xor_scalar(lhs, rhs)
    }

    fn bitwise_not(tensor: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_not(tensor)
    }

    // Shift amounts masked to type width via wrapping_shl/wrapping_shr.
    fn bitwise_left_shift(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_left_shift(lhs, rhs)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_left_shift_scalar(lhs, rhs)
    }

    fn bitwise_right_shift(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_right_shift(lhs, rhs)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Host>, rhs: Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::bitwise_right_shift_scalar(lhs, rhs)
    }

    fn int_cast(tensor: IntTensor<Host>, dtype: IntDType) -> IntTensor<Host> {
        ruprim_host::cast::int_cast(tensor, dtype)
    }

    fn int_unfold(
        tensor: IntTensor<Host>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Host> {
        crate::ops::unfold::unfold_int(tensor, dim, size, step)
    }

    fn int_neg(tensor: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::unary::dispatch_int::int_neg(tensor)
    }

    fn int_clamp(tensor: IntTensor<Host>, min: Scalar, max: Scalar) -> IntTensor<Host> {
        ruprim_host::unary::dispatch_int::int_clamp(tensor, min, max)
    }

    fn int_clamp_min(tensor: IntTensor<Host>, min: Scalar) -> IntTensor<Host> {
        ruprim_host::unary::dispatch_int::int_clamp_min(tensor, min)
    }

    fn int_clamp_max(tensor: IntTensor<Host>, max: Scalar) -> IntTensor<Host> {
        ruprim_host::unary::dispatch_int::int_clamp_max(tensor, max)
    }

    fn int_sign(tensor: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::unary::dispatch_int::int_sign(tensor)
    }

    fn int_mean(tensor: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::reduce::dispatch::int_mean(tensor)
    }

    fn int_max(tensor: IntTensor<Host>) -> IntTensor<Host> {
        crate::ops::reduce::max(tensor)
    }

    fn int_max_dim(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        crate::ops::reduce::max_dim(tensor, dim)
    }

    fn int_min(tensor: IntTensor<Host>) -> IntTensor<Host> {
        crate::ops::reduce::min(tensor)
    }

    fn int_min_dim(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        crate::ops::reduce::min_dim(tensor, dim)
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<Host>,
        dim: usize,
    ) -> (IntTensor<Host>, IntTensor<Host>) {
        crate::ops::reduce::max_dim_with_indices(tensor, dim)
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<Host>,
        dim: usize,
    ) -> (IntTensor<Host>, IntTensor<Host>) {
        crate::ops::reduce::min_dim_with_indices(tensor, dim)
    }

    fn int_any(tensor: IntTensor<Host>, out_dtype: ruda_tensor::BoolDType) -> BoolTensor<Host> {
        crate::ops::comparison::any_int(tensor, out_dtype)
    }

    fn int_any_dim(
        tensor: IntTensor<Host>,
        dim: usize,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::any_int_dim(tensor, dim, out_dtype)
    }

    fn int_all(tensor: IntTensor<Host>, out_dtype: ruda_tensor::BoolDType) -> BoolTensor<Host> {
        crate::ops::comparison::all_int(tensor, out_dtype)
    }

    fn int_all_dim(
        tensor: IntTensor<Host>,
        dim: usize,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::all_int_dim(tensor, dim, out_dtype)
    }

    fn int_powi(lhs: IntTensor<Host>, rhs: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_powi(lhs, rhs)
    }

    fn int_zeros(shape: Shape, _device: &Device<Host>, dtype: IntDType) -> IntTensor<Host> {
        HostTensor::zeros(shape, dtype.into())
    }

    fn int_ones(shape: Shape, _device: &Device<Host>, dtype: IntDType) -> IntTensor<Host> {
        ruprim_host::fill::int_ones(shape, dtype)
    }

    fn int_full(
        shape: Shape,
        fill_value: ruda_tensor::Scalar,
        _device: &Device<Host>,
        dtype: IntDType,
    ) -> IntTensor<Host> {
        ruprim_host::fill::int_full(shape, fill_value, dtype)
    }

    fn int_transpose(tensor: IntTensor<Host>) -> IntTensor<Host> {
        let ndims = tensor.layout().num_dims();
        if ndims < 2 {
            return tensor;
        }
        tensor.transpose(ndims - 2, ndims - 1)
    }

    fn int_repeat_dim(tensor: IntTensor<Host>, dim: usize, times: usize) -> IntTensor<Host> {
        crate::ops::repeat_dim::repeat_dim(tensor, dim, times)
    }

    fn int_not_equal(
        lhs: IntTensor<Host>,
        rhs: IntTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::int_not_equal(lhs, rhs, out_dtype)
    }

    fn int_not_equal_elem(
        lhs: IntTensor<Host>,
        rhs: ruda_tensor::Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::int_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_sort(tensor: IntTensor<Host>, dim: usize, descending: bool) -> IntTensor<Host> {
        crate::ops::sort::sort(tensor, dim, descending)
    }

    fn int_sort_with_indices(
        tensor: IntTensor<Host>,
        dim: usize,
        descending: bool,
    ) -> (IntTensor<Host>, IntTensor<Host>) {
        crate::ops::sort::sort_with_indices(tensor, dim, descending)
    }

    fn int_argsort(tensor: IntTensor<Host>, dim: usize, descending: bool) -> IntTensor<Host> {
        crate::ops::sort::argsort(tensor, dim, descending)
    }

    fn int_powi_scalar(lhs: IntTensor<Host>, rhs: ruda_tensor::Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_powi_scalar(lhs, rhs)
    }

    fn int_powi_scalar_impl(lhs: IntTensor<Host>, rhs: ruda_tensor::Scalar) -> IntTensor<Host> {
        ruprim_host::binary::dispatch_int::int_powi_scalar_impl(lhs, rhs)
    }

    fn int_max_abs(tensor: IntTensor<Host>) -> IntTensor<Host> {
        ruprim_host::reduce::dispatch::int_max_abs(tensor)
    }

    fn int_max_abs_dim(tensor: IntTensor<Host>, dim: usize) -> IntTensor<Host> {
        ruprim_host::reduce::dispatch::int_max_abs_dim(tensor, dim)
    }

    fn int_arange(
        range: core::ops::Range<i64>,
        _device: &Device<Host>,
        dtype: IntDType,
    ) -> IntTensor<Host> {
        Self::int_arange_step(range, 1, &Default::default(), dtype)
    }

    fn int_arange_step(
        range: core::ops::Range<i64>,
        step: usize,
        _device: &Device<Host>,
        dtype: IntDType,
    ) -> IntTensor<Host> {
        ruprim_host::fill::int_arange_step(range, step, dtype)
    }
}

// Tests kept here exercise flex-specific behavior: dtype storage
// selection for every int width (I16/I32/U8/U16/U32/I64/U64), and edge
// cases of the dtype-specific kernels (u64 wrap, i64::MIN abs/neg, bit
// shift at width). Plain int arithmetic, scalar ops, bool->int cast
// smokes, and negative-stride (flipped/transposed) variants have been
// migrated to ruda-backend-tests so they run against every backend.
// When adding new tests, keep them here only if they probe flex dtype
// storage; otherwise add them to
// crates/ruda-backend-tests/tests/tensor/int/ops/.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use ruda_tensor::TensorData;
    use ruda_tensor::ops::IntTensorOps;

    use crate::Host;
    use crate::HostTensor;

    #[test]
    fn test_u64_div_large_values() {
        let a = HostTensor::from_data(TensorData::new(vec![u64::MAX], [1]));
        let b = HostTensor::from_data(TensorData::new(vec![2u64], [1]));
        let result = Host::int_div(a, b);
        let values: Vec<u64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], u64::MAX / 2);
    }

    #[test]
    fn test_u64_remainder_large_values() {
        let a = HostTensor::from_data(TensorData::new(vec![u64::MAX], [1]));
        let b = HostTensor::from_data(TensorData::new(vec![2u64], [1]));
        let result = Host::int_remainder(a, b);
        let values: Vec<u64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], u64::MAX % 2);
    }

    #[test]
    fn test_int_abs_min_value() {
        // i64::MIN.abs() panics in debug; wrapping_abs returns MIN (matches PyTorch)
        let a = HostTensor::from_data(TensorData::new(vec![i64::MIN], [1]));
        let result = Host::int_abs(a);
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], i64::MIN.wrapping_abs());
    }

    #[test]
    fn test_int_neg_min_value() {
        // i64::MIN negation panics in debug; wrapping_neg returns MIN (matches PyTorch)
        let a = HostTensor::from_data(TensorData::new(vec![i64::MIN], [1]));
        let result = Host::int_neg(a);
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], i64::MIN.wrapping_neg());
    }

    #[test]
    fn test_int_shift_large_amount() {
        // Shift by >= bit width panics without wrapping; should not crash
        let a = HostTensor::from_data(TensorData::new(vec![1i64], [1]));
        let b = HostTensor::from_data(TensorData::new(vec![64i64], [1]));
        let _left = Host::bitwise_left_shift(a.clone(), b.clone());
        let _right = Host::bitwise_right_shift(a, b);
    }

    #[test]
    fn test_int_into_float_f64() {
        use ruda_tensor::ops::IntTensorOps;
        use ruda_tensor::FloatDType;

        let t = HostTensor::from_data(TensorData::new(vec![1i64, 2, -3], [3]));
        let result = Host::int_into_float(t, FloatDType::F64);
        assert_eq!(result.dtype(), ruda_tensor::DType::F64);
        let data: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0f64, 2.0, -3.0]);
    }

    #[test]
    fn test_u64_add_scalar_large() {
        let t = HostTensor::from_data(TensorData::new(vec![1u64, 2, 3], [3]));
        let big: u64 = (i64::MAX as u64) + 100;
        let result = Host::int_add_scalar(t, ruda_tensor::Scalar::from(big));
        let data: Vec<u64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![big + 1, big + 2, big + 3]);
    }

    #[test]
    fn test_u64_greater_elem_large() {
        let big: u64 = (i64::MAX as u64) + 100;
        let t = HostTensor::from_data(TensorData::new(vec![big, big + 1, big - 1], [3]));
        let result = Host::int_greater_elem(
            t,
            ruda_tensor::Scalar::from(big),
            ruda_tensor::BoolStore::Native,
        );
        let data: Vec<bool> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![false, true, false]);
    }

    #[test]
    fn test_int_mask_fill_i32() {
        let t = HostTensor::from_data(TensorData::new(vec![1i32, 2, 3, 4], [4]));
        let mask = HostTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));
        let result = Host::int_mask_fill(t, mask, ruda_tensor::Scalar::from(0i64));
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0, 2, 0, 4]);
    }

    #[test]
    fn test_int_mask_fill_i16() {
        let t = HostTensor::from_data(TensorData::new(vec![10i16, 20, 30, 40], [4]));
        let mask = HostTensor::from_data(TensorData::new(vec![false, true, false, true], [4]));
        let result = Host::int_mask_fill(t, mask, ruda_tensor::Scalar::from(-1i64));
        let data: Vec<i16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10, -1, 30, -1]);
    }

    #[test]
    fn test_int_mask_fill_u8() {
        let t = HostTensor::from_data(TensorData::new(vec![1u8, 2, 3, 4], [4]));
        let mask = HostTensor::from_data(TensorData::new(vec![true, true, false, false], [4]));
        let result = Host::int_mask_fill(t, mask, ruda_tensor::Scalar::from(255i64));
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![255, 255, 3, 4]);
    }

    #[test]
    fn test_int_mask_fill_u32() {
        let t = HostTensor::from_data(TensorData::new(vec![100u32, 200, 300], [3]));
        let mask = HostTensor::from_data(TensorData::new(vec![true, false, true], [3]));
        let result = Host::int_mask_fill(t, mask, ruda_tensor::Scalar::from(0i64));
        let data: Vec<u32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0, 200, 0]);
    }

    #[test]
    fn test_int_mask_where_i32() {
        let t = HostTensor::from_data(TensorData::new(vec![1i32, 2, 3, 4], [4]));
        let mask = HostTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));
        let v = HostTensor::from_data(TensorData::new(vec![10i32, 20, 30, 40], [4]));
        let result = Host::int_mask_where(t, mask, v);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10, 2, 30, 4]);
    }

    #[test]
    fn test_int_mask_where_u8() {
        let t = HostTensor::from_data(TensorData::new(vec![1u8, 2, 3, 4], [4]));
        let mask = HostTensor::from_data(TensorData::new(vec![false, true, false, true], [4]));
        let v = HostTensor::from_data(TensorData::new(vec![10u8, 20, 30, 40], [4]));
        let result = Host::int_mask_where(t, mask, v);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, 20, 3, 40]);
    }

    #[test]
    fn test_int_gather_i32() {
        let t = HostTensor::from_data(TensorData::new(vec![10i32, 20, 30, 40, 50, 60], [2, 3]));
        let indices = HostTensor::from_data(TensorData::new(vec![2i64, 0, 1, 2], [2, 2]));
        let result = Host::int_gather(1, t, indices);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![30, 10, 50, 60]);
    }

    #[test]
    fn test_int_select_u16() {
        let t = HostTensor::from_data(TensorData::new(vec![10u16, 20, 30, 40, 50, 60], [2, 3]));
        let indices = HostTensor::from_data(TensorData::new(vec![0i64, 1], [2]));
        let result = Host::int_select(t, 1, indices);
        let data: Vec<u16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10, 20, 40, 50]);
    }

    #[test]
    fn test_int_cumsum_i32() {
        let t = HostTensor::from_data(TensorData::new(vec![1i32, 2, 3, 4], [4]));
        let result = Host::int_cumsum(t, 0);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_int_cumprod_u8() {
        let t = HostTensor::from_data(TensorData::new(vec![1u8, 2, 3, 4], [4]));
        let result = Host::int_cumprod(t, 0);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, 2, 6, 24]);
    }

    #[test]
    fn test_int_cummin_i32() {
        let t = HostTensor::from_data(TensorData::new(vec![3i32, 1, 4, 1, 5], [5]));
        let result = Host::int_cummin(t, 0);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3, 1, 1, 1, 1]);
    }

    #[test]
    fn test_int_cummax_u16() {
        let t = HostTensor::from_data(TensorData::new(vec![3u16, 1, 4, 1, 5], [5]));
        let result = Host::int_cummax(t, 0);
        let data: Vec<u16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3, 3, 4, 4, 5]);
    }

    #[test]
    fn test_int_scatter_add_i32() {
        let t = HostTensor::from_data(TensorData::new(vec![0i32, 0, 0], [1, 3]));
        let indices = HostTensor::from_data(TensorData::new(vec![0i64, 2, 1], [1, 3]));
        let values = HostTensor::from_data(TensorData::new(vec![10i32, 20, 30], [1, 3]));
        let result = Host::int_scatter_add(1, t, indices, values);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10, 30, 20]);
    }

    #[test]
    fn test_int_select_add_u8() {
        let t = HostTensor::from_data(TensorData::new(vec![1u8, 2, 3], [3]));
        let indices = HostTensor::from_data(TensorData::new(vec![0i64, 2], [2]));
        let values = HostTensor::from_data(TensorData::new(vec![10u8, 20], [2]));
        let result = Host::int_select_add(t, 0, indices, values);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![11, 2, 23]);
    }

    #[test]
    fn test_int_random_i32() {
        use ruda_tensor::{DType, Distribution, ops::IntTensorOps};
        use ruda_tensor::{IntDType, Shape};

        let shape = Shape::from(vec![100]);
        let dist = Distribution::Uniform(0.0, 10.0);
        let device = crate::HostDevice;
        let t = Host::int_random(shape, dist, &device, IntDType::I32);
        assert_eq!(t.dtype(), DType::I32);
        let data: Vec<i32> = t.into_data().to_vec().unwrap();
        assert!(data.iter().all(|&v| (0..=10).contains(&v)));
    }

    #[test]
    fn test_int_random_u8() {
        use ruda_tensor::{DType, Distribution, ops::IntTensorOps};
        use ruda_tensor::{IntDType, Shape};

        let shape = Shape::from(vec![50]);
        let dist = Distribution::Uniform(0.0, 100.0);
        let device = crate::HostDevice;
        let t = Host::int_random(shape, dist, &device, IntDType::U8);
        assert_eq!(t.dtype(), DType::U8);
    }

    #[test]
    fn test_int_mean_i32() {
        use ruda_tensor::{DType, ops::IntTensorOps};

        let t = HostTensor::from_data(TensorData::new(vec![10i32, 20, 30], [3]));
        let result = Host::int_mean(t);
        assert_eq!(result.dtype(), DType::I32);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![20]); // (10 + 20 + 30) / 3 = 20
    }
}
