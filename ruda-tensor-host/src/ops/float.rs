//! Float tensor operations for the Host backend.

use alloc::vec::Vec;
use ruda_tensor::{
    Distribution, ExecutionError, FloatDType, Scalar, TensorData,
    ops::{FloatTensorOps, GridSampleOptions},
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use ruda_tensor::{Shape, Slice};


use crate::ops::matmul;
use crate::ops::unary;
use crate::{Host, HostTensor};

impl FloatTensorOps<Host> for Host {
    fn float_from_data(data: TensorData, _device: &Device<Host>) -> FloatTensor<Host> {
        HostTensor::from_data(data)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &Device<Host>,
        dtype: FloatDType,
    ) -> FloatTensor<Host> {
        rurand_host::float_random(shape, distribution, dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Host>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn float_device(_tensor: &FloatTensor<Host>) -> Device<Host> {
        // CPU backend: all tensors are on the default device
        Default::default()
    }

    fn float_to_device(tensor: FloatTensor<Host>, _device: &Device<Host>) -> FloatTensor<Host> {
        // CPU backend: no-op, tensors are always on CPU
        tensor
    }

    fn float_detach(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        tensor
    }

    fn float_into_int(tensor: FloatTensor<Host>, out_dtype: ruda_tensor::IntDType) -> IntTensor<Host> {
        ruprim_host::cast::float_into_int(tensor, out_dtype)
    }

    fn float_empty(shape: Shape, _device: &Device<Host>, dtype: FloatDType) -> FloatTensor<Host> {
        HostTensor::empty(shape, dtype.into())
    }

    fn float_add(lhs: FloatTensor<Host>, rhs: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_add(lhs, rhs)
    }

    fn float_add_scalar(lhs: FloatTensor<Host>, rhs: Scalar) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_add_scalar(lhs, rhs)
    }

    fn float_sub(lhs: FloatTensor<Host>, rhs: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_sub(lhs, rhs)
    }

    fn float_sub_scalar(lhs: FloatTensor<Host>, rhs: Scalar) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_sub_scalar(lhs, rhs)
    }

    fn float_mul(lhs: FloatTensor<Host>, rhs: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_mul(lhs, rhs)
    }

    fn float_mul_scalar(lhs: FloatTensor<Host>, rhs: Scalar) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_mul_scalar(lhs, rhs)
    }

    fn float_div(lhs: FloatTensor<Host>, rhs: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_div(lhs, rhs)
    }

    fn float_div_scalar(lhs: FloatTensor<Host>, rhs: Scalar) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_div_scalar(lhs, rhs)
    }

    fn float_remainder(lhs: FloatTensor<Host>, rhs: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_remainder(lhs, rhs)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Host>, rhs: Scalar) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_remainder_scalar(lhs, rhs)
    }

    fn float_matmul(lhs: FloatTensor<Host>, rhs: FloatTensor<Host>) -> FloatTensor<Host> {
        matmul::matmul(lhs, rhs)
    }

    fn float_cross(
        lhs: FloatTensor<Host>,
        rhs: FloatTensor<Host>,
        dim: usize,
    ) -> FloatTensor<Host> {
        rublas_host::cross(lhs, rhs, dim)
    }

    fn float_recip(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::recip(tensor)
    }

    fn float_swap_dims(tensor: FloatTensor<Host>, dim1: usize, dim2: usize) -> FloatTensor<Host> {
        tensor.transpose(dim1, dim2)
    }

    fn float_permute(tensor: FloatTensor<Host>, axes: &[usize]) -> FloatTensor<Host> {
        tensor.permute(axes)
    }

    fn float_flip(tensor: FloatTensor<Host>, axes: &[usize]) -> FloatTensor<Host> {
        crate::ops::flip::flip(tensor, axes)
    }

    fn float_cat(tensors: Vec<FloatTensor<Host>>, dim: usize) -> FloatTensor<Host> {
        crate::ops::cat::cat(tensors, dim)
    }

    fn float_reshape(tensor: FloatTensor<Host>, shape: Shape) -> FloatTensor<Host> {
        tensor.reshape(shape)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Host>,
        indices: IntTensor<Host>,
    ) -> FloatTensor<Host> {
        ruprim_host::gather_scatter::dispatch_float::float_gather(dim, tensor, indices)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Host>,
        indices: IntTensor<Host>,
        value: FloatTensor<Host>,
    ) -> FloatTensor<Host> {
        ruprim_host::gather_scatter::dispatch_float::float_scatter_add(dim, tensor, indices, value)
    }

    fn float_scatter_nd(
        data: FloatTensor<Host>,
        indices: IntTensor<Host>,
        values: FloatTensor<Host>,
        reduction: ruda_tensor::tensor::IndexingUpdateOp,
    ) -> FloatTensor<Host> {
        ruprim_host::gather_scatter::dispatch_float::float_scatter_nd(data, indices, values, reduction)
    }

    fn float_gather_nd(data: FloatTensor<Host>, indices: IntTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::gather_scatter::dispatch_float::float_gather_nd(data, indices)
    }

    fn float_select(
        tensor: FloatTensor<Host>,
        dim: usize,
        indices: IntTensor<Host>,
    ) -> FloatTensor<Host> {
        ruprim_host::gather_scatter::dispatch_float::float_select(tensor, dim, indices)
    }

    fn float_select_add(
        tensor: FloatTensor<Host>,
        dim: usize,
        indices: IntTensor<Host>,
        value: FloatTensor<Host>,
    ) -> FloatTensor<Host> {
        ruprim_host::gather_scatter::dispatch_float::float_select_add(tensor, dim, indices, value)
    }

    fn float_slice(tensor: FloatTensor<Host>, slices: &[Slice]) -> FloatTensor<Host> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Host>,
        slices: &[Slice],
        value: FloatTensor<Host>,
    ) -> FloatTensor<Host> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<Host>,
        mask: BoolTensor<Host>,
        value: FloatTensor<Host>,
    ) -> FloatTensor<Host> {
        ruprim_host::mask::dispatch::float_mask_where(tensor, mask, value)
    }

    fn float_mask_fill(
        tensor: FloatTensor<Host>,
        mask: BoolTensor<Host>,
        value: Scalar,
    ) -> FloatTensor<Host> {
        ruprim_host::mask::dispatch::float_mask_fill(tensor, mask, value)
    }

    fn float_equal(
        lhs: FloatTensor<Host>,
        rhs: FloatTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::equal(lhs, rhs, out_dtype)
    }

    fn float_equal_elem(
        lhs: FloatTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::float_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_greater(
        lhs: FloatTensor<Host>,
        rhs: FloatTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::greater(lhs, rhs, out_dtype)
    }

    fn float_greater_elem(
        lhs: FloatTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::float_greater_elem(lhs, rhs, out_dtype)
    }

    fn float_greater_equal(
        lhs: FloatTensor<Host>,
        rhs: FloatTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::greater_equal(lhs, rhs, out_dtype)
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::float_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_lower(
        lhs: FloatTensor<Host>,
        rhs: FloatTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::lower(lhs, rhs, out_dtype)
    }

    fn float_lower_elem(
        lhs: FloatTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::float_lower_elem(lhs, rhs, out_dtype)
    }

    fn float_lower_equal(
        lhs: FloatTensor<Host>,
        rhs: FloatTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::lower_equal(lhs, rhs, out_dtype)
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::float_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_not_equal(
        lhs: FloatTensor<Host>,
        rhs: FloatTensor<Host>,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::not_equal(lhs, rhs, out_dtype)
    }

    fn float_not_equal_elem(
        lhs: FloatTensor<Host>,
        rhs: Scalar,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::comparison::dispatch::float_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_neg(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::unary::dispatch_float::float_neg(tensor)
    }

    fn float_clamp(tensor: FloatTensor<Host>, min: Scalar, max: Scalar) -> FloatTensor<Host> {
        ruprim_host::unary::dispatch_float::float_clamp(tensor, min, max)
    }

    fn float_clamp_min(tensor: FloatTensor<Host>, min: Scalar) -> FloatTensor<Host> {
        ruprim_host::unary::dispatch_float::float_clamp_min(tensor, min)
    }

    fn float_clamp_max(tensor: FloatTensor<Host>, max: Scalar) -> FloatTensor<Host> {
        ruprim_host::unary::dispatch_float::float_clamp_max(tensor, max)
    }

    fn float_sign(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::unary::dispatch_float::float_sign(tensor)
    }

    fn float_mean(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        crate::ops::reduce::mean(tensor)
    }

    fn float_max(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        crate::ops::reduce::max(tensor)
    }

    fn float_max_dim(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        crate::ops::reduce::max_dim(tensor, dim)
    }

    fn float_min(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        crate::ops::reduce::min(tensor)
    }

    fn float_min_dim(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        crate::ops::reduce::min_dim(tensor, dim)
    }

    fn float_max_dim_with_indices(
        tensor: FloatTensor<Host>,
        dim: usize,
        indices_dtype: ruda_tensor::IntDType,
    ) -> (FloatTensor<Host>, IntTensor<Host>) {
        ruprim_host::reduce::dispatch::float_max_dim_with_indices(tensor, dim, indices_dtype)
    }

    fn float_min_dim_with_indices(
        tensor: FloatTensor<Host>,
        dim: usize,
        indices_dtype: ruda_tensor::IntDType,
    ) -> (FloatTensor<Host>, IntTensor<Host>) {
        ruprim_host::reduce::dispatch::float_min_dim_with_indices(tensor, dim, indices_dtype)
    }

    fn float_any(tensor: FloatTensor<Host>, out_dtype: ruda_tensor::BoolDType) -> BoolTensor<Host> {
        crate::ops::comparison::any_float(tensor, out_dtype)
    }

    fn float_any_dim(
        tensor: FloatTensor<Host>,
        dim: usize,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::any_float_dim(tensor, dim, out_dtype)
    }

    fn float_all(tensor: FloatTensor<Host>, out_dtype: ruda_tensor::BoolDType) -> BoolTensor<Host> {
        crate::ops::comparison::all_float(tensor, out_dtype)
    }

    fn float_all_dim(
        tensor: FloatTensor<Host>,
        dim: usize,
        out_dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        crate::ops::comparison::all_float_dim(tensor, dim, out_dtype)
    }

    fn float_sum(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        crate::ops::reduce::sum(tensor)
    }

    fn float_sum_dim(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        crate::ops::reduce::sum_dim(tensor, dim)
    }

    fn float_mean_dim(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        crate::ops::reduce::mean_dim(tensor, dim)
    }

    fn float_prod(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        crate::ops::reduce::prod(tensor)
    }

    fn float_prod_dim(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        crate::ops::reduce::prod_dim(tensor, dim)
    }

    fn float_cumsum(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        ruprim_host::cumulative::dispatch::float_cumsum(tensor, dim)
    }

    fn float_cumprod(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        ruprim_host::cumulative::dispatch::float_cumprod(tensor, dim)
    }

    fn float_cummin(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        ruprim_host::cumulative::dispatch::float_cummin(tensor, dim)
    }

    fn float_cummax(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        ruprim_host::cumulative::dispatch::float_cummax(tensor, dim)
    }

    fn float_cast(tensor: FloatTensor<Host>, dtype: FloatDType) -> FloatTensor<Host> {
        ruprim_host::cast::float_cast(tensor, dtype)
    }

    fn float_exp(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::exp(tensor)
    }

    fn float_log(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::log(tensor)
    }

    fn float_log1p(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::log1p(tensor)
    }

    fn float_powf(lhs: FloatTensor<Host>, rhs: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_powf(lhs, rhs)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Host>, value: Scalar) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_powf_scalar_impl(tensor, value)
    }

    fn float_sqrt(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::sqrt(tensor)
    }

    fn float_abs(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::abs(tensor)
    }

    fn float_cos(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::cos(tensor)
    }

    fn float_sin(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::sin(tensor)
    }

    fn float_tan(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::tan(tensor)
    }

    fn float_cosh(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::cosh(tensor)
    }

    fn float_sinh(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::sinh(tensor)
    }

    fn float_tanh(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::tanh(tensor)
    }

    fn float_acos(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::acos(tensor)
    }

    fn float_acosh(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::acosh(tensor)
    }

    fn float_asin(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::asin(tensor)
    }

    fn float_asinh(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::asinh(tensor)
    }

    fn float_atan(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::atan(tensor)
    }

    fn float_atanh(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::atanh(tensor)
    }

    fn float_atan2(lhs: FloatTensor<Host>, rhs: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_atan2(lhs, rhs)
    }

    fn float_round(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::round(tensor)
    }

    fn float_floor(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::floor(tensor)
    }

    fn float_ceil(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::ceil(tensor)
    }

    fn float_trunc(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::trunc(tensor)
    }

    fn float_erf(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        unary::erf(tensor)
    }

    fn float_argmax(
        tensor: FloatTensor<Host>,
        dim: usize,
        out_dtype: ruda_tensor::IntDType,
    ) -> IntTensor<Host> {
        ruprim_host::reduce::dispatch::float_argmax(tensor, dim, out_dtype)
    }

    fn float_argtopk(
        tensor: FloatTensor<Host>,
        dim: usize,
        k: usize,
        out_dtype: ruda_tensor::IntDType,
    ) -> IntTensor<Host> {
        ruprim_host::sort::dispatch::float_argtopk(tensor, dim, k, out_dtype)
    }

    fn float_argmin(
        tensor: FloatTensor<Host>,
        dim: usize,
        out_dtype: ruda_tensor::IntDType,
    ) -> IntTensor<Host> {
        ruprim_host::reduce::dispatch::float_argmin(tensor, dim, out_dtype)
    }

    fn float_expand(tensor: FloatTensor<Host>, shape: Shape) -> FloatTensor<Host> {
        crate::ops::expand::expand(tensor, shape)
    }

    fn float_unfold(
        tensor: FloatTensor<Host>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Host> {
        // unfold is now type-agnostic (zero-copy strided view)
        crate::ops::unfold::unfold(tensor, dim, size, step)
    }

    fn float_grid_sample_2d(
        tensor: FloatTensor<Host>,
        grid: FloatTensor<Host>,
        options: GridSampleOptions,
    ) -> FloatTensor<Host> {
        crate::ops::grid_sample::grid_sample_2d(tensor, grid, options)
    }

    fn float_zeros(shape: Shape, _device: &Device<Host>, dtype: FloatDType) -> FloatTensor<Host> {
        HostTensor::zeros(shape, dtype.into())
    }

    fn float_ones(shape: Shape, _device: &Device<Host>, dtype: FloatDType) -> FloatTensor<Host> {
        ruprim_host::fill::float_ones(shape, dtype)
    }

    fn float_full(
        shape: Shape,
        fill_value: Scalar,
        _device: &Device<Host>,
        dtype: FloatDType,
    ) -> FloatTensor<Host> {
        ruprim_host::fill::float_full(shape, fill_value, dtype)
    }

    fn float_transpose(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        let ndims = tensor.layout().num_dims();
        if ndims < 2 {
            return tensor;
        }
        tensor.transpose(ndims - 2, ndims - 1)
    }

    fn float_repeat_dim(tensor: FloatTensor<Host>, dim: usize, times: usize) -> FloatTensor<Host> {
        crate::ops::repeat_dim::repeat_dim(tensor, dim, times)
    }

    fn float_sort(tensor: FloatTensor<Host>, dim: usize, descending: bool) -> FloatTensor<Host> {
        crate::ops::sort::sort(tensor, dim, descending)
    }

    fn float_sort_with_indices(
        tensor: FloatTensor<Host>,
        dim: usize,
        descending: bool,
        indices_dtype: ruda_tensor::IntDType,
    ) -> (FloatTensor<Host>, IntTensor<Host>) {
        ruprim_host::sort::dispatch::float_sort_with_indices(tensor, dim, descending, indices_dtype)
    }

    fn float_argsort(
        tensor: FloatTensor<Host>,
        dim: usize,
        descending: bool,
        out_dtype: ruda_tensor::IntDType,
    ) -> IntTensor<Host> {
        ruprim_host::sort::dispatch::float_argsort(tensor, dim, descending, out_dtype)
    }

    fn float_powi(lhs: FloatTensor<Host>, rhs: IntTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_powi(lhs, rhs)
    }

    fn float_powi_scalar(lhs: FloatTensor<Host>, rhs: Scalar) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_powi_scalar(lhs, rhs)
    }

    fn float_powi_scalar_impl(lhs: FloatTensor<Host>, rhs: Scalar) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_powi_scalar(lhs, rhs)
    }

    fn float_powf_scalar(tensor: FloatTensor<Host>, value: Scalar) -> FloatTensor<Host> {
        ruprim_host::binary::dispatch_float::float_powf_scalar(tensor, value)
    }

    fn float_max_abs(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        ruprim_host::reduce::dispatch::float_max_abs(tensor)
    }

    fn float_max_abs_dim(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        ruprim_host::reduce::dispatch::float_max_abs_dim(tensor, dim)
    }

    fn float_is_nan(tensor: FloatTensor<Host>, out_dtype: ruda_tensor::BoolDType) -> BoolTensor<Host> {
        ruprim_host::unary::dispatch_float::float_is_nan(tensor, out_dtype)
    }

    fn float_is_inf(tensor: FloatTensor<Host>, out_dtype: ruda_tensor::BoolDType) -> BoolTensor<Host> {
        ruprim_host::unary::dispatch_float::float_is_inf(tensor, out_dtype)
    }
}

// Tests kept here exercise flex-specific behavior: direct `Host::`
// backend-op calls with explicit IntDType/FloatDType to pin dtype storage
// selection (U8/I32/I64, F16/F64). Plain arithmetic, math, cast, cross,
// unfold, and random smoke tests have been dropped in favor of the
// equivalent coverage in ruda-backend-tests, which exercises every backend.
// When adding new tests, keep them here only if they probe flex dtype
// storage or flex internals; otherwise add them to
// crates/ruda-backend-tests/tests/tensor/float/ops/.
#[cfg(test)]
mod tests {
    use ruda_tensor::TensorData;

    use crate::Host;

    #[test]
    fn test_float_into_int_i32() {
        use ruda_tensor::ops::FloatTensorOps;
        use ruda_tensor::IntDType;

        let t = crate::HostTensor::from_data(TensorData::from([1.5f32, -2.7, 0.0, 255.9]));
        let result = Host::float_into_int(t, IntDType::I32);
        assert_eq!(result.dtype(), ruda_tensor::DType::I32);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, -2, 0, 255]);
    }

    #[test]
    fn test_float_into_int_u8() {
        use ruda_tensor::ops::FloatTensorOps;
        use ruda_tensor::IntDType;

        let t = crate::HostTensor::from_data(TensorData::from([0.0f32, 1.9, 127.5, 255.0]));
        let result = Host::float_into_int(t, IntDType::U8);
        assert_eq!(result.dtype(), ruda_tensor::DType::U8);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0, 1, 127, 255]);
    }

    #[test]
    fn test_float_argmax_i32_out_dtype() {
        use ruda_tensor::ops::FloatTensorOps;
        use ruda_tensor::IntDType;

        let t = crate::HostTensor::from_data(TensorData::from([[1.0f32, 3.0, 2.0]]));
        let result = Host::float_argmax(t, 1, IntDType::I32);
        assert_eq!(result.dtype(), ruda_tensor::DType::I32);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1]);
    }

    #[test]
    fn test_float_argmin_i32_out_dtype() {
        use ruda_tensor::ops::FloatTensorOps;
        use ruda_tensor::IntDType;

        let t = crate::HostTensor::from_data(TensorData::from([[3.0f32, 1.0, 2.0]]));
        let result = Host::float_argmin(t, 1, IntDType::I32);
        assert_eq!(result.dtype(), ruda_tensor::DType::I32);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1]);
    }

    #[test]
    fn test_float_argmax_i64_out_dtype() {
        use ruda_tensor::ops::FloatTensorOps;
        use ruda_tensor::IntDType;

        let t = crate::HostTensor::from_data(TensorData::from([[1.0f32, 3.0, 2.0]]));
        let result = Host::float_argmax(t, 1, IntDType::I64);
        assert_eq!(result.dtype(), ruda_tensor::DType::I64);
        let data: Vec<i64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1]);
    }

    #[test]
    fn test_float_max_dim_with_indices_i32() {
        use ruda_tensor::ops::FloatTensorOps;
        use ruda_tensor::IntDType;

        let t = crate::HostTensor::from_data(TensorData::from([[1.0f32, 5.0], [3.0, 2.0]]));
        let (values, indices) = Host::float_max_dim_with_indices(t, 1, IntDType::I32);
        assert_eq!(indices.dtype(), ruda_tensor::DType::I32);
        let idx: Vec<i32> = indices.into_data().to_vec().unwrap();
        assert_eq!(idx, vec![1, 0]);
        let vals: Vec<f32> = values.into_data().to_vec().unwrap();
        assert_eq!(vals, vec![5.0, 3.0]);
    }

    #[test]
    fn test_float_min_dim_with_indices_i32() {
        use ruda_tensor::ops::FloatTensorOps;
        use ruda_tensor::IntDType;

        let t = crate::HostTensor::from_data(TensorData::from([[1.0f32, 5.0], [3.0, 2.0]]));
        let (values, indices) = Host::float_min_dim_with_indices(t, 1, IntDType::I32);
        assert_eq!(indices.dtype(), ruda_tensor::DType::I32);
        let idx: Vec<i32> = indices.into_data().to_vec().unwrap();
        assert_eq!(idx, vec![0, 1]);
        let vals: Vec<f32> = values.into_data().to_vec().unwrap();
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn test_float_random_f64() {
        use ruda_tensor::{DType, FloatDType, ops::FloatTensorOps};

        let shape = ruda_tensor::Shape::from(vec![100]);
        let dist = ruda_tensor::Distribution::Uniform(0.0, 1.0);
        let device = crate::HostDevice;
        let t = Host::float_random(shape, dist, &device, FloatDType::F64);
        assert_eq!(t.dtype(), DType::F64);
        let data: Vec<f64> = t.into_data().to_vec().unwrap();
        assert!(data.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_float_random_f16() {
        use ruda_tensor::{DType, FloatDType, ops::FloatTensorOps};

        let shape = ruda_tensor::Shape::from(vec![100]);
        let dist = ruda_tensor::Distribution::Uniform(0.0, 1.0);
        let device = crate::HostDevice;
        let t = Host::float_random(shape, dist, &device, FloatDType::F16);
        assert_eq!(t.dtype(), DType::F16);
    }
}
