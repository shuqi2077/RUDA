//! Bool tensor operations for the Host backend.

use alloc::vec::Vec;
use ruda_tensor::{
    DType, ExecutionError, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use ruda_tensor::{IntDType, Shape, Slice};

use crate::{Host, HostTensor};

impl BoolTensorOps<Host> for Host {
    fn bool_from_data(data: TensorData, _device: &Device<Host>) -> BoolTensor<Host> {
        HostTensor::from_data(data)
    }

    async fn bool_into_data(tensor: BoolTensor<Host>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn bool_device(_tensor: &BoolTensor<Host>) -> Device<Host> {
        Default::default()
    }

    fn bool_to_device(tensor: BoolTensor<Host>, _device: &Device<Host>) -> BoolTensor<Host> {
        tensor
    }

    fn bool_cat(tensors: Vec<BoolTensor<Host>>, dim: usize) -> BoolTensor<Host> {
        crate::ops::cat::cat(tensors, dim)
    }

    fn bool_reshape(tensor: BoolTensor<Host>, shape: Shape) -> BoolTensor<Host> {
        tensor.reshape(shape)
    }

    fn bool_slice(tensor: BoolTensor<Host>, slices: &[Slice]) -> BoolTensor<Host> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn bool_empty(
        shape: Shape,
        _device: &Device<Host>,
        dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        HostTensor::empty(shape, DType::from(dtype))
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Host>,
        slices: &[Slice],
        value: BoolTensor<Host>,
    ) -> BoolTensor<Host> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    fn bool_into_int(tensor: BoolTensor<Host>, out_dtype: ruda_tensor::IntDType) -> IntTensor<Host> {
        ruprim_host::cast::bool_into_int(tensor, out_dtype)
    }

    fn bool_into_float(
        tensor: BoolTensor<Host>,
        out_dtype: ruda_tensor::FloatDType,
    ) -> FloatTensor<Host> {
        ruprim_host::cast::bool_into_float(tensor, out_dtype)
    }

    fn bool_swap_dims(tensor: BoolTensor<Host>, dim1: usize, dim2: usize) -> BoolTensor<Host> {
        tensor.transpose(dim1, dim2)
    }

    fn bool_permute(tensor: BoolTensor<Host>, axes: &[usize]) -> BoolTensor<Host> {
        tensor.permute(axes)
    }

    fn bool_flip(tensor: BoolTensor<Host>, axes: &[usize]) -> BoolTensor<Host> {
        crate::ops::flip::flip(tensor, axes)
    }

    fn bool_equal(lhs: BoolTensor<Host>, rhs: BoolTensor<Host>) -> BoolTensor<Host> {
        ruprim_host::boolean::bool_equal(lhs, rhs)
    }

    fn bool_not(mut tensor: BoolTensor<Host>) -> BoolTensor<Host> {
        ruprim_host::boolean::bool_not(tensor)
    }

    fn bool_and(lhs: BoolTensor<Host>, rhs: BoolTensor<Host>) -> BoolTensor<Host> {
        ruprim_host::boolean::bool_and(lhs, rhs)
    }

    fn bool_or(lhs: BoolTensor<Host>, rhs: BoolTensor<Host>) -> BoolTensor<Host> {
        ruprim_host::boolean::bool_or(lhs, rhs)
    }

    fn bool_xor(lhs: BoolTensor<Host>, rhs: BoolTensor<Host>) -> BoolTensor<Host> {
        ruprim_host::boolean::bool_xor(lhs, rhs)
    }

    fn bool_expand(tensor: BoolTensor<Host>, shape: Shape) -> BoolTensor<Host> {
        crate::ops::expand::expand(tensor, shape)
    }

    // Missing methods
    fn bool_zeros(
        shape: Shape,
        device: &Device<Host>,
        dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        Self::bool_empty(shape, device, dtype)
    }

    fn bool_ones(
        shape: Shape,
        _device: &Device<Host>,
        dtype: ruda_tensor::BoolDType,
    ) -> BoolTensor<Host> {
        ruprim_host::boolean::bool_ones(shape, dtype)
    }

    fn bool_mask_where(
        tensor: BoolTensor<Host>,
        mask: BoolTensor<Host>,
        value: BoolTensor<Host>,
    ) -> BoolTensor<Host> {
        crate::ops::mask::mask_where_bool(tensor, mask, value)
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Host>,
        mask: BoolTensor<Host>,
        value: ruda_tensor::Scalar,
    ) -> BoolTensor<Host> {
        let value: bool = value.elem();
        crate::ops::mask::mask_fill_bool(tensor, mask, value)
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Host>,
        indices: IntTensor<Host>,
    ) -> BoolTensor<Host> {
        crate::ops::gather_scatter::gather_bool(tensor, dim, indices)
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Host>,
        indices: IntTensor<Host>,
        value: BoolTensor<Host>,
    ) -> BoolTensor<Host> {
        crate::ops::gather_scatter::scatter_or(tensor, dim, indices, value)
    }

    fn bool_equal_elem(lhs: BoolTensor<Host>, rhs: ruda_tensor::Scalar) -> BoolTensor<Host> {
        ruprim_host::boolean::bool_equal_elem(lhs, rhs)
    }

    fn bool_unfold(
        tensor: BoolTensor<Host>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Host> {
        crate::ops::unfold::unfold_bool(tensor, dim, size, step)
    }

    fn bool_not_equal(lhs: BoolTensor<Host>, rhs: BoolTensor<Host>) -> BoolTensor<Host> {
        let out_dtype = ruda_tensor::BoolDType::from(lhs.dtype());
        crate::ops::comparison::bool_not_equal(lhs, rhs, out_dtype)
    }

    fn bool_not_equal_elem(lhs: BoolTensor<Host>, rhs: ruda_tensor::Scalar) -> BoolTensor<Host> {
        let out_dtype = ruda_tensor::BoolDType::from(lhs.dtype());
        let rhs: bool = rhs.elem();
        crate::ops::comparison::bool_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn bool_any(tensor: BoolTensor<Host>) -> BoolTensor<Host> {
        let out_dtype = ruda_tensor::BoolDType::from(tensor.dtype());
        crate::ops::comparison::any_bool(tensor, out_dtype)
    }

    fn bool_any_dim(tensor: BoolTensor<Host>, dim: usize) -> BoolTensor<Host> {
        let out_dtype = ruda_tensor::BoolDType::from(tensor.dtype());
        crate::ops::comparison::any_bool_dim(tensor, dim, out_dtype)
    }

    fn bool_all(tensor: BoolTensor<Host>) -> BoolTensor<Host> {
        let out_dtype = ruda_tensor::BoolDType::from(tensor.dtype());
        crate::ops::comparison::all_bool(tensor, out_dtype)
    }

    fn bool_all_dim(tensor: BoolTensor<Host>, dim: usize) -> BoolTensor<Host> {
        let out_dtype = ruda_tensor::BoolDType::from(tensor.dtype());
        crate::ops::comparison::all_bool_dim(tensor, dim, out_dtype)
    }

    fn bool_select(
        tensor: BoolTensor<Host>,
        dim: usize,
        indices: IntTensor<Host>,
    ) -> BoolTensor<Host> {
        crate::ops::gather_scatter::select::<u8>(tensor, dim, indices)
    }

    fn bool_select_or(
        tensor: BoolTensor<Host>,
        dim: usize,
        indices: IntTensor<Host>,
        value: BoolTensor<Host>,
    ) -> BoolTensor<Host> {
        ruprim_host::boolean::bool_select_or(tensor, dim, indices, value)
    }

    fn bool_transpose(tensor: BoolTensor<Host>) -> BoolTensor<Host> {
        let ndims = tensor.layout().num_dims();
        if ndims < 2 {
            return tensor;
        }
        tensor.transpose(ndims - 2, ndims - 1)
    }

    fn bool_repeat_dim(tensor: BoolTensor<Host>, dim: usize, times: usize) -> BoolTensor<Host> {
        crate::ops::repeat_dim::repeat_dim(tensor, dim, times)
    }

    async fn bool_argwhere(tensor: BoolTensor<Host>, out_dtype: IntDType) -> IntTensor<Host> {
        ruprim_host::boolean::bool_argwhere(tensor, out_dtype).await
    }
}

// Tests kept here exercise flex-specific dtype storage selection via
// explicit IntDType/FloatDType. Plain bool ops, bool-to-int/float
// casts, and negative-stride (flipped) bool coverage have been migrated
// to crates/ruda-backend-tests/tests/tensor/bool/ops/{logical,cast}.rs
// so they run against every backend. When adding new tests, keep them
// here only if they probe flex dtype dispatch; otherwise add them
// there.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use ruda_tensor::TensorData;
    use ruda_tensor::ops::BoolTensorOps;
    use ruda_tensor::{FloatDType, IntDType};

    use crate::{Host, HostTensor};

    #[test]
    fn test_bool_into_int_u8() {
        let t = HostTensor::from_data(TensorData::from([true, false, true]));
        let result = Host::bool_into_int(t, IntDType::U8);
        assert_eq!(result.dtype(), ruda_tensor::DType::U8);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1u8, 0, 1]);
    }

    #[test]
    fn test_bool_into_float_f64() {
        let t = HostTensor::from_data(TensorData::from([true, false, true]));
        let result = Host::bool_into_float(t, FloatDType::F64);
        assert_eq!(result.dtype(), ruda_tensor::DType::F64);
        let data: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0f64, 0.0, 1.0]);
    }
}
