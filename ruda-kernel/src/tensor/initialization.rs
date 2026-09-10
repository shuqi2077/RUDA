use crate::dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*};
use crate::library::tensor::layout::linear::LinearView;
use ruda_core::tensor::{DType, Shape};
use super::{RudaTensor, element::TensorElement, allocation::empty_device_dtype, layout::{address_type, max_vector_size}};

/// Creates a tensor filled with `value`
pub fn full<R: Runtime, E: TensorElement>(
    shape: Shape,
    device: &R::Device,
    value: E,
) -> RudaTensor<R> {
    let client = R::client(device);

    full_client::<R, E>(client, shape, device.clone(), value)
}

/// Creates a tensor filled with `value`
pub fn full_client<R: Runtime, E: TensorElement>(
    client: ComputeClient<R>,
    shape: Shape,
    device: R::Device,
    value: E,
) -> RudaTensor<R> {
    let dtype = E::dtype();
    full_device_dtype(client, shape, device, InputScalar::new(value, dtype), dtype)
}

/// Creates a tensor filled with `value`
pub fn full_device_dtype<R: Runtime>(
    client: ComputeClient<R>,
    shape: Shape,
    device: R::Device,
    value: InputScalar,
    dtype: DType,
) -> RudaTensor<R> {
    let empty = empty_device_dtype(client, device, shape, dtype);

    #[ruda(launch_unchecked, address_type = "dynamic")]
    pub fn full_kernel<C: Numeric, N: Size>(
        tensor: &mut LinearView<Vector<C, N>, ReadWrite>,
        value: InputScalar,
        #[define(C)] _dtype: StorageType,
    ) {
        if !tensor.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }

        tensor[ABSOLUTE_POS] = Vector::new(value.get::<C>());
    }

    let num_elems = empty.meta.num_elements();
    if num_elems == 0 {
        return empty;
    }
    let vector_size = max_vector_size(&empty);

    let working_units = num_elems / vector_size as usize;
    let ruda_dim = RudaDim::new(empty.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&empty.client, working_units, ruda_dim);

    unsafe {
        full_kernel::launch_unchecked(
            &empty.client,
            ruda_count,
            ruda_dim,
            address_type!(empty),
            vector_size,
            empty.clone().into_linear_view(),
            value,
            empty.dtype.into(),
        );
    }

    empty
}

/// Creates a tensor filled with zeros
pub fn zeros<R: Runtime>(device: R::Device, shape: Shape, dtype: DType) -> RudaTensor<R> {
    let client = R::client(&device);
    full_device_dtype(client, shape, device, InputScalar::new(0u32, dtype), dtype)
}

/// Creates a tensor filled with ones
pub fn ones<R: Runtime>(device: R::Device, shape: Shape, dtype: DType) -> RudaTensor<R> {
    let client = R::client(&device);
    full_device_dtype(client, shape, device, InputScalar::new(1u32, dtype), dtype)
}

/// Creates a tensor filled with zeros
pub fn zeros_client<R: Runtime>(
    client: ComputeClient<R>,
    device: R::Device,
    shape: Shape,
    dtype: DType,
) -> RudaTensor<R> {
    full_device_dtype(client, shape, device, InputScalar::new(0u32, dtype), dtype)
}

/// Creates a tensor filled with ones
pub fn ones_client<R: Runtime>(
    client: ComputeClient<R>,
    device: R::Device,
    shape: Shape,
    dtype: DType,
) -> RudaTensor<R> {
    full_device_dtype(client, shape, device, InputScalar::new(1u32, dtype), dtype)
}
