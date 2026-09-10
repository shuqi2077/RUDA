use ruda_kernel::dsl as kernel_dsl;
use ruda_test_runtime::TestRuntime;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::TensorHandle;
use ruda_kernel::library::tensor::ViewOperationsMut;
use ruda_kernel::library::tensor::ViewOperationsMutExpand;
use ruda_kernel::dsl::zspace::Shape;
use ruda_kernel::dsl::zspace::Strides;

use crate::test_tensor::base::BaseInputSpec;

#[ruda(launch)]
fn arange_launch<T: Numeric>(
    tensor: &mut Tensor<T>,
    scale: InputScalar,
    #[define(T)] _types: StorageType,
) {
    let linear = ABSOLUTE_POS;

    if linear >= tensor.len() {
        terminate!();
    }

    let mut remaining = linear;
    let mut offset = 0;

    for d in 0..tensor.rank() {
        let dim = tensor.shape(tensor.rank() - 1 - d);
        let idx = remaining % dim;
        remaining /= dim;
        offset += idx * tensor.stride(tensor.rank() - 1 - d);
    }

    tensor.write_checked(offset, T::cast_from(linear) * scale.get::<T>());
}

fn new_arange(
    client: &ComputeClient<TestRuntime>,
    shape: Shape,
    strides: Strides,
    dtype: StorageType,
    scale: f32,
) -> TensorHandle<TestRuntime> {
    let num_elems = shape.iter().product::<usize>();

    // Performance is not important here and this simplifies greatly the problem
    let vector_size = 1;

    let working_units: u32 = num_elems as u32 / vector_size as u32;
    let ruda_dim = RudaDim::new(client.properties(), working_units as usize);
    let ruda_count = working_units.div_ceil(ruda_dim.num_elems());

    let out = TensorHandle::new(
        client.empty(dtype.size() * num_elems),
        shape,
        strides,
        dtype,
    );

    arange_launch::launch::<TestRuntime>(
        client,
        RudaCount::new_1d(ruda_count),
        ruda_dim,
        out.clone().into_arg(),
        InputScalar::new(scale, dtype),
        dtype,
    );

    out
}

pub(crate) fn build_arange(spec: BaseInputSpec, scale: Option<f32>) -> TensorHandle<TestRuntime> {
    new_arange(
        &spec.client,
        spec.shape.clone(),
        spec.strides(),
        spec.dtype,
        scale.unwrap_or(1.),
    )
}
