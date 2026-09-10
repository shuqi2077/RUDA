use ruda_kernel::dsl as kernel_dsl;
use ruda_test_runtime::TestRuntime;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::TensorHandle;
use ruda_kernel::library::tensor::ViewOperationsMut;
use ruda_kernel::library::tensor::ViewOperationsMutExpand;
use ruda_kernel::dsl::zspace::Shape;
use ruda_kernel::dsl::zspace::Strides;

use crate::BaseInputSpec;

#[ruda(launch)]
fn eye_launch<T: Numeric, N: Size>(
    tensor: &mut Tensor<Vector<T, N>>,
    #[define(T)] _types: StorageType,
) {
    let batch = RUDA_POS_Z as usize;
    let i = ABSOLUTE_POS_X as usize;
    let j = ABSOLUTE_POS_Y as usize;

    let rank = tensor.rank();
    let rows = tensor.shape(rank - 2);
    let cols = tensor.shape(rank - 1);
    if i >= rows || j >= cols {
        terminate!();
    }

    let idx =
        batch * tensor.stride(rank - 3) + i * tensor.stride(rank - 2) + j * tensor.stride(rank - 1);

    tensor.write_checked(idx, Vector::cast_from(i == j));
}

#[allow(unused)]
fn new_eyed(
    client: &ComputeClient<TestRuntime>,
    shape: Shape,
    rows: usize,
    cols: usize,
    total_batches: usize,
    dtype: StorageType,
    strides: Strides,
) -> TensorHandle<TestRuntime> {
    // Performance is not important here and this simplifies greatly the problem
    let vector_size = 1;

    let dim_x = 32;
    let dim_y = 32;
    let ruda_dim = RudaDim::new_2d(dim_x, dim_y);
    let ruda_count = RudaCount::new_3d(
        (rows as u32).div_ceil(dim_x),
        (cols as u32).div_ceil(dim_y),
        total_batches as u32,
    );

    let out = TensorHandle::new(
        client.empty(dtype.size() * shape.iter().product::<usize>()),
        shape.clone(),
        strides,
        dtype,
    );

    eye_launch::launch::<TestRuntime>(
        client,
        ruda_count,
        ruda_dim,
        vector_size,
        out.clone().into_arg(),
        dtype,
    );

    out
}

pub(crate) fn build_eye(spec: BaseInputSpec) -> TensorHandle<TestRuntime> {
    let (batches, matrix) = spec.shape.split_at(spec.shape.len() - 2);
    let rows = matrix[0];
    let cols = matrix[1];
    let total_batches = batches.iter().product::<usize>();

    new_eyed(
        &spec.client,
        spec.shape.clone(),
        rows,
        cols,
        total_batches,
        spec.dtype,
        spec.strides(),
    )
}
