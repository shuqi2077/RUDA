use ruda_kernel::dsl as kernel_dsl;
use ruda_test_runtime::TestRuntime;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::TensorHandle;
use ruda_kernel::library::tensor::ViewOperations;
use ruda_kernel::library::tensor::ViewOperationsExpand;
use ruda_kernel::library::tensor::ViewOperationsMut;
use ruda_kernel::library::tensor::ViewOperationsMutExpand;
use ruda_kernel::dsl::tensor_vector_size_parallel;
use ruda_kernel::dsl::zspace::shape;
use ruda_kernel::dsl::zspace::strides;

#[ruda(launch)]
fn cast_launch<From: Numeric, To: Numeric, N: Size>(
    from: &Tensor<Vector<From, N>>,
    to: &mut Tensor<Vector<To, N>>,
    #[define(From, To)] _types: [StorageType; 2],
) {
    cast_inner::<From, To, N>(from, to);
}

#[ruda]
fn cast_inner<From: Numeric, To: Numeric, N: Size>(
    from: &Tensor<Vector<From, N>>,
    to: &mut Tensor<Vector<To, N>>,
) {
    to.write_checked(
        ABSOLUTE_POS,
        Vector::cast_from(from.read_checked(ABSOLUTE_POS)),
    )
}

pub fn copy_casted(
    client: &ComputeClient<TestRuntime>,
    original: TensorHandle<TestRuntime>,
    target_type: StorageType,
) -> TensorHandle<TestRuntime> {
    if target_type == original.dtype {
        return TensorHandle::new_contiguous(
            original.shape().clone(),
            original.handle.clone(),
            target_type,
        );
    }

    let num_elems: usize = original.shape().num_elements();

    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(target_type.size()),
        &shape![num_elems],
        &strides![1],
        0,
    );

    let working_units: u32 = num_elems as u32 / vector_size as u32;
    let ruda_dim = RudaDim::new(client.properties(), working_units as usize);
    let ruda_count = working_units.div_ceil(ruda_dim.num_elems());

    let out = TensorHandle::new_contiguous(
        original.shape().clone(),
        client.empty(target_type.size() * num_elems),
        target_type,
    );

    let dtype = original.dtype;

    cast_launch::launch::<TestRuntime>(
        client,
        RudaCount::Static(ruda_count, 1, 1),
        ruda_dim,
        vector_size,
        original.into_arg(),
        out.clone().into_arg(),
        [dtype, target_type],
    );

    out
}
