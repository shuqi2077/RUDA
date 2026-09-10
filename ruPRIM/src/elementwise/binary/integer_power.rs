use ruda_core::tensor::{DType, TensorMetadata, element::Scalar as TensorScalar};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{
    RudaTensor,
    allocation::empty_device_dtype,
    layout::{address_type, broadcast_shape, max_vector_size},
};

#[ruda(launch_unchecked, address_type = "dynamic")]
fn tensor_kernel<F: Float, I: Int, N: Size>(
    lhs: &LinearView<Vector<F, N>>,
    rhs: &LinearView<Vector<I, N>>,
    out: &mut LinearView<Vector<F, N>, ReadWrite>,
    #[define(F, I)] _dtypes: [StorageType; 2],
) where F: Powi<I> {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    out[ABSOLUTE_POS] = Vector::powi(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn scalar_kernel<F: Float, I: Int, N: Size>(
    input: &LinearView<Vector<F, N>>,
    exponent: InputScalar,
    out: &mut LinearView<Vector<F, N>, ReadWrite>,
    #[define(F, I)] _dtypes: [StorageType; 2],
) where F: Powi<I> {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    out[ABSOLUTE_POS] = Vector::powi(input[ABSOLUTE_POS], Vector::new(exponent.get::<I>()));
}

pub fn tensor<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    let vector_size = max_vector_size(&lhs).min(max_vector_size(&rhs));
    let shape = broadcast_shape(&[&lhs, &rhs]);
    let dtype = lhs.dtype;
    let dtypes = [dtype.into(), rhs.dtype.into()];
    let client = lhs.client.clone();
    let working_units = shape.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&client, working_units, ruda_dim);

    unsafe {
        if lhs.can_mut_broadcast(&rhs) {
            tensor_kernel::launch_unchecked::<R>(
                &client,
                ruda_count,
                ruda_dim,
                address_type!(lhs, rhs),
                vector_size,
                lhs.clone().into_linear_view(),
                rhs.into_linear_view_like(&lhs),
                lhs.as_linear_view_alias(0),
                dtypes,
            );
            lhs
        } else {
            let output = empty_device_dtype(client.clone(), lhs.device.clone(), shape, dtype);
            tensor_kernel::launch_unchecked::<R>(
                &client,
                ruda_count,
                ruda_dim,
                address_type!(lhs, rhs, output),
                vector_size,
                lhs.into_linear_view_like(&output),
                rhs.into_linear_view_like(&output),
                output.clone().into_linear_view(),
                dtypes,
            );
            output
        }
    }
}

pub fn scalar<R: Runtime>(input: RudaTensor<R>, exponent: TensorScalar) -> RudaTensor<R> {
    let (exponent, exponent_dtype) = match exponent {
        TensorScalar::UInt(value) => {
            let dtype = if u32::try_from(value).is_ok() { DType::U32 } else { DType::U64 };
            (TensorScalar::UInt(value), dtype)
        }
        value => {
            let value = value.elem::<i64>();
            let dtype = if i32::try_from(value).is_ok() { DType::I32 } else { DType::I64 };
            (TensorScalar::Int(value), dtype)
        }
    };
    let exponent = InputScalar::new(exponent, exponent_dtype);
    let vector_size = max_vector_size(&input);
    let client = input.client.clone();
    let working_units = input.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&client, working_units, ruda_dim);
    let dtypes = [input.dtype.into(), exponent_dtype.into()];

    unsafe {
        if input.can_mut() && input.is_nonoverlapping() {
            scalar_kernel::launch_unchecked::<R>(
                &client,
                ruda_count,
                ruda_dim,
                address_type!(input),
                vector_size,
                input.clone().into_linear_view(),
                exponent,
                input.as_linear_view_alias(0),
                dtypes,
            );
            input
        } else {
            let output = empty_device_dtype(
                client.clone(), input.device.clone(), input.shape(), input.dtype,
            );
            scalar_kernel::launch_unchecked::<R>(
                &client,
                ruda_count,
                ruda_dim,
                address_type!(input, output),
                vector_size,
                input.into_linear_view(),
                exponent,
                output.clone().into_linear_view(),
                dtypes,
            );
            output
        }
    }
}
