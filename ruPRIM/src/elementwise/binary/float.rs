use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::broadcast_shape;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::linear::LinearView;

pub trait BinaryOpFloatFamily: Send + Sync + 'static {
    type BinaryOp<C: Float, N: Size>: BinaryOpFloat<C, N>;
}

#[ruda]
pub trait BinaryOpFloat<C: Float, N: Size>: 'static + Send + Sync {
    /// Execute a binary operation.
    fn execute(lhs: Vector<C, N>, rhs: Vector<C, N>) -> Vector<C, N>;
}

pub struct ArcTan2Op;

impl BinaryOpFloatFamily for ArcTan2Op {
    type BinaryOp<C: Float, N: Size> = Self;
}

#[ruda]
impl<T: Float, N: Size> BinaryOpFloat<T, N> for ArcTan2Op {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        Vector::atan2(lhs, rhs)
    }
}

#[ruda(launch_unchecked, address_type = "dynamic")]
pub fn kernel_binop<C: Float, N: Size, O: BinaryOpFloatFamily>(
    lhs: &LinearView<Vector<C, N>>,
    rhs: &LinearView<Vector<C, N>>,
    out: &mut LinearView<Vector<C, N>, ReadWrite>,
    #[define(C)] _dtype: StorageType,
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    out[ABSOLUTE_POS] = O::BinaryOp::<C, N>::execute(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
}

pub fn launch_binop_float<R: Runtime, O: BinaryOpFloatFamily>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
) -> RudaTensor<R> {
    let vector_size_lhs = max_vector_size(&lhs);
    let vector_size_rhs = max_vector_size(&rhs);
    let vector_size = Ord::min(vector_size_lhs, vector_size_rhs);

    let shape_out = broadcast_shape(&[&lhs, &rhs]);
    let dtype = lhs.dtype;

    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();
    let working_units = num_elems / vector_size as usize;

    let ruda_dim = RudaDim::new(lhs.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&lhs.client, working_units, ruda_dim);

    unsafe {
        if lhs.can_mut_broadcast(&rhs) {
            kernel_binop::launch_unchecked::<O, R>(
                &client,
                ruda_count,
                ruda_dim,
                address_type!(lhs, rhs),
                vector_size,
                lhs.clone().into_linear_view(),
                rhs.clone().into_linear_view_like(&lhs),
                lhs.as_linear_view_alias(0),
                dtype.into(),
            );

            lhs
        } else if rhs.can_mut_broadcast(&lhs) {
            kernel_binop::launch_unchecked::<O, R>(
                &client,
                ruda_count,
                ruda_dim,
                address_type!(lhs, rhs),
                vector_size,
                lhs.into_linear_view_like(&rhs),
                rhs.clone().into_linear_view(),
                rhs.as_linear_view_alias(1),
                dtype.into(),
            );

            rhs
        } else {
            let output =
                empty_device_dtype(lhs.client.clone(), lhs.device.clone(), shape_out, dtype);

            kernel_binop::launch_unchecked::<O, R>(
                &client,
                ruda_count,
                ruda_dim,
                address_type!(lhs, rhs, output),
                vector_size,
                lhs.into_linear_view_like(&output),
                rhs.into_linear_view_like(&output),
                output.clone().into_linear_view(),
                dtype.into(),
            );

            output
        }
    }
}

/// Calculate the four-quadrant inverse tangent of `lhs / rhs`.
pub fn atan2<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop_float::<R, ArcTan2Op>(lhs, rhs)
}
