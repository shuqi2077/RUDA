use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, element::TensorElement, layout::address_type};
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand};
use super::{RudaPrimitiveError, check_type, empty_like, iteration};

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn difference_kernel<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    input: &LinearView<T>, output: &mut LinearView<T, ReadWrite>, op: &O,
    #[comptime] right: bool,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() {
        let mut value = input[index];
        if right {
            if index + 1 < input.shape() { value = op.combine(value, input[index + 1]); }
        } else if index > 0 {
            value = op.combine(value, input[index - 1]);
        }
        output[index] = value;
    }
}

/// Apply `op(current, neighbour)`, preserving the left or right edge value.
/// In-place operation first snapshots the input so block scheduling cannot
/// change which neighbour is observed.
pub fn difference<R, T, O>(
    input: &RudaTensor<R>, op: O::RuntimeArg<R>, right: bool, in_place: bool,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaBinaryOp<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    let source = if in_place { iteration::copy::<R, T>(input)? } else { input.clone() };
    let output = if in_place { input.clone() } else { empty_like(input) };
    let count = input.meta.num_elements();
    if count > 0 {
        let dim = RudaDim::new(input.client.properties(), count);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            difference_kernel::launch_unchecked::<T, O, R>(
                &input.client, grid, dim, address_type!(source, output),
                source.into_linear_view(), output.clone().into_linear_view(), op, right,
            );
        }
    }
    Ok(output)
}
