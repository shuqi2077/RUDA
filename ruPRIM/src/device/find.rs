use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaCompare, RudaCompareExpand, RudaMinimum, RudaMinimumLaunch};
use super::{RudaPrimitiveError, check_type, select::RudaPredicate, reduce};
use super::select::RudaPredicateExpand;

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn candidates<T: Numeric, P: RudaPredicate<T> + LaunchArg>(
    input: &LinearView<T>, output: &mut LinearView<u64, ReadWrite>, predicate: &P,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() {
        let mut candidate = input.shape() as u64;
        if predicate.test(input[index]) { candidate = index as u64; }
        output[index] = candidate;
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn bound_kernel<T: Numeric, C: RudaCompare<T> + LaunchArg>(
    input: &LinearView<T>, output: &mut LinearView<u64, ReadWrite>, value: InputScalar,
    compare: &C, #[comptime] upper: bool,
) {
    if ABSOLUTE_POS == 0 {
        let key = value.get::<T>();
        let mut low = 0usize;
        let mut high = input.shape();
        while low < high {
            let mid = low + (high - low) / 2;
            let before = if upper { !compare.before(key, input[mid]) } else { compare.before(input[mid], key) };
            if before { low = mid + 1; } else { high = mid; }
        }
        output[0] = low as u64;
    }
}

/// Return the first matching index in a one-element U64 device tensor.
/// When no item matches, return the input length, including zero for empty input.
pub fn find_if<R, T, P>(input: &RudaTensor<R>, predicate: P::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, P: RudaPredicate<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    let size = input.meta.num_elements();
    let indices = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([size]), DType::U64);
    if size > 0 {
        let dim = RudaDim::new(input.client.properties(), size);
        let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
        unsafe {
            candidates::launch_unchecked::<T, P, R>(&input.client, grid, dim, address_type!(input, indices),
                input.clone().into_linear_view(), indices.clone().into_linear_view(), predicate);
        }
    }
    reduce::reduce::<R, u64, RudaMinimum>(&indices, size as u64, RudaMinimumLaunch::new(), threads)
}

/// Lower or upper insertion bound in a sequence sorted by `compare`.
pub fn bound<R, T, C>(input: &RudaTensor<R>, value: T, compare: C::RuntimeArg<R>, upper: bool) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, C: RudaCompare<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([1]), DType::U64);
    unsafe {
        bound_kernel::launch_unchecked::<T, C, R>(
            &input.client, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), address_type!(input, output),
            input.clone().into_linear_view(), output.clone().into_linear_view(), InputScalar::new(value, input.dtype), compare, upper,
        );
    }
    Ok(output)
}
