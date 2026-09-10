use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

// Using plane operations, return the lowest coordinate for each vector element
// for which the item equal the target.
#[ruda]
pub(crate) fn lowest_coordinate_matching<E: Scalar, N: Size>(
    target: Vector<E, N>,
    item: Vector<E, N>,
    coordinate: Vector<u32, N>,
) -> Vector<u32, N> {
    let is_candidate = item.equal(target);
    let candidate_coordinate =
        select_many(is_candidate, coordinate, Vector::empty().fill(u32::MAX));
    plane_min(candidate_coordinate)
}
