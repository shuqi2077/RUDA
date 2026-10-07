use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl::ir::{ElemType, FloatKind, ManagedVariable};

fn has_infinite_bounds(elem: ElemType) -> bool {
    matches!(
        elem,
        ElemType::Float(
            FloatKind::F16
                | FloatKind::BF16
                | FloatKind::F32
                | FloatKind::F64
                | FloatKind::Flex32
                | FloatKind::TF32
                | FloatKind::E5M2
        )
    )
}

#[ruda]
pub(crate) fn max_identity<E: Numeric>() -> E {
    intrinsic!(|scope| {
        let elem = E::as_type(scope).elem_type();
        let value = if has_infinite_bounds(elem) {
            elem.constant(f64::NEG_INFINITY.into())
        } else {
            elem.min_variable()
        };
        E::from_expand_elem(ManagedVariable::Plain(value))
    })
}

#[ruda]
pub(crate) fn min_identity<E: Numeric>() -> E {
    intrinsic!(|scope| {
        let elem = E::as_type(scope).elem_type();
        let value = if has_infinite_bounds(elem) {
            elem.constant(f64::INFINITY.into())
        } else {
            elem.max_variable()
        };
        E::from_expand_elem(ManagedVariable::Plain(value))
    })
}

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
