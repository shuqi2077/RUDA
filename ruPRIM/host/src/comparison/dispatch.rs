use ruda_core::tensor::{DType, host::HostTensor, element::Scalar};
use num_traits::ToPrimitive;

/// Convert a Scalar to (i64, u64) pair for the given dtype.
/// Only the matching type's conversion is validated; the other gets a dummy 0.
fn scalar_to_int_pair(dtype: DType, rhs: &Scalar) -> (i64, u64) {
    if dtype == DType::U64 {
        (0, rhs.to_u64().unwrap())
    } else {
        (rhs.to_i64().unwrap(), 0)
    }
}

pub fn float_equal_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    crate::comparison::equal_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
}

pub fn float_greater_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    crate::comparison::greater_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
}

pub fn float_greater_equal_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    crate::comparison::greater_equal_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
}

pub fn float_lower_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    crate::comparison::lower_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
}

pub fn float_lower_equal_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    crate::comparison::lower_equal_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
}

pub fn float_not_equal_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    crate::comparison::not_equal_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
}

pub fn int_equal_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
    crate::comparison::int_equal_elem(lhs, i, u, out_dtype)
}

pub fn int_greater_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
    crate::comparison::int_greater_elem(lhs, i, u, out_dtype)
}

pub fn int_greater_equal_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
    crate::comparison::int_greater_equal_elem(lhs, i, u, out_dtype)
}

pub fn int_lower_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
    crate::comparison::int_lower_elem(lhs, i, u, out_dtype)
}

pub fn int_lower_equal_elem(
    lhs: HostTensor,
    rhs: Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
    crate::comparison::int_lower_equal_elem(lhs, i, u, out_dtype)
}

pub fn int_not_equal_elem(
    lhs: HostTensor,
    rhs: ruda_core::tensor::element::Scalar,
    out_dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    let (i, u) = scalar_to_int_pair(lhs.dtype(), &rhs);
    crate::comparison::int_not_equal_elem(lhs, i, u, out_dtype)
}

