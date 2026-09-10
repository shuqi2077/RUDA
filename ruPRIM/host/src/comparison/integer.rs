use super::*;

// Integer comparison functions

fn compare_int<I64Cmp, U64Cmp>(
    lhs: HostTensor,
    rhs: HostTensor,
    out_dtype: BoolDType,
    i64_cmp: I64Cmp,
    u64_cmp: U64Cmp,
) -> HostTensor
where
    I64Cmp: Fn(i64, i64) -> bool,
    U64Cmp: Fn(u64, u64) -> bool,
{
    let (lhs, rhs) = crate::expand::broadcast_binary(lhs, rhs);

    match lhs.dtype() {
        DType::I64 => compare_typed(lhs, &rhs, out_dtype, i64_cmp),
        DType::U64 => compare_typed(lhs, &rhs, out_dtype, u64_cmp),
        DType::I32 => compare_typed(lhs, &rhs, out_dtype, |a: i32, b: i32| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::I16 => compare_typed(lhs, &rhs, out_dtype, |a: i16, b: i16| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::I8 => compare_typed(lhs, &rhs, out_dtype, |a: i8, b: i8| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U32 => compare_typed(lhs, &rhs, out_dtype, |a: u32, b: u32| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U16 => compare_typed(lhs, &rhs, out_dtype, |a: u16, b: u16| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U8 => compare_typed(lhs, &rhs, out_dtype, |a: u8, b: u8| {
            i64_cmp(a as i64, b as i64)
        }),
        other => panic!("compare_int: unsupported dtype {:?}", other),
    }
}

fn compare_int_elem<I64Cmp, U64Cmp>(
    lhs: HostTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
    i64_cmp: I64Cmp,
    u64_cmp: U64Cmp,
) -> HostTensor
where
    I64Cmp: Fn(i64, i64) -> bool,
    U64Cmp: Fn(u64, u64) -> bool,
{
    match lhs.dtype() {
        DType::I64 => compare_elem_typed(lhs, i64_rhs, out_dtype, i64_cmp),
        DType::U64 => compare_elem_typed(lhs, u64_rhs, out_dtype, u64_cmp),
        DType::I32 => compare_elem_typed(lhs, i64_rhs as i32, out_dtype, |a: i32, b: i32| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::I16 => compare_elem_typed(lhs, i64_rhs as i16, out_dtype, |a: i16, b: i16| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::I8 => compare_elem_typed(lhs, i64_rhs as i8, out_dtype, |a: i8, b: i8| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U32 => compare_elem_typed(lhs, i64_rhs as u32, out_dtype, |a: u32, b: u32| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U16 => compare_elem_typed(lhs, i64_rhs as u16, out_dtype, |a: u16, b: u16| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U8 => compare_elem_typed(lhs, i64_rhs as u8, out_dtype, |a: u8, b: u8| {
            i64_cmp(a as i64, b as i64)
        }),
        other => panic!("compare_int_elem: unsupported dtype {:?}", other),
    }
}

pub fn int_greater(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a > b, |a, b| a > b)
}

pub fn int_greater_elem(
    lhs: HostTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> HostTensor {
    compare_int_elem(lhs, i64_rhs, u64_rhs, out_dtype, |a, b| a > b, |a, b| a > b)
}

pub fn int_greater_equal(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a >= b, |a, b| a >= b)
}

pub fn int_greater_equal_elem(
    lhs: HostTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> HostTensor {
    compare_int_elem(
        lhs,
        i64_rhs,
        u64_rhs,
        out_dtype,
        |a, b| a >= b,
        |a, b| a >= b,
    )
}

pub fn int_lower(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a < b, |a, b| a < b)
}

pub fn int_lower_elem(
    lhs: HostTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> HostTensor {
    compare_int_elem(lhs, i64_rhs, u64_rhs, out_dtype, |a, b| a < b, |a, b| a < b)
}

pub fn int_lower_equal(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a <= b, |a, b| a <= b)
}

pub fn int_lower_equal_elem(
    lhs: HostTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> HostTensor {
    compare_int_elem(
        lhs,
        i64_rhs,
        u64_rhs,
        out_dtype,
        |a, b| a <= b,
        |a, b| a <= b,
    )
}

pub fn int_equal(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a == b, |a, b| a == b)
}

pub fn int_equal_elem(
    lhs: HostTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> HostTensor {
    compare_int_elem(
        lhs,
        i64_rhs,
        u64_rhs,
        out_dtype,
        |a, b| a == b,
        |a, b| a == b,
    )
}

pub fn int_not_equal(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a != b, |a, b| a != b)
}

pub fn int_not_equal_elem(
    lhs: HostTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> HostTensor {
    compare_int_elem(
        lhs,
        i64_rhs,
        u64_rhs,
        out_dtype,
        |a, b| a != b,
        |a, b| a != b,
    )
}

pub fn bool_not_equal(lhs: HostTensor, rhs: HostTensor, out_dtype: BoolDType) -> HostTensor {
    let (lhs, rhs) = crate::expand::broadcast_binary(lhs, rhs);
    let shape = lhs.layout().shape().clone();
    let lhs_data: &[u8] = lhs.bytes();
    let rhs_data: &[u8] = rhs.bytes();
    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((ls, le)), Some((rs, re))) => lhs_data[ls..le]
            .iter()
            .zip(&rhs_data[rs..re])
            .map(|(&a, &b)| if a != b { 1 } else { 0 })
            .collect(),
        _ => {
            let lhs = lhs.to_contiguous();
            let rhs = rhs.to_contiguous();
            lhs.bytes()
                .iter()
                .zip(rhs.bytes())
                .map(|(&a, &b)| if a != b { 1 } else { 0 })
                .collect()
        }
    };
    make_bool_tensor(result, shape, out_dtype)
}

pub fn bool_not_equal_elem(lhs: HostTensor, rhs: bool, out_dtype: BoolDType) -> HostTensor {
    let rhs_val: u8 = if rhs { 1 } else { 0 };
    let shape = lhs.layout().shape().clone();
    let lhs = lhs.to_contiguous();
    let data: &[u8] = lhs.bytes();
    let result: Vec<u8> = data
        .iter()
        .map(|&a| if a != rhs_val { 1 } else { 0 })
        .collect();
    make_bool_tensor(result, shape, out_dtype)
}

