use crate::dsl::{comptime, ruda, prelude::*};

#[ruda]
pub(crate) fn small_int_reverse<I: Int, N: Size>(
    x: Vector<I, N>,
    #[comptime] width: u32,
) -> Vector<I, N> {
    let shift = comptime!(32 - width);

    let reversed = Vector::reverse_bits(Vector::<u32, N>::cast_from(x));
    Vector::cast_from(reversed >> Vector::new(shift))
}

#[ruda]
pub(crate) fn u64_reverse<I: Int, N: Size>(x: Vector<I, N>) -> Vector<I, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);

    let low_rev = Vector::reverse_bits(low);
    let high_rev = Vector::reverse_bits(high);
    // Swap low and high values
    let high = Vector::cast_from(low_rev) << shift;
    high | Vector::cast_from(high_rev)
}

#[ruda]
pub(crate) fn u64_count_bits<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);

    let low_cnt = Vector::<u32, N>::cast_from(Vector::count_ones(low));
    let high_cnt = Vector::<u32, N>::cast_from(Vector::count_ones(high));
    low_cnt + high_cnt
}

#[ruda]
pub(crate) fn u64_leading_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);
    let low_zeros = Vector::leading_zeros(low);
    let high_zeros = Vector::leading_zeros(high);

    select_many(
        high_zeros.equal(Vector::new(32)),
        low_zeros + high_zeros,
        high_zeros,
    )
}

/// There are three possible outcomes:
/// * low has any set -> return low
/// * low is empty, high has any set -> return high + 32
/// * low and high are empty -> return 0
#[ruda]
pub(crate) fn u64_ffs<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);
    let low_ffs = Vector::find_first_set(low);
    let high_ffs = Vector::find_first_set(high);

    let high_ffs = select_many(
        high_ffs.equal(Vector::new(0)),
        high_ffs,
        high_ffs + Vector::new(32),
    );
    select_many(low_ffs.equal(Vector::new(0)), high_ffs, low_ffs)
}

/// Subtract extra leading zeros after normalizing
#[ruda]
pub(crate) fn u16_u8_leading_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let width = I::type_size_bits().comptime() as u32;
    let over_width = Vector::new(32 - width);

    let x = Vector::<u32, N>::cast_from(x);
    let lz = x.leading_zeros();
    lz - over_width
}

/// There are three possible outcomes:
/// * low has any set -> return low
/// * low is empty, high has any set -> return high + 32
/// * low and high are empty -> return 0
#[ruda]
pub(crate) fn u64_trailing_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);
    let low_tz = Vector::trailing_zeros(low);
    let high_tz = Vector::trailing_zeros(high);

    let high_tz = select_many(
        high_tz.equal(Vector::new(32)),
        Vector::new(64),
        high_tz + Vector::new(32),
    );
    select_many(low_tz.equal(Vector::new(32)), high_tz, low_tz)
}

/// Clamp to width
#[ruda]
pub(crate) fn u16_u8_trailing_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let width = Vector::new(I::type_size_bits().comptime() as u32);

    let x = Vector::<u32, N>::cast_from(x);
    let lz = x.trailing_zeros();
    lz.min(width)
}
