use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
pub(crate) fn taus_step_0(z: u32) -> u32 {
    taus_step(z, 13u32, 19u32, 12u32, 4294967294u32)
}

#[ruda]
pub(crate) fn taus_step_1(z: u32) -> u32 {
    taus_step(z, 2u32, 25u32, 4u32, 4294967288u32)
}

#[ruda]
pub(crate) fn taus_step_2(z: u32) -> u32 {
    taus_step(z, 3u32, 11u32, 17u32, 4294967280u32)
}

#[ruda]
fn taus_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
    let b = z << s1;
    let b = b ^ z;
    let b = b >> s2;
    let z = (z & m) << s3;
    z ^ b
}

#[ruda]
pub(crate) fn lcg_step(z: u32) -> u32 {
    let a = 1664525u32;
    let b = 1013904223u32;

    z * a + b
}

/// Converts a `u32` into a `f32` in the unit interval `[0.0, 1.0)`.
/// Used for generating random floats.
#[ruda]
pub fn to_unit_interval_closed_open(int_random: u32) -> f32 {
    // Use upper 24 bits for f32 precision
    // https://lemire.me/blog/2017/02/28/how-many-floating-point-numbers-are-in-the-interval-01/
    let shifted = int_random >> 8;
    f32::cast_from(shifted) / 16777216.0 // 2^24
}

/// Converts a `u32` into a `f32` in the unit interval `(0.0, 1.0)`.
/// Used for generating random floats.
#[ruda]
pub fn to_unit_interval_open(int_random: u32) -> f32 {
    // Use upper 23 bits to leave room for the offset
    let shifted = int_random >> 9;
    (f32::cast_from(shifted) + 1.0) / 8388609.0 // 2^23 + 1
}
