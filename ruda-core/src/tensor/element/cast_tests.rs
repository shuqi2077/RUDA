#[allow(unused_imports)]
use super::*;

#[test]
fn to_element_float() {
    let f32_toolarge = 1e39f64;
    assert_eq!(f32_toolarge.to_f32(), f32::INFINITY);
    assert_eq!((-f32_toolarge).to_f32(), f32::NEG_INFINITY);
    assert_eq!((f32::MAX as f64).to_f32(), f32::MAX);
    assert_eq!((-f32::MAX as f64).to_f32(), -f32::MAX);
    assert_eq!(f64::INFINITY.to_f32(), f32::INFINITY);
    assert_eq!((f64::NEG_INFINITY).to_f32(), f32::NEG_INFINITY);
    assert!((f64::NAN).to_f32().is_nan());
}

#[test]
#[should_panic]
fn to_element_signed_to_u8_underflow() {
    let _x = (-1i8).to_u8();
}

#[test]
#[should_panic]
fn to_element_signed_to_u16_underflow() {
    let _x = (-1i8).to_u16();
}

#[test]
#[should_panic]
fn to_element_signed_to_u32_underflow() {
    let _x = (-1i8).to_u32();
}

#[test]
#[should_panic]
fn to_element_signed_to_u64_underflow() {
    let _x = (-1i8).to_u64();
}

#[test]
#[should_panic]
fn to_element_signed_to_u128_underflow() {
    let _x = (-1i8).to_u128();
}

#[test]
#[should_panic]
fn to_element_signed_to_usize_underflow() {
    let _x = (-1i8).to_usize();
}

#[test]
#[should_panic]
fn to_element_unsigned_to_u8_overflow() {
    let _x = 256.to_u8();
}

#[test]
#[should_panic]
fn to_element_unsigned_to_u16_overflow() {
    let _x = 65_536.to_u16();
}

#[test]
#[should_panic]
fn to_element_unsigned_to_u32_overflow() {
    let _x = 4_294_967_296u64.to_u32();
}

#[test]
#[should_panic]
fn to_element_unsigned_to_u64_overflow() {
    let _x = 18_446_744_073_709_551_616u128.to_u64();
}

#[test]
fn to_element_int_to_float() {
    assert_eq!((-1).to_f32(), -1.0);
    assert_eq!((-1).to_f64(), -1.0);
    assert_eq!(255.to_f32(), 255.0);
    assert_eq!(65_535.to_f64(), 65_535.0);
}

#[test]
fn to_element_float_to_int() {
    assert_eq!((-1.0).to_i8(), -1);
    assert_eq!(1.0.to_u8(), 1);
    assert_eq!(1.8.to_u16(), 1);
    assert_eq!(123.456.to_u32(), 123);
}
