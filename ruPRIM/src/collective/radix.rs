use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use half::{bf16, f16};

/// Order-preserving radix encoding. Signed keys flip their sign bit; floating
/// keys invert negative encodings and flip the sign of positive encodings.
/// Floating signed zeros share an encoding so stable sorts preserve their order.
#[ruda]
pub trait RudaRadixKey: Numeric {
    fn ordered_bits(value: Self) -> u64;
}

macro_rules! unsigned_key {
    ($($ty:ty),* $(,)?) => {$(
        #[ruda]
        impl RudaRadixKey for $ty {
            fn ordered_bits(value: Self) -> u64 { u64::cast_from(value) }
        }
    )*};
}
unsigned_key!(u8, u16, u32, u64);

macro_rules! signed_key {
    ($ty:ident, $bits:ident, $sign:expr) => {
        #[ruda]
        impl RudaRadixKey for $ty {
            fn ordered_bits(value: Self) -> u64 {
                u64::cast_from($bits::reinterpret(value) ^ $sign)
            }
        }
    };
}
signed_key!(i8, u8, 0x80u8);
signed_key!(i16, u16, 0x8000u16);
signed_key!(i32, u32, 0x8000_0000u32);
signed_key!(i64, u64, 0x8000_0000_0000_0000u64);

macro_rules! float_key {
    ($ty:ident, $bits:ident, $sign:expr) => {
        #[ruda]
        impl RudaRadixKey for $ty {
            fn ordered_bits(value: Self) -> u64 {
                let raw = $bits::reinterpret(value);
                let mut bits = raw;
                if value == $ty::from_int(0) { bits = 0; }
                let ordered = if (bits & $sign) != 0 { !bits } else { bits ^ $sign };
                u64::cast_from(ordered)
            }
        }
    };
}
float_key!(f32, u32, 0x8000_0000u32);
float_key!(f64, u64, 0x8000_0000_0000_0000u64);
float_key!(f16, u16, 0x8000u16);
float_key!(bf16, u16, 0x8000u16);
