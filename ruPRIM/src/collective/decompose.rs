use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use super::radix::RudaRadixKey;

pub trait RudaDecomposition<K: RudaType> {
    const BITS: usize;
}

/// Bit zero is the least significant bit of the last decomposed field.
#[ruda]
pub trait RudaDecomposer<K: RudaType>: RudaType + RudaDecomposition<K> {
    fn bit(&self, key: K, index: usize) -> bool;
}

#[derive(Clone, Copy, RudaType, RudaLaunch)]
pub struct RudaScalarDecomposer;

impl<R: Runtime> Clone for RudaScalarDecomposerLaunch<R> {
    fn clone(&self) -> Self { Self::new() }
}

#[ruda]
impl<K: RudaRadixKey> RudaDecomposer<K> for RudaScalarDecomposer {
    fn bit(&self, key: K, index: usize) -> bool {
        ((K::ordered_bits(key) >> index as u64) & 1u64) != 0
    }
}

impl<K: RudaRadixKey> RudaDecomposition<K> for RudaScalarDecomposer {
    const BITS: usize = core::mem::size_of::<K>() * 8;
}

/// Define a lexicographic arithmetic-field decomposition of any width.
/// Fields are listed most significant first; padding is never a sort key.
#[macro_export]
macro_rules! ruda_decomposer {
    ($vis:vis $name:ident, $launch:ident for $key:ty { $($field:ident : $ty:ty),+ $(,)? }) => {
        #[derive(Clone, Copy, ruda_kernel::dsl::RudaType, ruda_kernel::dsl::RudaLaunch)]
        $vis struct $name;
        impl<R: ruda_kernel::dsl::prelude::Runtime> Clone for $launch<R> {
            fn clone(&self) -> Self { Self::new() }
        }
        impl $crate::collective::decompose::RudaDecomposition<$key> for $name {
            const BITS: usize = 0 $(+ core::mem::size_of::<$ty>() * 8)+;
        }
        #[ruda_kernel::dsl::ruda]
        impl $crate::collective::decompose::RudaDecomposer<$key> for $name {
            fn bit(&self, key: $key, index: usize) -> bool {
                let index = usize::cast_from(index);
                let lower = ruda_kernel::dsl::comptime![0 $(+ core::mem::size_of::<$ty>() * 8)+];
                let mut result = false;
                $(
                    let lower = ruda_kernel::dsl::comptime![lower - core::mem::size_of::<$ty>() * 8];
                    if index >= lower && index < lower + ruda_kernel::dsl::comptime![core::mem::size_of::<$ty>() * 8] {
                        result = ((<$ty as $crate::collective::radix::RudaRadixKey>::ordered_bits(key.$field)
                            >> (index - lower) as u64) & 1u64) != 0;
                    }
                )+
                result
            }
        }
    };
}
