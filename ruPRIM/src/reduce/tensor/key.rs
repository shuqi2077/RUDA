use ruda_core::tensor::DType;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::AutotuneKey;
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of sum versions
pub struct SumAutotuneKey {
    /// The type of the tensor
    dtype: DType,
    /// The anchored length of the tensor
    #[autotune(anchor)]
    length: usize,
}


#[cfg(feature = "tensor-reduce-autotune")]
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub(super) enum SumTuneKey {
    Sum(SumAutotuneKey),
}

#[cfg(feature = "tensor-reduce-autotune")]
impl core::fmt::Display for SumTuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Sum(key) => core::fmt::Debug::fmt(key, f),
        }
    }
}

#[cfg(feature = "tensor-reduce-autotune")]
impl ruda::runtime::tune::AutotuneKey for SumTuneKey {}
