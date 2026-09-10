use super::{RuntimeError, RuntimeErrorKind, TensorDType};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum BackendKind { Nvidia, Amd, Host, Custom(String) }

/// Backend is part of identity: CUDA:0 and HIP:0 must never alias.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceKey { pub backend: BackendKind, pub ordinal: u32 }

/// CUDA API version returned by cuDriverGetVersion, NOT a release like 580.65.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaApiVersion(u32);
impl CudaApiVersion {
    pub fn from_encoded(value: u32) -> Result<Self, RuntimeError> {
        if value < 1000 || value % 10 != 0 {
            return Err(RuntimeError::invalid("invalid CUDA API version encoding"));
        }
        Ok(Self(value))
    }
    pub const fn encoded(self) -> u32 { self.0 }
    pub const fn major(self) -> u32 { self.0 / 1000 }
    pub const fn minor(self) -> u32 { (self.0 % 1000) / 10 }
}

#[derive(Debug, Clone)]
pub struct CudaRequirements {
    pub minimum_api: CudaApiVersion,
    /// Minimum driver API proven to load this build's PTX. None means unknown,
    /// not "all drivers are compatible". The API version is not a PTX version.
    pub minimum_ptx_api: Option<CudaApiVersion>,
    pub uses_ptx: bool,
}
impl CudaRequirements {
    /// Conservative preflight only. Successful module load / feature probing is
    /// still required; driver API numbers alone cannot prove binary support.
    pub fn check(&self, driver_api: CudaApiVersion) -> Result<(), RuntimeError> {
        if driver_api < self.minimum_api {
            return Err(RuntimeError::new(RuntimeErrorKind::DriverIncompatible,
                format!("driver CUDA API {} is below required {}", driver_api.encoded(), self.minimum_api.encoded())));
        }
        if self.uses_ptx {
            let minimum = self.minimum_ptx_api.ok_or_else(|| RuntimeError::new(
                RuntimeErrorKind::DriverIncompatible,
                "PTX minimum driver requirement is unknown; probe module load or provide a verified requirement",
            ))?;
            if driver_api < minimum {
                return Err(RuntimeError::new(RuntimeErrorKind::DriverIncompatible,
                    "driver is below this PTX build's verified requirement"));
            }
        }
        Ok(())
    }
}

/// Full version/build identity for performance decisions; changing any field
/// naturally invalidates a cached decision. No persisted global hardware cache.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceFingerprint {
    pub device: DeviceKey,
    pub architecture: String,
    pub driver: String,
    pub runtime: String,
    pub compiler_build: String,
}

#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub identity: DeviceFingerprint,
    /// Only list dtypes whose arithmetic AND storage are implemented.
    pub arithmetic_dtypes: Vec<TensorDType>,
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    pub subgroup_reductions: bool,
    pub packed_subgroups: bool,
    pub max_threads_x: u32,
    pub max_threads_per_block: u32,
    pub max_grid_x: u32,
}
impl DeviceCapabilities {
    pub fn supports_dtype(&self, dtype: TensorDType) -> bool {
        self.arithmetic_dtypes.contains(&dtype)
    }
    /// One full, fixed-size subgroup avoids partial/multiple-wave reductions.
    pub fn reduction_width(&self) -> Option<u32> {
        fixed_reduction_width(self.subgroup_min, self.subgroup_max,
            self.subgroup_reductions, self.packed_subgroups,
            self.max_threads_x, self.max_threads_per_block)
    }
}

/// Shared by the host planner and the actual packed-kernel launch path.
pub fn fixed_reduction_width(
    minimum: u32, maximum: u32, reductions: bool, packed: bool,
    max_threads_x: u32, max_threads_per_block: u32,
) -> Option<u32> {
    (minimum != 0 && minimum == maximum && minimum.is_power_of_two()
        && reductions && packed && minimum <= max_threads_x
        && minimum <= max_threads_per_block).then_some(minimum)
}

/// The dedicated decode kernel holds two values per lane. Do not generalize
/// its launch to an arbitrary head dimension without changing those loads.
pub fn supports_two_lane_decode(
    subgroup: Option<u32>, head_dimension: usize, query_sequence: usize,
) -> bool {
    query_sequence == 1 && subgroup.and_then(|n| n.checked_mul(2))
        .is_some_and(|n| head_dimension == n as usize)
}
