pub use ruda_core::device::*;
use ruda_core::tensor::{BoolDType, DType, FloatDType, IntDType};
use crate::Backend;
pub use ruda_core::tensor::device_settings::{DeviceSettings, DeviceError};
use ruda_core::tensor::device_settings::DeviceSettingsRegistry;

#[cfg(feature = "std")]
pub use std::collections::HashMap;
#[cfg(not(feature = "std"))]
pub use hashbrown::HashMap;

/// Device trait for all ruda backend devices.
pub trait DeviceOps: Clone + Default + PartialEq + Send + Sync + core::fmt::Debug + Device {
    /// Returns the [device id](DeviceId).
    fn id(&self) -> DeviceId {
        self.to_id()
    }

    /// Returns the inner device without autodiff enabled.
    ///
    /// For most devices this is a no-op that returns `self`. For autodiff-enabled
    /// devices, this returns the underlying inner device.
    fn inner(&self) -> &Self {
        self
    }
}

/// Get the [`device`'s settings](DeviceSettings).
pub fn get_device_settings<B: Backend>(device: &B::Device) -> DeviceSettings {
    let default_settings = || {
        DeviceSettings::new(
            default_float::<B>(),
            default_int::<B>(),
            default_bool::<B>(device),
        )
    };
    DeviceSettingsRegistry::get_or_insert(device, default_settings)
}

fn default_bool<B: Backend>(device: &B::Device) -> BoolDType {
    // NOTE: this fallback logic is mostly tied to the dispatch backend since we still have associated
    // element types. Once they're removed, we need to have some sort of `DeviceDefaults` trait that provides
    // per-device defaults instead.

    // dtype.into() handles u8/u32 conversion to Bool(..)
    let default_bool: BoolDType = <B::BoolElem as crate::Element>::dtype().into();
    ruda_core::tensor::device_settings::select_bool_dtype(default_bool, |dtype| B::supports_dtype(device, dtype))
}

fn default_float<B: Backend>() -> FloatDType {
    <B::FloatElem as crate::Element>::dtype().into()
}

fn default_int<B: Backend>() -> IntDType {
    <B::IntElem as crate::Element>::dtype().into()
}

fn check_dtype_support<B: Backend>(
    device: &B::Device,
    dtype: impl Into<DType>,
) -> Result<(), DeviceError> {
    let dtype = dtype.into();
    // Default dtypes should have `DTypeUsage::general()`. Types restricted to specialized
    // operations should not be used as default.
    if B::supports_dtype(device, dtype) {
        Ok(())
    } else {
        Err(DeviceError::unsupported_dtype(device, dtype))
    }
}

/// Sets the default data types for the device.
///
/// This updates the device's default data types used for tensor creation.
///
/// Settings can only be initialized once per device. Subsequent calls for
/// the same device return [`DeviceError::AlreadyInitialized`].
///
/// # Note
///
/// Initialization must happen before any tensor creation on the device.
/// The first tensor operation will lock the device to its defaults, causing
/// any subsequent initialization attempt to return [`DeviceError::AlreadyInitialized`].
///
/// # Example
///
/// ```rust, ignore
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     
///     // Update the device settings
///     set_default_dtypes::<B>(&device, DType::F16, DType::I32);
///     
///     // All float tensors created after this will use F16 by default
///     let tensor = Tensor::<B, 2>::zeros([2, 3], &device);
///     // All int tensors created after this will use I32 default
///     let tensor = Tensor::<B, 2, Int>::zeros([2, 3], &device);
/// }
/// ```
pub fn set_default_dtypes<B: Backend>(
    device: &B::Device,
    float_dtype: impl Into<FloatDType>,
    int_dtype: impl Into<IntDType>,
) -> Result<(), DeviceError> {
    let float_dtype = float_dtype.into();
    let int_dtype = int_dtype.into();
    check_dtype_support::<B>(device, float_dtype)?;
    check_dtype_support::<B>(device, int_dtype)?;

    let settings = DeviceSettings::new(float_dtype, int_dtype, default_bool::<B>(device));

    initialize_unchecked(device, settings)?;
    Ok(())
}

/// Sets the default floating-point data type for the device.
///
/// This updates the device's default data types used for tensor creation.
///
/// Settings can only be initialized once per device. Subsequent calls for
/// the same device return [`DeviceError::AlreadyInitialized`].
///
/// # Note
///
/// Initialization must happen before any tensor creation on the device.
/// The first tensor operation will lock the device to its defaults, causing
/// any subsequent initialization attempt to return [`DeviceError::AlreadyInitialized`].
///
/// # Example
///
/// ```rust, ignore
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     
///     // Update the device settings
///     set_default_float_dtype::<B>(&device, DType::F16);
///     
///     // All float tensors created after this will use F16 by default
///     let tensor = Tensor::<B, 2>::zeros([2, 3], &device);
/// }
/// ```
pub fn set_default_float_dtype<B: Backend>(
    device: &B::Device,
    dtype: impl Into<FloatDType>,
) -> Result<(), DeviceError> {
    let dtype = dtype.into();
    check_dtype_support::<B>(device, dtype)?;

    let settings = DeviceSettings::new(dtype, default_int::<B>(), default_bool::<B>(device));

    initialize_unchecked(device, settings)?;
    Ok(())
}

/// Sets the default integer data type for the device.
///
/// This updates the device's default data types used for tensor creation.
///
/// Settings can only be initialized once per device. Subsequent calls for
/// the same device return [`DeviceError::AlreadyInitialized`].
///
/// # Note
///
/// Initialization must happen before any tensor creation on the device.
/// The first tensor operation will lock the device to its defaults, causing
/// any subsequent initialization attempt to return [`DeviceError::AlreadyInitialized`].
///
/// # Example
///
/// ```rust, ignore
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     
///     // Update the device settings
///     set_default_int_dtype::<B>(&device, DType::I32);
///     
///     // All int tensors created after this will use I32 default
///     let tensor = Tensor::<B, 2, Int>::zeros([2, 3], &device);
/// }
/// ```
pub fn set_default_int_dtype<B: Backend>(
    device: &B::Device,
    dtype: impl Into<IntDType>,
) -> Result<(), DeviceError> {
    let dtype = dtype.into();
    check_dtype_support::<B>(device, dtype)?;

    let settings = DeviceSettings::new(default_float::<B>(), dtype, default_bool::<B>(device));

    initialize_unchecked(device, settings)?;
    Ok(())
}

// Unchecked dtypes
fn initialize_unchecked<D: DeviceOps>(
    device: &D,
    settings: DeviceSettings,
) -> Result<(), DeviceError> {
    DeviceSettingsRegistry::init(device, settings)
}

mod adapters;
