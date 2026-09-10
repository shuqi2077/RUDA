use crate::device::{Device, DeviceId};
use crate::tensor::{BoolDType, BoolStore, DType, FloatDType, IntDType};

use alloc::format;
use alloc::string::String;
use crate::stub::RwLock;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;
use thiserror::Error;

use core::any::TypeId;

#[cfg(feature = "tensor-device-settings-std")]
pub use std::collections::HashMap;
#[cfg(feature = "tensor-device-settings-std")]
use std::sync::{LazyLock, OnceLock};

#[cfg(not(feature = "tensor-device-settings-std"))]
pub use hashbrown::HashMap;
#[cfg(not(feature = "tensor-device-settings-std"))]
use spin::{Lazy as LazyLock, Once as OnceLock};

/// Settings controlling the default data types for a specific device.
///
/// These settings are managed in a global registry that enforces strict initialization semantics:
///
/// 1. Manual Initialization: You can set these once at the start of your program using `set_default_dtypes`.
/// 2. Default Initialization: If an operation (like creating a tensor) occurs before manual initialization,
///    the settings are permanently locked to their default values.
/// 3. Immutability: Once initialized, settings cannot be changed. This ensures consistent behavior across
///    all threads and operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSettings {
    /// Default floating-point data type.
    pub float_dtype: FloatDType,
    /// Default integer data type.
    pub int_dtype: IntDType,
    /// Default bool data type.
    pub bool_dtype: BoolDType,
}

impl DeviceSettings {
    pub fn new(
        float_dtype: impl Into<FloatDType>,
        int_dtype: impl Into<IntDType>,
        bool_dtype: impl Into<BoolDType>,
    ) -> Self {
        Self {
            float_dtype: float_dtype.into(),
            int_dtype: int_dtype.into(),
            bool_dtype: bool_dtype.into(),
        }
    }
}

/// Key for the registry: physical device type + device id
type RegistryKey = (DeviceId, TypeId);

/// Global registry mapping devices to their settings.
///
/// Each value is wrapped in a `OnceLock` to enforce that settings are initialized only once
/// per device.
static REGISTRY: LazyLock<RwLock<HashMap<RegistryKey, Arc<OnceLock<DeviceSettings>>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

pub struct DeviceSettingsRegistry;

impl DeviceSettingsRegistry {
    /// Returns the settings for the given device, inserting the default if absent.
    pub fn get_or_insert<D: Device>(
        device: &D,
        default_fn: impl FnOnce() -> DeviceSettings,
    ) -> DeviceSettings {
        let key = Self::key(device);
        #[cfg(feature = "tensor-device-settings-std")]
        {
            let cached = LOCAL_CACHE.with(|cache| cache.borrow().get(&key).copied());
            if let Some(settings) = cached {
                return settings;
            }

            // Entry does not exist in cache
            let settings = {
                let read = REGISTRY.read().unwrap();
                read.get(&key).cloned()
            }
            .unwrap_or_else(|| {
                let mut map = REGISTRY.write().unwrap();
                Arc::clone(map.entry(key).or_default())
            });

            let settings = *settings.get_or_init(default_fn);

            LOCAL_CACHE.with(|cache| {
                cache.borrow_mut().insert(key, settings);
            });

            settings
        }
        #[cfg(not(feature = "tensor-device-settings-std"))]
        {
            let settings = {
                let read = REGISTRY.read().unwrap();
                read.get(&key).cloned()
            }
            .unwrap_or_else(|| {
                let mut map = REGISTRY.write().unwrap();
                Arc::clone(map.entry(key).or_default())
            });

            settings.call_once(default_fn);
            *settings.get().unwrap()
        }
    }

    /// Initializes the settings for the given device.
    ///
    /// Returns `Err` with the existing settings if already initialized.
    pub fn init<D: Device>(device: &D, settings: DeviceSettings) -> Result<(), DeviceError> {
        let key = Self::key(device);
        let mut map = REGISTRY.write().unwrap();
        let cell = map.entry(key).or_insert_with(|| Arc::new(OnceLock::new()));

        #[cfg(feature = "tensor-device-settings-std")]
        return cell
            .set(settings)
            .map_err(|_| DeviceError::already_initialized(device));

        #[cfg(not(feature = "tensor-device-settings-std"))]
        if cell.get().is_some() {
            Err(DeviceError::already_initialized(device))
        } else {
            cell.call_once(|| settings);
            Ok(())
        }
    }

    /// Returns the device registry key.
    fn key<D: Device>(device: &D) -> RegistryKey {
        (device.to_id(), TypeId::of::<D>())
    }
}

#[cfg(feature = "tensor-device-settings-std")]
std::thread_local! {
    /// Thread-local cache access to initialized device settings is lock-free.
    static LOCAL_CACHE: core::cell::RefCell<HashMap<RegistryKey, DeviceSettings>> =
        core::cell::RefCell::new(HashMap::new());
}

/// Errors that can occur during device-related operations.
///
/// This covers errors related to hardware capability mismatches, such as
/// requesting a data type not supported by the device, and configuration
/// errors like attempting to change a settings in an invalid context.
#[derive(Debug, Error)]
pub enum DeviceError {
    /// Unsupported data type by the device.
    #[error("Device {device} does not support the requested data type {dtype:?}")]
    UnsupportedDType {
        /// The string representation of the device.
        device: String,
        /// The data type that caused the error.
        dtype: DType,
    },
    /// Device settings have already been initialized.
    #[error("Device {device} settings have already been initialized")]
    AlreadyInitialized {
        /// The string representation of the device.
        device: String,
    },
}

impl DeviceError {
    /// Helper to create a [`DeviceError::UnsupportedDType`] from any device.
    pub fn unsupported_dtype<D: Device>(device: &D, dtype: DType) -> Self {
        Self::UnsupportedDType {
            device: format!("{device:?}"),
            dtype,
        }
    }

    /// Helper to create a [`DeviceError::AlreadyInitialized`] from any device.
    pub fn already_initialized<D: Device>(device: &D) -> Self {
        Self::AlreadyInitialized {
            device: format!("{device:?}"),
        }
    }
}

#[cfg(all(test, feature = "tensor-device-settings-std"))]
mod tests {
    use alloc::vec;
    use serial_test::serial;

    use super::*;

    fn clear_registry() {
        REGISTRY.write().unwrap().clear();
    }

    #[derive(Clone, Debug, Default, PartialEq, new)]
    pub struct TestDeviceA {
        index: u16,
    }

    impl Device for TestDeviceA {
        fn from_id(device_id: DeviceId) -> Self {
            Self {
                index: device_id.index_id,
            }
        }

        fn to_id(&self) -> DeviceId {
            DeviceId {
                type_id: 0,
                index_id: self.index,
            }
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, new)]
    pub struct TestDeviceB {
        index: u16,
    }

    impl Device for TestDeviceB {
        fn from_id(device_id: DeviceId) -> Self {
            Self {
                index: device_id.index_id,
            }
        }

        fn to_id(&self) -> DeviceId {
            DeviceId {
                type_id: 0,
                index_id: self.index,
            }
        }
    }

    // Test defaults
    impl DeviceSettings {
        fn defaults() -> Self {
            DeviceSettings::new(FloatDType::F32, IntDType::I32, BoolDType::Native)
        }
    }

    fn get_test_device_settings<D: Device>(device: &D) -> DeviceSettings {
        DeviceSettingsRegistry::get_or_insert(device, DeviceSettings::defaults)
    }

    #[test]
    #[serial]
    fn default_settings_returned_when_uninitialized() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        let s1 = get_test_device_settings(&device);
        let s2 = get_test_device_settings(&device);

        assert_eq!(s1, s2);
        assert_eq!(s1, DeviceSettings::defaults());
    }

    #[test]
    #[serial]
    fn initialized_settings_are_returned() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);
        let settings = DeviceSettings::new(FloatDType::BF16, IntDType::I32, BoolDType::Native);

        DeviceSettingsRegistry::init(&device, settings).unwrap();
        let s1 = get_test_device_settings(&device);
        let s2 = get_test_device_settings(&device);

        assert_eq!(s1, s2);
        assert_eq!(s1, settings);
        assert_eq!(s2, settings);
    }

    #[test]
    #[serial]
    fn settings_are_device_id_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceA::new(1);
        let settings = DeviceSettings::new(FloatDType::F16, IntDType::I64, BoolDType::Native);

        DeviceSettingsRegistry::init(&d1, settings).unwrap();

        let s1 = get_test_device_settings(&d1);
        let s2 = get_test_device_settings(&d2);

        assert_ne!(s1, s2);
        assert_eq!(s1, settings);
        assert_eq!(s2, DeviceSettings::defaults());
    }

    #[test]
    #[serial]
    fn settings_are_device_type_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceB::new(0);
        let settings = DeviceSettings::new(FloatDType::F16, IntDType::I64, BoolDType::Native);

        DeviceSettingsRegistry::init(&d2, settings).unwrap();

        let s1 = get_test_device_settings(&d1);
        let s2 = get_test_device_settings(&d2);

        assert_ne!(s1, s2);
        assert_eq!(s1, DeviceSettings::defaults());
        assert_eq!(s2, settings);
    }

    #[test]
    #[serial]
    fn initialization_after_default_returns_error() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);
        // Settings are set to default on first access, which forces consistency
        let _before = get_test_device_settings(&device);

        let settings = DeviceSettings::new(FloatDType::BF16, IntDType::I64, BoolDType::Native);
        let result = DeviceSettingsRegistry::init(&device, settings);

        assert!(matches!(
            result,
            Err(DeviceError::AlreadyInitialized { .. })
        ));
    }

    #[test]
    #[serial]
    fn second_initialization_returns_error() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);
        let settings = DeviceSettings::new(FloatDType::F16, IntDType::I32, BoolDType::Native);
        DeviceSettingsRegistry::init(&device, settings).unwrap();

        let result = DeviceSettingsRegistry::init(&device, DeviceSettings::defaults());
        assert!(matches!(
            result,
            Err(DeviceError::AlreadyInitialized { .. })
        ));
    }

    #[cfg(feature = "tensor-device-settings-std")]
    #[test]
    #[serial]
    fn initialized_settings_are_global() {
        clear_registry();

        let device = TestDeviceA::new(0);
        let settings = DeviceSettings::new(FloatDType::F16, IntDType::I32, BoolDType::Native);

        DeviceSettingsRegistry::init(&device, settings).unwrap();
        let settings_actual = get_test_device_settings(&device);
        assert_eq!(settings_actual, settings);

        // The other thread will see the initialized settings
        let seen_by_new_thread =
            std::thread::spawn(move || get_test_device_settings(&TestDeviceA::new(0)))
                .join()
                .unwrap();
        assert_eq!(seen_by_new_thread, settings_actual);
    }
}

pub fn select_bool_dtype(default_bool: BoolDType, supports_dtype: impl Fn(DType) -> bool) -> BoolDType {
    let bool_as_dtype = default_bool.into();
    if supports_dtype(bool_as_dtype) {
        default_bool
    } else if !matches!(bool_as_dtype, DType::Bool(BoolStore::U8))
        && supports_dtype(DType::Bool(BoolStore::U8))
    {
        BoolDType::U8
    } else if !matches!(bool_as_dtype, DType::Bool(BoolStore::U32))
        && supports_dtype(DType::Bool(BoolStore::U32))
    {
        BoolDType::U32
    } else if !matches!(bool_as_dtype, DType::Bool(BoolStore::Native))
        && supports_dtype(DType::Bool(BoolStore::Native))
    {
        BoolDType::Native
    } else {
        unreachable!()
    }
}
