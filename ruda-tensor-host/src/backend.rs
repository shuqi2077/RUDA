use alloc::string::String;
use core::marker::PhantomData;

use ruda_tensor::{Backend, BackendTypes, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps};
use ruda_tensor::graph::{BackendIr, HandleKind, TensorHandle};
use ruda_core::device::Device;

use crate::qtensor::HostQTensor;
use crate::tensor::HostTensor;

pub use rurand_host::HostRng as HostRng;

/// CPU device for the Host backend.
///
/// Unit struct since there's only one CPU device.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HostDevice;

impl Device for HostDevice {
    fn to_id(&self) -> DeviceId {
        DeviceId::new(0, 0)
    }

    fn from_id(_id: DeviceId) -> Self {
        Self
    }
}

impl DeviceOps for HostDevice {}

impl core::fmt::Display for HostDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Cpu")
    }
}

impl core::fmt::Debug for HostDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

/// The Host backend, a fast, portable CPU backend for Ruda.
///
/// The `E` and `I` type parameters exist purely to match the shape of other Ruda
/// backends (e.g. `NdArray<E, I, Q>`) so `Host` slots into `ruda-dispatch`'s
/// generic dispatch macros. The body of `Host` uses runtime `DType` dispatch, so
/// both parameters are phantom and unused at runtime.
///
/// # Limitations of the phantom generics
///
/// The `Backend` impl is provided only for the default instantiation
/// `Host<f32, i32>`. Writing `Host` (with no arguments) resolves to the default
/// and works exactly as before. Writing `Host<f64, i64>` or any other non-default
/// combination is a valid Rust type but will not satisfy trait bounds requiring
/// `Backend`, producing errors like:
///
/// ```text
/// the trait bound `Host<f64, i64>: Backend` is not satisfied
/// ```
///
/// This is a deliberate compromise for the initial migration: making `Host`
/// generic over element types at the trait-impl level is a follow-up that would
/// require rewriting all `impl FooOps<Host> for Host` blocks plus internal
/// `Host::method()` calls (tracked in
/// [#4762](https://github.com/shuqi2077/RUDA/blob/main/THIRD_PARTY_NOTICES.md)). Until then, treat
/// the generic parameters as opaque shape placeholders; real element-type
/// selection happens at runtime via `DType`.
///
/// The bound is locked in by a compile-fail doctest so that if someone later
/// makes the `Backend` impl generic over `E`/`I`, this documentation gets
/// flagged as out of date:
///
/// ```compile_fail
/// use ruda_tensor::Backend;
/// use ruda_tensor_host::Host;
/// fn requires_backend<B: Backend>() {}
/// requires_backend::<Host<f64, i64>>();
/// ```
#[derive(Clone, Copy, Default)]
pub struct Host<E = f32, I = i32> {
    _e: PhantomData<E>,
    _i: PhantomData<I>,
}

impl<E: core::fmt::Debug, I: core::fmt::Debug> core::fmt::Debug for Host<E, I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Flex")
            .field("_e", &self._e)
            .field("_i", &self._i)
            .finish()
    }
}

impl BackendTypes for Host {
    type Device = HostDevice;

    type FloatTensorPrimitive = HostTensor;
    /// Default float element type. Determines the dtype for `.float()` conversions and
    /// `Tensor::from_data` when no explicit dtype is provided.
    /// Prefer explicit dtypes via `(&device, DType::F32)`.
    type FloatElem = f32;

    type IntTensorPrimitive = HostTensor;
    /// Default int element type. Determines the dtype for `.int()` conversions and
    /// `Tensor::from_data` when no explicit dtype is provided.
    /// Set to i32 to match ruda's ecosystem default (test suite, record settings, ruda-tensor-remote).
    /// Prefer explicit dtypes via `(&device, DType::I32)`.
    type IntElem = i32;

    type BoolTensorPrimitive = HostTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = HostQTensor;
}

impl Backend for Host {
    fn name(_device: &Self::Device) -> String {
        "flex".into()
    }

    fn seed(_device: &Self::Device, seed: u64) {
        rurand_host::seed(seed)
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        match dtype {
            // Full support for standard types
            DType::F64 | DType::F32 | DType::F16 | DType::BF16 => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            // Bool storage: flex stores bools as 1 byte per element, so Native and
            // U8 are both supported (they share the same layout, only the tag
            // differs). Bool(U32) would require 4-byte-per-element storage
            // throughout the backend and is not yet implemented.
            DType::Bool(ruda_tensor::BoolStore::Native | ruda_tensor::BoolStore::U8) => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::Bool(ruda_tensor::BoolStore::U32) => DTypeUsageSet::empty(),
            // Quantized types: storage only for now
            DType::QFloat(_) => DTypeUsage::Storage.into(),
            _ => DTypeUsageSet::empty(),
        }
    }
}

impl BackendIr for Host {
    type Handle = HandleKind<Self>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> HostTensor {
        match handle.handle {
            HandleKind::Float(t) => t,
            _ => panic!("Expected float handle, got {}", handle.handle.name()),
        }
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> HostTensor {
        match handle.handle {
            HandleKind::Int(t) => t,
            _ => panic!("Expected int handle, got {}", handle.handle.name()),
        }
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> HostTensor {
        match handle.handle {
            HandleKind::Bool(t) => t,
            _ => panic!("Expected bool handle, got {}", handle.handle.name()),
        }
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> HostQTensor {
        match handle.handle {
            HandleKind::Quantized(t) => t,
            _ => panic!("Expected quantized handle, got {}", handle.handle.name()),
        }
    }

    fn float_tensor_handle(tensor: HostTensor) -> Self::Handle {
        HandleKind::Float(tensor)
    }

    fn int_tensor_handle(tensor: HostTensor) -> Self::Handle {
        HandleKind::Int(tensor)
    }

    fn bool_tensor_handle(tensor: HostTensor) -> Self::Handle {
        HandleKind::Bool(tensor)
    }

    fn quantized_tensor_handle(tensor: HostQTensor) -> Self::Handle {
        HandleKind::Quantized(tensor)
    }
}

// Ops traits are implemented in the ops module

#[cfg(test)]
mod tests {
    use ruda_tensor::{Backend, DType};
    use ruda_tensor::BoolStore;

    use super::*;

    #[test]
    fn supports_bool_native() {
        let device = HostDevice;
        assert!(Host::supports_dtype(
            &device,
            DType::Bool(BoolStore::Native)
        ));
    }

    #[test]
    fn supports_bool_u8() {
        let device = HostDevice;
        assert!(Host::supports_dtype(&device, DType::Bool(BoolStore::U8)));
    }

    #[test]
    fn does_not_support_bool_u32() {
        let device = HostDevice;
        assert!(
            !Host::supports_dtype(&device, DType::Bool(BoolStore::U32)),
            "Bool(U32) should not be supported: flex stores bools as 1 byte per element"
        );
    }

    #[test]
    fn bool_empty_preserves_native_dtype() {
        use ruda_tensor::ops::BoolTensorOps;
        let shape = ruda_tensor::Shape::from(alloc::vec![3]);
        let t = Host::bool_empty(shape, &HostDevice, ruda_tensor::BoolDType::Native);
        assert_eq!(t.dtype(), DType::Bool(BoolStore::Native));
    }

    #[test]
    fn bool_empty_preserves_u8_dtype() {
        use ruda_tensor::ops::BoolTensorOps;
        let shape = ruda_tensor::Shape::from(alloc::vec![3]);
        let t = Host::bool_empty(shape, &HostDevice, ruda_tensor::BoolDType::U8);
        assert_eq!(t.dtype(), DType::Bool(BoolStore::U8));
    }

    #[test]
    fn device_prints_as_cpu() {
        use alloc::format;
        assert_eq!(format!("{:?}", HostDevice), "Cpu");
        assert_eq!(format!("{}", HostDevice), "Cpu");
    }

    #[test]
    fn comparison_preserves_out_dtype_native() {
        let lhs = HostTensor::from_data(ruda_tensor::TensorData::from([1.0f32, 2.0, 3.0]));
        let rhs = HostTensor::from_data(ruda_tensor::TensorData::from([2.0f32, 2.0, 1.0]));
        let result = crate::ops::comparison::greater(lhs, rhs, ruda_tensor::BoolDType::Native);
        assert_eq!(result.dtype(), DType::Bool(BoolStore::Native));
    }

    #[test]
    fn comparison_preserves_out_dtype_u8() {
        let lhs = HostTensor::from_data(ruda_tensor::TensorData::from([1.0f32, 2.0, 3.0]));
        let rhs = HostTensor::from_data(ruda_tensor::TensorData::from([2.0f32, 2.0, 1.0]));
        let result = crate::ops::comparison::greater(lhs, rhs, ruda_tensor::BoolDType::U8);
        assert_eq!(result.dtype(), DType::Bool(BoolStore::U8));
    }

    #[test]
    #[should_panic(expected = "Bool(U32)")]
    fn comparison_u32_panics() {
        let lhs = HostTensor::from_data(ruda_tensor::TensorData::from([1.0f32, 2.0]));
        let rhs = HostTensor::from_data(ruda_tensor::TensorData::from([2.0f32, 1.0]));
        let _ = crate::ops::comparison::greater(lhs, rhs, ruda_tensor::BoolDType::U32);
    }

    #[test]
    fn bool_not_preserves_u8_dtype() {
        use ruda_tensor::ops::BoolTensorOps;
        // Construct a Bool(U8) tensor directly to verify bool_not preserves
        // the dtype tag across the op. from_data would produce Bool(Native),
        // so we use make_bool_tensor to get the U8 tag.
        let t_u8 = crate::ops::comparison::make_bool_tensor(
            alloc::vec![1, 0, 1],
            ruda_tensor::Shape::from(alloc::vec![3]),
            ruda_tensor::BoolDType::U8,
        );
        let result = Host::bool_not(t_u8);
        assert_eq!(result.dtype(), DType::Bool(BoolStore::U8));
        let data: &[u8] = result.bytes();
        assert_eq!(&data[..3], &[0, 1, 0]);
    }
}
