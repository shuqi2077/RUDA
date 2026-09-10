use crate::DeviceBackend;
pub use ruda_driver_cuda::CudaDevice;
use ruda_driver_cuda::CudaRuntime;

#[cfg(not(feature = "cuda-fusion"))]
pub type Cuda<F = f32, I = i32> = DeviceBackend<CudaRuntime, F, I, u8>;

#[cfg(feature = "cuda-fusion")]
pub type Cuda<F = f32, I = i32> = ruda_fusion::Fusion<DeviceBackend<CudaRuntime, F, I, u8>>;

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;
    use ruda_tensor::{Backend, BoolStore, DType, QTensorPrimitive};
    use ruda_kernel::tensor::RudaTensor;

    #[test]
    fn should_support_dtypes() {
        type B = Cuda;
        let device = Default::default();

        assert!(B::supports_dtype(&device, DType::F32));
        assert!(B::supports_dtype(&device, DType::Flex32));
        assert!(B::supports_dtype(&device, DType::F16));
        assert!(B::supports_dtype(&device, DType::BF16));
        assert!(B::supports_dtype(&device, DType::I64));
        assert!(B::supports_dtype(&device, DType::I32));
        assert!(B::supports_dtype(&device, DType::I16));
        assert!(B::supports_dtype(&device, DType::I8));
        assert!(B::supports_dtype(&device, DType::U64));
        assert!(B::supports_dtype(&device, DType::U32));
        assert!(B::supports_dtype(&device, DType::U16));
        assert!(B::supports_dtype(&device, DType::U8));
        assert!(B::supports_dtype(&device, DType::Bool(BoolStore::Native)));
        assert!(B::supports_dtype(
            &device,
            DType::QFloat(RudaTensor::<CudaRuntime>::default_scheme())
        ));

        // Currently not registered in supported types
        assert!(!B::supports_dtype(&device, DType::F64));
    }
}
