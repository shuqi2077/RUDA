use alloc::vec::Vec;

use ruda_core::tensor::{DType, QTensorPrimitive, TensorMetadata, quantization::{QuantStore, params_shape}};
use ruda_core::tensor::{QuantScheme, Shape};

use ruda_core::tensor::host::HostTensor;

/// Quantized tensor for the Flex backend.
///
/// Stores quantized i8 values in the tensor and keeps scales separately
/// for efficient dequantization without reparsing bytes.
#[derive(Clone)]
pub struct HostQTensor {
    /// The underlying quantized data (stored as i8).
    pub(crate) tensor: HostTensor,
    /// Quantization scheme.
    pub(crate) scheme: QuantScheme,
    /// Per-tensor or per-block scale factors.
    pub(crate) scales: Vec<f32>,
}

impl HostQTensor {
    /// Create a new quantized tensor.
    ///
    /// The tensor must store i8 data and scales must match the quantization parameter shape.
    pub fn new(tensor: HostTensor, scheme: QuantScheme, scales: Vec<f32>) -> Self {
        assert_eq!(
            tensor.dtype(),
            DType::I8,
            "quantized tensor must store i8 data, got {:?}",
            tensor.dtype()
        );
        assert_eq!(
            scales.len(),
            params_shape(&tensor.shape(), scheme.level).num_elements(),
            "quantized scale count must match the parameter shape"
        );
        Self {
            tensor,
            scheme,
            scales,
        }
    }

    /// Get the underlying tensor.
    pub fn tensor(&self) -> &HostTensor {
        &self.tensor
    }

    /// Get the quantization scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
}

impl QTensorPrimitive for HostQTensor {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }

    fn default_scheme() -> QuantScheme {
        QuantScheme::default().with_store(QuantStore::Native)
    }
}

impl TensorMetadata for HostQTensor {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.tensor.shape()
    }

    fn rank(&self) -> usize {
        self.tensor.rank()
    }
}

impl core::fmt::Debug for HostQTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FlexQTensor")
            .field("tensor", &self.tensor)
            .field("scheme", &self.scheme)
            .field("scales", &self.scales)
            .finish()
    }
}
