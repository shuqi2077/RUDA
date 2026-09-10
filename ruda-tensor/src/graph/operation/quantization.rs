use super::*;


/// Quantization parameters intermediate representation.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationParametersIr {
    /// The scaling factor.
    pub scales: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct QuantizeOpIr {
    pub tensor: TensorIr,
    pub qparams: QuantizationParametersIr,
    pub scheme: QuantScheme,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DequantizeOpIr {
    pub input: TensorIr,
    pub out: TensorIr,
}
