use super::*;

/// Custom operation in fusion stream, declaring its inputs and outputs.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct CustomOpIr {
    /// Unique identifier of the operation.
    pub id: String,
    /// Input tensors used in the custom operation.
    pub inputs: Vec<TensorIr>,
    /// Output tensors used in the custom operation.
    pub outputs: Vec<TensorIr>,
}

/// Swap dim operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct SwapDimsOpIr {
    /// Input tensor intermediate representation.
    pub input: TensorIr,
    /// Output tensor intermediate representation.
    pub out: TensorIr,
    /// The first dim to swap.
    pub dim1: usize,
    /// The second dim to swap.
    pub dim2: usize,
}

/// Permute operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct PermuteOpIr {
    /// Input tensor intermediate representation.
    pub input: TensorIr,
    /// Output tensor intermediate representation.
    pub out: TensorIr,
    /// The new order of the dimensions.
    pub axes: Vec<usize>,
}

/// Shape operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct ShapeOpIr {
    /// Input tensor intermediate representation.
    pub input: TensorIr,
    /// Output tensor intermediate representation with the new shape.
    pub out: TensorIr,
}

/// Unfold operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct UnfoldOpIr {
    /// Input tensor intermediate representation.
    pub input: TensorIr,
    /// Output tensor intermediate representation.
    pub out: TensorIr,

    /// The selected dim.
    pub dim: usize,
    /// The window size.
    pub size: usize,
    /// The window step along dim.
    pub step: usize,
}

/// Flip operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct FlipOpIr {
    /// Input tensor intermediate representation.
    pub input: TensorIr,
    /// Output tensor intermediate representation.
    pub out: TensorIr,
    /// The dimensions to flip.
    pub axes: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct RandomOpIr {
    pub out: TensorIr,
    pub distribution: Distribution,
}

/// Creation operation intermediate representation.
/// As opposed to [InitOperationIr], creation operations are lazy initialized.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct CreationOpIr {
    /// Output tensor intermediate representation.
    pub out: TensorIr,
}

/// Full operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct FullOpIr {
    /// Output tensor intermediate representation.
    pub out: TensorIr,
    /// Fill value.
    pub value: ScalarIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
/// Declares a tensor has been initialized.
///
/// It is necessary to register for proper orphan detection and avoid memory leak.
pub struct InitOperationIr {
    /// The initialized tensor.
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct BinaryOpIr {
    pub lhs: TensorIr,
    pub rhs: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct UnaryOpIr {
    pub input: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarOpIr {
    pub lhs: TensorIr,
    // TODO: Make that an enum with `Value` and `Id` variants for relative/global
    // conversion.
    pub rhs: ScalarIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Hash)]
#[allow(missing_docs)]
pub struct ReduceOpIr {
    pub input: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Hash)]
#[allow(missing_docs)]
pub struct ReduceDimOpIr {
    pub input: TensorIr,
    pub out: TensorIr,
    pub axis: usize,
    pub accumulator_len: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct CastOpIr {
    pub input: TensorIr,
    pub out: TensorIr,
}

/// IR for operations that operate along a dimension without reducing it.
/// Unlike `ReduceDimOpIr`, the output shape is the same as the input shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Hash)]
#[allow(missing_docs)]
pub struct DimOpIr {
    pub input: TensorIr,
    pub out: TensorIr,
    pub axis: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct GatherOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub indices: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScatterOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub indices: TensorIr,
    pub value: TensorIr,
    pub update: IndexingUpdateOp,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScatterNdOpIr {
    pub data: TensorIr,
    pub indices: TensorIr,
    pub values: TensorIr,
    pub reduction: IndexingUpdateOp,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct GatherNdOpIr {
    pub data: TensorIr,
    pub indices: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SelectOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub indices: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SelectAssignOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub indices: TensorIr,
    pub value: TensorIr,
    pub update: IndexingUpdateOp,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SliceOpIr {
    pub tensor: TensorIr,
    pub ranges: Vec<Slice>,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SliceAssignOpIr {
    pub tensor: TensorIr,
    pub ranges: Vec<crate::Slice>,
    pub value: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaskWhereOpIr {
    pub tensor: TensorIr,
    pub mask: TensorIr,
    pub value: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaskFillOpIr {
    pub tensor: TensorIr,
    pub mask: TensorIr,
    pub value: ScalarIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ClampOpIr {
    pub tensor: TensorIr,
    pub min: ScalarIr,
    pub max: ScalarIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct RepeatDimOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub times: usize,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct CatOpIr {
    pub tensors: Vec<TensorIr>,
    pub dim: usize,
    pub out: TensorIr,
}

#[cfg(feature = "graph-distributed")]
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AllReduceOpIr {
    pub tensor: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ReduceDimWithIndicesOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub out: TensorIr,
    pub out_indices: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct EmbeddingOpIr {
    pub weights: TensorIr,
    pub indices: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct EmbeddingBackwardOpIr {
    pub weights: TensorIr,
    pub out_grad: TensorIr,
    pub indices: TensorIr,
    pub out: TensorIr,
}
