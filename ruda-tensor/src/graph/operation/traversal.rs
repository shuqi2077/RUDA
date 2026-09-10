use super::*;


impl CustomOpIr {
    /// Create a new custom operation intermediate representation.
    pub fn new(id: &'static str, inputs: &[TensorIr], outputs: &[TensorIr]) -> Self {
        Self {
            id: id.to_owned(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        }
    }

    /// Cast the intermediate representation, and get the in and output tensors.
    pub fn as_fixed<const N_IN: usize, const N_OUT: usize>(
        &self,
    ) -> (&[TensorIr; N_IN], &[TensorIr; N_OUT]) {
        (
            self.inputs.as_slice().try_into().expect(
                "Wrong number of inputs expected (expected {D}, is {}), check your implementation",
            ),
            self.outputs.as_slice().try_into().expect(
                "Wrong number of outputs expected (expected {D}, is {}), check your implementation",
            ),
        )
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        Box::new(self.inputs.iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        Box::new(self.outputs.iter())
    }
}

#[allow(missing_docs)]
impl RfftOpIr {
    pub fn create<F>(signal: TensorIr, dim: usize, n: Option<usize>, mut new_id: F) -> Self
    where
        F: FnMut() -> crate::graph::TensorId,
    {
        // `n` is required to be a power of two at the public API boundary, so
        // the output has `n / 2 + 1` bins (matching scipy/torch for pow2 n).
        let mut shape = signal.shape.clone();
        let fft_len = n.unwrap_or(shape[dim]);
        shape[dim] = fft_len / 2 + 1;
        let dtype = signal.dtype;

        Self {
            signal,
            dim,
            n,
            out_re: TensorIr::uninit(new_id(), shape.clone(), dtype),
            out_im: TensorIr::uninit(new_id(), shape, dtype),
        }
    }
}

#[allow(missing_docs)]
impl IRfftOpIr {
    pub fn create<F>(
        input_re: TensorIr,
        input_im: TensorIr,
        dim: usize,
        n: Option<usize>,
        mut new_id: F,
    ) -> Self
    where
        F: FnMut() -> crate::graph::TensorId,
    {
        debug_assert!(
            input_re.shape[dim] >= 1,
            "IRfftOpIr: input spectrum dimension must be >= 1"
        );
        debug_assert!(
            !matches!(n, Some(0)),
            "IRfftOpIr: n must be >= 1 when specified"
        );
        let mut shape = input_re.shape.clone();
        shape[dim] = n.unwrap_or((shape[dim] - 1) * 2);
        let dtype = input_re.dtype;

        Self {
            input_re,
            input_im,
            dim,
            n,
            out_signal: TensorIr::uninit(new_id(), shape, dtype),
        }
    }
}

impl OperationIr {
    /// Get all input [tensors](TensorIr) involved with the current operation.
    pub fn inputs(&self) -> impl Iterator<Item = &TensorIr> {
        match self {
            OperationIr::BaseFloat(repr) => repr.inputs(),
            OperationIr::BaseInt(repr) => repr.inputs(),
            OperationIr::BaseBool(repr) => repr.inputs(),
            OperationIr::NumericFloat(_dtype, repr) => repr.inputs(),
            OperationIr::NumericInt(_dtype, repr) => repr.inputs(),
            OperationIr::Bool(repr) => repr.inputs(),
            OperationIr::Int(repr) => repr.inputs(),
            OperationIr::Float(_dtype, repr) => repr.inputs(),
            OperationIr::Module(repr) => repr.inputs(),
            OperationIr::Init(repr) => repr.inputs(),
            OperationIr::Custom(repr) => repr.inputs(),
            OperationIr::Drop(repr) => Box::new([repr].into_iter()),
            #[cfg(feature = "graph-distributed")]
            OperationIr::Distributed(repr) => repr.inputs(),
        }
    }

    /// Get all output [tensors](TensorIr) involved with the current operation.
    pub fn outputs(&self) -> impl Iterator<Item = &TensorIr> {
        match self {
            OperationIr::BaseFloat(repr) => repr.outputs(),
            OperationIr::BaseInt(repr) => repr.outputs(),
            OperationIr::BaseBool(repr) => repr.outputs(),
            OperationIr::NumericFloat(_dtype, repr) => repr.outputs(),
            OperationIr::NumericInt(_dtype, repr) => repr.outputs(),
            OperationIr::Bool(repr) => repr.outputs(),
            OperationIr::Int(repr) => repr.outputs(),
            OperationIr::Float(_dtype, repr) => repr.outputs(),
            OperationIr::Module(repr) => repr.outputs(),
            OperationIr::Init(repr) => repr.outputs(),
            OperationIr::Custom(repr) => repr.outputs(),
            OperationIr::Drop(_repr) => Box::new([].into_iter()),
            #[cfg(feature = "graph-distributed")]
            OperationIr::Distributed(repr) => repr.outputs(),
        }
    }

    /// Get all [tensor](TensorIr) involved with the current operation.
    pub fn nodes(&self) -> Vec<&TensorIr> {
        self.inputs().chain(self.outputs()).collect()
    }

    /// Set the given nodes that are [read write](super::TensorStatus::ReadWrite) to
    /// [read only](super::TensorStatus::ReadOnly) in the current operation.
    ///
    /// Returns the tensor that were updated with their original representation.
    pub fn mark_read_only(&mut self, nodes: &[TensorId]) -> Vec<TensorIr> {
        match self {
            OperationIr::BaseFloat(repr) => repr.mark_read_only(nodes),
            OperationIr::BaseInt(repr) => repr.mark_read_only(nodes),
            OperationIr::BaseBool(repr) => repr.mark_read_only(nodes),
            OperationIr::NumericFloat(_dtype, repr) => repr.mark_read_only(nodes),
            OperationIr::NumericInt(_dtype, repr) => repr.mark_read_only(nodes),
            OperationIr::Bool(repr) => repr.mark_read_only(nodes),
            OperationIr::Int(repr) => repr.mark_read_only(nodes),
            OperationIr::Float(_dtype, repr) => repr.mark_read_only(nodes),
            OperationIr::Module(repr) => repr.mark_read_only(nodes),
            OperationIr::Init(_) => Vec::new(),
            OperationIr::Drop(repr) => {
                let mut output = Vec::new();
                repr.mark_read_only(nodes, &mut output);
                output
            }
            OperationIr::Custom(repr) => {
                let mut output = Vec::new();

                for input in repr.inputs.iter_mut() {
                    input.mark_read_only(nodes, &mut output);
                }

                output
            }
            #[cfg(feature = "graph-distributed")]
            OperationIr::Distributed(repr) => repr.mark_read_only(nodes),
        }
    }
}

impl BaseOperationIr {
    fn inputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            BaseOperationIr::Reshape(repr) => Box::new([&repr.input].into_iter()),
            BaseOperationIr::SwapDims(repr) => Box::new([&repr.input].into_iter()),
            BaseOperationIr::Permute(repr) => Box::new([&repr.input].into_iter()),
            BaseOperationIr::Expand(repr) => Box::new([&repr.input].into_iter()),
            BaseOperationIr::Flip(repr) => Box::new([&repr.input].into_iter()),
            BaseOperationIr::Slice(repr) => Box::new([&repr.tensor].into_iter()),
            BaseOperationIr::SliceAssign(repr) => Box::new([&repr.tensor, &repr.value].into_iter()),
            BaseOperationIr::Gather(repr) => Box::new([&repr.tensor, &repr.indices].into_iter()),
            BaseOperationIr::Scatter(repr) => {
                Box::new([&repr.tensor, &repr.indices, &repr.value].into_iter())
            }
            BaseOperationIr::ScatterNd(repr) => {
                Box::new([&repr.data, &repr.indices, &repr.values].into_iter())
            }
            BaseOperationIr::GatherNd(repr) => Box::new([&repr.data, &repr.indices].into_iter()),
            BaseOperationIr::Select(repr) => Box::new([&repr.tensor, &repr.indices].into_iter()),
            BaseOperationIr::SelectAssign(repr) => {
                Box::new([&repr.tensor, &repr.indices, &repr.value].into_iter())
            }
            BaseOperationIr::MaskWhere(repr) => {
                Box::new([&repr.tensor, &repr.mask, &repr.value].into_iter())
            }
            BaseOperationIr::MaskFill(repr) => Box::new([&repr.tensor, &repr.mask].into_iter()),
            BaseOperationIr::Equal(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            BaseOperationIr::EqualElem(repr) => Box::new([&repr.lhs].into_iter()),
            BaseOperationIr::RepeatDim(repr) => Box::new([&repr.tensor].into_iter()),
            BaseOperationIr::Cat(repr) => Box::new(repr.tensors.iter()),
            BaseOperationIr::Cast(repr) => Box::new([&repr.input].into_iter()),
            BaseOperationIr::Unfold(repr) => Box::new([&repr.input].into_iter()),
            BaseOperationIr::Empty(_repr) => Box::new([].into_iter()),
            BaseOperationIr::Ones(_repr) => Box::new([].into_iter()),
            BaseOperationIr::Zeros(_repr) => Box::new([].into_iter()),
        }
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            BaseOperationIr::Reshape(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::SwapDims(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Permute(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Expand(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Flip(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Slice(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::SliceAssign(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Gather(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Scatter(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::ScatterNd(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::GatherNd(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Select(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::SelectAssign(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::MaskWhere(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::MaskFill(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Equal(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::EqualElem(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::RepeatDim(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Cat(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Cast(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Unfold(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Empty(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Ones(repr) => Box::new([&repr.out].into_iter()),
            BaseOperationIr::Zeros(repr) => Box::new([&repr.out].into_iter()),
        }
    }

    fn mark_read_only(&mut self, nodes: &[TensorId]) -> Vec<TensorIr> {
        let mut output = Vec::new();

        match self {
            BaseOperationIr::Reshape(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::SwapDims(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::Permute(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }

            BaseOperationIr::Expand(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }

            BaseOperationIr::Flip(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::Slice(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::SliceAssign(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
                repr.value.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::Gather(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
                repr.indices.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::Scatter(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
                repr.indices.mark_read_only(nodes, &mut output);
                repr.value.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::ScatterNd(repr) => {
                repr.data.mark_read_only(nodes, &mut output);
                repr.indices.mark_read_only(nodes, &mut output);
                repr.values.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::GatherNd(repr) => {
                repr.data.mark_read_only(nodes, &mut output);
                repr.indices.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::Select(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
                repr.indices.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::SelectAssign(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
                repr.indices.mark_read_only(nodes, &mut output);
                repr.value.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::MaskWhere(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
                repr.mask.mark_read_only(nodes, &mut output);
                repr.value.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::MaskFill(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
                repr.mask.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::Equal(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::EqualElem(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::RepeatDim(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::Cat(repr) => {
                for t in repr.tensors.iter_mut() {
                    t.mark_read_only(nodes, &mut output);
                }
            }
            BaseOperationIr::Cast(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::Unfold(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            BaseOperationIr::Empty(_) => {}
            BaseOperationIr::Zeros(_) => {}
            BaseOperationIr::Ones(_) => {}
        };

        output
    }
}

impl NumericOperationIr {
    fn inputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            NumericOperationIr::Add(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::AddScalar(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::Sub(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::SubScalar(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::Mul(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::MulScalar(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::Div(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::DivScalar(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::Rem(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::RemScalar(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::GreaterElem(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::GreaterEqualElem(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::LowerElem(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::LowerEqualElem(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::Greater(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::GreaterEqual(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::Lower(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::LowerEqual(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::ArgMax(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::ArgTopK(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::TopK(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::ArgMin(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::Clamp(repr) => Box::new([&repr.tensor].into_iter()),
            NumericOperationIr::Abs(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::Full(_repr) => Box::new([].into_iter()),
            NumericOperationIr::MeanDim(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::Mean(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::Sum(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::SumDim(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::Prod(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::ProdDim(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::Max(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::MaxDimWithIndices(repr) => Box::new([&repr.tensor].into_iter()),
            NumericOperationIr::MinDimWithIndices(repr) => Box::new([&repr.tensor].into_iter()),
            NumericOperationIr::Min(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::MaxDim(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::MinDim(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::MaxAbs(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::MaxAbsDim(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::IntRandom(_repr) => Box::new([].into_iter()),
            NumericOperationIr::Powi(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            NumericOperationIr::PowiScalar(repr) => Box::new([&repr.lhs].into_iter()),
            NumericOperationIr::CumMin(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::CumMax(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::CumProd(repr) => Box::new([&repr.input].into_iter()),
            NumericOperationIr::CumSum(repr) => Box::new([&repr.input].into_iter()),
        }
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            NumericOperationIr::Add(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::AddScalar(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Sub(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::SubScalar(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Mul(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::MulScalar(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Div(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::DivScalar(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Rem(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::RemScalar(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::GreaterElem(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::GreaterEqualElem(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::LowerElem(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::LowerEqualElem(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Greater(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::GreaterEqual(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Lower(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::LowerEqual(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::ArgMax(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::ArgTopK(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::TopK(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::ArgMin(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Clamp(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Abs(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Full(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::MeanDim(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Mean(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Sum(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::SumDim(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Prod(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::ProdDim(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Max(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::MaxDimWithIndices(repr) => {
                Box::new([&repr.out, &repr.out_indices].into_iter())
            }
            NumericOperationIr::MinDimWithIndices(repr) => {
                Box::new([&repr.out, &repr.out_indices].into_iter())
            }
            NumericOperationIr::Min(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::MaxDim(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::MinDim(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::MaxAbs(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::MaxAbsDim(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::IntRandom(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::Powi(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::PowiScalar(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::CumMin(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::CumMax(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::CumProd(repr) => Box::new([&repr.out].into_iter()),
            NumericOperationIr::CumSum(repr) => Box::new([&repr.out].into_iter()),
        }
    }
    fn mark_read_only(&mut self, nodes: &[TensorId]) -> Vec<TensorIr> {
        let mut output = Vec::new();

        match self {
            NumericOperationIr::Add(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::AddScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Sub(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::SubScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Mul(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::MulScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Div(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::DivScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Rem(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::RemScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::GreaterElem(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::GreaterEqualElem(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::LowerElem(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::LowerEqualElem(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Greater(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::GreaterEqual(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Lower(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::LowerEqual(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::ArgMax(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::ArgTopK(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::TopK(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::ArgMin(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Clamp(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Abs(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Full(_) => {}
            NumericOperationIr::MeanDim(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Mean(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Sum(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::SumDim(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Prod(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::ProdDim(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Max(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::MaxDimWithIndices(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::MinDimWithIndices(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::Min(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::MaxDim(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::MinDim(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::MaxAbs(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::MaxAbsDim(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::IntRandom(_) => {}
            NumericOperationIr::Powi(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::PowiScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::CumSum(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::CumProd(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::CumMin(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            NumericOperationIr::CumMax(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
        };

        output
    }
}

impl FloatOperationIr {
    fn inputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            FloatOperationIr::Matmul(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            FloatOperationIr::Cross(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            FloatOperationIr::Random(_repr) => Box::new([].into_iter()),
            FloatOperationIr::Exp(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Log(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Log1p(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Erf(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Recip(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::PowfScalar(repr) => Box::new([&repr.lhs].into_iter()),
            FloatOperationIr::Sqrt(repr)
            | FloatOperationIr::Rsqrt(repr)
            | FloatOperationIr::Silu(repr) => {
                Box::new([&repr.input].into_iter())
            }
            FloatOperationIr::Cos(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Sin(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Tanh(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Round(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Floor(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Ceil(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Trunc(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::IntoInt(repr) | FloatOperationIr::QuantizeDynamic(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Quantize(repr) => {
                Box::new([&repr.tensor, &repr.qparams.scales].into_iter())
            }
            FloatOperationIr::Dequantize(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::IsNan(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::IsInf(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::GridSample2d(repr) => {
                Box::new([&repr.tensor, &repr.grid].into_iter())
            }
            FloatOperationIr::Tan(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Cosh(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::Sinh(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::ArcCos(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::ArcCosh(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::ArcSin(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::ArcSinh(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::ArcTan(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::ArcTanh(repr) => Box::new([&repr.input].into_iter()),
            FloatOperationIr::ArcTan2(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            FloatOperationIr::Powf(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            FloatOperationIr::Matmul(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Cross(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Random(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Exp(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Log(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Log1p(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Erf(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Recip(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::PowfScalar(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Sqrt(repr)
            | FloatOperationIr::Rsqrt(repr)
            | FloatOperationIr::Silu(repr) => {
                Box::new([&repr.out].into_iter())
            }
            FloatOperationIr::Cos(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Sin(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Tanh(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Round(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Floor(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Ceil(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Trunc(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::IntoInt(repr) | FloatOperationIr::QuantizeDynamic(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Quantize(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Dequantize(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::IsNan(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::IsInf(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::GridSample2d(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Tan(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Cosh(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Sinh(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::ArcCos(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::ArcCosh(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::ArcSin(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::ArcSinh(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::ArcTan(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::ArcTanh(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::ArcTan2(repr) => Box::new([&repr.out].into_iter()),
            FloatOperationIr::Powf(repr) => Box::new([&repr.out].into_iter()),
        }
    }

    fn mark_read_only(&mut self, nodes: &[TensorId]) -> Vec<TensorIr> {
        let mut output = Vec::new();

        match self {
            FloatOperationIr::Matmul(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Cross(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Random(_) => {}
            FloatOperationIr::Exp(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Log(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Log1p(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Erf(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Recip(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::PowfScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Sqrt(repr)
            | FloatOperationIr::Rsqrt(repr)
            | FloatOperationIr::Silu(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Cos(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Sin(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Tanh(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Round(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Floor(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Ceil(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Trunc(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Quantize(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
                repr.qparams.scales.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Dequantize(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::IntoInt(repr) | FloatOperationIr::QuantizeDynamic(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::IsNan(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::IsInf(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::GridSample2d(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
                repr.grid.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Tan(repr) => repr.input.mark_read_only(nodes, &mut output),
            FloatOperationIr::Cosh(repr) => repr.input.mark_read_only(nodes, &mut output),
            FloatOperationIr::Sinh(repr) => repr.input.mark_read_only(nodes, &mut output),
            FloatOperationIr::ArcCos(repr) => repr.input.mark_read_only(nodes, &mut output),
            FloatOperationIr::ArcCosh(repr) => repr.input.mark_read_only(nodes, &mut output),
            FloatOperationIr::ArcSin(repr) => repr.input.mark_read_only(nodes, &mut output),
            FloatOperationIr::ArcSinh(repr) => repr.input.mark_read_only(nodes, &mut output),
            FloatOperationIr::ArcTan(repr) => repr.input.mark_read_only(nodes, &mut output),
            FloatOperationIr::ArcTanh(repr) => repr.input.mark_read_only(nodes, &mut output),
            FloatOperationIr::ArcTan2(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            FloatOperationIr::Powf(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
        };

        output
    }
}

impl IntOperationIr {
    fn inputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            IntOperationIr::Matmul(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            IntOperationIr::IntoFloat(repr) => Box::new([&repr.input].into_iter()),
            IntOperationIr::BitwiseAnd(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            IntOperationIr::BitwiseAndScalar(repr) => Box::new([&repr.lhs].into_iter()),
            IntOperationIr::BitwiseOr(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            IntOperationIr::BitwiseOrScalar(repr) => Box::new([&repr.lhs].into_iter()),
            IntOperationIr::BitwiseXor(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            IntOperationIr::BitwiseXorScalar(repr) => Box::new([&repr.lhs].into_iter()),
            IntOperationIr::BitwiseNot(repr) => Box::new([&repr.input].into_iter()),
            IntOperationIr::BitwiseLeftShift(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            IntOperationIr::BitwiseLeftShiftScalar(repr) => Box::new([&repr.lhs].into_iter()),
            IntOperationIr::BitwiseRightShift(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            IntOperationIr::BitwiseRightShiftScalar(repr) => Box::new([&repr.lhs].into_iter()),
        }
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            IntOperationIr::Matmul(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::IntoFloat(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseAnd(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseAndScalar(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseOr(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseOrScalar(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseXor(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseXorScalar(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseNot(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseLeftShift(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseLeftShiftScalar(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseRightShift(repr) => Box::new([&repr.out].into_iter()),
            IntOperationIr::BitwiseRightShiftScalar(repr) => Box::new([&repr.out].into_iter()),
        }
    }

    fn mark_read_only(&mut self, nodes: &[TensorId]) -> Vec<TensorIr> {
        let mut output = Vec::new();

        match self {
            IntOperationIr::Matmul(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::IntoFloat(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseAnd(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseAndScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseOr(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseOrScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseXor(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseXorScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseNot(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseLeftShift(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseLeftShiftScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseRightShift(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            IntOperationIr::BitwiseRightShiftScalar(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
            }
        };

        output
    }
}

impl BoolOperationIr {
    fn inputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            BoolOperationIr::IntoFloat(repr) => Box::new([&repr.input].into_iter()),
            BoolOperationIr::IntoInt(repr) => Box::new([&repr.input].into_iter()),
            BoolOperationIr::Not(repr) => Box::new([&repr.input].into_iter()),
            BoolOperationIr::And(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
            BoolOperationIr::Or(repr) => Box::new([&repr.lhs, &repr.rhs].into_iter()),
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            BoolOperationIr::IntoFloat(repr) => Box::new([&repr.out].into_iter()),
            BoolOperationIr::IntoInt(repr) => Box::new([&repr.out].into_iter()),
            BoolOperationIr::Not(repr) => Box::new([&repr.out].into_iter()),
            BoolOperationIr::And(repr) => Box::new([&repr.out].into_iter()),
            BoolOperationIr::Or(repr) => Box::new([&repr.out].into_iter()),
        }
    }
    fn mark_read_only(&mut self, nodes: &[TensorId]) -> Vec<TensorIr> {
        let mut output = Vec::new();

        match self {
            BoolOperationIr::IntoFloat(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            BoolOperationIr::IntoInt(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            BoolOperationIr::Not(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            BoolOperationIr::And(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
            BoolOperationIr::Or(repr) => {
                repr.lhs.mark_read_only(nodes, &mut output);
                repr.rhs.mark_read_only(nodes, &mut output);
            }
        };

        output
    }
}

impl ModuleOperationIr {
    fn inputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            ModuleOperationIr::Embedding(repr) => {
                Box::new([&repr.weights, &repr.indices].into_iter())
            }
            ModuleOperationIr::EmbeddingBackward(repr) => {
                Box::new([&repr.weights, &repr.out_grad, &repr.indices].into_iter())
            }
            ModuleOperationIr::Linear(repr) => {
                if let Some(bias) = &repr.bias {
                    Box::new([&repr.x, &repr.weight, bias].into_iter())
                } else {
                    Box::new([&repr.x, &repr.weight].into_iter())
                }
            }
            ModuleOperationIr::LinearXBackward(repr) => {
                Box::new([&repr.weight, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::LinearWeightBackward(repr) => {
                Box::new([&repr.x, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::LinearBiasBackward(repr) => {
                Box::new([&repr.output_grad].into_iter())
            }
            ModuleOperationIr::Conv1d(repr) => {
                if let Some(bias) = &repr.bias {
                    Box::new([&repr.x, &repr.weight, bias].into_iter())
                } else {
                    Box::new([&repr.x, &repr.weight].into_iter())
                }
            }
            ModuleOperationIr::Conv1dXBackward(repr) => {
                Box::new([&repr.x, &repr.weight, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::Conv1dWeightBackward(repr) => {
                Box::new([&repr.x, &repr.weight, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::Conv1dBiasBackward(repr) => {
                Box::new([&repr.x, &repr.bias, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::Conv2d(repr) => {
                if let Some(bias) = &repr.bias {
                    Box::new([&repr.x, &repr.weight, bias].into_iter())
                } else {
                    Box::new([&repr.x, &repr.weight].into_iter())
                }
            }
            ModuleOperationIr::Conv2dXBackward(repr) => {
                Box::new([&repr.x, &repr.weight, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::Conv2dWeightBackward(repr) => {
                Box::new([&repr.x, &repr.weight, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::Conv2dBiasBackward(repr) => {
                Box::new([&repr.x, &repr.bias, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::Conv3d(repr) => {
                if let Some(bias) = &repr.bias {
                    Box::new([&repr.x, &repr.weight, bias].into_iter())
                } else {
                    Box::new([&repr.x, &repr.weight].into_iter())
                }
            }
            ModuleOperationIr::Conv3dXBackward(repr) => {
                Box::new([&repr.x, &repr.weight, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::Conv3dWeightBackward(repr) => {
                Box::new([&repr.x, &repr.weight, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::Conv3dBiasBackward(repr) => {
                Box::new([&repr.x, &repr.bias, &repr.output_grad].into_iter())
            }
            ModuleOperationIr::DeformableConv2d(repr) => match (&repr.mask, &repr.bias) {
                (Some(mask), Some(bias)) => {
                    Box::new([&repr.x, &repr.offset, &repr.weight, mask, bias].into_iter())
                }
                (Some(mask), None) => {
                    Box::new([&repr.x, &repr.offset, &repr.weight, mask].into_iter())
                }
                (None, Some(bias)) => {
                    Box::new([&repr.x, &repr.offset, &repr.weight, bias].into_iter())
                }
                (None, None) => Box::new([&repr.x, &repr.offset, &repr.weight].into_iter()),
            },
            ModuleOperationIr::DeformableConv2dBackward(repr) => match (&repr.mask, &repr.bias) {
                (Some(mask), Some(bias)) => Box::new(
                    [
                        &repr.x,
                        &repr.offset,
                        &repr.weight,
                        &repr.out_grad,
                        mask,
                        bias,
                    ]
                    .into_iter(),
                ),
                (Some(mask), None) => Box::new(
                    [&repr.x, &repr.offset, &repr.weight, &repr.out_grad, mask].into_iter(),
                ),
                (None, Some(bias)) => Box::new(
                    [&repr.x, &repr.offset, &repr.weight, &repr.out_grad, bias].into_iter(),
                ),
                (None, None) => {
                    Box::new([&repr.x, &repr.offset, &repr.weight, &repr.out_grad].into_iter())
                }
            },
            ModuleOperationIr::ConvTranspose1d(repr) => {
                if let Some(bias) = &repr.bias {
                    Box::new([&repr.x, &repr.weight, bias].into_iter())
                } else {
                    Box::new([&repr.x, &repr.weight].into_iter())
                }
            }
            ModuleOperationIr::ConvTranspose2d(repr) => {
                if let Some(bias) = &repr.bias {
                    Box::new([&repr.x, &repr.weight, bias].into_iter())
                } else {
                    Box::new([&repr.x, &repr.weight].into_iter())
                }
            }
            ModuleOperationIr::ConvTranspose3d(repr) => {
                if let Some(bias) = &repr.bias {
                    Box::new([&repr.x, &repr.weight, bias].into_iter())
                } else {
                    Box::new([&repr.x, &repr.weight].into_iter())
                }
            }
            ModuleOperationIr::AvgPool1d(repr) => Box::new([&repr.x].into_iter()),
            ModuleOperationIr::AvgPool2d(repr) => Box::new([&repr.x].into_iter()),
            ModuleOperationIr::AvgPool1dBackward(repr) => {
                Box::new([&repr.x, &repr.grad].into_iter())
            }
            ModuleOperationIr::AvgPool2dBackward(repr) => {
                Box::new([&repr.x, &repr.grad].into_iter())
            }
            ModuleOperationIr::AdaptiveAvgPool1d(repr) => Box::new([&repr.x].into_iter()),
            ModuleOperationIr::AdaptiveAvgPool2d(repr) => Box::new([&repr.x].into_iter()),
            ModuleOperationIr::AdaptiveAvgPool1dBackward(repr) => {
                Box::new([&repr.x, &repr.grad].into_iter())
            }
            ModuleOperationIr::AdaptiveAvgPool2dBackward(repr) => {
                Box::new([&repr.x, &repr.grad].into_iter())
            }
            ModuleOperationIr::MaxPool1d(repr) => Box::new([&repr.x].into_iter()),
            ModuleOperationIr::MaxPool1dWithIndices(repr) => Box::new([&repr.x].into_iter()),
            ModuleOperationIr::MaxPool1dWithIndicesBackward(repr) => {
                Box::new([&repr.x, &repr.indices, &repr.grad].into_iter())
            }
            ModuleOperationIr::MaxPool2d(repr) => Box::new([&repr.x].into_iter()),
            ModuleOperationIr::MaxPool2dWithIndices(repr) => Box::new([&repr.x].into_iter()),
            ModuleOperationIr::MaxPool2dWithIndicesBackward(repr) => {
                Box::new([&repr.x, &repr.indices, &repr.grad].into_iter())
            }
            ModuleOperationIr::Interpolate(repr) => Box::new([&repr.x].into_iter()),
            ModuleOperationIr::InterpolateBackward(repr) => {
                Box::new([&repr.x, &repr.grad].into_iter())
            }
            ModuleOperationIr::Rfft(repr) => Box::new([&repr.signal].into_iter()),
            ModuleOperationIr::IRfft(repr) => {
                Box::new([&repr.input_re, &repr.input_im].into_iter())
            }
            ModuleOperationIr::Attention(repr) => {
                if let Some(mask) = &repr.mask {
                    if let Some(attn_bias) = &repr.attn_bias {
                        Box::new([&repr.query, &repr.key, &repr.value, mask, attn_bias].into_iter())
                    } else {
                        Box::new([&repr.query, &repr.key, &repr.value, mask].into_iter())
                    }
                } else if let Some(attn_bias) = &repr.attn_bias {
                    Box::new([&repr.query, &repr.key, &repr.value, attn_bias].into_iter())
                } else {
                    Box::new([&repr.query, &repr.key, &repr.value].into_iter())
                }
            }
            ModuleOperationIr::CtcLoss(repr) => Box::new(
                [
                    &repr.log_probs,
                    &repr.targets,
                    &repr.input_lengths,
                    &repr.target_lengths,
                ]
                .into_iter(),
            ),
            ModuleOperationIr::CtcLossBackward(repr) => Box::new(
                [
                    &repr.log_probs,
                    &repr.targets,
                    &repr.input_lengths,
                    &repr.target_lengths,
                    &repr.grad_loss,
                ]
                .into_iter(),
            ),
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            ModuleOperationIr::Embedding(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::EmbeddingBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Linear(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::LinearXBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::LinearWeightBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::LinearBiasBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv1d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv1dXBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv1dWeightBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv1dBiasBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv2d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv2dXBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv2dWeightBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv2dBiasBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv3d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv3dXBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv3dWeightBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Conv3dBiasBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::DeformableConv2d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::DeformableConv2dBackward(repr) => {
                match (&repr.mask_grad, &repr.bias_grad) {
                    (Some(mask_grad), Some(bias_grad)) => Box::new(
                        [
                            &repr.input_grad,
                            &repr.offset_grad,
                            &repr.weight_grad,
                            mask_grad,
                            bias_grad,
                        ]
                        .into_iter(),
                    ),
                    (Some(mask_grad), None) => Box::new(
                        [
                            &repr.input_grad,
                            &repr.offset_grad,
                            &repr.weight_grad,
                            mask_grad,
                        ]
                        .into_iter(),
                    ),
                    (None, Some(bias_grad)) => Box::new(
                        [
                            &repr.input_grad,
                            &repr.offset_grad,
                            &repr.weight_grad,
                            bias_grad,
                        ]
                        .into_iter(),
                    ),
                    (None, None) => Box::new(
                        [&repr.input_grad, &repr.offset_grad, &repr.weight_grad].into_iter(),
                    ),
                }
            }
            ModuleOperationIr::ConvTranspose1d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::ConvTranspose2d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::ConvTranspose3d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::AvgPool1d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::AvgPool2d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::AvgPool1dBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::AvgPool2dBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::AdaptiveAvgPool1d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::AdaptiveAvgPool2d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::AdaptiveAvgPool1dBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::AdaptiveAvgPool2dBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::MaxPool1d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::MaxPool1dWithIndices(repr) => {
                Box::new([&repr.out, &repr.out_indices].into_iter())
            }
            ModuleOperationIr::MaxPool1dWithIndicesBackward(repr) => {
                Box::new([&repr.out].into_iter())
            }
            ModuleOperationIr::MaxPool2d(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::MaxPool2dWithIndices(repr) => {
                Box::new([&repr.out, &repr.out_indices].into_iter())
            }
            ModuleOperationIr::MaxPool2dWithIndicesBackward(repr) => {
                Box::new([&repr.out].into_iter())
            }
            ModuleOperationIr::Interpolate(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::InterpolateBackward(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::Rfft(repr) => Box::new([&repr.out_re, &repr.out_im].into_iter()),
            ModuleOperationIr::IRfft(repr) => Box::new([&repr.out_signal].into_iter()),
            ModuleOperationIr::Attention(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::CtcLoss(repr) => Box::new([&repr.out].into_iter()),
            ModuleOperationIr::CtcLossBackward(repr) => Box::new([&repr.out].into_iter()),
        }
    }

    fn mark_read_only(&mut self, nodes: &[TensorId]) -> Vec<TensorIr> {
        let mut output = Vec::new();

        match self {
            ModuleOperationIr::Embedding(repr) => {
                repr.weights.mark_read_only(nodes, &mut output);
                repr.indices.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::EmbeddingBackward(repr) => {
                repr.weights.mark_read_only(nodes, &mut output);
                repr.out_grad.mark_read_only(nodes, &mut output);
                repr.indices.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Linear(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);

                if let Some(bias) = &mut repr.bias {
                    bias.mark_read_only(nodes, &mut output);
                }
            }
            ModuleOperationIr::LinearXBackward(repr) => {
                repr.weight.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::LinearWeightBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::LinearBiasBackward(repr) => {
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Conv1d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);

                if let Some(bias) = &mut repr.bias {
                    bias.mark_read_only(nodes, &mut output);
                }
            }
            ModuleOperationIr::Conv1dXBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Conv1dWeightBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Conv1dBiasBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.bias.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Conv2d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);

                if let Some(bias) = &mut repr.bias {
                    bias.mark_read_only(nodes, &mut output);
                }
            }
            ModuleOperationIr::Conv2dXBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Conv2dWeightBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Conv2dBiasBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.bias.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Conv3d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);

                if let Some(bias) = &mut repr.bias {
                    bias.mark_read_only(nodes, &mut output);
                }
            }
            ModuleOperationIr::Conv3dXBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Conv3dWeightBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Conv3dBiasBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.bias.mark_read_only(nodes, &mut output);
                repr.output_grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::DeformableConv2d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);
                repr.offset.mark_read_only(nodes, &mut output);

                match (&mut repr.mask, &mut repr.bias) {
                    (Some(mask), Some(bias)) => {
                        mask.mark_read_only(nodes, &mut output);
                        bias.mark_read_only(nodes, &mut output);
                    }
                    (Some(mask), None) => {
                        mask.mark_read_only(nodes, &mut output);
                    }
                    (None, Some(bias)) => {
                        bias.mark_read_only(nodes, &mut output);
                    }
                    (None, None) => {}
                };
            }
            ModuleOperationIr::DeformableConv2dBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);
                repr.offset.mark_read_only(nodes, &mut output);
                repr.out_grad.mark_read_only(nodes, &mut output);

                if let Some(mask) = repr.mask.as_mut() {
                    mask.mark_read_only(nodes, &mut output);
                }
                if let Some(bias) = repr.bias.as_mut() {
                    bias.mark_read_only(nodes, &mut output);
                }
            }
            ModuleOperationIr::ConvTranspose1d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);

                if let Some(bias) = &mut repr.bias {
                    bias.mark_read_only(nodes, &mut output);
                }
            }
            ModuleOperationIr::ConvTranspose2d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);

                if let Some(bias) = &mut repr.bias {
                    bias.mark_read_only(nodes, &mut output);
                }
            }
            ModuleOperationIr::ConvTranspose3d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.weight.mark_read_only(nodes, &mut output);

                if let Some(bias) = &mut repr.bias {
                    bias.mark_read_only(nodes, &mut output);
                }
            }
            ModuleOperationIr::AvgPool1d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AvgPool2d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AvgPool1dBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AvgPool2dBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AdaptiveAvgPool1d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AdaptiveAvgPool2d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AdaptiveAvgPool1dBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AdaptiveAvgPool2dBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::MaxPool1d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::MaxPool1dWithIndices(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::MaxPool1dWithIndicesBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::MaxPool2d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::MaxPool2dWithIndices(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::MaxPool2dWithIndicesBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Interpolate(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::InterpolateBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Rfft(repr) => {
                repr.signal.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::IRfft(repr) => {
                repr.input_re.mark_read_only(nodes, &mut output);
                repr.input_im.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Attention(repr) => {
                repr.query.mark_read_only(nodes, &mut output);
                repr.key.mark_read_only(nodes, &mut output);
                repr.value.mark_read_only(nodes, &mut output);
                if let Some(mask) = &mut repr.mask {
                    mask.mark_read_only(nodes, &mut output);
                }
                if let Some(attn_bias) = &mut repr.attn_bias {
                    attn_bias.mark_read_only(nodes, &mut output);
                }
            }
            ModuleOperationIr::CtcLoss(repr) => {
                repr.log_probs.mark_read_only(nodes, &mut output);
                repr.targets.mark_read_only(nodes, &mut output);
                repr.input_lengths.mark_read_only(nodes, &mut output);
                repr.target_lengths.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::CtcLossBackward(repr) => {
                repr.log_probs.mark_read_only(nodes, &mut output);
                repr.targets.mark_read_only(nodes, &mut output);
                repr.input_lengths.mark_read_only(nodes, &mut output);
                repr.target_lengths.mark_read_only(nodes, &mut output);
                repr.grad_loss.mark_read_only(nodes, &mut output);
            }
        };

        output
    }
}

#[cfg(feature = "graph-distributed")]
impl DistributedOperationIr {
    fn inputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            DistributedOperationIr::AllReduce(repr) => Box::new([&repr.tensor].into_iter()),
        }
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        match self {
            DistributedOperationIr::AllReduce(repr) => Box::new([&repr.out].into_iter()),
        }
    }

    fn mark_read_only(&mut self, nodes: &[TensorId]) -> Vec<TensorIr> {
        let mut output = Vec::new();

        match self {
            DistributedOperationIr::AllReduce(repr) => {
                repr.tensor.mark_read_only(nodes, &mut output);
            }
        }

        output
    }
}

impl InitOperationIr {
    fn inputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        Box::new([].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = &TensorIr> + '_> {
        Box::new([&self.out].into_iter())
    }
}

impl TensorIr {
    fn mark_read_only(&mut self, nodes: &[TensorId], output: &mut Vec<TensorIr>) {
        if self.status == TensorStatus::ReadWrite && nodes.contains(&self.id) {
            output.push(self.clone());
            self.status = TensorStatus::ReadOnly;
        }
    }
}

impl core::hash::Hash for RandomOpIr {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.out.hash(state);

        match self.distribution {
            Distribution::Default => 1u8.hash(state),
            Distribution::Bernoulli(_) => 2u8.hash(state),
            Distribution::Uniform(_, _) => 3u8.hash(state),
            Distribution::Normal(_, _) => 4u8.hash(state),
        }
    }
}

/// Extension trait to extract outputs when registering an operation.
pub trait OperationOutput<O> {
    /// Extract a single output.
    fn output(self) -> O;

    /// Extract a fixed number of outputs.
    fn outputs<const N: usize>(self) -> [O; N];
}

impl<O: core::fmt::Debug> OperationOutput<O> for Vec<O> {
    fn output(self) -> O {
        let [tensor] = self.outputs();
        tensor
    }

    fn outputs<const N: usize>(self) -> [O; N] {
        self.try_into().unwrap()
    }
}
