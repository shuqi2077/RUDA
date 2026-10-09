use super::*;

struct OptionalTensors<'a, const N: usize> {
    tensors: core::array::IntoIter<Option<&'a TensorIr>, N>,
}

impl<'a, const N: usize> Iterator for OptionalTensors<'a, N> {
    type Item = &'a TensorIr;

    fn next(&mut self) -> Option<Self::Item> {
        self.tensors.find_map(core::convert::identity)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tensors.as_slice().iter().filter(|tensor| tensor.is_some()).count();
        (remaining, Some(remaining))
    }
}

macro_rules! tensor_iter {
    ($($fixed:ident: $n:literal),*; $($optional:ident: $m:literal),*) => {
        enum TensorIter<'a> {
            Slice(core::slice::Iter<'a, TensorIr>),
            $($fixed(core::array::IntoIter<&'a TensorIr, $n>),)*
            $($optional(OptionalTensors<'a, $m>),)*
        }

        $(impl<'a> From<[&'a TensorIr; $n]> for TensorIter<'a> {
            fn from(tensors: [&'a TensorIr; $n]) -> Self {
                Self::$fixed(tensors.into_iter())
            }
        })*

        $(impl<'a> From<[Option<&'a TensorIr>; $m]> for TensorIter<'a> {
            fn from(tensors: [Option<&'a TensorIr>; $m]) -> Self {
                Self::$optional(OptionalTensors { tensors: tensors.into_iter() })
            }
        })*

        impl<'a> Iterator for TensorIter<'a> {
            type Item = &'a TensorIr;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    Self::Slice(tensors) => tensors.next(),
                    $(Self::$fixed(tensors) => tensors.next(),)*
                    $(Self::$optional(tensors) => tensors.next(),)*
                }
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                match self {
                    Self::Slice(tensors) => tensors.nth(n),
                    $(Self::$fixed(tensors) => tensors.nth(n),)*
                    $(Self::$optional(tensors) => tensors.nth(n),)*
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                match self {
                    Self::Slice(tensors) => tensors.size_hint(),
                    $(Self::$fixed(tensors) => tensors.size_hint(),)*
                    $(Self::$optional(tensors) => tensors.size_hint(),)*
                }
            }
        }
    };
}

tensor_iter! {
    Fixed0: 0, Fixed1: 1, Fixed2: 2, Fixed3: 3, Fixed4: 4, Fixed5: 5, Fixed6: 6;
    Optional2: 2, Optional3: 3, Optional5: 5
}


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

    fn inputs(&self) -> TensorIter<'_> {
        TensorIter::Slice(self.inputs.iter())
    }

    fn outputs(&self) -> TensorIter<'_> {
        TensorIter::Slice(self.outputs.iter())
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
            OperationIr::Drop(repr) => TensorIter::from([repr]),
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
            OperationIr::Drop(_repr) => TensorIter::from([]),
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
    fn inputs(&self) -> TensorIter<'_> {
        match self {
            BaseOperationIr::Reshape(repr) => TensorIter::from([&repr.input]),
            BaseOperationIr::SwapDims(repr) => TensorIter::from([&repr.input]),
            BaseOperationIr::Permute(repr) => TensorIter::from([&repr.input]),
            BaseOperationIr::Expand(repr) => TensorIter::from([&repr.input]),
            BaseOperationIr::Flip(repr) => TensorIter::from([&repr.input]),
            BaseOperationIr::Slice(repr) => TensorIter::from([&repr.tensor]),
            BaseOperationIr::SliceAssign(repr) => TensorIter::from([&repr.tensor, &repr.value]),
            BaseOperationIr::Gather(repr) => TensorIter::from([&repr.tensor, &repr.indices]),
            BaseOperationIr::Scatter(repr) => {
                TensorIter::from([&repr.tensor, &repr.indices, &repr.value])
            }
            BaseOperationIr::ScatterNd(repr) => {
                TensorIter::from([&repr.data, &repr.indices, &repr.values])
            }
            BaseOperationIr::GatherNd(repr) => TensorIter::from([&repr.data, &repr.indices]),
            BaseOperationIr::Select(repr) => TensorIter::from([&repr.tensor, &repr.indices]),
            BaseOperationIr::SelectAssign(repr) => {
                TensorIter::from([&repr.tensor, &repr.indices, &repr.value])
            }
            BaseOperationIr::MaskWhere(repr) => {
                TensorIter::from([&repr.tensor, &repr.mask, &repr.value])
            }
            BaseOperationIr::MaskFill(repr) => TensorIter::from([&repr.tensor, &repr.mask]),
            BaseOperationIr::Equal(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            BaseOperationIr::EqualElem(repr) => TensorIter::from([&repr.lhs]),
            BaseOperationIr::RepeatDim(repr) => TensorIter::from([&repr.tensor]),
            BaseOperationIr::Cat(repr) => TensorIter::Slice(repr.tensors.iter()),
            BaseOperationIr::Cast(repr) => TensorIter::from([&repr.input]),
            BaseOperationIr::Unfold(repr) => TensorIter::from([&repr.input]),
            BaseOperationIr::Empty(_repr) => TensorIter::from([]),
            BaseOperationIr::Ones(_repr) => TensorIter::from([]),
            BaseOperationIr::Zeros(_repr) => TensorIter::from([]),
        }
    }

    fn outputs(&self) -> TensorIter<'_> {
        match self {
            BaseOperationIr::Reshape(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::SwapDims(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Permute(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Expand(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Flip(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Slice(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::SliceAssign(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Gather(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Scatter(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::ScatterNd(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::GatherNd(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Select(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::SelectAssign(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::MaskWhere(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::MaskFill(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Equal(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::EqualElem(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::RepeatDim(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Cat(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Cast(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Unfold(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Empty(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Ones(repr) => TensorIter::from([&repr.out]),
            BaseOperationIr::Zeros(repr) => TensorIter::from([&repr.out]),
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
    fn inputs(&self) -> TensorIter<'_> {
        match self {
            NumericOperationIr::Add(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::AddScalar(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::Sub(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::SubScalar(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::Mul(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::MulScalar(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::Div(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::DivScalar(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::Rem(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::RemScalar(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::GreaterElem(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::GreaterEqualElem(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::LowerElem(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::LowerEqualElem(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::Greater(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::GreaterEqual(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::Lower(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::LowerEqual(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::ArgMax(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::ArgTopK(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::TopK(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::ArgMin(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::Clamp(repr) => TensorIter::from([&repr.tensor]),
            NumericOperationIr::Abs(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::Full(_repr) => TensorIter::from([]),
            NumericOperationIr::MeanDim(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::Mean(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::Sum(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::SumDim(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::Prod(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::ProdDim(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::Max(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::MaxDimWithIndices(repr) => TensorIter::from([&repr.tensor]),
            NumericOperationIr::MinDimWithIndices(repr) => TensorIter::from([&repr.tensor]),
            NumericOperationIr::Min(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::MaxDim(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::MinDim(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::MaxAbs(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::MaxAbsDim(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::IntRandom(_repr) => TensorIter::from([]),
            NumericOperationIr::Powi(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            NumericOperationIr::PowiScalar(repr) => TensorIter::from([&repr.lhs]),
            NumericOperationIr::CumMin(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::CumMax(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::CumProd(repr) => TensorIter::from([&repr.input]),
            NumericOperationIr::CumSum(repr) => TensorIter::from([&repr.input]),
        }
    }

    fn outputs(&self) -> TensorIter<'_> {
        match self {
            NumericOperationIr::Add(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::AddScalar(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Sub(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::SubScalar(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Mul(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::MulScalar(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Div(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::DivScalar(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Rem(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::RemScalar(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::GreaterElem(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::GreaterEqualElem(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::LowerElem(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::LowerEqualElem(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Greater(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::GreaterEqual(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Lower(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::LowerEqual(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::ArgMax(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::ArgTopK(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::TopK(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::ArgMin(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Clamp(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Abs(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Full(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::MeanDim(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Mean(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Sum(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::SumDim(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Prod(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::ProdDim(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Max(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::MaxDimWithIndices(repr) => {
                TensorIter::from([&repr.out, &repr.out_indices])
            }
            NumericOperationIr::MinDimWithIndices(repr) => {
                TensorIter::from([&repr.out, &repr.out_indices])
            }
            NumericOperationIr::Min(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::MaxDim(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::MinDim(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::MaxAbs(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::MaxAbsDim(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::IntRandom(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::Powi(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::PowiScalar(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::CumMin(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::CumMax(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::CumProd(repr) => TensorIter::from([&repr.out]),
            NumericOperationIr::CumSum(repr) => TensorIter::from([&repr.out]),
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
    fn inputs(&self) -> TensorIter<'_> {
        match self {
            FloatOperationIr::Matmul(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            FloatOperationIr::Cross(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            FloatOperationIr::Random(_repr) => TensorIter::from([]),
            FloatOperationIr::Exp(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Log(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Log1p(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Erf(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Recip(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::PowfScalar(repr) => TensorIter::from([&repr.lhs]),
            FloatOperationIr::Sqrt(repr)
            | FloatOperationIr::Rsqrt(repr)
            | FloatOperationIr::Silu(repr) => {
                TensorIter::from([&repr.input])
            }
            FloatOperationIr::Cos(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Sin(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Tanh(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Round(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Floor(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Ceil(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Trunc(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::IntoInt(repr) | FloatOperationIr::QuantizeDynamic(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Quantize(repr) => {
                TensorIter::from([&repr.tensor, &repr.qparams.scales])
            }
            FloatOperationIr::Dequantize(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::IsNan(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::IsInf(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::GridSample2d(repr) => {
                TensorIter::from([&repr.tensor, &repr.grid])
            }
            FloatOperationIr::Tan(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Cosh(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::Sinh(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::ArcCos(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::ArcCosh(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::ArcSin(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::ArcSinh(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::ArcTan(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::ArcTanh(repr) => TensorIter::from([&repr.input]),
            FloatOperationIr::ArcTan2(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            FloatOperationIr::Powf(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
        }
    }
    fn outputs(&self) -> TensorIter<'_> {
        match self {
            FloatOperationIr::Matmul(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Cross(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Random(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Exp(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Log(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Log1p(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Erf(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Recip(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::PowfScalar(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Sqrt(repr)
            | FloatOperationIr::Rsqrt(repr)
            | FloatOperationIr::Silu(repr) => {
                TensorIter::from([&repr.out])
            }
            FloatOperationIr::Cos(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Sin(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Tanh(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Round(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Floor(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Ceil(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Trunc(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::IntoInt(repr) | FloatOperationIr::QuantizeDynamic(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Quantize(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Dequantize(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::IsNan(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::IsInf(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::GridSample2d(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Tan(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Cosh(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Sinh(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::ArcCos(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::ArcCosh(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::ArcSin(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::ArcSinh(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::ArcTan(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::ArcTanh(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::ArcTan2(repr) => TensorIter::from([&repr.out]),
            FloatOperationIr::Powf(repr) => TensorIter::from([&repr.out]),
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
    fn inputs(&self) -> TensorIter<'_> {
        match self {
            IntOperationIr::Matmul(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            IntOperationIr::IntoFloat(repr) => TensorIter::from([&repr.input]),
            IntOperationIr::BitwiseAnd(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            IntOperationIr::BitwiseAndScalar(repr) => TensorIter::from([&repr.lhs]),
            IntOperationIr::BitwiseOr(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            IntOperationIr::BitwiseOrScalar(repr) => TensorIter::from([&repr.lhs]),
            IntOperationIr::BitwiseXor(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            IntOperationIr::BitwiseXorScalar(repr) => TensorIter::from([&repr.lhs]),
            IntOperationIr::BitwiseNot(repr) => TensorIter::from([&repr.input]),
            IntOperationIr::BitwiseLeftShift(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            IntOperationIr::BitwiseLeftShiftScalar(repr) => TensorIter::from([&repr.lhs]),
            IntOperationIr::BitwiseRightShift(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            IntOperationIr::BitwiseRightShiftScalar(repr) => TensorIter::from([&repr.lhs]),
        }
    }

    fn outputs(&self) -> TensorIter<'_> {
        match self {
            IntOperationIr::Matmul(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::IntoFloat(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseAnd(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseAndScalar(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseOr(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseOrScalar(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseXor(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseXorScalar(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseNot(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseLeftShift(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseLeftShiftScalar(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseRightShift(repr) => TensorIter::from([&repr.out]),
            IntOperationIr::BitwiseRightShiftScalar(repr) => TensorIter::from([&repr.out]),
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
    fn inputs(&self) -> TensorIter<'_> {
        match self {
            BoolOperationIr::IntoFloat(repr) => TensorIter::from([&repr.input]),
            BoolOperationIr::IntoInt(repr) => TensorIter::from([&repr.input]),
            BoolOperationIr::Not(repr) => TensorIter::from([&repr.input]),
            BoolOperationIr::And(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
            BoolOperationIr::Or(repr) => TensorIter::from([&repr.lhs, &repr.rhs]),
        }
    }
    fn outputs(&self) -> TensorIter<'_> {
        match self {
            BoolOperationIr::IntoFloat(repr) => TensorIter::from([&repr.out]),
            BoolOperationIr::IntoInt(repr) => TensorIter::from([&repr.out]),
            BoolOperationIr::Not(repr) => TensorIter::from([&repr.out]),
            BoolOperationIr::And(repr) => TensorIter::from([&repr.out]),
            BoolOperationIr::Or(repr) => TensorIter::from([&repr.out]),
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
    fn inputs(&self) -> TensorIter<'_> {
        match self {
            ModuleOperationIr::Embedding(repr) => {
                TensorIter::from([&repr.weights, &repr.indices])
            }
            ModuleOperationIr::EmbeddingBackward(repr) => {
                TensorIter::from([&repr.weights, &repr.out_grad, &repr.indices])
            }
            ModuleOperationIr::Linear(repr) => {
                if let Some(bias) = &repr.bias {
                    TensorIter::from([&repr.x, &repr.weight, bias])
                } else {
                    TensorIter::from([&repr.x, &repr.weight])
                }
            }
            ModuleOperationIr::LinearXBackward(repr) => {
                TensorIter::from([&repr.weight, &repr.output_grad])
            }
            ModuleOperationIr::LinearWeightBackward(repr) => {
                TensorIter::from([&repr.x, &repr.output_grad])
            }
            ModuleOperationIr::LinearBiasBackward(repr) => {
                TensorIter::from([&repr.output_grad])
            }
            ModuleOperationIr::Conv1d(repr) => {
                if let Some(bias) = &repr.bias {
                    TensorIter::from([&repr.x, &repr.weight, bias])
                } else {
                    TensorIter::from([&repr.x, &repr.weight])
                }
            }
            ModuleOperationIr::Conv1dXBackward(repr) => {
                TensorIter::from([&repr.x, &repr.weight, &repr.output_grad])
            }
            ModuleOperationIr::Conv1dWeightBackward(repr) => {
                TensorIter::from([&repr.x, &repr.weight, &repr.output_grad])
            }
            ModuleOperationIr::Conv1dBiasBackward(repr) => {
                TensorIter::from([&repr.x, &repr.bias, &repr.output_grad])
            }
            ModuleOperationIr::Conv2d(repr) => {
                if let Some(bias) = &repr.bias {
                    TensorIter::from([&repr.x, &repr.weight, bias])
                } else {
                    TensorIter::from([&repr.x, &repr.weight])
                }
            }
            ModuleOperationIr::Conv2dXBackward(repr) => {
                TensorIter::from([&repr.x, &repr.weight, &repr.output_grad])
            }
            ModuleOperationIr::Conv2dWeightBackward(repr) => {
                TensorIter::from([&repr.x, &repr.weight, &repr.output_grad])
            }
            ModuleOperationIr::Conv2dBiasBackward(repr) => {
                TensorIter::from([&repr.x, &repr.bias, &repr.output_grad])
            }
            ModuleOperationIr::Conv3d(repr) => {
                if let Some(bias) = &repr.bias {
                    TensorIter::from([&repr.x, &repr.weight, bias])
                } else {
                    TensorIter::from([&repr.x, &repr.weight])
                }
            }
            ModuleOperationIr::Conv3dXBackward(repr) => {
                TensorIter::from([&repr.x, &repr.weight, &repr.output_grad])
            }
            ModuleOperationIr::Conv3dWeightBackward(repr) => {
                TensorIter::from([&repr.x, &repr.weight, &repr.output_grad])
            }
            ModuleOperationIr::Conv3dBiasBackward(repr) => {
                TensorIter::from([&repr.x, &repr.bias, &repr.output_grad])
            }
            ModuleOperationIr::DeformableConv2d(repr) => match (&repr.mask, &repr.bias) {
                (Some(mask), Some(bias)) => {
                    TensorIter::from([&repr.x, &repr.offset, &repr.weight, mask, bias])
                }
                (Some(mask), None) => {
                    TensorIter::from([&repr.x, &repr.offset, &repr.weight, mask])
                }
                (None, Some(bias)) => {
                    TensorIter::from([&repr.x, &repr.offset, &repr.weight, bias])
                }
                (None, None) => TensorIter::from([&repr.x, &repr.offset, &repr.weight]),
            },
            ModuleOperationIr::DeformableConv2dBackward(repr) => match (&repr.mask, &repr.bias) {
                (Some(mask), Some(bias)) => TensorIter::from([
                        &repr.x,
                        &repr.offset,
                        &repr.weight,
                        &repr.out_grad,
                        mask,
                        bias,
                    ]),
                (Some(mask), None) => TensorIter::from([&repr.x, &repr.offset, &repr.weight, &repr.out_grad, mask]),
                (None, Some(bias)) => TensorIter::from([&repr.x, &repr.offset, &repr.weight, &repr.out_grad, bias]),
                (None, None) => {
                    TensorIter::from([&repr.x, &repr.offset, &repr.weight, &repr.out_grad])
                }
            },
            ModuleOperationIr::ConvTranspose1d(repr) => {
                if let Some(bias) = &repr.bias {
                    TensorIter::from([&repr.x, &repr.weight, bias])
                } else {
                    TensorIter::from([&repr.x, &repr.weight])
                }
            }
            ModuleOperationIr::ConvTranspose2d(repr) => {
                if let Some(bias) = &repr.bias {
                    TensorIter::from([&repr.x, &repr.weight, bias])
                } else {
                    TensorIter::from([&repr.x, &repr.weight])
                }
            }
            ModuleOperationIr::ConvTranspose3d(repr) => {
                if let Some(bias) = &repr.bias {
                    TensorIter::from([&repr.x, &repr.weight, bias])
                } else {
                    TensorIter::from([&repr.x, &repr.weight])
                }
            }
            ModuleOperationIr::AvgPool1d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::AvgPool2d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::AvgPool1dBackward(repr) => {
                TensorIter::from([&repr.x, &repr.grad])
            }
            ModuleOperationIr::AvgPool2dBackward(repr) => {
                TensorIter::from([&repr.x, &repr.grad])
            }
            ModuleOperationIr::AdaptiveAvgPool1d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::AdaptiveAvgPool2d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::AdaptiveAvgPool3d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::AvgPool3d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::AvgPool3dBackward(repr) => TensorIter::from([&repr.x, &repr.grad]),
            ModuleOperationIr::AdaptiveAvgPool3dBackward(repr) => {
                TensorIter::from([&repr.x, &repr.grad])
            }
            ModuleOperationIr::AdaptiveAvgPool1dBackward(repr) => {
                TensorIter::from([&repr.x, &repr.grad])
            }
            ModuleOperationIr::AdaptiveAvgPool2dBackward(repr) => {
                TensorIter::from([&repr.x, &repr.grad])
            }
            ModuleOperationIr::MaxPool1d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::MaxPool1dWithIndices(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::MaxPool1dWithIndicesBackward(repr) => {
                TensorIter::from([&repr.x, &repr.indices, &repr.grad])
            }
            ModuleOperationIr::MaxPool2d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::MaxPool3d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::MaxPool3dWithIndices(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::MaxPool3dWithIndicesBackward(repr) => {
                TensorIter::from([&repr.x, &repr.indices, &repr.grad])
            }
            ModuleOperationIr::MaxPool2dWithIndices(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::MaxPool2dWithIndicesBackward(repr) => {
                TensorIter::from([&repr.x, &repr.indices, &repr.grad])
            }
            ModuleOperationIr::Interpolate(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::Interpolate1d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::Interpolate3d(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::Interpolate1dBackward(repr) => TensorIter::from([&repr.x, &repr.grad]),
            ModuleOperationIr::Interpolate3dBackward(repr) => TensorIter::from([&repr.x, &repr.grad]),
            ModuleOperationIr::LayerNorm(repr) => TensorIter::from([Some(&repr.x), Some(&repr.gamma), repr.beta.as_ref()]),
            ModuleOperationIr::LayerNormBackward(repr) => {
                TensorIter::from([&repr.x, &repr.gamma, &repr.grad, &repr.mean, &repr.rstd])
            }
            ModuleOperationIr::RmsNorm(repr) => TensorIter::from([&repr.x, &repr.gamma]),
            ModuleOperationIr::Softmax(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::SiluNative(repr) => TensorIter::from([&repr.input]),
            ModuleOperationIr::GeluNative(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::GeluNativeBackward(repr) => TensorIter::from([&repr.x, &repr.grad]),
            ModuleOperationIr::ExponentialReluNative(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::ExponentialReluNativeBackward(repr) => TensorIter::from([&repr.x, &repr.grad]),
            ModuleOperationIr::LeakyReluNative(repr) => TensorIter::from([&repr.x]),
            ModuleOperationIr::LeakyReluNativeBackward(repr) => TensorIter::from([&repr.x, &repr.grad]),
            ModuleOperationIr::PreluNative(repr) => TensorIter::from([&repr.x, &repr.alpha]),
            ModuleOperationIr::PreluNativeBackwardSelect(repr) => TensorIter::from([&repr.x, &repr.alpha, &repr.grad]),
            ModuleOperationIr::GroupNorm(repr) => TensorIter::from([Some(&repr.x), repr.gamma.as_ref(), repr.beta.as_ref()]),
            ModuleOperationIr::GroupNormBackwardSelect(repr) => {
                TensorIter::from([Some(&repr.x), Some(&repr.grad), Some(&repr.mean), Some(&repr.rstd), repr.gamma.as_ref()])
            }
            ModuleOperationIr::SiluNativeBackward(repr) => TensorIter::from([&repr.x, &repr.grad]),
            ModuleOperationIr::SoftmaxBackward(repr) => TensorIter::from([&repr.working, &repr.grad]),
            ModuleOperationIr::RmsNormBackward(repr) => TensorIter::from([&repr.x, &repr.gamma, &repr.grad, &repr.rstd]),
            ModuleOperationIr::RmsNormBackwardSelect(repr) => TensorIter::from([&repr.x, &repr.gamma, &repr.grad, &repr.rstd]),
            ModuleOperationIr::LayerNormBackwardSelect(repr) => {
                TensorIter::from([&repr.x, &repr.gamma, &repr.grad, &repr.mean, &repr.rstd])
            }
            ModuleOperationIr::InterpolateBackward(repr) => {
                TensorIter::from([&repr.x, &repr.grad])
            }
            ModuleOperationIr::Rfft(repr) => TensorIter::from([&repr.signal]),
            ModuleOperationIr::IRfft(repr) => {
                TensorIter::from([&repr.input_re, &repr.input_im])
            }
            ModuleOperationIr::Attention(repr) => {
                if let Some(mask) = &repr.mask {
                    if let Some(attn_bias) = &repr.attn_bias {
                        TensorIter::from([&repr.query, &repr.key, &repr.value, mask, attn_bias])
                    } else {
                        TensorIter::from([&repr.query, &repr.key, &repr.value, mask])
                    }
                } else if let Some(attn_bias) = &repr.attn_bias {
                    TensorIter::from([&repr.query, &repr.key, &repr.value, attn_bias])
                } else {
                    TensorIter::from([&repr.query, &repr.key, &repr.value])
                }
            }
            ModuleOperationIr::CtcLoss(repr) => TensorIter::from([
                    &repr.log_probs,
                    &repr.targets,
                    &repr.input_lengths,
                    &repr.target_lengths,
                ]),
            ModuleOperationIr::CtcLossBackward(repr) => TensorIter::from([
                    &repr.log_probs,
                    &repr.targets,
                    &repr.input_lengths,
                    &repr.target_lengths,
                    &repr.grad_loss,
                ]),
        }
    }
    fn outputs(&self) -> TensorIter<'_> {
        match self {
            ModuleOperationIr::Embedding(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::EmbeddingBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Linear(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::LinearXBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::LinearWeightBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::LinearBiasBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv1d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv1dXBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv1dWeightBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv1dBiasBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv2d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv2dXBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv2dWeightBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv2dBiasBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv3d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv3dXBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv3dWeightBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Conv3dBiasBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::DeformableConv2d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::DeformableConv2dBackward(repr) => {
                match (&repr.mask_grad, &repr.bias_grad) {
                    (Some(mask_grad), Some(bias_grad)) => TensorIter::from([
                            &repr.input_grad,
                            &repr.offset_grad,
                            &repr.weight_grad,
                            mask_grad,
                            bias_grad,
                        ]),
                    (Some(mask_grad), None) => TensorIter::from([
                            &repr.input_grad,
                            &repr.offset_grad,
                            &repr.weight_grad,
                            mask_grad,
                        ]),
                    (None, Some(bias_grad)) => TensorIter::from([
                            &repr.input_grad,
                            &repr.offset_grad,
                            &repr.weight_grad,
                            bias_grad,
                        ]),
                    (None, None) => TensorIter::from([&repr.input_grad, &repr.offset_grad, &repr.weight_grad]),
                }
            }
            ModuleOperationIr::ConvTranspose1d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::ConvTranspose2d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::ConvTranspose3d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AvgPool1d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AvgPool2d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AvgPool1dBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AvgPool2dBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AdaptiveAvgPool1d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AdaptiveAvgPool2d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AdaptiveAvgPool3d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AvgPool3d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AvgPool3dBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AdaptiveAvgPool3dBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AdaptiveAvgPool1dBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::AdaptiveAvgPool2dBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::MaxPool1d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::MaxPool1dWithIndices(repr) => {
                TensorIter::from([&repr.out, &repr.out_indices])
            }
            ModuleOperationIr::MaxPool1dWithIndicesBackward(repr) => {
                TensorIter::from([&repr.out])
            }
            ModuleOperationIr::MaxPool2d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::MaxPool3d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::MaxPool3dWithIndices(repr) => {
                TensorIter::from([&repr.out, &repr.out_indices])
            }
            ModuleOperationIr::MaxPool3dWithIndicesBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::MaxPool2dWithIndices(repr) => {
                TensorIter::from([&repr.out, &repr.out_indices])
            }
            ModuleOperationIr::MaxPool2dWithIndicesBackward(repr) => {
                TensorIter::from([&repr.out])
            }
            ModuleOperationIr::Interpolate(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Interpolate1d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Interpolate3d(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Interpolate1dBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Interpolate3dBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::LayerNorm(repr) => TensorIter::from([&repr.out, &repr.mean, &repr.rstd]),
            ModuleOperationIr::LayerNormBackward(repr) => {
                TensorIter::from([&repr.input_grad, &repr.weight_grad, &repr.bias_grad])
            }
            ModuleOperationIr::RmsNorm(repr) => TensorIter::from([&repr.out, &repr.rstd]),
            ModuleOperationIr::Softmax(repr) => TensorIter::from([&repr.out, &repr.working]),
            ModuleOperationIr::SiluNative(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::GeluNative(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::GeluNativeBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::ExponentialReluNative(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::ExponentialReluNativeBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::LeakyReluNative(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::LeakyReluNativeBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::PreluNative(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::PreluNativeBackwardSelect(repr) => TensorIter::from([repr.input_grad.as_ref(), repr.weight_grad.as_ref()]),
            ModuleOperationIr::GroupNorm(repr) => TensorIter::from([&repr.out, &repr.mean, &repr.rstd]),
            ModuleOperationIr::GroupNormBackwardSelect(repr) => {
                TensorIter::from([repr.input_grad.as_ref(), repr.weight_grad.as_ref(), repr.bias_grad.as_ref()])
            }
            ModuleOperationIr::SiluNativeBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::SoftmaxBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::RmsNormBackward(repr) => TensorIter::from([&repr.input_grad, &repr.weight_grad]),
            ModuleOperationIr::RmsNormBackwardSelect(repr) => TensorIter::from([repr.input_grad.as_ref(), repr.weight_grad.as_ref()]),
            ModuleOperationIr::LayerNormBackwardSelect(repr) => {
                TensorIter::from([repr.input_grad.as_ref(), repr.weight_grad.as_ref(), repr.bias_grad.as_ref()])
            }
            ModuleOperationIr::InterpolateBackward(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::Rfft(repr) => TensorIter::from([&repr.out_re, &repr.out_im]),
            ModuleOperationIr::IRfft(repr) => TensorIter::from([&repr.out_signal]),
            ModuleOperationIr::Attention(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::CtcLoss(repr) => TensorIter::from([&repr.out]),
            ModuleOperationIr::CtcLossBackward(repr) => TensorIter::from([&repr.out]),
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
            ModuleOperationIr::AdaptiveAvgPool3d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AvgPool3d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AvgPool3dBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::AdaptiveAvgPool3dBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
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
            ModuleOperationIr::MaxPool3d(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::MaxPool3dWithIndices(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::MaxPool3dWithIndicesBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
                repr.indices.mark_read_only(nodes, &mut output);
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
            ModuleOperationIr::Interpolate1d(repr) => { repr.x.mark_read_only(nodes, &mut output); }
            ModuleOperationIr::Interpolate3d(repr) => { repr.x.mark_read_only(nodes, &mut output); }
            ModuleOperationIr::Interpolate1dBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Interpolate3dBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::LayerNorm(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.gamma.mark_read_only(nodes, &mut output);
                if let Some(beta) = &mut repr.beta { beta.mark_read_only(nodes, &mut output); }
            }
            ModuleOperationIr::LayerNormBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.gamma.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
                repr.mean.mark_read_only(nodes, &mut output);
                repr.rstd.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::RmsNorm(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.gamma.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::Softmax(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::SiluNative(repr) => {
                repr.input.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::GeluNative(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::GeluNativeBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::ExponentialReluNative(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::ExponentialReluNativeBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::LeakyReluNative(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::LeakyReluNativeBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::PreluNative(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.alpha.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::PreluNativeBackwardSelect(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.alpha.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::GroupNorm(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                for value in repr.gamma.iter_mut().chain(repr.beta.iter_mut()) { value.mark_read_only(nodes, &mut output); }
            }
            ModuleOperationIr::GroupNormBackwardSelect(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
                repr.mean.mark_read_only(nodes, &mut output);
                repr.rstd.mark_read_only(nodes, &mut output);
                if let Some(gamma) = &mut repr.gamma { gamma.mark_read_only(nodes, &mut output); }
            }
            ModuleOperationIr::SiluNativeBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::SoftmaxBackward(repr) => {
                repr.working.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::RmsNormBackward(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.gamma.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
                repr.rstd.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::RmsNormBackwardSelect(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.gamma.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
                repr.rstd.mark_read_only(nodes, &mut output);
            }
            ModuleOperationIr::LayerNormBackwardSelect(repr) => {
                repr.x.mark_read_only(nodes, &mut output);
                repr.gamma.mark_read_only(nodes, &mut output);
                repr.grad.mark_read_only(nodes, &mut output);
                repr.mean.mark_read_only(nodes, &mut output);
                repr.rstd.mark_read_only(nodes, &mut output);
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
    fn inputs(&self) -> TensorIter<'_> {
        match self {
            DistributedOperationIr::AllReduce(repr) => TensorIter::from([&repr.tensor]),
        }
    }

    fn outputs(&self) -> TensorIter<'_> {
        match self {
            DistributedOperationIr::AllReduce(repr) => TensorIter::from([&repr.out]),
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
    fn inputs(&self) -> TensorIter<'_> {
        TensorIter::from([])
    }
    fn outputs(&self) -> TensorIter<'_> {
        TensorIter::from([&self.out])
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
