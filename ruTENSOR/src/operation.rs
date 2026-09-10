use crate::{OperandDescriptor, TensorDescriptor, Mode, Error, Result};
use ruda_core::tensor::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Identity, Negate, Abs, Sqrt, Exp, Log, Sin, Cos, Tanh, Relu, Reciprocal,
    /// Conjugation is the identity for the real storage types accepted by ruTENSOR.
    Conjugate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp { Add, Mul, Min, Max }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionOp { Sum, Product, Min, Max }

/// Arithmetic precision, including operand transforms and scalar coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComputeType { F32, F64 }

impl ComputeType {
    pub fn dtype(self) -> DType {
        match self { Self::F32 => DType::F32, Self::F64 => DType::F64 }
    }

    pub(crate) fn for_operands(inputs: &[OperandDescriptor]) -> Self {
        if inputs.iter().any(|x| x.tensor.dtype == DType::F64) { Self::F64 } else { Self::F32 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kind {
    Contraction,
    Reduction(ReductionOp),
    Permutation,
    Elementwise(BinaryOp, BinaryOp),
}

/// A buffer-independent operation. Input order is the order passed to its constructor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationDescriptor {
    pub(crate) inputs: Vec<OperandDescriptor>,
    pub(crate) output: TensorDescriptor,
    pub(crate) output_modes: Vec<Mode>,
    pub(crate) compute: ComputeType,
    pub(crate) kind: Kind,
    pub(crate) terms: usize,
    pub(crate) addend: bool,
}

impl OperationDescriptor {
    /// D = alpha * sum(op(A) * op(B)) + beta * op(C).
    /// C, if supplied, follows A and B in the execution input list.
    pub fn contraction(
        a: OperandDescriptor, b: OperandDescriptor, c: Option<OperandDescriptor>,
        output: TensorDescriptor, modes: &[Mode], compute: ComputeType,
    ) -> Result<Self> {
        Self::sum_product(vec![a, b], c, output, modes, compute)
    }

    /// General multi-operand sum of products. Modes absent from D are summed.
    pub fn sum_product(
        mut inputs: Vec<OperandDescriptor>, c: Option<OperandDescriptor>,
        output: TensorDescriptor, modes: &[Mode], compute: ComputeType,
    ) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::InvalidOperation("a sum of products requires at least one input".into()));
        }
        let terms = inputs.len();
        let addend = c.is_some();
        inputs.extend(c);
        Self::new(inputs, output, modes, compute, Kind::Contraction, terms, addend)
    }

    /// D = alpha * reduce(op(A)) + beta * op(C).
    pub fn reduction(
        a: OperandDescriptor, c: Option<OperandDescriptor>, output: TensorDescriptor,
        modes: &[Mode], reduction: ReductionOp, compute: ComputeType,
    ) -> Result<Self> {
        let addend = c.is_some();
        let mut inputs = vec![a];
        inputs.extend(c);
        Self::new(inputs, output, modes, compute, Kind::Reduction(reduction), 1, addend)
    }

    /// D = alpha * op(A), with a physical copy into the specified output layout.
    pub fn permutation(
        a: OperandDescriptor, output: TensorDescriptor, modes: &[Mode], compute: ComputeType,
    ) -> Result<Self> {
        Self::new(vec![a], output, modes, compute, Kind::Permutation, 1, false)
    }

    /// D = combine(alpha * op(A), beta * op(B)).
    pub fn elementwise_binary(
        a: OperandDescriptor, b: OperandDescriptor, output: TensorDescriptor,
        modes: &[Mode], combine: BinaryOp, compute: ComputeType,
    ) -> Result<Self> {
        Self::new(vec![a, b], output, modes, compute,
            Kind::Elementwise(combine, BinaryOp::Add), 2, false)
    }

    /// D = combine_abc(combine_ab(alpha * op(A), beta * op(B)), gamma * op(C)).
    pub fn elementwise_trinary(
        inputs: [OperandDescriptor; 3], output: TensorDescriptor, modes: &[Mode],
        combine_ab: BinaryOp, combine_abc: BinaryOp, compute: ComputeType,
    ) -> Result<Self> {
        Self::new(inputs.into(), output, modes, compute,
            Kind::Elementwise(combine_ab, combine_abc), 3, false)
    }

    fn new(
        inputs: Vec<OperandDescriptor>, output: TensorDescriptor, modes: &[Mode],
        compute: ComputeType, kind: Kind, terms: usize, addend: bool,
    ) -> Result<Self> {
        if output.rank() != modes.len() {
            return Err(Error::InvalidDescriptor("output modes and axes differ".into()));
        }
        for (i, mode) in modes.iter().enumerate() {
            if modes[..i].contains(mode) {
                return Err(Error::InvalidDescriptor("output modes must be unique".into()));
            }
        }
        if !output.is_nonoverlapping() {
            return Err(Error::InvalidDescriptor("output strides overlap".into()));
        }
        if compute == ComputeType::F32 && inputs.iter().any(|x| x.tensor.dtype == DType::F64) {
            return Err(Error::InvalidOperation("F64 inputs require F64 computation".into()));
        }
        Ok(Self { inputs, output, output_modes: modes.to_vec(), compute, kind, terms, addend })
    }

    pub fn inputs(&self) -> &[OperandDescriptor] { &self.inputs }
    pub fn output(&self) -> &TensorDescriptor { &self.output }
    pub fn output_modes(&self) -> &[Mode] { &self.output_modes }
    pub fn compute_type(&self) -> ComputeType { self.compute }
    pub fn scalar_count(&self) -> usize {
        match self.kind {
            Kind::Elementwise(_, _) => self.terms,
            _ => 1 + usize::from(self.addend),
        }
    }
}
