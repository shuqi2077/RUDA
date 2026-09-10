use core::fmt::Display;

use alloc::{format, vec::Vec};

use crate::ir::{IndexAssignOperator, IndexOperator, TypeHash};

use crate::ir::{BinaryOperator, OperationArgs, OperationReflect, UnaryOperator, Variable};

/// Operators available on the GPU
#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = OperatorOpCode)]
pub enum Operator {
    /// Native byte address of an element in a bound array (U64 result).
    #[operation(pure)]
    NativeAddress(BinaryOperator),
    /// Read through a native U64 byte address. Not a pure operation.
    NativeLoad(UnaryOperator),
    /// Write rhs through the native U64 byte address lhs; no result.
    NativeStore(BinaryOperator),
    #[operation(pure)]
    Index(IndexOperator),
    CopyMemory(CopyMemoryOperator),
    CopyMemoryBulk(CopyMemoryBulkOperator),
    #[operation(pure)]
    UncheckedIndex(IndexOperator),
    IndexAssign(IndexAssignOperator),
    UncheckedIndexAssign(IndexAssignOperator),
    #[operation(pure)]
    InitVector(VectorInitOperator),
    #[operation(commutative, pure)]
    And(BinaryOperator),
    #[operation(commutative, pure)]
    Or(BinaryOperator),
    #[operation(pure)]
    Not(UnaryOperator),
    #[operation(pure)]
    Cast(UnaryOperator),
    #[operation(pure)]
    Reinterpret(UnaryOperator),
    /// A select statement/ternary
    #[operation(pure)]
    Select(Select),
}

impl Display for Operator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Operator::NativeAddress(op) => write!(f, "address({}[{}])", op.lhs, op.rhs),
            Operator::NativeLoad(op) => write!(f, "load_native({})", op.input),
            Operator::NativeStore(op) => write!(f, "store_native({}, {})", op.lhs, op.rhs),
            Operator::Index(op) => write!(f, "{}[{}]", op.list, op.index),
            Operator::CopyMemory(op) => {
                write!(f, "[{}] = {}[{}]", op.out_index, op.input, op.in_index)
            }
            Operator::CopyMemoryBulk(op) => write!(
                f,
                "memcpy([{}], {}[{}], {})",
                op.input, op.in_index, op.out_index, op.len
            ),
            Operator::UncheckedIndex(op) => {
                write!(f, "unchecked {}[{}]", op.list, op.index)
            }
            Operator::IndexAssign(op) => write!(f, "[{}] = {}", op.index, op.value),
            Operator::UncheckedIndexAssign(op) => {
                write!(f, "unchecked [{}] = {}", op.index, op.value)
            }
            Operator::And(op) => write!(f, "{} && {}", op.lhs, op.rhs),
            Operator::Or(op) => write!(f, "{} || {}", op.lhs, op.rhs),
            Operator::Not(op) => write!(f, "!{}", op.input),
            Operator::InitVector(init) => {
                let inits = init
                    .inputs
                    .iter()
                    .map(|input| format!("{input}"))
                    .collect::<Vec<_>>();
                write!(f, "vec({})", inits.join(", "))
            }
            Operator::Select(op) => {
                write!(f, "{} ? {} : {}", op.cond, op.then, op.or_else)
            }
            Operator::Cast(op) => write!(f, "cast({})", op.input),
            Operator::Reinterpret(op) => write!(f, "reinterpret({})", op.input),
        }
    }
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct SliceOperator {
    pub input: Variable,
    pub start: Variable,
    pub end: Variable,
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct ReinterpretSliceOperator {
    pub input: Variable,
    pub vector_size: u32,
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct VectorInitOperator {
    pub inputs: Vec<Variable>,
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct CopyMemoryOperator {
    pub out_index: Variable,
    pub input: Variable,
    pub in_index: Variable,
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct CopyMemoryBulkOperator {
    pub out_index: Variable,
    pub input: Variable,
    pub in_index: Variable,
    pub len: usize,
    pub offset_input: Variable,
    pub offset_out: Variable,
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct Select {
    pub cond: Variable,
    pub then: Variable,
    pub or_else: Variable,
}
