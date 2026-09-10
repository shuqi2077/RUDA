use ruda_core::ir::{
    ElemType, FloatKind, Instruction, IntKind, ManagedVariable, Operator, UIntKind, UnaryOperator,
    Variable,
};

use crate::dsl::prelude::*;

define_scalar!(Compute);
define_scalar!(Exponent);
define_scalar!(Magnitude);
define_size!(Width);

#[ruda]
fn integer_power<F: Float, I: Int, U: Int, N: Size>(
    lhs: Vector<F, N>,
    rhs: Vector<I, N>,
    #[comptime] signed: bool,
) -> Vector<F, N> {
    let mut output = Vector::<F, N>::from_int(1);
    #[unroll]
    for lane in 0..N::value() {
        let mut base = lhs[lane];
        let exponent = rhs[lane];
        let mut magnitude = U::cast_from(exponent);
        if comptime![signed] {
            if exponent < I::from_int(0) {
                magnitude = U::from_int(0) - magnitude;
                base = F::from_int(1) / base;
            }
        }
        let mut result = F::from_int(1);
        while magnitude != U::from_int(0) {
            if (magnitude & U::from_int(1)) != U::from_int(0) {
                result = result * base;
            }
            magnitude = magnitude >> U::from_int(1);
            if magnitude != U::from_int(0) {
                base = base * base;
            }
        }
        output[lane] = result;
    }
    output
}

#[allow(missing_docs)]
pub fn expand_integer_power(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
    let compute = match lhs.elem_type() {
        ElemType::Float(FloatKind::F16 | FloatKind::BF16) => FloatKind::F32,
        ElemType::Float(kind @ (FloatKind::F32 | FloatKind::Flex32 | FloatKind::TF32 | FloatKind::F64)) => kind,
        _ => panic!("integer power lowering requires a supported floating base"),
    };
    let (signed, magnitude) = match rhs.elem_type() {
        ElemType::Int(IntKind::I64) => (true, UIntKind::U64),
        ElemType::UInt(UIntKind::U64) => (false, UIntKind::U64),
        ElemType::Int(_) => (true, UIntKind::U32),
        ElemType::UInt(_) => (false, UIntKind::U32),
        _ => panic!("integer power lowering requires an integer exponent"),
    };
    let width = out.vector_size();
    let exponent_elem = match rhs.elem_type() {
        ElemType::Int(IntKind::I8 | IntKind::I16) => ElemType::Int(IntKind::I32),
        ElemType::UInt(UIntKind::U8 | UIntKind::U16) => ElemType::UInt(UIntKind::U32),
        elem => elem,
    };
    scope.register_type::<Compute>(compute.into());
    scope.register_type::<Exponent>(exponent_elem.into());
    scope.register_type::<Magnitude>(magnitude.into());
    scope.register_size::<Width>(width);
    let base = scope.create_local(Type::new(compute.into()).with_vector_size(width));
    scope.register(Instruction::new(Operator::Cast(UnaryOperator { input: lhs }), *base));
    let exponent = scope.create_local(Type::scalar(exponent_elem).with_vector_size(width));
    scope.register(Instruction::new(Operator::Cast(UnaryOperator { input: rhs }), *exponent));
    let result = integer_power::expand::<Compute, Exponent, Magnitude, Width>(
        scope, base.into(), exponent.into(), signed,
    );
    let result: ManagedVariable = result.into();
    scope.register(Instruction::new(Operator::Cast(UnaryOperator { input: *result }), out));
}
