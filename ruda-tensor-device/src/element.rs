use ruda_tensor::{bf16, f16};
use ruda_kernel::dsl::{
    flex32,
    prelude::{Float, Int},
};
use rublas::kernel_ir::definition::{MatmulPrecision, MatrixPrecision};
use ruprim::reduce::ReducePrecision;

pub use ruda_kernel::tensor::element::TensorElement;

/// Element that can be used for matrix multiplication. Includes ints and floats.
pub trait MatmulElement:
    TensorElement + MatmulPrecision<Acc: MatrixPrecision<Global: TensorElement>>
{
}

/// The float element type for the jit backend.
pub trait FloatElement: MatmulElement + Float {}

/// The int element type for the jit backend.
pub trait IntElement:
    MatmulElement + Int + ReducePrecision<EI: TensorElement, EA: TensorElement>
{
}

/// The element type for booleans for the jit backend.
pub trait BoolElement: TensorElement + Int {
    /// The true value for the boolean element.
    fn true_val() -> Self {
        Self::from_int(1)
    }

    /// The false value for the boolean element.
    fn false_val() -> Self {
        Self::from_int(0)
    }

    /// New bool element from Rust bool.
    fn new_bool(val: bool) -> Self {
        match val {
            true => Self::true_val(),
            false => Self::false_val(),
        }
    }
}


impl FloatElement for f64 {}
impl FloatElement for f32 {}
impl FloatElement for flex32 {}
impl FloatElement for bf16 {}
impl FloatElement for f16 {}
impl IntElement for i64 {}
impl IntElement for i32 {}
impl IntElement for i16 {}
impl IntElement for i8 {}
impl IntElement for u64 {}
impl IntElement for u32 {}
impl IntElement for u16 {}
impl IntElement for u8 {}

impl BoolElement for u8 {}
impl BoolElement for u32 {}

impl MatmulElement for f64 {}
impl MatmulElement for f32 {}
impl MatmulElement for flex32 {}
impl MatmulElement for bf16 {}
impl MatmulElement for f16 {}

impl MatmulElement for i64 {}
impl MatmulElement for i32 {}
impl MatmulElement for i16 {}
impl MatmulElement for i8 {}
impl MatmulElement for u64 {}
impl MatmulElement for u32 {}
impl MatmulElement for u16 {}
impl MatmulElement for u8 {}
