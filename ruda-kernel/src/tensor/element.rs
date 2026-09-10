use crate::dsl::{RudaElement as RudaElem, flex32, prelude::Numeric};
use ruda_core::tensor::element::Element;
use half::{bf16, f16};

/// The base element trait for the jit backend.
pub trait TensorElement: Element + RudaElem + PartialEq + Numeric {}

impl TensorElement for u64 {}
impl TensorElement for u32 {}
impl TensorElement for u16 {}
impl TensorElement for u8 {}
impl TensorElement for i64 {}
impl TensorElement for i32 {}
impl TensorElement for i16 {}
impl TensorElement for i8 {}
impl TensorElement for f64 {}
impl TensorElement for f32 {}
impl TensorElement for flex32 {}
impl TensorElement for f16 {}
impl TensorElement for bf16 {}
