use ruda_core::{e4m3, e5m2, ue8m0};
use ruda_core::ir::{ConstantValue, ElemType, FloatKind, Scope, Type};

use crate::dsl::prelude::*;

impl RudaType for e4m3 {
    type ExpandType = NativeExpand<e4m3>;
}

impl Scalar for e4m3 {}
impl RudaPrimitive for e4m3 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::E4M3).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        e4m3::from_f64(value)
    }
}

impl IntoRuntime for e4m3 {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl NativeAssign for e4m3 {}

impl RudaType for e5m2 {
    type ExpandType = NativeExpand<e5m2>;
}

impl Scalar for e5m2 {}
impl RudaPrimitive for e5m2 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::E5M2).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        e5m2::from_f64(value)
    }
}

impl IntoRuntime for e5m2 {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl NativeAssign for e5m2 {}

impl RudaType for ue8m0 {
    type ExpandType = NativeExpand<ue8m0>;
}

impl Scalar for ue8m0 {}
impl RudaPrimitive for ue8m0 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::UE8M0).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        ue8m0::from_f64(value)
    }
}

impl IntoRuntime for ue8m0 {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl NativeAssign for ue8m0 {}
