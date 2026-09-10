use crate::dsl::unexpanded;
use crate::dsl::{
    expand_assert,
    ir::{Instruction, Operator, Scope, UnaryOperator},
};
use crate::dsl::{
    expand_error,
    frontend::{RudaPrimitive, RudaType, cast},
};

use super::NativeExpand;

/// Enable elegant casting from any to any `RudaElem`
pub trait Cast: RudaPrimitive {
    fn cast_from<From: RudaPrimitive>(value: From) -> Self;

    fn __expand_cast_from<From: RudaPrimitive>(
        scope: &mut Scope,
        value: NativeExpand<From>,
    ) -> <Self as RudaType>::ExpandType {
        if Self::as_type(scope) == value.expand.ty {
            return value.expand.into();
        }
        let vec_in = value.expand.vector_size();
        let elems_in = vec_in * value.expand.ty.packing_factor();
        let elems_out = Self::__expand_vector_size(scope) * Self::__expand_packing_factor(scope);
        if vec_in > 1 && elems_in != elems_out {
            expand_error!("Cast element count must match if input is not scalar");
        }
        let new_var = scope.create_local(<Self as RudaPrimitive>::as_type(scope));
        cast::expand::<From, Self>(scope, value, new_var.clone().into());
        new_var.into()
    }
}

impl<P: RudaPrimitive> Cast for P {
    fn cast_from<From: RudaPrimitive>(_value: From) -> Self {
        unexpanded!()
    }
}

/// Enables reinterpetring the bits from any value to any other type of the same size.
pub trait Reinterpret: RudaPrimitive {
    /// Reinterpret the bits of another primitive as this primitive without conversion.
    #[allow(unused_variables)]
    fn reinterpret<From: RudaPrimitive>(value: From) -> Self {
        unexpanded!()
    }

    /// Calculates the expected vectorization for the reinterpret target
    fn reinterpret_vectorization<From: RudaPrimitive>() -> usize {
        unexpanded!()
    }

    fn __expand_reinterpret<From: RudaPrimitive>(
        scope: &mut Scope,
        value: NativeExpand<From>,
    ) -> <Self as RudaType>::ExpandType {
        let size_in = value.expand.ty.size();
        let size_out = Self::__expand_type_size(scope);
        expand_assert!(size_in == size_out, "Reinterpret type sizes must match");
        let new_var = scope.create_local(<Self as RudaPrimitive>::as_type(scope));
        scope.register(Instruction::new(
            Operator::Reinterpret(UnaryOperator {
                input: *value.expand,
            }),
            *new_var.clone(),
        ));
        new_var.into()
    }

    fn __expand_reinterpret_vectorization<From: RudaPrimitive>(scope: &mut Scope) -> usize {
        let type_size = From::__expand_type_size(scope);
        type_size / Self::Scalar::__expand_type_size(scope)
    }
}

impl<P: RudaPrimitive> Reinterpret for P {}
