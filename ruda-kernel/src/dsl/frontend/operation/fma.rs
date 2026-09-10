use crate::dsl::{prelude::*, unexpanded};

/// Fused multiply-add `A*B+C`.
#[allow(unused_variables)]
pub fn fma<C: RudaPrimitive>(a: C, b: C, c: C) -> C {
    unexpanded!()
}

/// Expand method of [`fma()`].
pub mod fma {
    use super::*;
    use ruda_core::ir::{Arithmetic, FmaOperator, Instruction, Scope};

    pub fn expand<C: RudaPrimitive>(
        scope: &mut Scope,
        a: NativeExpand<C>,
        b: NativeExpand<C>,
        c: NativeExpand<C>,
    ) -> NativeExpand<C> {
        let output = scope.create_local(a.expand.ty);
        let out = *output;
        let a = *a.expand;
        let b = *b.expand;
        let c = *c.expand;

        scope.register(Instruction::new(
            Arithmetic::Fma(FmaOperator { a, b, c }),
            out,
        ));

        output.into()
    }
}
