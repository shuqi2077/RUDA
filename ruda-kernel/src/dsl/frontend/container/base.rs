use crate::dsl::prelude::RudaPrimitive;
use ruda_core::ir::{Instruction, Metadata, Scope, Variable};

pub fn expand_length_native(scope: &mut Scope, list: Variable) -> Variable {
    let out = scope.create_local(usize::as_type(scope));
    scope.register(Instruction::new(
        Metadata::Length { var: list },
        out.clone().into(),
    ));
    out.into()
}
