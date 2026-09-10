use super::*;

#[test]
fn native_memory_preserves_load_store_and_space_conversion() {
    for address in [UIntKind::U32, UIntKind::U64] {
        for shared in [false, true] {
            let mut k = kernel("native_roundtrip");
            let ty: Type = UIntKind::U32.into();
            let input = buffer(&mut k, ty, Visibility::ReadWrite);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let storage = if shared { *k.body.create_shared_array(ty, 64, Some(16)) } else { input };
            let index = Variable::builtin(Builtin::UnitPos, address.into());
            let stored = Variable::builtin(Builtin::UnitPos, UIntKind::U32.into());
            let pointer = *k.body.create_local(UIntKind::U64.into());
            let value = *k.body.create_local(ty);
            k.body.register(Instruction::new(Operator::NativeAddress(BinaryOperator { lhs: storage, rhs: index }), pointer));
            k.body.register(Instruction::no_out(Operator::NativeStore(BinaryOperator { lhs: pointer, rhs: stored })));
            k.body.register(Instruction::new(Operator::NativeLoad(UnaryOperator { input: pointer }), value));
            store(&mut k.body, output, index, value);
            let compiled = compile(k, ExecutionMode::Unchecked, address).unwrap();
            assert!(compiled.source.contains(if shared { "cvta.shared.u64" } else { "cvta.global.u64" }));
            assert!(compiled.source.contains("ld.u32"));
            assert!(compiled.source.contains("st.u32"));
        }
    }
}

#[test]
fn native_pointer_requires_u64() {
    let mut k = kernel("invalid_native_pointer");
    let pointer = local(0, UIntKind::U32.into());
    let value = local(1, UIntKind::U32.into());
    k.body.register(Instruction::new(Operator::NativeLoad(UnaryOperator { input: pointer }), value));
    assert!(compile(k, ExecutionMode::Unchecked, UIntKind::U32).is_err());
}
