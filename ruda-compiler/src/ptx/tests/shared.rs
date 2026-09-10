use super::*;

fn shared_kernel(address: UIntKind) -> KernelDefinition {
    let mut k = kernel(&format!("shared_{address:?}"));
    let ty: Type = UIntKind::U64.into();
    let input = buffer(&mut k, ty, Visibility::Read);
    let output = buffer(&mut k, ty, Visibility::ReadWrite);
    let index = Variable::builtin(Builtin::UnitPos, address.into());
    let first = *k.body.create_shared_array(ty, 64, Some(128));
    let second = *k.body.create_shared_array(ty, 64, Some(128));
    let value = *k.body.create_local(ty);
    load(&mut k.body, input, index, value);
    store(&mut k.body, first, index, value);
    k.body.register(Synchronization::SyncRuda);
    load(&mut k.body, first, index, value);
    k.body.register(Synchronization::SyncRuda);
    k.body.register(Marker::Free(first));
    store(&mut k.body, second, index, value);
    k.body.register(Synchronization::SyncRuda);
    load(&mut k.body, second, index, value);
    store(&mut k.body, output, index, value);
    k
}

#[test]
fn shared_liveness_reuses_freed_storage_in_both_address_modes() {
    for address in [UIntKind::U32, UIntKind::U64] {
        let result = compile(shared_kernel(address), ExecutionMode::Checked, address).unwrap();
        assert_eq!(result.shared_memory_bytes, 512);
        assert!(
            result
                .source
                .contains(".extern .shared .align 128 .b8 dynamic_shared_mem[]")
        );
        assert_eq!(result.source.matches("st.shared.u64").count(), 2);
        assert_eq!(result.source.matches("ld.shared.u64").count(), 2);
        assert_eq!(result.source.matches("bar.sync 0;").count(), 3);
        // Only the global load/store have the existing checked-I/O guards.
        assert_eq!(result.source.matches("setp.ge.").count(), 2);
        export(&result);
    }
}

#[test]
fn shared_allocations_without_free_remain_disjoint() {
    let mut k = shared_kernel(UIntKind::U32);
    k.body
        .instructions
        .retain(|op| !matches!(op.operation, Operation::Marker(_)));
    let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert_eq!(result.shared_memory_bytes, 1024);
}

#[test]
fn freeing_unused_shared_storage_does_not_allocate() {
    let mut k = kernel("unused_shared");
    let shared = *k
        .body
        .create_shared_array(Type::from(UIntKind::U64), 64, None);
    k.body.register(Marker::Free(shared));
    let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert_eq!(result.shared_memory_bytes, 0);
    assert!(!result.source.contains("dynamic_shared_mem"));
}

#[test]
fn invalid_shared_layouts_fail_before_allocation() {
    for (alignment, unroll_factor) in [(0, 1), (3, 1), (4, 1), (128, 0)] {
        let mut k = kernel("invalid_shared");
        let ty: Type = UIntKind::U64.into();
        let variable = Variable::new(
            VariableKind::SharedArray {
                id: 0,
                length: 64,
                unroll_factor,
                alignment: Some(alignment),
            },
            ty,
        );
        let value = *k.body.create_local(ty);
        load(&mut k.body, variable, 0u32.into(), value);
        assert!(compile(k, ExecutionMode::Checked, UIntKind::U32).is_err());
    }
}

#[test]
fn shared_unroll_factor_scales_allocation() {
    let mut k = kernel("unrolled_shared");
    let ty: Type = UIntKind::U64.into();
    let variable = Variable::new(
        VariableKind::SharedArray {
            id: 0,
            length: 64,
            unroll_factor: 2,
            alignment: Some(128),
        },
        ty,
    );
    let value = *k.body.create_local(ty);
    load(&mut k.body, variable, 0u32.into(), value);
    let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert_eq!(result.shared_memory_bytes, 64 * 2 * 8);
    assert!(result.source.contains(".extern .shared .align 128 .b8 dynamic_shared_mem[]"));
    assert!(result.source.contains("ld.shared.u64"));
}

#[test]
fn unsigned_size_comparisons_promote_to_u64() {
    for (lhs_ty, rhs_ty) in [
        (UIntKind::U32, UIntKind::U64),
        (UIntKind::U64, UIntKind::U32),
    ] {
        let mut k = kernel("size_comparison");
        let lhs = Variable::constant(ConstantValue::UInt(u32::MAX as u64), Type::from(lhs_ty));
        let rhs = Variable::constant(ConstantValue::UInt(1), Type::from(rhs_ty));
        k.body.register(Instruction::new(
            Comparison::Greater(BinaryOperator { lhs, rhs }),
            local(0, Type::scalar(ElemType::Bool)),
        ));
        let result = compile(k, ExecutionMode::Checked, UIntKind::U64).unwrap();
        assert!(result.source.contains("cvt.u64.u32"));
        assert!(result.source.contains("setp.gt.u64"));
    }
}
