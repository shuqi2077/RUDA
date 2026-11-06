use super::*;

#[test]
fn native_integer_warp_reductions_preserve_type_scan_and_target_contracts() {
    for ty in [Type::from(IntKind::I32), Type::from(UIntKind::U32),
        Type::from(IntKind::I64), Type::from(FloatKind::F32)] {
        for (operation, mnemonic) in [(0, "add"), (1, "min"), (2, "max"), (3, "add"), (4, "add"), (5, "mul")] {
            for (sm, version) in [(75, (7, 0)), (80, (6, 5)), (80, (7, 0)), (89, (8, 0))] {
                let mut k = kernel("warp_reduction");
                let input = buffer(&mut k, ty, Visibility::Read);
                let output = buffer(&mut k, ty, Visibility::ReadWrite);
                let index = Variable::builtin(Builtin::UnitPos, UIntKind::U32.into());
                let value = *k.body.create_local(ty);
                let result = *k.body.create_local(ty);
                load(&mut k.body, input, index, value);
                let unary = UnaryOperator { input: value };
                let op = match operation {
                    0 => Plane::Sum(unary), 1 => Plane::Min(unary), 2 => Plane::Max(unary),
                    3 => Plane::InclusiveSum(unary), 4 => Plane::ExclusiveSum(unary),
                    _ => Plane::Prod(unary),
                };
                k.body.register(Instruction::new(op, result));
                store(&mut k.body, output, index, result);
                let compiled = PtxCompiler.compile(k, &PtxCompilationOptions {
                    target: Some(PtxTarget { sm, version }),
                }, ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap();
                let native = operation < 3 && sm >= 80 && version >= (7, 0)
                    && matches!(ty, Type::Scalar(StorageType::Scalar(ElemType::Int(IntKind::I32) | ElemType::UInt(UIntKind::U32))));
                assert_eq!(compiled.source.contains("redux.sync."), native);
                if native {
                    let suffix = if ty == Type::from(IntKind::I32) { "s32" } else { "u32" };
                    assert!(compiled.source.contains(&format!("redux.sync.{mnemonic}.{suffix}")));
                    assert!(compiled.source.contains(", 0xffffffff;"));
                    assert!(!compiled.source.contains("shfl.sync"));
                }
                if version >= (7, 0) { export(&compiled); }
            }
        }
    }
}
