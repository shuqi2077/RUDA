use super::*;

#[test]
fn unsigned_power_of_two_division_and_remainders_are_exact_integer_operations() {
    for (kind, name, bits) in [(UIntKind::U32, "u32", 32u32), (UIntKind::U64, "u64", 64u32)] {
        let ty = Type::from(kind);
        for shift in 0..bits {
            let rhs = ty.constant(ConstantValue::UInt(1u64 << shift));
            for operation in ["div", "mod", "rem"] {
                let mut k = kernel(&format!("unsigned_power2_{name}_{operation}_s{shift}"));
                let input = buffer(&mut k, ty, Visibility::Read);
                let output = buffer(&mut k, ty, Visibility::ReadWrite);
                let index = Variable::builtin(Builtin::UnitPos, UIntKind::U32.into());
                let lhs = *k.body.create_local(ty);
                let result = *k.body.create_local(ty);
                load(&mut k.body, input, index, lhs);
                let op = BinaryOperator { lhs, rhs };
                k.body.register(Instruction::new(match operation {
                    "div" => Arithmetic::Div(op),
                    "mod" => Arithmetic::Modulo(op),
                    _ => Arithmetic::Remainder(op),
                }, result));
                store(&mut k.body, output, index, result);
                let compiled = PtxCompiler.compile(k, &PtxCompilationOptions {
                    target: Some(PtxTarget { sm: 80, version: (8, 0) }),
                }, ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap();
                assert!(compiled.source.contains(if operation == "div" {
                    if bits == 32 { "shr.u32" } else { "shr.u64" }
                } else if bits == 32 { "and.b32" } else { "and.b64" }));
                assert!(!compiled.source.contains("div.u"));
                assert!(!compiled.source.contains("rem.u"));
                assert!(!compiled.source.contains("cvt.f"));
                export(&compiled);
            }
        }
        for divisor in [0, 3, 7, u64::MAX] {
            let mut k = kernel("unsigned_non_power2");
            let input = buffer(&mut k, ty, Visibility::Read);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let lhs = *k.body.create_local(ty);
            let result = *k.body.create_local(ty);
            load(&mut k.body, input, 0u32.into(), lhs);
            k.body.register(Instruction::new(Arithmetic::Div(BinaryOperator {
                lhs, rhs: ty.constant(ConstantValue::UInt(divisor)),
            }), result));
            store(&mut k.body, output, 0u32.into(), result);
            let source = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap().source;
            if divisor == 0 {
                assert!(source.contains(&format!("div.{name}")));
            } else {
                assert!(!source.contains("div.u"));
                assert!(source.contains(&format!("mul.hi.{name}"))
                    || source.contains(&format!("selp.{name}")));
            }
            assert!(!source.contains("cvt.f"));
        }
    }
}
