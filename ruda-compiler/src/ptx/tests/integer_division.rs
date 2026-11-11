use super::*;

#[test]
fn constant_unsigned_division_preserves_full_width() {
    for (bits, kind) in [(32, UIntKind::U32), (64, UIntKind::U64)] {
        let ty = Type::scalar(ElemType::UInt(kind));
        for divisor in [1, 2, 3, 4, 5, 7, 10, 31, 32, 96, 136, 192, (1u64 << (bits - 1)) + 1, if bits == 32 { u32::MAX as u64 } else { u64::MAX }] {
            for (name, operation) in [
                ("div", Arithmetic::Div as fn(BinaryOperator) -> Arithmetic),
                ("mod", Arithmetic::Modulo), ("rem", Arithmetic::Remainder),
            ] {
                let mut k = kernel(&format!("constant_{name}_{bits}_{divisor}"));
                let input = buffer(&mut k, ty, Visibility::Read);
                let output = buffer(&mut k, ty, Visibility::ReadWrite);
                let index = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
                let value = local(0, ty);
                load(&mut k.body, input, index, value);
                // Use an in-place output to exercise source/output aliasing.
                k.body.instructions.push(Instruction::new(operation(BinaryOperator {
                    lhs: value, rhs: Variable::constant(ConstantValue::UInt(divisor), ty),
                }), value));
                store(&mut k.body, output, index, value);
                let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
                assert!(!compiled.source.contains(&format!("div.u{bits}")));
                assert!(!compiled.source.contains(&format!("rem.u{bits}")));
                if divisor.is_power_of_two() {
                    assert!(compiled.source.contains(&format!("{}", if name == "div" { format!("shr.u{bits}") } else { format!("and.b{bits}") })));
                } else {
                    assert!(compiled.source.contains(&format!("mul.hi.u{bits}"))
                        || compiled.source.contains(&format!("selp.u{bits}")));
                    assert!(!compiled.source.contains("cvt.f"));
                }
                export(&compiled);
            }
        }
    }
}

#[test]
fn zero_and_signed_divisors_keep_existing_lowering() {
    for (ty, rhs, suffix) in [
        (Type::scalar(ElemType::UInt(UIntKind::U32)), ConstantValue::UInt(0), "u32"),
        (Type::scalar(ElemType::Int(IntKind::I32)), ConstantValue::Int(4), "s32"),
    ] {
        let mut k = add_kernel("unchanged_division", ty, UIntKind::U32);
        k.body.instructions[2] = Instruction::new(Arithmetic::Div(BinaryOperator {
            lhs: local(0, ty), rhs: Variable::constant(rhs, ty),
        }), local(2, ty));
        let compiled = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(compiled.source.contains(&format!("div.{suffix}")));
    }
}
