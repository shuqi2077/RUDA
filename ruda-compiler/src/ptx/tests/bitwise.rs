use super::*;

fn unary_kernel(name: &str, input: ElemType, output: ElemType, op: fn(UnaryOperator) -> Bitwise) -> KernelDefinition {
    let mut k = kernel(name);
    let source = buffer(&mut k, Type::scalar(input), Visibility::Read);
    let target = buffer(&mut k, Type::scalar(output), Visibility::ReadWrite);
    let index = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
    let value = local(0, Type::scalar(input));
    let result = local(1, Type::scalar(output));
    load(&mut k.body, source, index, value);
    k.body.instructions.push(Instruction::new(op(UnaryOperator { input: value }), result));
    store(&mut k.body, target, index, result);
    k
}

#[test]
fn emits_integer_bitwise_and_count_operations() {
    for (name, input, suffix) in [
        ("u32", ElemType::UInt(UIntKind::U32), "b32"),
        ("i32", ElemType::Int(IntKind::I32), "b32"),
        ("u64", ElemType::UInt(UIntKind::U64), "b64"),
        ("i64", ElemType::Int(IntKind::I64), "b64"),
    ] {
        let operations: [(&str, fn(UnaryOperator) -> Bitwise, &str, bool); 6] = [
            ("not", Bitwise::BitwiseNot, "not", false),
            ("reverse", Bitwise::ReverseBits, "brev", false),
            ("ones", Bitwise::CountOnes, "popc", true),
            ("leading", Bitwise::LeadingZeros, "clz", true),
            ("trailing", Bitwise::TrailingZeros, "clz", true),
            ("first", Bitwise::FindFirstSet, "clz", true),
        ];
        for (operation, op, opcode, count) in operations {
            let output = if count { ElemType::UInt(UIntKind::U32) } else { input };
            let k = unary_kernel(&format!("{operation}_{name}"), input, output, op);
            let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
            assert!(result.source.contains(&format!("{opcode}.{suffix}")));
            if operation == "trailing" || operation == "first" {
                assert!(result.source.contains(&format!("brev.{suffix}")));
            }
            if operation == "first" {
                assert!(result.source.contains("selp.u32"));
                assert!(result.source.contains("add.u32"));
            }
            export(&result);
        }
    }
}

#[test]
fn emits_binary_bits_and_wide_shift_counts() {
    for elem in [ElemType::UInt(UIntKind::U32), ElemType::Int(IntKind::I32),
                 ElemType::UInt(UIntKind::U64), ElemType::Int(IntKind::I64)] {
        let operations: [(&str, fn(BinaryOperator) -> Bitwise); 5] = [
            ("and", Bitwise::BitwiseAnd), ("or", Bitwise::BitwiseOr),
            ("xor", Bitwise::BitwiseXor), ("shl", Bitwise::ShiftLeft), ("shr", Bitwise::ShiftRight),
        ];
        for (name, op) in operations {
            let ty = Type::scalar(elem);
            let mut k = add_kernel("integer_binary", ty, UIntKind::U32);
            k.body.instructions[2] = Instruction::new(op(BinaryOperator {
                lhs: local(0, ty), rhs: local(1, ty),
            }), local(2, ty));
            let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
            assert!(result.source.contains(&format!("{name}.")));
            if elem.size() == 8 && (name == "shl" || name == "shr") {
                assert!(result.source.contains("min.u64"));
                assert!(result.source.contains("cvt.u32.u64"));
            }
        }
    }
}

#[test]
fn rejects_invalid_bitwise_types() {
    let f32_ty = ElemType::Float(FloatKind::F32);
    let u64_ty = ElemType::UInt(UIntKind::U64);
    for k in [
        unary_kernel("float_not", f32_ty, f32_ty, Bitwise::BitwiseNot),
        unary_kernel("wide_count_output", u64_ty, u64_ty, Bitwise::CountOnes),
    ] {
        assert!(matches!(compile(k, ExecutionMode::Checked, UIntKind::U32), Err(CompilationError::Validation { .. })));
    }
}

#[test]
fn predicate_select_uses_predicate_logic_not_selp_pred() {
    let mut k = kernel("predicate_select");
    let ty = Type::scalar(ElemType::Bool);
    let condition = local(0, ty);
    let yes = Variable::constant(ConstantValue::Bool(true), ty);
    let no = Variable::constant(ConstantValue::Bool(false), ty);
    k.body.instructions.push(Instruction::new(Operation::Copy(yes), condition));
    k.body.instructions.push(Instruction::new(Operator::Select(Select {
        cond: condition, then: no, or_else: yes,
    }), condition));
    let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert!(!result.source.contains("selp.pred"));
    assert!(result.source.contains("and.pred"));
    assert!(result.source.contains("not.pred"));
    assert!(result.source.contains("or.pred"));
    export(&result);
}
