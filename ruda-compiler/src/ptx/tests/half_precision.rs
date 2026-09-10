use super::*;

#[test]
fn half_constants_do_not_double_round() {
    for (kind, midpoint, tiny, base) in [
        (FloatKind::F16, 1.00048828125f64, 2f64.powi(-25), 0x3c00u16),
        (FloatKind::BF16, 1.00390625f64, 2f64.powi(-134), 0x3f80u16),
    ] {
        let mut k = kernel(&format!("constants_{kind:?}"));
        k.ruda_dim = RudaDim::new_1d(1);
        let ty = Type::scalar(ElemType::Float(kind));
        let output = buffer(&mut k, ty, Visibility::ReadWrite);
        let cases = [
            (midpoint, base),
            (f64::from_bits(midpoint.to_bits() + 1), base + 1),
            (f64::from_bits(midpoint.to_bits() - 1), base),
            (tiny, 0),
            (f64::from_bits(tiny.to_bits() + 1), 1),
            (-0.0, 0x8000),
        ];
        for (index, (value, bits)) in cases.into_iter().enumerate() {
            let scalar = types::Scalar::of(ty).unwrap();
            assert_eq!(
                scalar.constant(ConstantValue::Float(value)).unwrap(),
                format!("0x{bits:04X}")
            );
            store(
                &mut k.body,
                output,
                (index as u32).into(),
                Variable::new(VariableKind::Constant(ConstantValue::Float(value)), ty),
            );
        }
        export(&compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap());
    }
}

#[test]
fn half_storage_and_arithmetic_use_16_bit_registers() {
    for kind in [FloatKind::F16, FloatKind::BF16] {
        let k = add_kernel(
            &format!("add_{kind:?}"),
            Type::scalar(ElemType::Float(kind)),
            UIntKind::U32,
        );
        let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(result.source.contains(".reg .b16"));
        assert!(result.source.contains("ld.global.b16"));
        assert!(result.source.contains("st.global.b16"));
        assert!(result.source.contains(if kind == FloatKind::F16 {
            "add.rn.f16"
        } else {
            "fma.rn.bf16"
        }));
        export(&result);
    }
}

#[test]
fn half_target_requirements_are_explicit() {
    for (kind, sm, version) in [
        (FloatKind::F16, 52, (8, 0)),
        (FloatKind::BF16, 75, (8, 0)),
        (FloatKind::BF16, 80, (6, 0)),
    ] {
        let k = add_kernel(
            "unsupported_target",
            Type::scalar(ElemType::Float(kind)),
            UIntKind::U32,
        );
        let options = PtxCompilationOptions {
            target: Some(PtxTarget { sm, version }),
        };
        assert!(matches!(
            PtxCompiler.compile(k, &options, ExecutionMode::Checked, UIntKind::U32.into()),
            Err(CompilationError::UnsupportedInstruction { .. })
        ));
    }
}

#[test]
fn half_division_is_not_emitted_as_invalid_ptx() {
    for kind in [FloatKind::F16, FloatKind::BF16] {
        let mut k = kernel("half_division");
        let ty = Type::scalar(ElemType::Float(kind));
        let a = Variable::constant(ConstantValue::Float(1.0), ty);
        k.body.register(Instruction::new(
            Arithmetic::Div(BinaryOperator { lhs: a, rhs: a }),
            local(0, ty),
        ));
        let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        let suffix = types::Scalar::of(ty).unwrap().suffix();
        if kind == FloatKind::F16 {
            assert_eq!(result.source.matches("cvt.f32.f16").count(), 2);
        } else {
            assert_eq!(result.source.matches("{0, ").count(), 2);
        }
        assert_eq!(result.source.matches("div.rn.f32").count(), 1);
        assert!(result.source.contains(&format!("cvt.rn.{suffix}.f32")));
        assert!(!result.source.contains("div.rn.f16"));
        assert!(!result.source.contains("div.rn.bf16"));
    }
}

#[test]
fn half_fma_preserves_existing_f32_intermediate() {
    for kind in [FloatKind::F16, FloatKind::BF16] {
        let mut k = kernel("half_fma");
        let ty = Type::scalar(ElemType::Float(kind));
        let a = Variable::constant(ConstantValue::Float(1.25), ty);
        k.body.register(Instruction::new(
            Arithmetic::Fma(FmaOperator { a, b: a, c: a }),
            local(0, ty),
        ));
        let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(result.source.contains("fma.rn.f32"));
        assert!(result.source.contains(&format!(
            "cvt.rn.{}.f32",
            types::Scalar::of(ty).unwrap().suffix()
        )));
        assert!(!result.source.contains("fma.rn.f16"));
        assert!(!result.source.contains("fma.rn.bf16"));
    }
}
