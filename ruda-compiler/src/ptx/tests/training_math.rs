use super::*;

#[test]
fn erf_and_powf_compile_for_training_storage() {
    for kind in [FloatKind::F32, FloatKind::F16, FloatKind::BF16] {
        let ty = Type::scalar(ElemType::Float(kind));
        for power in [false, true] {
            let mut k = kernel(&format!("training_math_{kind:?}_{power}"));
            let input = buffer(&mut k, ty, Visibility::Read);
            let exponent = buffer(&mut k, ty, Visibility::Read);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let idx = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
            let x = local(0, ty);
            let y = local(1, ty);
            let z = local(2, ty);
            load(&mut k.body, input, idx, x);
            load(&mut k.body, exponent, idx, y);
            let op = if power { Arithmetic::Powf(BinaryOperator { lhs: x, rhs: y }) }
                else { Arithmetic::Erf(UnaryOperator { input: x }) };
            k.body.instructions.push(Instruction::new(op, z));
            store(&mut k.body, output, idx, z);
            let mut target = options();
            if kind != FloatKind::BF16 { target.target.as_mut().unwrap().sm = 75; }
            let result = PtxCompiler.compile(k, &target, ExecutionMode::Checked, UIntKind::U32.into()).unwrap();
            assert!(!result.source.contains("call.uni"));
            assert!(result.source.contains(if power { "mul.rn.f64" } else { "fma.rn.f32" }));
            export(&result);
        }
    }
}
