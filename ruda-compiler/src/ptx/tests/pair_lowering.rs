use super::*;

fn generate(k: KernelDefinition, sm: u32, version: (u32, u32)) -> PtxKernel {
    PtxCompiler
        .compile(
            k,
            &PtxCompilationOptions {
                target: Some(PtxTarget { sm, version }),
            },
            ExecutionMode::Unchecked,
            UIntKind::U32.into(),
        )
        .unwrap()
}

#[test]
fn pairs_follow_type_target_and_vector_width() {
    for (kind, sm, version, packed) in [
        (FloatKind::F32, 89, (8, 1), false),
        (FloatKind::F32, 100, (8, 6), true),
        (FloatKind::F16, 75, (7, 0), true),
        (FloatKind::BF16, 80, (7, 0), true),
        (FloatKind::BF16, 90, (7, 8), true),
        (FloatKind::F16, 89, (8, 1), true),
    ] {
        for width in [2, 3, 8] {
            let ty = Type::from(kind).with_vector_size(width);
            let mut k = kernel(&format!("arithmetic_{kind:?}_{width}_sm{sm}"));
            let a = buffer(&mut k, ty, Visibility::Read);
            let b = buffer(&mut k, ty, Visibility::Read);
            let c = buffer(&mut k, ty, Visibility::ReadWrite);
            let index = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
            let x = local(0, ty);
            let y = local(1, ty);
            let acc = local(2, ty);
            load(&mut k.body, a, index, x);
            load(&mut k.body, b, index, y);
            k.body.register(Instruction::new(
                Arithmetic::Add(BinaryOperator { lhs: x, rhs: y }),
                acc,
            ));
            let broadcast = Variable::constant(ConstantValue::Float(0.75), Type::from(kind));
            k.body.register(Instruction::new(
                Arithmetic::Sub(BinaryOperator {
                    lhs: acc,
                    rhs: broadcast,
                }),
                acc,
            ));
            k.body.register(Instruction::new(
                Arithmetic::Mul(BinaryOperator { lhs: acc, rhs: y }),
                acc,
            ));
            k.body.register(Instruction::new(
                Arithmetic::Fma(FmaOperator { a: x, b: y, c: acc }),
                acc,
            ));
            store(&mut k.body, c, index, acc);
            let p = generate(k, sm, version);
            let ops: Vec<_> = p
                .source
                .lines()
                .filter(|l| l.contains(".rn.") && l.contains("x2 "))
                .collect();
            assert_eq!(ops.len(), if packed { 4 * (width / 2) } else { 0 });
            if kind == FloatKind::BF16 && sm == 80 {
                assert!(ops.iter().all(|op| op.contains("fma.rn.bf16x2")));
            }
            if packed && width == 3 {
                assert_eq!(
                    p.source
                        .lines()
                        .filter(|l| l.contains(".rn.") && !l.contains("x2 "))
                        .count(),
                    4
                );
            }
            export(&p);
        }
    }
}

#[test]
fn round_trip_casts_use_both_lanes_and_preserve_target_errors() {
    for (src, dst) in [
        (FloatKind::F32, FloatKind::BF16),
        (FloatKind::F32, FloatKind::F16),
        (FloatKind::F32, FloatKind::E4M3),
        (FloatKind::F32, FloatKind::E5M2),
        (FloatKind::F16, FloatKind::E4M3),
    ] {
        for width in [3, 8] {
            let mut k = kernel(&format!("convert_{src:?}_{dst:?}_{width}"));
            let source_ty = Type::from(src).with_vector_size(width);
            let packed_ty = Type::from(dst).with_vector_size(width);
            let target_ty = Type::from(FloatKind::F32).with_vector_size(width);
            let input = buffer(&mut k, source_ty, Visibility::Read);
            let output = buffer(&mut k, target_ty, Visibility::ReadWrite);
            let index = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
            let x = local(0, source_ty);
            let y = local(1, packed_ty);
            let z = local(2, target_ty);
            load(&mut k.body, input, index, x);
            k.body.register(Instruction::new(
                Operator::Cast(UnaryOperator { input: x }),
                y,
            ));
            k.body.register(Instruction::new(
                Operator::Cast(UnaryOperator { input: y }),
                z,
            ));
            store(&mut k.body, output, index, z);
            let p = generate(k.clone(), 89, (8, 1));
            let fp8 = matches!(dst, FloatKind::E4M3 | FloatKind::E5M2);
            if fp8 {
                let options = PtxCompilationOptions {
                    target: Some(PtxTarget {
                        sm: 89,
                        version: (8, 0),
                    }),
                };
                assert!(
                    PtxCompiler
                        .compile(k, &options, ExecutionMode::Unchecked, UIntKind::U32.into())
                        .is_err()
                );
                assert_eq!(
                    p.source.matches("cvt.rn.satfinite.").count(),
                    width.div_ceil(2)
                );
                assert_eq!(
                    p.source.matches("cvt.rn.f16x2.e").count(),
                    width.div_ceil(2)
                );
            } else {
                assert_eq!(p.source.matches("x2.f32 ").count(), width / 2);
            }
            export(&p);
        }
    }
}

#[test]
fn whole_partial_and_divergent_warps() {
    for (threads, sparse) in [(64, false), (19, false), (64, true)] {
        let mut k = kernel(&format!("masked_sum_{threads}_{}", u8::from(sparse)));
        k.ruda_dim = RudaDim::new_1d(threads);
        let ty = Type::from(FloatKind::F32);
        let input = buffer(&mut k, ty, Visibility::Read);
        let output = buffer(&mut k, ty, Visibility::ReadWrite);
        let idx = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
        if sparse {
            let u32_ty = Type::from(UIntKind::U32);
            let low = local(2, u32_ty);
            k.body.register(Instruction::new(
                Bitwise::BitwiseAnd(BinaryOperator {
                    lhs: idx,
                    rhs: 3u32.into(),
                }),
                low,
            ));
            let skip = local(3, Type::scalar(ElemType::Bool));
            k.body.register(Instruction::new(
                Comparison::Equal(BinaryOperator {
                    lhs: low,
                    rhs: 0u32.into(),
                }),
                skip,
            ));
            let mut early = k.body.child();
            early.register(Branch::Return);
            k.body.register(Branch::If(Box::new(If {
                cond: skip,
                scope: early,
            })));
        }
        let x = local(0, ty);
        let y = local(1, ty);
        load(&mut k.body, input, idx, x);
        k.body
            .register(Instruction::new(Plane::Sum(UnaryOperator { input: x }), y));
        store(&mut k.body, output, idx, y);
        let p = generate(k, 89, (8, 1));
        assert_eq!(p.source.contains("%lanemask_lt"), sparse);
        assert_eq!(p.source.contains("activemask.b32"), sparse);
        if threads == 19 {
            assert!(p.source.contains("0x0007ffff"));
            assert!(!p.source.contains("popc.b32"));
        }
        export(&p);
    }
}

#[test]
fn branch_join_recovers_mask_but_not_inside_a_divergent_arm() {
    let mut k = kernel("joined_planes");
    let ty = Type::from(FloatKind::F32);
    let input = buffer(&mut k, ty, Visibility::Read);
    let output = buffer(&mut k, ty, Visibility::ReadWrite);
    let idx = Variable::builtin(Builtin::UnitPos, UIntKind::U32.into());
    let x = local(0, ty);
    let part = local(1, ty);
    let cond = local(2, Type::scalar(ElemType::Bool));
    load(&mut k.body, input, idx, x);
    k.body.register(Instruction::new(
        Comparison::Lower(BinaryOperator {
            lhs: idx,
            rhs: 5u32.into(),
        }),
        cond,
    ));
    let mut arm = k.body.child();
    arm.register(Instruction::new(
        Plane::Sum(UnaryOperator { input: x }),
        part,
    ));
    store(&mut arm, output, idx, part);
    k.body
        .register(Branch::If(Box::new(If { cond, scope: arm })));
    k.body.register(Instruction::new(
        Plane::Sum(UnaryOperator { input: x }),
        part,
    ));
    store(&mut k.body, output, idx, part);
    let p = generate(k, 89, (8, 1));
    assert_eq!(p.source.matches("activemask.b32").count(), 1);
}

#[test]
fn repeated_access_reuses_buffer_alignment_without_assuming_the_view_is_aligned() {
    let mut k = kernel("repeated_vector_input");
    let ty = Type::from(FloatKind::F32).with_vector_size(4);
    let a = buffer(&mut k, ty, Visibility::Read);
    let b = buffer(&mut k, ty, Visibility::ReadWrite);
    let idx = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
    let x = local(0, ty);
    let y = local(1, ty);
    load(&mut k.body, a, idx, x);
    load(&mut k.body, a, 0u32.into(), y);
    k.body.register(Instruction::new(
        Arithmetic::Add(BinaryOperator { lhs: x, rhs: y }),
        y,
    ));
    store(&mut k.body, b, idx, y);
    let p = generate(k, 89, (8, 1));
    assert_eq!(p.source.matches("setp.ne.u64").count(), 2); // two bases, three accesses
    assert!(p.source.contains("ld.global.f32"));
    assert!(p.source.contains("st.global.f32"));
    export(&p);
}

#[test]
fn row_layouts_compile_only_the_requested_specializations() {
    use super::row::{compile_row_variants, row_layouts};
    assert!(row_layouts(0, 1024).is_empty());
    assert!(row_layouts(896, 1024).iter().any(|l| l.threads_per_row() == 1024));
    assert!(row_layouts(256, 1024).iter().all(|l| l.threads_per_row() <= 256));
    assert!(row_layouts(200, 16).is_empty());
    let short = row_layouts(13, 128);
    assert!(
        short
            .iter()
            .all(|l| l.threads_per_row() == 32 && l.block().x <= 128)
    );
    let compiled = compile_row_variants(
        49,
        128,
        &options(),
        ExecutionMode::Unchecked,
        UIntKind::U32.into(),
        |layout| {
            if layout.rows_per_block() == 1 {
                return None;
            }
            let mut k = kernel("generic_rows");
            k.ruda_dim = layout.block();
            Some(k)
        },
    )
    .unwrap();
    assert!(!compiled.is_empty());
    use std::time::Duration;
    assert!(super::row::select_row_variant(&compiled, |_, _| None).is_none());
    let selected = super::row::select_row_variant(&compiled, |layout, _| {
        (layout.threads_per_row() == 64).then_some(Duration::from_nanos(10))
    })
    .unwrap();
    assert_eq!(selected.0.threads_per_row(), 64);
    for (layout, code) in compiled {
        assert_eq!(layout.block(), code.ruda_dim);
        assert_eq!(layout.blocks(0), 0);
        assert_eq!(layout.blocks(layout.rows_per_block() + 1), 2);
    }
    assert!(
        compile_row_variants(
            49,
            128,
            &options(),
            ExecutionMode::Unchecked,
            UIntKind::U32.into(),
            |_| Some(kernel("wrong_dimensions"))
        )
        .is_err()
    );
}
