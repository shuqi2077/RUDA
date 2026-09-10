use super::*;

#[test]
fn mixed_array_tensor_metadata_uses_extended_positions_and_scalar_padding() {
    for address in [UIntKind::U32, UIntKind::U64] {
        for dimension in [UIntKind::U32, UIntKind::U64] {
            let mut k = kernel(&format!("metadata_{address:?}_{dimension:?}"));
            let ty = Type::scalar(ElemType::UInt(address));
            for _ in 0..5 {
                buffer(&mut k, ty, Visibility::Read);
            }
            k.buffers[1].has_extended_meta = true;
            k.buffers[3].has_extended_meta = true;
            k.scalars.push(ScalarKernelArg {
                ty: FloatKind::F32.into(),
                count: 1,
            });
            let var = Variable::new(VariableKind::GlobalInputArray(3), ty);
            let dim = Variable::builtin(Builtin::UnitPosX, dimension.into());
            for (id, op) in [
                Metadata::Rank { var },
                Metadata::Shape { var, dim },
                Metadata::Stride { var, dim },
            ]
            .into_iter()
            .enumerate()
            {
                k.body
                    .instructions
                    .push(Instruction::new(op, local(id as u32, ty)));
            }
            let result = compile(k, ExecutionMode::Checked, address).unwrap();
            let bytes = if address == UIntKind::U32 { 4 } else { 8 };
            assert_eq!(result.dynamic_metadata_index, Some(5));
            assert!(result.source.contains(&format!("info[{}]", 8 + 16 * bytes)));
            for field in [11, 13, 15] {
                assert!(
                    result
                        .source
                        .contains(&format!("[info+{}]", 8 + field * bytes))
                );
            }
            assert!(
                result.source.find("buffer_4,").unwrap()
                    < result.source.find(".param .u64 dynamic_meta").unwrap()
            );
            assert!(
                result.source.find(".param .u64 dynamic_meta").unwrap()
                    < result.source.find(".param .align 8 .b8 info").unwrap()
            );
            export(&result);
        }
    }
}

#[test]
fn rank_requires_extended_metadata() {
    let mut k = kernel("array_rank");
    let ty = Type::scalar(ElemType::UInt(UIntKind::U32));
    let var = buffer(&mut k, ty, Visibility::Read);
    k.body
        .instructions
        .push(Instruction::new(Metadata::Rank { var }, local(0, ty)));
    assert!(matches!(
        compile(k, ExecutionMode::Checked, UIntKind::U32),
        Err(CompilationError::Validation { .. })
    ));
}

#[test]
fn array_only_abi_has_no_dynamic_pointer() {
    let k = add_kernel(
        "array",
        Type::scalar(ElemType::Float(FloatKind::F32)),
        UIntKind::U32,
    );
    let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert_eq!(result.dynamic_metadata_index, None);
    assert!(!result.source.contains("dynamic_meta"));
}
