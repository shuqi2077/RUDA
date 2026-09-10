use super::*;
use ruda_core::{
    ir::*,
    kernel::{KernelArg, KernelOptions, ScalarKernelArg, Visibility},
};

mod metadata;
mod shared;
mod half_precision;
mod bitwise;

fn options() -> PtxCompilationOptions {
    PtxCompilationOptions {
        target: Some(PtxTarget {
            version: (8, 0),
            sm: 89,
        }),
    }
}

fn local(id: u32, ty: Type) -> Variable {
    Variable::new(VariableKind::LocalMut { id }, ty)
}

fn kernel(name: &str) -> KernelDefinition {
    KernelDefinition {
        buffers: vec![],
        tensor_maps: vec![],
        scalars: vec![],
        ruda_dim: RudaDim::new_1d(64),
        body: Scope::root(false),
        options: KernelOptions {
            kernel_name: name.into(),
            ..Default::default()
        },
    }
}

fn buffer(kernel: &mut KernelDefinition, ty: Type, visibility: Visibility) -> Variable {
    let id = kernel.buffers.len() as u32;
    kernel.buffers.push(KernelArg {
        id,
        visibility,
        ty,
        size: None,
        has_extended_meta: false,
    });
    Variable::new(
        if visibility == Visibility::Read {
            VariableKind::GlobalInputArray(id)
        } else {
            VariableKind::GlobalOutputArray(id)
        },
        ty,
    )
}

fn load(scope: &mut Scope, list: Variable, index: Variable, out: Variable) {
    scope.instructions.push(Instruction::new(
        Operator::Index(IndexOperator {
            list,
            index,
            vector_size: 0,
            unroll_factor: 1,
        }),
        out,
    ));
}

fn store(scope: &mut Scope, array: Variable, index: Variable, value: Variable) {
    scope.instructions.push(Instruction::new(
        Operator::IndexAssign(IndexAssignOperator {
            index,
            value,
            vector_size: 0,
            unroll_factor: 1,
        }),
        array,
    ));
}

fn add_kernel(name: &str, ty: Type, index: UIntKind) -> KernelDefinition {
    let mut k = kernel(name);
    let a = buffer(&mut k, ty, Visibility::Read);
    let b = buffer(&mut k, ty, Visibility::Read);
    let c = buffer(&mut k, ty, Visibility::ReadWrite);
    let idx = Variable::builtin(Builtin::AbsolutePosX, index.into());
    let x = local(0, ty);
    let y = local(1, ty);
    let z = local(2, ty);
    load(&mut k.body, a, idx, x);
    load(&mut k.body, b, idx, y);
    k.body.instructions.push(Instruction::new(
        Arithmetic::Add(BinaryOperator { lhs: x, rhs: y }),
        z,
    ));
    store(&mut k.body, c, idx, z);
    k
}

fn compile(k: KernelDefinition, mode: ExecutionMode, address: UIntKind) -> Result<PtxKernel> {
    PtxCompiler.compile(k, &options(), mode, address.into())
}

fn export(k: &PtxKernel) {
    if let Some(directory) = std::env::var_os("RUDA_PTX_TEST_OUTPUT") {
        let path = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join(format!("{}.ptx", k.entrypoint)), &k.source).unwrap();
    }
}

#[test]
fn emits_existing_ir_and_argument_abi() {
    let ty = Type::scalar(ElemType::Float(FloatKind::F32));
    for (name, index, address) in [
        ("add_f32_i32_a32", UIntKind::U32, UIntKind::U32),
        ("add_f32_i64_a64", UIntKind::U64, UIntKind::U64),
        ("add_f32_i32_a64", UIntKind::U32, UIntKind::U64),
        ("add_f32_i64_a32", UIntKind::U64, UIntKind::U32),
    ] {
        let result = compile(add_kernel(name, ty, index), ExecutionMode::Checked, address).unwrap();
        assert!(
            result
                .source
                .starts_with(".version 8.0\n.target sm_89\n.address_size 64")
        );
        assert!(result.source.contains("add.rn.f32"));
        let bytes = if address == UIntKind::U32 { 24 } else { 48 };
        assert!(
            result
                .source
                .contains(&format!(".param .align 8 .b8 info[{bytes}]"))
        );
        assert_eq!(result.ruda_dim, RudaDim::new_1d(64));
        assert_eq!(result.shared_memory_bytes, 0);
        assert_eq!(result.to_string(), result.source);
        assert_eq!(result.source.matches("bra L").count(), 3);
        export(&result);
    }
}

#[test]
fn scalar_numeric_types() {
    for (name, ty, op) in [
        ("add_f64", ElemType::Float(FloatKind::F64), "add.rn.f64"),
        ("add_i32", ElemType::Int(IntKind::I32), "add.s32"),
        ("add_i64", ElemType::Int(IntKind::I64), "add.s64"),
        ("add_u32", ElemType::UInt(UIntKind::U32), "add.u32"),
        ("add_u64", ElemType::UInt(UIntKind::U64), "add.u64"),
    ] {
        let result = compile(
            add_kernel(name, Type::scalar(ty), UIntKind::U32),
            ExecutionMode::Checked,
            UIntKind::U32,
        )
        .unwrap();
        assert!(result.source.contains(op));
        export(&result);
    }
}

#[test]
fn checked_and_unchecked_are_distinct() {
    let k = add_kernel(
        "add_unchecked",
        Type::scalar(ElemType::Float(FloatKind::F32)),
        UIntKind::U32,
    );
    let result = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap();
    assert!(!result.source.contains("setp.ge"));
    assert!(result.source.contains("ld.global.f32"));
    export(&result);
}

#[test]
fn requires_explicit_target() {
    assert!(matches!(
        PtxCompiler.compile(
            kernel("empty"),
            &PtxCompilationOptions::default(),
            ExecutionMode::Checked,
            UIntKind::U32.into()
        ),
        Err(CompilationError::Validation { .. })
    ));
}

#[test]
fn validation_mode_is_not_silently_unchecked() {
    let k = add_kernel("validate", Type::scalar(ElemType::Float(FloatKind::F32)), UIntKind::U32);
    let checked = compile(k.clone(), ExecutionMode::Checked, UIntKind::U32).unwrap();
    let validated = compile(k, ExecutionMode::Validate, UIntKind::U32).unwrap();
    assert_eq!(validated.source.matches("setp.ge.").count(), 3);
    assert_eq!(validated.source.matches("call.uni (printf_status), vprintf").count(), 3);
    assert!(!checked.source.contains("vprintf"));
    assert_eq!(validated.source.matches("ld.global.f32").count(), 2);
    assert_eq!(validated.source.matches("st.global.f32").count(), 1);
}

#[test]
fn unsupported_ir_is_not_sent_to_nvrtc() {
    let k = add_kernel(
        "unsupported_fp8",
        Type::scalar(ElemType::Float(FloatKind::E4M3)),
        UIntKind::U32,
    );
    assert!(matches!(
        compile(k, ExecutionMode::Checked, UIntKind::U32),
        Err(CompilationError::UnsupportedInstruction { .. })
    ));
    let mut k = kernel("unsupported");
    let ty = Type::scalar(ElemType::Float(FloatKind::F64));
    k.body.instructions.push(Instruction::new(
        Arithmetic::Exp(UnaryOperator {
            input: Variable::constant(ConstantValue::Float(1.0), ty),
        }),
        local(0, ty),
    ));
    assert!(matches!(
        compile(k, ExecutionMode::Checked, UIntKind::U32),
        Err(CompilationError::UnsupportedInstruction { .. })
    ));
}

#[test]
fn f32_exponential_emits_native_ptx() {
    let mut k = kernel("exp_f32");
    let ty = Type::scalar(ElemType::Float(FloatKind::F32));
    k.body.instructions.push(Instruction::new(
        Arithmetic::Exp(UnaryOperator {
            input: Variable::constant(ConstantValue::Float(1.0), ty),
        }),
        local(0, ty),
    ));
    let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert!(result.source.contains("ld.const.u64"));
    assert!(result.source.contains("cvt.rn.f32.f64"));
    assert!(result.source.contains("setp.neu.f32"));
    assert!(result.source.contains("setp.gt.f32"));
    assert!(result.source.contains("setp.lt.f32"));
    assert!(!result.source.contains("ex2.approx"));
}

#[test]
fn rejects_unsupported_index_layout() {
    let mut k = add_kernel(
        "invalid",
        Type::scalar(ElemType::UInt(UIntKind::U32)),
        UIntKind::U32,
    );
    if let Operation::Operator(Operator::Index(op)) = &mut k.body.instructions[0].operation {
        op.unroll_factor = 2;
    }
    assert!(matches!(
        compile(k, ExecutionMode::Checked, UIntKind::U32),
        Err(CompilationError::UnsupportedInstruction { .. })
    ));
}

#[test]
fn validates_invalid_ir_and_read_only_stores() {
    let mut k = kernel("bad-name");
    assert!(matches!(
        compile(k.clone(), ExecutionMode::Checked, UIntKind::U32),
        Err(CompilationError::Validation { .. })
    ));
    k.options.kernel_name = "valid".into();
    k.body.instructions.push(Instruction::no_out(Branch::Break));
    assert!(matches!(
        compile(k, ExecutionMode::Checked, UIntKind::U32),
        Err(CompilationError::Validation { .. })
    ));
    let mut k = add_kernel(
        "readonly",
        Type::scalar(ElemType::UInt(UIntKind::U32)),
        UIntKind::U32,
    );
    k.buffers[2].visibility = Visibility::Read;
    assert!(matches!(
        compile(k, ExecutionMode::Checked, UIntKind::U32),
        Err(CompilationError::Validation { .. })
    ));
    let mut k = kernel("overflow");
    k.ruda_dim = RudaDim::new_3d(u32::MAX, u32::MAX, u32::MAX);
    assert!(matches!(
        compile(k, ExecutionMode::Checked, UIntKind::U32),
        Err(CompilationError::Validation { .. })
    ));
}

#[test]
fn multidimensional_builtins_match_existing_ir_layout() {
    for (name, kind) in [
        ("builtins_u32", UIntKind::U32),
        ("builtins_u64", UIntKind::U64),
    ] {
        let ty = Type::scalar(ElemType::UInt(kind));
        let mut k = kernel(name);
        k.ruda_dim = RudaDim::new_3d(4, 4, 4);
        let output = buffer(&mut k, ty, Visibility::ReadWrite);
        let base = local(0, ty);
        let absolute = Variable::builtin(Builtin::AbsolutePos, kind.into());
        k.body.instructions.push(Instruction::new(
            Arithmetic::Mul(BinaryOperator {
                lhs: absolute,
                rhs: Variable::constant(ConstantValue::UInt(9), ty),
            }),
            base,
        ));
        for (i, builtin) in [
            Builtin::UnitPos,
            Builtin::RudaPos,
            Builtin::RudaDim,
            Builtin::RudaCount,
            Builtin::PlanePos,
            Builtin::UnitPosPlane,
            Builtin::AbsolutePosX,
            Builtin::AbsolutePosY,
            Builtin::AbsolutePosZ,
        ]
        .into_iter()
        .enumerate()
        {
            let index = local(i as u32 + 1, ty);
            k.body.instructions.push(Instruction::new(
                Arithmetic::Add(BinaryOperator {
                    lhs: base,
                    rhs: Variable::constant(ConstantValue::UInt(i as u64), ty),
                }),
                index,
            ));
            store(
                &mut k.body,
                output,
                index,
                Variable::builtin(builtin, kind.into()),
            );
        }
        let result = compile(k, ExecutionMode::Checked, kind).unwrap();
        assert!(result.source.contains(".reqntid 4, 4, 4"));
        export(&result);
    }
}

#[test]
fn compiler_does_not_retain_kernel_state() {
    let mut compiler = PtxCompiler;
    let k = add_kernel(
        "same",
        Type::scalar(ElemType::Float(FloatKind::F32)),
        UIntKind::U32,
    );
    let first = compiler
        .compile(
            k.clone(),
            &options(),
            ExecutionMode::Checked,
            UIntKind::U32.into(),
        )
        .unwrap();
    let second = compiler
        .compile(k, &options(), ExecutionMode::Checked, UIntKind::U32.into())
        .unwrap();
    assert_eq!(first.source, second.source);
}

#[test]
fn exact_float_constants() {
    use super::types::Scalar;
    assert_eq!(
        Scalar::F32.constant(ConstantValue::Float(-0.0)).unwrap(),
        "0f80000000"
    );
    assert_eq!(
        Scalar::F32
            .constant(ConstantValue::Float(f64::INFINITY))
            .unwrap(),
        "0f7F800000"
    );
    assert_eq!(
        Scalar::F64.constant(ConstantValue::Float(-0.0)).unwrap(),
        "0d8000000000000000"
    );
}

#[test]
fn scalar_arguments_nan_select_and_control_flow() {
    let mut k = kernel("classify");
    let f32_ty = Type::scalar(ElemType::Float(FloatKind::F32));
    let u32_ty = Type::scalar(ElemType::UInt(UIntKind::U32));
    let pred_ty = Type::scalar(ElemType::Bool);
    let a = buffer(&mut k, f32_ty, Visibility::Read);
    let c = buffer(&mut k, f32_ty, Visibility::ReadWrite);
    k.scalars.push(ScalarKernelArg {
        ty: f32_ty.storage_type(),
        count: 1,
    });
    let scalar = Variable::new(VariableKind::GlobalScalar(0), f32_ty);
    let idx = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
    let x = local(0, f32_ty);
    let pred = local(1, pred_ty);
    let selected = local(2, f32_ty);
    let iteration = local(3, u32_ty);
    let done = local(4, pred_ty);
    let zero = Variable::constant(ConstantValue::UInt(0), u32_ty);
    let one = Variable::constant(ConstantValue::UInt(1), u32_ty);
    load(&mut k.body, a, idx, x);
    k.body.instructions.push(Instruction::new(
        Comparison::NotEqual(BinaryOperator { lhs: x, rhs: x }),
        pred,
    ));
    k.body.instructions.push(Instruction::new(
        Operator::Select(Select {
            cond: pred,
            then: scalar,
            or_else: x,
        }),
        selected,
    ));
    k.body
        .instructions
        .push(Instruction::new(Operation::Copy(zero), iteration));
    let mut inner = k.body.child();
    inner.instructions.push(Instruction::new(
        Comparison::Equal(BinaryOperator {
            lhs: iteration,
            rhs: one,
        }),
        done,
    ));
    let mut exit = inner.child();
    exit.instructions.push(Instruction::no_out(Branch::Break));
    inner
        .instructions
        .push(Instruction::no_out(Branch::If(Box::new(If {
            cond: done,
            scope: exit,
        }))));
    inner.instructions.push(Instruction::new(
        Arithmetic::Add(BinaryOperator {
            lhs: iteration,
            rhs: one,
        }),
        iteration,
    ));
    let mut yes = inner.child();
    yes.instructions
        .push(Instruction::new(Operation::Copy(scalar), selected));
    let mut no = inner.child();
    no.instructions.push(Instruction::new(
        Arithmetic::Add(BinaryOperator {
            lhs: selected,
            rhs: scalar,
        }),
        selected,
    ));
    inner
        .instructions
        .push(Instruction::no_out(Branch::IfElse(Box::new(IfElse {
            cond: pred,
            scope_if: yes,
            scope_else: no,
        }))));
    k.body
        .instructions
        .push(Instruction::no_out(Branch::Loop(Box::new(Loop {
            scope: inner,
        }))));
    store(&mut k.body, c, idx, selected);
    let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
    assert!(result.source.contains("setp.neu.f32"));
    assert!(result.source.contains("selp.f32"));
    assert!(result.source.contains(".b8 info[24]"));
    assert!(result.source.contains("[info+0]"));
    export(&result);
}
