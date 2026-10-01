//! Tests of the production Rust lowerer, allocator and emitter. These require
//! cargo; the Python host reference is deliberately a separate validation tier.
use super::{rows::{self, BindingKind, Binary, Node, Reduction, Unary}, row_programs::{self, RowProgram}, *};
use ruda_core::{compiler::Compiler, ir::*, kernel::*, launch::RudaDim};
fn opts(rows: u64, width: u32) -> AscendOptions { AscendOptions { target: Some(AscendTarget::Ascend950DT),
    elements: rows * width as u64, row_width: Some(width), ..Default::default() } }
fn compile(op: RowProgram, rows: u64, width: u32) -> Result<AscendKernel> {
    AscendCompiler.compile(row_programs::definition(op, width, 1e-5)?, &opts(rows, width), ExecutionMode::Checked, UIntKind::U64.into())
}
fn mutated(k: KernelDefinition) -> Result<AscendKernel> {
    AscendCompiler.compile(k, &opts(3, 64), ExecutionMode::Checked, UIntKind::U64.into())
}
#[test] fn row_programs_compile_at_all_declared_widths() {
    for op in RowProgram::ALL { for w in [32, 64, 96, 256, 1024, 4096] {
        let k = compile(op, 3, w).unwrap(); assert_eq!(k.row_width(), Some(w));
        assert!(k.ub_bytes() <= 131072); assert!(k.source().contains("GetBlockIdx"));
        assert!(k.build_contract().contains("common-row.v1"));
    }}
}
#[test] fn row_schema_cannot_be_confused_with_old_map_schema() {
    let k = compile(RowProgram::Softmax, 3, 64).unwrap();
    assert!(k.build_contract().contains("logical_plane=32"));
    assert!(!k.build_contract().contains("common-map.v1"));
}
#[test] fn per_row_stat_and_shared_weight_lengths_are_exact() {
    let k = compile(RowProgram::RmsNorm, 3, 64).unwrap();
    assert_eq!(k.bindings().iter().map(|b| b.bytes).collect::<Vec<_>>(), [768, 256, 768, 12]);
    let k = compile(RowProgram::RmsNormInputBackward, 3, 64).unwrap();
    assert_eq!(k.bindings().iter().map(|b| b.bytes).collect::<Vec<_>>(), [768, 768, 256, 12, 768]);
}
#[test] fn empty_batch_has_no_reduction_or_write_at_runtime() {
    let k = compile(RowProgram::RmsNorm, 0, 64).unwrap();
    assert!(k.source().contains("if (rows == 0) { return; }"));
    assert_eq!(k.bindings().iter().map(|b| b.bytes).collect::<Vec<_>>(), [0, 256, 0, 0]);
}
#[test] fn invalid_widths_and_fractional_rows_fail() {
    for w in [0, 1, 31, 33, 4097, 8192] { assert!(row_programs::definition(RowProgram::Softmax, w, 1e-5).is_err()); }
    let mut o = opts(1, 64); o.elements = 65;
    assert!(AscendCompiler.compile(row_programs::definition(RowProgram::Softmax, 64, 1e-5).unwrap(), &o, ExecutionMode::Checked, UIntKind::U64.into()).is_err());
}
#[test] fn invalid_epsilon_rejected() {
    for e in [0.0, -1.0, f32::NAN, f32::INFINITY] { assert!(row_programs::definition(RowProgram::RmsNorm, 64, e).is_err()); }
}
#[test] fn multiple_logical_planes_are_never_collapsed() {
    for lanes in [1, 16, 64, 128] { let mut k = row_programs::definition(RowProgram::Softmax, 64, 1e-5).unwrap();
        k.ruda_dim = RudaDim::new_1d(lanes); assert!(mutated(k).is_err()); }
}
#[test] fn reduction_is_exactly_one_logical_plane_not_full_row() {
    let k = compile(RowProgram::Softmax, 3, 256).unwrap();
    for line in k.source().lines().filter(|l| l.contains("AscendC::Reduce")) {
        assert!(line.contains(", 32)") || line.contains(", 32, false)"));
    }
    assert!(k.source().contains("ruda_row_sync<AscendC::HardEvent::V_S>(pipe)"));
    assert!(k.source().contains("ruda_row_sync<AscendC::HardEvent::S_V>(pipe)"));
    assert!(k.source().contains("pipe.FetchEventID(event)"));
}
#[test] fn unsupported_plane_op_rejected() {
    let mut k = row_programs::definition(RowProgram::Sum, 64, 1e-5).unwrap();
    for i in &mut k.body.instructions { if let Operation::Plane(Plane::Sum(x)) = i.operation.clone() {
        i.operation = Plane::InclusiveSum(x).into(); break;
    }} assert!(mutated(k).is_err());
}
#[test] fn partial_output_is_rejected_before_emitting() {
    let mut k = row_programs::definition(RowProgram::Softmax, 64, 1e-5).unwrap();
    k.body.instructions.pop(); assert!(mutated(k).is_err());
}
#[test] fn duplicate_chunk_store_is_rejected() {
    let mut k = row_programs::definition(RowProgram::Softmax, 64, 1e-5).unwrap();
    k.body.instructions.push(k.body.instructions.last().unwrap().clone()); assert!(mutated(k).is_err());
}
#[test] fn unpredicated_scalar_store_is_rejected() {
    let mut k = row_programs::definition(RowProgram::Sum, 64, 1e-5).unwrap();
    let i = k.body.instructions.pop().unwrap();
    if let Operation::Branch(Branch::If(b)) = i.operation { k.body.instructions.extend(b.scope.instructions); } else { panic!(); }
    assert!(mutated(k).is_err());
}
#[test] fn plane_arithmetic_inside_single_lane_branch_is_rejected() {
    let mut k = row_programs::definition(RowProgram::Sum, 64, 1e-5).unwrap();
    let plane = k.body.instructions.iter().find(|i| matches!(i.operation, Operation::Plane(_))).unwrap().clone();
    if let Operation::Branch(Branch::If(b)) = &mut k.body.instructions.last_mut().unwrap().operation { b.scope.instructions.insert(0, plane); }
    assert!(mutated(k).is_err());
}
#[test] fn nonuniform_scalar_store_is_rejected() {
    let mut k = row_programs::definition(RowProgram::Sum, 64, 1e-5).unwrap();
    let x = k.body.instructions.iter().find(|i| matches!(i.operation, Operation::Operator(Operator::Index(_)))).unwrap().out.unwrap();
    if let Operation::Branch(Branch::If(b)) = &mut k.body.instructions.last_mut().unwrap().operation {
        if let Operation::Operator(Operator::IndexAssign(op)) = &mut b.scope.instructions[0].operation { op.value = x; }
    } assert!(mutated(k).is_err());
}
#[test] fn low_precision_is_not_silently_promoted() {
    let mut k = row_programs::definition(RowProgram::Softmax, 64, 1e-5).unwrap();
    k.buffers[0].ty = Type::scalar(ElemType::Float(FloatKind::F16)); assert!(mutated(k).is_err());
}
#[test] fn static_lengths_are_validated_per_binding() {
    let mut k = row_programs::definition(RowProgram::RmsNorm, 64, 1e-5).unwrap();
    k.buffers[1].size = Some(192); assert!(mutated(k).is_err());
}
#[test] fn mixed_binding_index_layout_is_rejected() {
    let mut k = row_programs::definition(RowProgram::RmsNorm, 64, 1e-5).unwrap();
    let gamma = Variable::new(VariableKind::GlobalInputArray(1), Type::scalar(ElemType::Float(FloatKind::F32)));
    let idx = k.body.instructions.iter().find_map(|i| if let Operation::Operator(Operator::Index(op)) = &i.operation { Some(op.index) } else { None }).unwrap();
    for i in &mut k.body.instructions { if let Operation::Operator(Operator::Index(op)) = &mut i.operation {
        if op.list == gamma { op.index = idx; break; }
    }} assert!(mutated(k).is_err());
}
#[test] fn map_compiler_does_not_accept_row_definitions_by_name() {
    let o = AscendOptions { target: Some(AscendTarget::Ascend950DT), elements: 192, ..Default::default() };
    assert!(AscendCompiler.compile(row_programs::definition(RowProgram::Softmax, 64, 1e-5).unwrap(), &o, ExecutionMode::Checked, UIntKind::U64.into()).is_err());
    let mut k = row_programs::definition(RowProgram::Softmax, 64, 1e-5).unwrap(); k.options.kernel_name = "custom_equations".into();
    assert!(mutated(k).is_ok());
}
#[test] fn ub_limit_failure_is_explicit() {
    let mut o = opts(3, 64); o.ub_limit_bytes = 32;
    assert!(AscendCompiler.compile(row_programs::definition(RowProgram::Softmax, 64, 1e-5).unwrap(), &o, ExecutionMode::Checked, UIntKind::U64.into()).is_err());
}
#[test] fn row_temporaries_never_alias_current_operands() {
    let p = rows::lower(row_programs::definition(RowProgram::RmsNormInputBackward, 256, 1e-5).unwrap(), 3*256, 256).unwrap();
    let a = rows::allocate(&p, true, 131072).unwrap(); let fresh = rows::allocate(&p, false, 131072).unwrap();
    assert!(a.slots < fresh.slots);
    for (n, node) in p.nodes.iter().enumerate() { for src in node.inputs() {
        if let (Some(x), Some(y)) = (a.node_slots[n], a.node_slots[src]) { assert_ne!(x, y); }
    }}
}
#[test] fn scalar_inputs_use_explicit_mte_to_scalar_dependency() {
    let k = compile(RowProgram::RmsNormInputBackward, 3, 64).unwrap();
    assert!(k.source().contains("HardEvent::MTE2_S"));
    assert!(k.source().contains("Copy(")); assert!(!k.source().contains("DataCopy(out"));
}
#[test] fn exact_same_row_ir_compiles_as_ptx_when_enabled() {
    #[cfg(feature = "ptx")] {
        use crate::ptx::*;
        for op in [RowProgram::RmsNorm, RowProgram::Softmax, RowProgram::RmsNormInputBackward] {
            let ir = row_programs::definition(op, 64, 1e-5).unwrap();
            let ptx = PtxCompiler.compile(ir.clone(), &PtxCompilationOptions { target: Some(PtxTarget { version: (8, 0), sm: 75 }) }, ExecutionMode::Checked, UIntKind::U64.into()).unwrap();
            assert!(ptx.source.contains(".entry")); assert!(mutated(ir).is_ok());
        }
    }
}
/// Test-only evaluator of the actual lowered nodes; never part of CannProgram.
fn evaluate(p: &rows::Program, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut output: Vec<Vec<f32>> = p.bindings.iter().filter(|b| b.arg.visibility == Visibility::ReadWrite)
        .map(|b| vec![f32::NAN; b.kind.count(p.rows, p.width) as usize]).collect();
    for r in 0..p.rows as usize {
        let mut values: Vec<Vec<f32>> = vec![];
        for node in &p.nodes {
            let y = match *node {
                Node::Input { binding, offset } => match p.bindings[binding].kind {
                    BindingKind::Matrix => input[binding][r*p.width as usize+offset as usize..r*p.width as usize+offset as usize+32].to_vec(),
                    BindingKind::SharedRow => input[binding][offset as usize..offset as usize+32].to_vec(),
                    BindingKind::RowScalar => vec![input[binding][r]; 32],
                },
                Node::Constant(bits) => vec![f32::from_bits(bits); 32],
                Node::Unary(op, x) => values[x].iter().map(|&x| match op { Unary::Neg => -x, Unary::Abs => x.abs(),
                    Unary::Exp => x.exp(), Unary::Log => x.ln(), Unary::Sqrt => x.sqrt(), Unary::Rsqrt => x.sqrt().recip(), Unary::Recip => x.recip() }).collect(),
                Node::Binary(op, x, y) => values[x].iter().zip(&values[y]).map(|(&x, &y)| match op {
                    Binary::Add => x+y, Binary::Sub => x-y, Binary::Mul => x*y, Binary::Div => x/y, Binary::Max => x.max(y) }).collect(),
                Node::Reduce(op, x) => { let v = match op { Reduction::Sum => values[x].iter().sum(),
                    Reduction::Max => values[x].iter().copied().fold(f32::NEG_INFINITY, f32::max) }; vec![v; 32] },
            }; values.push(y);
        }
        for &(binding, col, src) in &p.stores {
            let n = p.bindings[..binding].iter().filter(|b| b.arg.visibility == Visibility::ReadWrite).count();
            if p.bindings[binding].kind == BindingKind::RowScalar { output[n][r] = values[src][0]; }
            else { let start = r*p.width as usize+col as usize; output[n][start..start+32].copy_from_slice(&values[src]); }
        }
    } output
}
fn eval(op: RowProgram, width: u32, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let p = rows::lower(row_programs::definition(op, width, 1e-5).unwrap(), input[0].len() as u64, width).unwrap(); evaluate(&p, input)
}
#[test] fn softmax_and_input_gradient_match_reference() {
    for width in [32, 96, 256] {
        let x: Vec<f32> = (0..3*width).map(|i| (i%41) as f32/7.0-2.0).collect();
        let dy: Vec<f32> = (0..3*width).map(|i| (i%13) as f32/9.0-0.5).collect();
        let y = eval(RowProgram::Softmax, width as u32, &[x.clone()])[0].clone();
        let dx = eval(RowProgram::SoftmaxBackward, width as u32, &[y.clone(), dy.clone()])[0].clone();
        for row in 0..3 { let off = row*width; let xx = &x[off..off+width];
            let max = xx.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let e: Vec<_> = xx.iter().map(|&x| (x as f64-max).exp()).collect(); let den: f64 = e.iter().sum();
            let dot: f64 = e.iter().zip(&dy[off..off+width]).map(|(v, &g)| v/den*g as f64).sum();
            for i in 0..width { let yy = e[i]/den; assert!((y[off+i] as f64-yy).abs()<2e-6);
                assert!((dx[off+i] as f64-yy*(dy[off+i] as f64-dot)).abs()<2e-6); }
        }
    }
}
#[test] fn rmsnorm_input_gradient_matches_finite_difference() {
    let width = 32; let x: Vec<f32> = (0..width).map(|i| (i%11) as f32/6.0-0.7).collect();
    let dy: Vec<f32> = (0..width).map(|i| (i%7) as f32/9.0-0.3).collect(); let w = vec![1.1; width];
    let fwd = eval(RowProgram::RmsNorm, width as u32, &[x.clone(), w.clone()]);
    let dx = eval(RowProgram::RmsNormInputBackward, width as u32, &[x.clone(), dy.clone(), w.clone(), fwd[1].clone()])[0].clone();
    let loss = |v: &[f64]| { let r = (v.iter().map(|x| x*x).sum::<f64>()/width as f64+1e-5).sqrt().recip();
        v.iter().zip(&w).zip(&dy).map(|((&x, &w), &g)| x*r*w as f64*g as f64).sum::<f64>() };
    for j in 0..width { let mut hi: Vec<f64> = x.iter().map(|&v| v as f64).collect(); let mut lo = hi.clone(); hi[j]+=1e-5; lo[j]-=1e-5;
        let fd = (loss(&hi)-loss(&lo))/2e-5; assert!((dx[j] as f64-fd).abs()<1e-5); }
}
#[test] fn mean_of_large_finite_values_has_no_early_low_precision_rounding() {
    let out = eval(RowProgram::Mean, 4096, &[vec![1000.0; 4096]]); assert_eq!(out, vec![vec![1000.0]]);
}
