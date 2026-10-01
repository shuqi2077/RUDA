use super::*;
use super::{lower::{Node,Binary,Unary},programs::{MapProgram,definition}};
use ruda_core::{ir::*,kernel::*,launch::RudaDim};
fn opt(n:u64)->AscendOptions{AscendOptions{target:Some(AscendTarget::Ascend950DT),elements:n,..Default::default()}}
fn compile(k:KernelDefinition,n:u64)->Result<AscendKernel>{AscendCompiler.compile(k,&opt(n),ExecutionMode::Checked,UIntKind::U64.into())}
fn f()->Type{Type::scalar(ElemType::Float(FloatKind::F32))}
fn v(id:u32)->Variable{Variable::new(VariableKind::LocalMut{id},f())}
#[test]fn real_common_ir_generates_vector_primitives(){let k=compile(definition(MapProgram::Add),1025).unwrap();assert!(k.source().contains("AscendC::Add("));assert!(k.source().contains("DataCopyPad"));assert!(k.source().contains("count * uint32_t(sizeof(float))"));assert!(!k.source().contains("aclnn"));assert!(!k.source().contains("deep_gemm"));assert_eq!(k.bindings().len(),3);assert_eq!(k.bindings()[2].bytes,4100);}
#[test]fn every_algorithm_is_a_common_kernel(){for p in [MapProgram::Copy,MapProgram::Add,MapProgram::Mul,MapProgram::Silu,MapProgram::SiluMul,MapProgram::SiluBackward,MapProgram::SiluMulBackward]{let k=compile(definition(p),513).unwrap();assert_eq!(k.bindings().len(),p.input_count()+p.output_count());assert!(k.ub_bytes()<=131072);}}
#[test]fn missing_target_is_not_guessed(){assert!(AscendCompiler.compile(definition(MapProgram::Add),&AscendOptions::default(),ExecutionMode::Checked,UIntKind::U64.into()).is_err());}
#[test]fn modes_are_not_silently_downgraded(){for m in [ExecutionMode::Unchecked,ExecutionMode::Validate]{assert!(AscendCompiler.compile(definition(MapProgram::Add),&opt(16),m,UIntKind::U64.into()).is_err());}}
#[test]fn cannot_accept_32_bit_pointer_contract(){assert!(AscendCompiler.compile(definition(MapProgram::Add),&opt(16),ExecutionMode::Checked,UIntKind::U32.into()).is_err());}
#[test]fn limits_are_checked(){for tile in [0,1,7,15,4097]{let mut o=opt(16);o.tile_elements=tile;assert!(AscendCompiler.compile(definition(MapProgram::Add),&o,ExecutionMode::Checked,UIntKind::U64.into()).is_err());}let mut o=opt(16);o.ub_limit_bytes=32;assert!(AscendCompiler.compile(definition(MapProgram::Silu),&o,ExecutionMode::Checked,UIntKind::U64.into()).is_err());}
#[test]fn code_injection_in_entry_rejected(){for name in ["", "1x", "x(){", "x\n", "x::y"]{let mut k=definition(MapProgram::Add);k.options.kernel_name=name.into();assert!(compile(k,16).is_err());}}
#[test]fn unsupported_buffer_precision_is_not_cast(){let mut k=definition(MapProgram::Add);k.buffers[0].ty=Type::scalar(ElemType::Float(FloatKind::F16));assert!(compile(k,16).is_err());}
#[test]fn output_is_not_loadable_as_an_input(){let mut k=definition(MapProgram::Add);if let Operation::Operator(Operator::Index(op))=&mut k.body.instructions[0].operation{op.list=Variable::new(VariableKind::GlobalOutputArray(2),f());}assert!(compile(k,16).is_err());}
#[test]fn duplicate_and_missing_stores_are_rejected(){let mut k=definition(MapProgram::Add);k.body.instructions.push(k.body.instructions.last().unwrap().clone());assert!(compile(k,16).is_err());let mut k=definition(MapProgram::Add);k.body.instructions.pop();assert!(compile(k,16).is_err());}
#[test]fn shifted_loads_are_rejected(){let mut k=definition(MapProgram::Add);if let Operation::Operator(Operator::Index(op))=&mut k.body.instructions[0].operation{op.index=0u32.into();}assert!(compile(k,16).is_err());}
#[test]fn undefined_local_rejected(){let mut k=definition(MapProgram::Add);if let Operation::Arithmetic(Arithmetic::Add(op))=&mut k.body.instructions[2].operation{op.lhs=v(999);}assert!(compile(k,16).is_err());}
#[test]fn unsupported_matrix_warp_sync_not_scalarized(){let mut k=definition(MapProgram::Add);k.body.instructions.insert(0,Instruction::no_out(Branch::Return));assert!(compile(k,16).is_err());let mut k=definition(MapProgram::Add);k.ruda_dim=RudaDim::new_2d(32,4);assert!(compile(k,16).is_err());}
#[test]fn canonical_length_guard_is_accepted(){let mut k=definition(MapProgram::Add);let arr=Variable::new(VariableKind::GlobalInputArray(0),f());let len=Variable::new(VariableKind::LocalMut{id:50},Type::scalar(ElemType::UInt(UIntKind::U32)));let outside=Variable::new(VariableKind::LocalMut{id:51},Type::scalar(ElemType::Bool));let index=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U32.into());let mut child=Scope::root(false);child.instructions.push(Instruction::no_out(Branch::Return));let mut prefix=vec![Instruction::new(Metadata::Length{var:arr},len),Instruction::new(Comparison::GreaterEqual(BinaryOperator{lhs:index,rhs:len}),outside),Instruction::no_out(Branch::If(Box::new(If{cond:outside,scope:child})))];prefix.append(&mut k.body.instructions);k.body.instructions=prefix;assert!(compile(k,17).is_ok());}
#[test]fn scalar_size_and_extended_metadata_fail_closed(){for ext in [false,true]{let mut k=definition(MapProgram::Add);k.buffers[0].has_extended_meta=ext;k.buffers[0].size=Some(18);assert!(compile(k,17).is_err());}}
#[test]fn signed_zero_constant_bits_survive(){let mut k=definition(MapProgram::Copy);let value=Variable::constant(ConstantValue::Float(-0.0),f());k.body.instructions.insert(1,Instruction::new(Arithmetic::Add(BinaryOperator{lhs:v(0),rhs:value}),v(10))); // use the actual loaded value type/kind below
if let Some(load)=k.body.instructions[0].out{if let Operation::Arithmetic(Arithmetic::Add(op))=&mut k.body.instructions[1].operation{op.lhs=load;}}
assert!(compile(k,9).unwrap().source().contains("0x80000000U"));}
#[test]fn zero_elements_have_explicit_no_work_body(){let k=compile(definition(MapProgram::Copy),0).unwrap();assert!(k.source().contains("elements == 0"));assert_eq!(k.bindings()[0].bytes,0);}
#[test]fn reuse_lowers_ub_without_overwriting_live_operands(){let k=definition(MapProgram::SiluMulBackward);let p=lower::lower(k.clone(),257).unwrap();let reused=plan::allocate(&p,true).unwrap();let fresh=plan::allocate(&p,false).unwrap();assert!(reused.slots<fresh.slots);for(i,node)in p.nodes.iter().enumerate(){for src in node.inputs(){if let(Some(x),Some(y))=(reused.node_slots[i],reused.node_slots[src]){assert_ne!(x,y);}}}let mut o=opt(257);o.reuse_temporaries=false;let old=AscendCompiler.compile(k.clone(),&o,ExecutionMode::Checked,UIntKind::U64.into()).unwrap();let new=compile(k,257).unwrap();assert!(new.ub_bytes()<old.ub_bytes());}
#[test]fn same_ir_can_use_ptx_backend(){
    #[cfg(feature="ptx")] {
        use crate::ptx::*;
        let k=definition(MapProgram::SiluMul);
        let p=PtxCompiler.compile(k.clone(),&PtxCompilationOptions{target:Some(PtxTarget{version:(8,0),sm:75})},ExecutionMode::Checked,UIntKind::U64.into()).unwrap();
        let a=compile(k,1025).unwrap();assert!(p.source.contains(".entry"));assert!(a.source().contains("AscendC::Mul"));
    }
}
/// Test-only reference of checked SSA expressions, never linked to a device executor.
fn reference(op:MapProgram,x:&[f32],y:&[f32],dy:&[f32])->Vec<Vec<f32>>{
    let p=lower::lower(definition(op),x.len()as u64).unwrap();let inputs=[x,y,dy];let mut nodes:Vec<Vec<f32>>=vec![];
    for node in &p.nodes{let z:Vec<f32>=match *node{
        Node::Input(i)=>inputs[i].to_vec(),Node::Constant(bits)=>vec![f32::from_bits(bits);x.len()],
        Node::Unary(u,a)=>nodes[a].iter().map(|&v|match u{Unary::Neg=>-v,Unary::Abs=>v.abs(),Unary::Exp=>v.exp(),Unary::Log=>v.ln(),Unary::Sqrt=>v.sqrt(),Unary::Rsqrt=>1.0/v.sqrt(),Unary::Recip=>1.0/v}).collect(),
        Node::Binary(b,a,c)=>nodes[a].iter().zip(&nodes[c]).map(|(&v,&w)|match b{Binary::Add=>v+w,Binary::Sub=>v-w,Binary::Mul=>v*w,Binary::Div=>v/w}).collect()};nodes.push(z);}
    p.stores.iter().map(|&(_,v)|nodes[v].clone()).collect()
}
#[test]fn backward_equations_match_finite_differences(){let x=[-2.0,-0.3,0.2,1.7];let up=[1.2,0.7,-0.9,2.0];let dy=[0.5,-1.0,0.2,0.8];let grads=reference(MapProgram::SiluMulBackward,&x,&up,&dy);let h=0.001;for j in 0..4{let mut hi=x;let mut lo=x;hi[j]+=h;lo[j]-=h;let a=reference(MapProgram::SiluMul,&hi,&up,&[]);let b=reference(MapProgram::SiluMul,&lo,&up,&[]);let finite=(a[0][j]-b[0][j])/(2.0*h)*dy[j];assert!((grads[0][j]-finite).abs()<0.001);let expected=dy[j]*x[j]/(1.0+(-x[j]).exp());assert!((grads[1][j]-expected).abs()<1e-6);}}
