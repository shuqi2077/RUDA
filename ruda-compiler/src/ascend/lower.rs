//! Validate common RUDA IR before scheduling vector instructions. No raw C++ IR.
use super::{Result, invalid, unsupported};
use ruda_core::{ir::*, kernel::{KernelArg, KernelDefinition, Visibility}};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Unary { Neg, Abs, Exp, Log, Sqrt, Rsqrt, Recip }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Binary { Add, Sub, Mul, Div }
#[derive(Clone, Copy, Debug)]
pub(super) enum Node { Input(usize), Constant(u32), Unary(Unary, usize), Binary(Binary, usize, usize) }
impl Node { pub fn inputs(&self) -> Vec<usize> { match self {
    Self::Unary(_, a) => vec![*a], Self::Binary(_, a,b) => vec![*a,*b], _ => vec![] } } }
#[derive(Debug)]
pub(super) struct Program {
    pub name: String, pub bindings: Vec<KernelArg>, pub nodes: Vec<Node>,
    pub stores: Vec<(usize, usize)>,
}
#[derive(Clone, Copy, Debug)]
enum Value { Lane, Length, Outside, Vector(usize) }
fn f32_type() -> Type { Type::scalar(ElemType::Float(FloatKind::F32)) }
fn is_index(ty: Type) -> bool { ty == Type::scalar(ElemType::UInt(UIntKind::U32)) || ty == Type::scalar(ElemType::UInt(UIntKind::U64)) }
fn valid_local(v: Variable) -> bool { matches!(v.kind, VariableKind::LocalMut{..} | VariableKind::LocalConst{..} | VariableKind::Versioned{..}) }
fn ident(s:&str)->bool { !["for","while","if","else","return","float","int","void","class","template","auto","const","extern","union","struct","namespace","operator","new","delete"].contains(&s) && !s.is_empty() && s.len()<=128 && s.as_bytes()[0].is_ascii_alphabetic() && s.bytes().all(|c|c.is_ascii_alphanumeric()||c==b'_') }

struct Lower { p: Program, values: HashMap<Variable,Value>, loads: HashMap<usize,usize>, constants: HashMap<u32,usize>, wrote: HashSet<usize> }
impl Lower {
    fn resolve(&self, v: Variable) -> Result<Value> {
        if matches!(v.kind, VariableKind::Builtin(Builtin::AbsolutePosX)) && is_index(v.ty) { return Ok(Value::Lane); }
        self.values.get(&v).copied().ok_or_else(|| invalid(format!("undefined or unsupported operand {v:?}")))
    }
    fn vector(&mut self,v:Variable)->Result<usize> {
        if v.ty != f32_type() { return Err(unsupported(format!("non-FP32 arithmetic operand {v:?}"))); }
        if let VariableKind::Constant(ConstantValue::Float(x)) = v.kind {
            let y=x as f32;
            if !y.is_finite() { return Err(unsupported("non-finite compile-time constants")); }
            let bits=y.to_bits();
            if let Some(&id)=self.constants.get(&bits) { return Ok(id); }
            let id=self.p.nodes.len(); self.p.nodes.push(Node::Constant(bits)); self.constants.insert(bits,id);return Ok(id);
        }
        match self.resolve(v)? { Value::Vector(n)=>Ok(n), _=>Err(unsupported("lane indices/metadata cannot become floating point data")) }
    }
    fn assign(&mut self,out:Variable,value:Value)->Result<()> {
        if !valid_local(out) { return Err(invalid(format!("not a local destination: {out:?}"))); }
        let ty_ok=match value { Value::Vector(_)=>out.ty==f32_type(),Value::Outside=>out.ty==Type::scalar(ElemType::Bool),_=>is_index(out.ty) };
        if !ty_ok { return Err(invalid("IR output type does not match operation")); }
        if self.values.contains_key(&out) && !matches!(out.kind,VariableKind::LocalMut{..}) { return Err(invalid("immutable local assigned twice")); }
        self.values.insert(out,value);Ok(())
    }
    fn array(&self,v:Variable,writable:bool)->Result<usize> {
        let id=match (v.kind,writable) {
            (VariableKind::GlobalInputArray(id),false)|(VariableKind::GlobalOutputArray(id),true)=>id,
            _=>return Err(unsupported("only readonly input loads and distinct output stores are supported")),
        };
        let i=self.p.bindings.iter().position(|b|b.id==id).ok_or_else(||invalid("unknown buffer id"))?;
        let b=&self.p.bindings[i];
        if b.ty!=v.ty || (b.visibility==Visibility::ReadWrite)!=writable { return Err(invalid("buffer visibility/type mismatch")); }
        Ok(i)
    }
    fn length_array(&self,v:Variable)->Result<()> {
        match v.kind { VariableKind::GlobalInputArray(_)=>{self.array(v,false)?;},VariableKind::GlobalOutputArray(_)=>{self.array(v,true)?;},_=>return Err(unsupported("length of non-global array")) }; Ok(())
    }
    fn add(&mut self,out:Variable,node:Node)->Result<()> { let n=self.p.nodes.len();self.p.nodes.push(node);self.assign(out,Value::Vector(n)) }
    fn instruction(&mut self,i:&Instruction)->Result<()> {
        if !i.modes.fp_math_mode.is_empty() { return Err(unsupported("fast-math modes must be lowered explicitly, not silently dropped")); }
        let out=i.out;
        match &i.operation {
            Operation::NonSemantic(_) => Ok(()),
            Operation::Copy(v) => {
                let dst=out.ok_or_else(||invalid("copy has no destination"))?;
                let value=if dst.ty==f32_type(){Value::Vector(self.vector(*v)?)}else{self.resolve(*v)?};
                self.assign(dst,value)
            },
            Operation::Metadata(Metadata::Length{var}|Metadata::BufferLength{var})=>{
                self.length_array(*var)?;self.assign(out.ok_or_else(||invalid("metadata output missing"))?,Value::Length)
            },
            Operation::Comparison(Comparison::GreaterEqual(op))=>{
                if matches!((self.resolve(op.lhs)?,self.resolve(op.rhs)?),(Value::Lane,Value::Length)) {
                    self.assign(out.ok_or_else(||invalid("comparison output missing"))?,Value::Outside)
                } else { Err(unsupported("only canonical index >= length guard is supported")) }
            },
            Operation::Branch(Branch::If(branch))=>{
                if !matches!(self.resolve(branch.cond)?,Value::Outside) { return Err(unsupported("data-dependent branch")); }
                let ops:Vec<_>=branch.scope.instructions.iter().filter(|i|!matches!(i.operation,Operation::NonSemantic(_))).collect();
                if ops.len()!=1 || !matches!(ops[0].operation,Operation::Branch(Branch::Return)) { return Err(unsupported("only a canonical tail-guard return is allowed")); }
                // The actual loader contract is equal lengths; lowering dispatches
                // only [0,elements), so this exact guard is redundant on every lane.
                Ok(())
            },
            Operation::Operator(Operator::Index(op))=>{
                if op.vector_size!=0 || op.unroll_factor!=1 || !matches!(self.resolve(op.index)?,Value::Lane) { return Err(unsupported("loads must index one scalar at AbsolutePosX")); }
                let a=self.array(op.list,false)?;
                let id=if let Some(&n)=self.loads.get(&a){n}else{let n=self.p.nodes.len();self.p.nodes.push(Node::Input(a));self.loads.insert(a,n);n};
                self.assign(out.ok_or_else(||invalid("load output missing"))?,Value::Vector(id))
            },
            Operation::Operator(Operator::IndexAssign(op))=>{
                if op.vector_size!=0 || op.unroll_factor!=1 || !matches!(self.resolve(op.index)?,Value::Lane) { return Err(unsupported("stores must index one scalar at AbsolutePosX")); }
                let a=self.array(out.ok_or_else(||invalid("store output missing"))?,true)?;
                if !self.wrote.insert(a){return Err(unsupported("multiple stores to one output"));}
                let v=self.vector(op.value)?;self.p.stores.push((a,v));Ok(())
            },
            Operation::Arithmetic(a)=>{
                let dst=out.ok_or_else(||invalid("arithmetic output missing"))?;
                let binary=match a { Arithmetic::Add(op)=>Some((Binary::Add,op)),Arithmetic::Sub(op)=>Some((Binary::Sub,op)),Arithmetic::Mul(op)=>Some((Binary::Mul,op)),Arithmetic::Div(op)=>Some((Binary::Div,op)),_=>None };
                if let Some((kind,op))=binary {let lhs=self.vector(op.lhs)?;let rhs=self.vector(op.rhs)?;return self.add(dst,Node::Binary(kind,lhs,rhs));}
                let (kind,op)=match a { Arithmetic::Neg(op)=>(Unary::Neg,op),Arithmetic::Abs(op)=>(Unary::Abs,op),Arithmetic::Exp(op)=>(Unary::Exp,op),Arithmetic::Log(op)=>(Unary::Log,op),Arithmetic::Sqrt(op)=>(Unary::Sqrt,op),Arithmetic::InverseSqrt(op)=>(Unary::Rsqrt,op),Arithmetic::Recip(op)=>(Unary::Recip,op),_=>return Err(unsupported(format!("arithmetic {a:?}"))) };
                let input=self.vector(op.input)?;self.add(dst,Node::Unary(kind,input))
            },
            other=>Err(unsupported(format!("operation {other:?}; no fallback to another kernel/compiler"))),
        }
    }
}

pub(super) fn lower(mut k:KernelDefinition,elements:u64)->Result<Program> {
    if !ident(&k.options.kernel_name) {return Err(invalid("kernel entry must be a short ASCII identifier beginning with a letter"));}
    if k.options.debug_symbols || k.options.cluster_dim.is_some() {return Err(unsupported("debug symbols/clusters"));}
    if k.ruda_dim.x==0 || k.ruda_dim.y!=1 || k.ruda_dim.z!=1 {return Err(unsupported("only nonzero one-dimensional logical lane launch is accepted"));}
    if !k.tensor_maps.is_empty()||!k.scalars.is_empty()||!k.body.const_arrays.is_empty(){return Err(unsupported("tensor maps, runtime scalars and constant arrays"));}
    let errors=k.body.pop_errors();if !errors.is_empty(){return Err(invalid(errors.join("; ")));}
    if k.buffers.is_empty()||k.buffers.len()>8{return Err(unsupported("expected 1..8 total buffers"));}
    let mut ids=HashSet::new();let mut inputs=0;let mut outputs=0;
    for b in &k.buffers {
        if !ids.insert(b.id){return Err(invalid("duplicate kernel buffer id"));}
        if b.ty!=f32_type(){return Err(unsupported("only scalar FP32 buffers; no implicit precision changes"));}
        if b.has_extended_meta||b.size.is_some_and(|n|n as u64!=elements){return Err(unsupported("extended metadata or static size mismatch"));}
        if b.visibility==Visibility::Read {inputs+=1}else{outputs+=1}
    }
    if inputs>4||outputs==0||outputs>4{return Err(unsupported("at most four inputs/four outputs, and at least one output"));}
    // Read actual scope instructions. Unused local declarations carry no effects;
    // every operation, operand and referenced special storage is checked below.
    if k.body.instructions.len()>4096{return Err(unsupported("instruction limit exceeded"));}
    let mut l=Lower{p:Program{name:k.options.kernel_name,bindings:k.buffers,nodes:vec![],stores:vec![]},values:HashMap::new(),loads:HashMap::new(),constants:HashMap::new(),wrote:HashSet::new()};
    for (n,i) in k.body.instructions.iter().enumerate(){l.instruction(i).map_err(|e|{
        let text=format!("instruction {n}: {e}");
        if matches!(e,ruda_core::compiler::CompilationError::UnsupportedInstruction{..}){unsupported(text)}else{invalid(text)}
    })?;}
    if l.wrote.len()!=outputs{return Err(invalid("every declared output must be written exactly once"));}
    if l.loads.len()!=inputs{return Err(unsupported("unused input bindings must be removed before lowering"));}
    Ok(l.p)
}
