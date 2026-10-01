//! Device algorithms authored with the existing shared RUDA Kernel IR types.
//! No target intrinsics or C++ source fragments appear here. These definitions
//! may also be given to PtxCompiler unchanged (tested with the `ptx` feature).
use ruda_core::{ir::*,kernel::{KernelArg,KernelDefinition,KernelOptions,Visibility},launch::RudaDim};
#[derive(Clone, Copy, Debug)]
pub enum MapProgram { Copy, Add, Mul, Silu, SiluMul, SiluBackward, SiluMulBackward }
impl MapProgram {
    pub fn name(self)->&'static str {match self {Self::Copy=>"copy",Self::Add=>"add",Self::Mul=>"mul",Self::Silu=>"silu",Self::SiluMul=>"silu_mul",Self::SiluBackward=>"silu_backward",Self::SiluMulBackward=>"silu_mul_backward"}}
    pub fn parse(s:&str)->Option<Self>{match s{"copy"=>Some(Self::Copy),"add"=>Some(Self::Add),"mul"=>Some(Self::Mul),"silu"=>Some(Self::Silu),"silu_mul"=>Some(Self::SiluMul),"silu_backward"=>Some(Self::SiluBackward),"silu_mul_backward"=>Some(Self::SiluMulBackward),_=>None}}
    pub fn input_count(self)->usize{match self{Self::Copy|Self::Silu=>1,Self::SiluMulBackward=>3,_=>2}}
    pub fn output_count(self)->usize{if matches!(self,Self::SiluMulBackward){2}else{1}}
}
struct Builder { k:KernelDefinition, next:u32, idx:Variable }
impl Builder {
    fn new(name:String)->Self {Self {k:KernelDefinition{buffers:vec![],tensor_maps:vec![],scalars:vec![],ruda_dim:RudaDim::new_1d(64),body:Scope::root(false),options:KernelOptions{kernel_name:name,..Default::default()}},next:0,idx:Variable::builtin(Builtin::AbsolutePosX,UIntKind::U32.into())}}
    fn ty()->Type{Type::scalar(ElemType::Float(FloatKind::F32))}
    fn local(&mut self,ty:Type)->Variable{let v=Variable::new(VariableKind::LocalConst{id:self.next},ty);self.next+=1;v}
    fn array(&mut self,w:bool)->Variable{let id=self.k.buffers.len() as u32;let ty=Self::ty();self.k.buffers.push(KernelArg{id,visibility:if w{Visibility::ReadWrite}else{Visibility::Read},ty,size:None,has_extended_meta:false});Variable::new(if w{VariableKind::GlobalOutputArray(id)}else{VariableKind::GlobalInputArray(id)},ty)}
    fn read(&mut self,a:Variable)->Variable {let o=self.local(Self::ty());self.k.body.instructions.push(Instruction::new(Operator::Index(IndexOperator{list:a,index:self.idx,vector_size:0,unroll_factor:1}),o));o}
    fn write(&mut self,a:Variable,x:Variable){self.k.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator{index:self.idx,value:x,vector_size:0,unroll_factor:1}),a));}
    fn arithmetic(&mut self,op:Arithmetic)->Variable{let out=self.local(Self::ty());self.k.body.instructions.push(Instruction::new(op,out));out}
    fn add(&mut self,a:Variable,b:Variable)->Variable{self.arithmetic(Arithmetic::Add(BinaryOperator{lhs:a,rhs:b}))}
    fn sub(&mut self,a:Variable,b:Variable)->Variable{self.arithmetic(Arithmetic::Sub(BinaryOperator{lhs:a,rhs:b}))}
    fn mul(&mut self,a:Variable,b:Variable)->Variable{self.arithmetic(Arithmetic::Mul(BinaryOperator{lhs:a,rhs:b}))}
    fn sigmoid(&mut self,x:Variable)->Variable{let neg=self.arithmetic(Arithmetic::Neg(UnaryOperator{input:x}));let exp=self.arithmetic(Arithmetic::Exp(UnaryOperator{input:neg}));let d=self.add(one(),exp);self.arithmetic(Arithmetic::Div(BinaryOperator{lhs:one(),rhs:d}))}
    fn derivative(&mut self,x:Variable,s:Variable)->Variable{let inv=self.sub(one(),s);let prod=self.mul(x,inv);let term=self.add(one(),prod);self.mul(s,term)}
}
fn one()->Variable{Variable::constant(ConstantValue::Float(1.0),Builder::ty())}
/// Returns a real common KernelDefinition, not the separate v38 matrix IR.
/// Backward programs consume saved input(s) plus upstream gradient explicitly;
/// this function does not register a new framework autograd backend.
pub fn definition(op:MapProgram)->KernelDefinition{
    let mut b=Builder::new(format!("ruda_cann_{}",op.name()));
    let arrays:Vec<_>=(0..op.input_count()).map(|_|b.array(false)).collect();
    let outs:Vec<_>=(0..op.output_count()).map(|_|b.array(true)).collect();
    let x=b.read(arrays[0]);
    match op{
        MapProgram::Copy=>b.write(outs[0],x),
        MapProgram::Add|MapProgram::Mul=>{let y=b.read(arrays[1]);let z=if matches!(op,MapProgram::Add){b.add(x,y)}else{b.mul(x,y)};b.write(outs[0],z);},
        MapProgram::Silu=>{let s=b.sigmoid(x);let y=b.mul(x,s);b.write(outs[0],y);},
        MapProgram::SiluMul=>{let up=b.read(arrays[1]);let s=b.sigmoid(x);let h=b.mul(x,s);let y=b.mul(h,up);b.write(outs[0],y);},
        MapProgram::SiluBackward=>{let dy=b.read(arrays[1]);let s=b.sigmoid(x);let ds=b.derivative(x,s);let dx=b.mul(dy,ds);b.write(outs[0],dx);},
        MapProgram::SiluMulBackward=>{let up=b.read(arrays[1]);let dy=b.read(arrays[2]);let s=b.sigmoid(x);let h=b.mul(x,s);let ds=b.derivative(x,s);let dh=b.mul(dy,up);let dx=b.mul(dh,ds);let du=b.mul(dy,h);b.write(outs[0],dx);b.write(outs[1],du);},
    }
    b.k
}
