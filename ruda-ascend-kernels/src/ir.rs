//! Small structured Rust device IR. There is deliberately no Raw/Cpp statement.
//! Kernel authors express loops, branches, loads and typed operations in Rust;
//! only the target printer knows CANN function names.
use std::{collections::BTreeSet, ops::{Add,Sub,Mul,Div,Rem,BitAnd,BitOr,BitXor,Shl}};
use crate::config::{Major,Spec};
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Imm(u64), Var(String), GroupEnd(Box<Expr>),
    Bin(Bin,Box<Expr>,Box<Expr>), Min(Box<Expr>,Box<Expr>),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bin {Add,Sub,Mul,Div,Rem,And,Or,Xor,Shl,Lt,Le,Eq,Ne,LogicalAnd,LogicalOr}
pub fn u(x:u64)->Expr {Expr::Imm(x)}
pub fn v(x:&str)->Expr {Expr::Var(x.into())}
impl Expr {
    fn binary(self, op:Bin, rhs:Self)->Self {Self::Bin(op,Box::new(self),Box::new(rhs))}
    pub fn lt(self,r:Self)->Self{self.binary(Bin::Lt,r)}
    pub fn le(self,r:Self)->Self{self.binary(Bin::Le,r)}
    pub fn eq(self,r:Self)->Self{self.binary(Bin::Eq,r)}
    pub fn ne(self,r:Self)->Self{self.binary(Bin::Ne,r)}
    pub fn and(self,r:Self)->Self{self.binary(Bin::LogicalAnd,r)}
    pub fn or(self,r:Self)->Self{self.binary(Bin::LogicalOr,r)}
    pub fn min(self,r:Self)->Self{Self::Min(Box::new(self),Box::new(r))}
    pub fn group_end(index:Self)->Self{Self::GroupEnd(Box::new(index))}
    pub fn check(&self,names:&BTreeSet<String>)->Result<(),String>{
        match self {
            Self::Imm(_)=>Ok(()),
            Self::Var(n) if names.contains(n)=>Ok(()),
            Self::Var(n)=>Err(format!("unbound scalar {n}")),
            Self::GroupEnd(i)=>i.check(names),
            Self::Bin(_,a,b)|Self::Min(a,b)=>{a.check(names)?;b.check(names)}
        }
    }
}
macro_rules! binary {
    ($trait:ident,$method:ident,$op:ident)=>{
        impl $trait for Expr {type Output=Expr;fn $method(self,rhs:Expr)->Expr{self.binary(Bin::$op,rhs)}}
    };
}
binary!(Add,add,Add); binary!(Sub,sub,Sub); binary!(Mul,mul,Mul);
binary!(Div,div,Div); binary!(Rem,rem,Rem); binary!(BitAnd,bitand,And);
binary!(BitOr,bitor,Or); binary!(BitXor,bitxor,Xor); binary!(Shl,shl,Shl);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pipe {Mte2,Mte1,Matrix}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {A,B}
#[derive(Clone, Debug)]
pub enum Op {
    Init,
    Signal{src:Pipe,dst:Pipe,event:Expr},
    Wait{src:Pipe,dst:Pipe,event:Expr},
    GlobalToL1{operand:Operand,major:Major,global_byte:Expr,local_byte:Expr,
        stride:Expr,valid_mn:Expr,valid_k:Expr,l1_mn:u64,l1_k:u64},
    L1ToL0{operand:Operand,major:Major,src_byte:Expr,dst_byte:Expr,
        mn_index:Expr,k_index:Expr,rows:Expr,cols:Expr,l1_mn:u64,l1_k:u64},
    Mmad{a:Expr,b:Expr,c:Expr,m:Expr,n:Expr,k:Expr,first:Expr,last:Expr},
    Store{global_byte:Expr,local_byte:Expr,m:Expr,n:Expr,stride:Expr},
}
impl Op {
    fn exprs(&self)->Vec<&Expr>{match self {
        Self::Init=>vec![],
        Self::Signal{event,..}|Self::Wait{event,..}=>vec![event],
        Self::GlobalToL1{global_byte,local_byte,stride,valid_mn,valid_k,..}=>vec![global_byte,local_byte,stride,valid_mn,valid_k],
        Self::L1ToL0{src_byte,dst_byte,mn_index,k_index,rows,cols,..}=>vec![src_byte,dst_byte,mn_index,k_index,rows,cols],
        Self::Mmad{a,b,c,m,n,k,first,last}=>vec![a,b,c,m,n,k,first,last],
        Self::Store{global_byte,local_byte,m,n,stride}=>vec![global_byte,local_byte,m,n,stride],
    }}
}
#[derive(Clone, Debug)]
pub enum Stmt {
    Let(String,Expr), Set(String,Expr),
    For{index:String,start:Expr,end:Expr,step:Expr,body:Block},
    While{condition:Expr,body:Block},
    If{condition:Expr,yes:Block,no:Block},
    Op(Op),
}
#[derive(Clone, Debug, Default)]
pub struct Block {pub statements:Vec<Stmt>}
impl Block {
    pub fn let_(&mut self,name:&str,e:Expr)->Expr {self.statements.push(Stmt::Let(name.into(),e));v(name)}
    pub fn set(&mut self,name:&str,e:Expr){self.statements.push(Stmt::Set(name.into(),e));}
    pub fn op(&mut self,op:Op){self.statements.push(Stmt::Op(op));}
    pub fn for_(&mut self,index:&str,start:Expr,end:Expr,step:Expr,f:impl FnOnce(&mut Block)){
        let mut body=Block::default();f(&mut body);
        self.statements.push(Stmt::For{index:index.into(),start,end,step,body});
    }
    pub fn while_(&mut self,condition:Expr,f:impl FnOnce(&mut Block)){
        let mut body=Block::default();f(&mut body);self.statements.push(Stmt::While{condition,body});
    }
    pub fn if_(&mut self,condition:Expr,yes:impl FnOnce(&mut Block),no:impl FnOnce(&mut Block)){
        let mut y=Block::default();let mut n=Block::default();yes(&mut y);no(&mut n);
        self.statements.push(Stmt::If{condition,yes:y,no:n});
    }
    pub fn check(&self,mut names:BTreeSet<String>)->Result<(),String>{
        fn valid(n:&str)->bool{!n.is_empty() && n.bytes().all(|c|c.is_ascii_alphanumeric()||c==b'_') && !n.as_bytes()[0].is_ascii_digit()}
        for s in &self.statements {match s {
            Stmt::Let(n,e)=>{e.check(&names)?;if !valid(n)||!names.insert(n.clone()){return Err(format!("duplicate/invalid scalar {n}"))}},
            Stmt::Set(n,e)=>{if !names.contains(n){return Err(format!("unknown assignment {n}"))}e.check(&names)?;},
            Stmt::For{index,start,end,step,body}=>{
                for e in [start,end,step]{e.check(&names)?;}
                if matches!(step,Expr::Imm(0)){return Err("zero loop step".into())}
                let mut inner=names.clone();if !valid(index)||!inner.insert(index.clone()){return Err("invalid loop variable".into())}
                body.check(inner)?;
            },
            Stmt::While{condition,body}=>{condition.check(&names)?;body.check(names.clone())?;},
            Stmt::If{condition,yes,no}=>{condition.check(&names)?;yes.check(names.clone())?;no.check(names.clone())?;},
            Stmt::Op(op)=>for e in op.exprs(){e.check(&names)?;},
        }}Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Program{pub spec:Spec,pub body:Block}
impl Program {
    pub fn inputs()->BTreeSet<String>{[
        "m","n","k","groups","core_id","a_addr","a_stride","a_batch",
        "b_addr","b_stride","b_batch","d_addr","d_stride","d_batch",
    ].iter().map(|x|(*x).to_owned()).collect()}
    pub fn check(&self)->Result<(),String>{self.spec.check()?;self.body.check(Self::inputs())}
}
