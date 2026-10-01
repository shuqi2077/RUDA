//! Execute the SAME Rust device IR in a test-only memory/pipe model.
//! This does not certify CANN lowering, asynchronous hardware or NPU performance.
use super::*;
use ir::{Bin,Block,Expr,Op,Operand,Pipe,Program,Stmt};
use std::collections::BTreeMap;

fn bf16(x:f32)->f32 {
    let b=x.to_bits();
    if b&0x7f800000==0x7f800000 {return f32::from_bits(b&0xffff0000)}
    f32::from_bits(b.wrapping_add(0x7fff+((b>>16)&1))&0xffff0000)
}
#[derive(Clone)]struct Tile{rows:usize,cols:usize,data:Vec<f32>}
struct Vm {
    vars:BTreeMap<String,u64>,ends:Vec<i32>,
    a:Vec<f32>,b:Vec<f32>,d:Vec<f32>,a_base:u64,b_base:u64,d_base:u64,
    l1:BTreeMap<u64,Tile>,l0a:BTreeMap<u64,Tile>,l0b:BTreeMap<u64,Tile>,acc:BTreeMap<u64,Tile>,
    flags:BTreeMap<(u8,u8,u64),bool>,output:Output,stores:usize,
}
fn pn(p:Pipe)->u8{match p{Pipe::Mte2=>0,Pipe::Mte1=>1,Pipe::Matrix=>2}}
impl Vm {
    fn e(&self,x:&Expr)->u64{match x{
        Expr::Imm(n)=>*n,Expr::Var(n)=>*self.vars.get(n).unwrap_or_else(||panic!("unknown {n}")),
        Expr::GroupEnd(i)=>u64::try_from(self.ends[self.e(i)as usize]).unwrap(),
        Expr::Min(a,b)=>self.e(a).min(self.e(b)),
        Expr::Bin(op,a,b)=>{let a=self.e(a);let b=self.e(b);match op{
            Bin::Add=>a.checked_add(b).unwrap(),Bin::Sub=>a.checked_sub(b).unwrap(),Bin::Mul=>a.checked_mul(b).unwrap(),
            Bin::Div=>a/b,Bin::Rem=>a%b,Bin::And=>a&b,Bin::Or=>a|b,Bin::Xor=>a^b,Bin::Shl=>a<<b,
            Bin::Lt=>u64::from(a<b),Bin::Le=>u64::from(a<=b),Bin::Eq=>u64::from(a==b),Bin::Ne=>u64::from(a!=b),
            Bin::LogicalAnd=>u64::from(a!=0&&b!=0),Bin::LogicalOr=>u64::from(a!=0||b!=0),
        }}
    }}
    fn exec(&mut self,b:&Block){
        let mut locals=Vec::new();
        for s in &b.statements{match s{
            Stmt::Let(n,x)=>{let z=self.e(x);let old=self.vars.insert(n.clone(),z);locals.push((n.clone(),old));},
            Stmt::Set(n,x)=>{assert!(self.vars.contains_key(n));let z=self.e(x);self.vars.insert(n.clone(),z);},
            Stmt::For{index,start,end,step,body}=>{
                let old=self.vars.get(index).copied();let mut i=self.e(start);let stop=self.e(end);let by=self.e(step);assert!(by>0);
                while i<stop{self.vars.insert(index.clone(),i);self.exec(body);i=i.checked_add(by).unwrap();}
                if let Some(x)=old{self.vars.insert(index.clone(),x);}else{self.vars.remove(index);}
            },
            Stmt::While{condition,body}=>{let mut n=0;while self.e(condition)!=0{self.exec(body);n+=1;assert!(n<10000,"loop did not terminate");}},
            Stmt::If{condition,yes,no}=>{if self.e(condition)!=0{self.exec(yes)}else{self.exec(no)}},
            Stmt::Op(o)=>self.op(o),
        }}
        for (n,old) in locals.into_iter().rev(){if let Some(x)=old{self.vars.insert(n,x);}else{self.vars.remove(&n);}}
    }
    fn op(&mut self,o:&Op){match o{
        Op::Init=>{},
        Op::Signal{src,dst,event}=>{let key=(pn(*src),pn(*dst),self.e(event));assert!(!self.flags.get(&key).copied().unwrap_or(false),"signalled busy flag");self.flags.insert(key,true);},
        Op::Wait{src,dst,event}=>{let key=(pn(*src),pn(*dst),self.e(event));assert!(self.flags.get(&key).copied().unwrap_or(false),"wait on unready flag");self.flags.insert(key,false);},
        Op::GlobalToL1{operand,major,global_byte,local_byte,stride,valid_mn,valid_k,l1_mn,l1_k}=>{
            let (input,base)=if *operand==Operand::A{(&self.a,self.a_base)}else{(&self.b,self.b_base)};
            let first=(self.e(global_byte)-base)/2;let stride=self.e(stride);
            let rows=self.e(valid_mn)as usize;let cols=self.e(valid_k)as usize;
            let mut tile=Tile{rows:*l1_mn as usize,cols:*l1_k as usize,data:vec![f32::NAN;(*l1_mn * *l1_k)as usize]};
            assert!(rows<=tile.rows&&cols<=tile.cols);
            for i in 0..rows{for j in 0..cols{
                let index=first+match major{Major::K=>i as u64*stride+j as u64,Major::Mn=>j as u64*stride+i as u64};
                tile.data[i*tile.cols+j]=input[index as usize];
            }}self.l1.insert(self.e(local_byte),tile);
        },
        Op::L1ToL0{operand,src_byte,dst_byte,mn_index,k_index,rows,cols,..}=>{
            let src=self.l1.get(&self.e(src_byte)).unwrap();let r=self.e(rows)as usize;let c=self.e(cols)as usize;
            let mi=self.e(mn_index)as usize;let ki=self.e(k_index)as usize;
            assert!(mi+r<=src.rows&&ki+c<=src.cols);
            let mut tile=Tile{rows:r,cols:c,data:vec![0.;r*c]};
            for i in 0..r{for j in 0..c{let x=src.data[(i+mi)*src.cols+j+ki];assert!(x.is_finite(),"read uninitialized L1");tile.data[i*c+j]=x;}}
            let dst=self.e(dst_byte);
            if *operand==Operand::A{self.l0a.insert(dst,tile);}else{self.l0b.insert(dst,tile);}
        },
        Op::Mmad{a,b,c,m,n,k,first,..}=>{
            let m=self.e(m)as usize;let n=self.e(n)as usize;let k=self.e(k)as usize;let dst=self.e(c);
            if self.e(first)!=0{self.acc.insert(dst,Tile{rows:m,cols:n,data:vec![0.;m*n]});}
            let a=self.l0a.get(&self.e(a)).unwrap();let b=self.l0b.get(&self.e(b)).unwrap();
            assert_eq!((a.rows,a.cols,b.rows,b.cols),(m,k,n,k));
            let acc=self.acc.get_mut(&dst).unwrap();assert_eq!((acc.rows,acc.cols),(m,n));
            for i in 0..m{for j in 0..n{for t in 0..k{acc.data[i*n+j]=a.data[i*k+t].mul_add(b.data[j*k+t],acc.data[i*n+j]);}}}
        },
        Op::Store{global_byte,local_byte,m,n,stride}=>{
            let m=self.e(m)as usize;let n=self.e(n)as usize;let stride=self.e(stride)as usize;
            let size=if self.output==Output::F32{4}else{2};let index=((self.e(global_byte)-self.d_base)/size)as usize;
            let acc=self.acc.get(&self.e(local_byte)).unwrap();assert_eq!((acc.rows,acc.cols),(m,n));
            for i in 0..m{for j in 0..n{let x=acc.data[i*n+j];assert!(self.d[index+i*stride+j].is_nan(),"overlapping tile store");self.d[index+i*stride+j]=if self.output==Output::Bf16{bf16(x)}else{x};}}
            self.stores+=1;
        }
    }}
}
fn run_case(spec:Spec,m:usize,n:usize,k:usize,ends:Vec<i32>,batches:usize){
    let grouped=spec.kind==Kind::MGrouped;let g=if grouped{ends.len()}else{batches};
    let a= (0..batches*m*k).map(|i|bf16(((i%23)as f32-11.)/16.)).collect::<Vec<_>>();
    let b= (0..g*n*k).map(|i|bf16(((i%17)as f32-8.)/8.)).collect::<Vec<_>>();
    let mut expected=vec![0.;batches*m*n];
    for batch in 0..batches{for i in 0..m{for j in 0..n{
        let group=if grouped{ends.iter().position(|&end|i<(end as usize)).unwrap()}else{batch};
        let mut x=0f64;
        for t in 0..k{
            let ai=batch*m*k+if spec.transpose_a{t*m+i}else{i*k+t};
            let bi=group*n*k+if spec.transpose_b{j*k+t}else{t*n+j};
            x+=f64::from(a[ai])*f64::from(b[bi]);
        }
        expected[batch*m*n+i*n+j]=if spec.output==Output::Bf16{bf16(x as f32)}else{x as f32};
    }}}
    let mut vm=Vm{vars:BTreeMap::new(),ends,a,b,d:vec![f32::NAN;batches*m*n],a_base:1<<40,b_base:2<<40,d_base:3<<40,
        l1:BTreeMap::new(),l0a:BTreeMap::new(),l0b:BTreeMap::new(),acc:BTreeMap::new(),flags:BTreeMap::new(),output:spec.output,stores:0};
    for (name,value) in [("m",m as u64),("n",n as u64),("k",k as u64),("groups",g as u64),
        ("a_addr",vm.a_base),("b_addr",vm.b_base),("d_addr",vm.d_base),
        ("a_stride",if spec.transpose_a{m as u64}else{k as u64}),("b_stride",if spec.transpose_b{k as u64}else{n as u64}),
        ("d_stride",n as u64),("a_batch",(m*k)as u64),("b_batch",(n*k)as u64),("d_batch",(m*n)as u64)]{
        vm.vars.insert(name.into(),value);
    }
    let p=kernel::build(spec).unwrap();
    for core in 0..layout::CORES{
        vm.vars.insert("core_id".into(),core);vm.exec(&p.body);
        assert!(vm.flags.values().all(|&x|!x));vm.l1.clear();vm.l0a.clear();vm.l0b.clear();vm.acc.clear();
    }
    for (i,(actual,expected)) in vm.d.iter().zip(&expected).enumerate(){assert!(actual.is_finite());assert!((actual-expected).abs()<1e-4,"{} index {i}: {actual} != {expected}",spec.key());}
}
#[test]fn every_bf16_variant_emits_without_foreign_kernel(){
    assert_eq!(Spec::all().len(),18);
    for s in Spec::all(){let text=emit(s).unwrap();assert!(text.contains("asc_mmad("));assert!(text.contains(&s.entry()));
        for forbidden in ["#include <deep_gemm/","bf16_gemm_impl","aclnnMatmul","torch","dlopen"]{assert!(!text.contains(forbidden));}}
}
#[test]fn test_only_ir_vm_all_dense_transposes_and_outputs(){
    for s in Spec::all().into_iter().filter(|s|s.kind==Kind::Dense){run_case(s,80,96,144,vec![],1);}
}
#[test]fn test_only_ir_vm_batches_and_short_k(){
    for s in Spec::all().into_iter().filter(|s|s.kind==Kind::Batched){run_case(s,16,32,16,vec![],3);}
}
#[test]fn test_only_ir_vm_grouped_empty_experts_and_l0b_reuse(){
    for s in Spec::all().into_iter().filter(|s|s.kind==Kind::MGrouped){run_case(s,768,16,144,vec![0,256,256,768],1);}
}
#[test]fn all_local_memory_budgets_fit(){for s in Spec::all(){let x=layout::TileLayout::new(s).unwrap();assert!(x.l1_used<=512*1024);assert!(x.acc_stages*x.acc_stage_bytes<=256*1024);}}
#[test]fn unsupported_grouped_transpose_is_rejected(){for a in [false,true]{for b in [false,true]{if !a&&b{continue}assert!(Spec{kind:Kind::MGrouped,transpose_a:a,transpose_b:b,output:Output::F32}.check().is_err());}}}
#[test]fn zero_divisor_rejected(){assert!(layout::ceil_div(1,0).is_err());assert_eq!(layout::ceil_div(u64::MAX,16).unwrap(),u64::MAX/16+1);}
#[test]fn unbound_ir_names_and_zero_steps_are_rejected(){
    let s=Spec::all()[0];let mut b=Block::default();b.let_("x",ir::v("missing"));assert!(Program{spec:s,body:b}.check().is_err());
    let mut b=Block::default();b.for_("i",ir::u(0),ir::u(1),ir::u(0),|_|{});assert!(Program{spec:s,body:b}.check().is_err());
}
#[test]fn swizzle_ir_is_a_bijection_with_both_tail_directions(){
    for m in [1u64,2,3,4,5,7,8,9,16,17,31,32,33]{for n in [1u64,2,3,7,8,9,16,17]{
        let mut vm=Vm{vars:BTreeMap::new(),ends:vec![],a:vec![],b:vec![],d:vec![],a_base:0,b_base:0,d_base:0,
            l1:BTreeMap::new(),l0a:BTreeMap::new(),l0b:BTreeMap::new(),acc:BTreeMap::new(),flags:BTreeMap::new(),output:Output::F32,stores:0};
        let mut body=Block::default();scheduler::swizzle(&mut body);
        let mut seen=std::collections::BTreeSet::new();
        for i in 0..m*n{
            for (name,x) in [("region_m",m),("n_tiles",n),("local_tile",i),("mapped_m",0),("mapped_n",0)]{vm.vars.insert(name.into(),x);}
            vm.exec(&body);let mi=vm.vars["mapped_m"];let ni=vm.vars["mapped_n"];
            assert!(mi<m&&ni<n);assert!(seen.insert((mi,ni)),"duplicate tile");
        }
        assert_eq!(seen.len(),(m*n)as usize);
    }}
}
