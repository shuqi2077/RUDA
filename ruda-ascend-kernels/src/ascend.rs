//! Target printer ONLY. All GEMM control flow is in kernel.rs/scheduler.rs.
//! This module maps typed operations to CANN C API intrinsics, never to a
//! DeepGEMM template or ACLNN matmul. Generated CCE is still required by Bisheng.
use crate::{config::{Major,Output},ir::{Bin,Block,Expr,Op,Operand,Pipe,Program,Stmt},layout::FRACTAL};
use std::fmt::Write;
fn e(x:&Expr)->String{match x {
    Expr::Imm(n)=>format!("{n}ULL"),Expr::Var(s)=>s.clone(),
    Expr::GroupEnd(i)=>format!("static_cast<uint64_t>(grouped_layout[{}])",e(i)),
    Expr::Min(a,b)=>format!("(({}) < ({}) ? ({}) : ({}))",e(a),e(b),e(a),e(b)),
    Expr::Bin(op,a,b)=>format!("({} {} {})",e(a),match op{
        Bin::Add=>"+",Bin::Sub=>"-",Bin::Mul=>"*",Bin::Div=>"/",Bin::Rem=>"%",
        Bin::And=>"&",Bin::Or=>"|",Bin::Xor=>"^",Bin::Shl=>"<<",Bin::Lt=>"<",
        Bin::Le=>"<=",Bin::Eq=>"==",Bin::Ne=>"!=",Bin::LogicalAnd=>"&&",Bin::LogicalOr=>"||",
    },e(b)),
}}
fn pipe(p:Pipe)->&'static str{match p{Pipe::Mte2=>"PIPE_MTE2",Pipe::Mte1=>"PIPE_MTE1",Pipe::Matrix=>"PIPE_M"}}
fn ptr(space:&str,ty:&str,x:&Expr)->String{format!("reinterpret_cast<{space} {ty}*>({})",e(x))}
fn line(out:&mut String,depth:usize,s:impl AsRef<str>){
    let _=writeln!(out,"{}{}","    ".repeat(depth),s.as_ref());
}
fn call(out:&mut String,d:usize,name:&str,args:Vec<String>){line(out,d,format!("{name}({});",args.join(", ")));}
fn op(out:&mut String,d:usize,o:&Op,output:Output){match o {
    Op::Init=>call(out,d,"asc_init",vec![]),
    Op::Signal{src,dst,event}|Op::Wait{src,dst,event}=>{
        let name=if matches!(o,Op::Signal{..}){"asc_sync_notify"}else{"asc_sync_wait"};
        call(out,d,name,vec![pipe(*src).into(),pipe(*dst).into(),format!("static_cast<event_t>({})",e(event))]);
    },
    Op::GlobalToL1{major,global_byte,local_byte,stride,valid_mn,valid_k,l1_mn,l1_k,..}=>{
        let nz_stride=if *major==Major::K{*l1_mn}else{*l1_k};
        // Batch=1, NZ-N stride=1, C0 stride in rows; exact CANN packed field.
        let config=1u64|(1u64<<16)|(nz_stride<<32);
        call(out,d,"asc_set_gm2l1_nz_para",vec![format!("{config}ULL")]);
        let (rows,cols)=if *major==Major::K{(valid_mn,valid_k)}else{(valid_k,valid_mn)};
        call(out,d,"asc_copy_gm2l1_nd2nz",vec![
            ptr("__cbuf__","bfloat16_t",local_byte),ptr("__gm__","bfloat16_t",global_byte),
            format!("({} * 2ULL)",e(stride)),"asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM".into(),
            e(rows),e(cols),"0ULL".into(),"false".into()]);
    },
    Op::L1ToL0{operand,major,src_byte,dst_byte,mn_index,k_index,rows,cols,l1_mn,l1_k}=>{
        let (name,space)=match (operand,major){
            (Operand::A,Major::K)=>("asc_copy_l12l0a","__ca__"),
            (Operand::B,Major::K)=>("asc_copy_l12l0b","__cb__"),
            (Operand::A,Major::Mn)=>("asc_copy_l12l0a_transpose","__ca__"),
            (Operand::B,Major::Mn)=>("asc_copy_l12l0b_transpose","__cb__"),
        };
        let mut a=vec![ptr(space,"bfloat16_t",dst_byte),ptr("__cbuf__","bfloat16_t",src_byte)];
        let div=|x:&Expr|format!("({} / {FRACTAL}ULL)",e(x));
        if *major==Major::K {
            a.extend([div(mn_index),div(k_index),div(rows),div(cols),format!("{}ULL",l1_mn/FRACTAL),div(rows)]);
        } else {
            a.extend([div(k_index),div(mn_index),div(cols),div(rows),format!("{}ULL",l1_k/FRACTAL),div(rows)]);
        }
        call(out,d,name,a);
    },
    Op::Mmad{a,b,c,m,n,k,first,last}=>{
        let flag=format!("static_cast<uint8_t>({} ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP)",e(last));
        call(out,d,"asc_mmad",vec![ptr("__cc__","float",c),ptr("__ca__","bfloat16_t",a),ptr("__cb__","bfloat16_t",b),
            e(m),e(k),e(n),flag,"true".into(),"false".into(),format!("({} != 0ULL)",e(first))]);
    },
    Op::Store{global_byte,local_byte,m,n,stride}=>{
        call(out,d,"asc_set_l0c2gm_nz2nd",vec!["1ULL".into(),"0ULL".into(),"0ULL".into()]);
        let (ty,quant)=if output==Output::F32{("float","asc_quant_mode::NoQuant")}else{("bfloat16_t","asc_quant_mode::F322BF16")};
        call(out,d,"asc_copy_l0c2gm",vec![ptr("__gm__",ty,global_byte),ptr("__cc__","float",local_byte),
            e(n),e(m),e(stride),e(m),"asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM".into(),
            "asc_unit_flag_mode::ENABLE_UPDATE".into(),quant.into(),"asc_relu_pre_mode::NONE".into(),
            "false".into(),"true".into(),"false".into(),"false".into()]);
    },
}}
fn block(out:&mut String,d:usize,b:&Block,output:Output){for s in &b.statements{match s {
    Stmt::Let(n,x)=>line(out,d,format!("uint64_t {n} = {};",e(x))),
    Stmt::Set(n,x)=>line(out,d,format!("{n} = {};",e(x))),
    Stmt::For{index,start,end,step,body}=>{
        line(out,d,format!("for (uint64_t {index} = {}; {index} < {}; {index} += {}) {{",e(start),e(end),e(step)));
        block(out,d+1,body,output);line(out,d,"}");
    },
    Stmt::While{condition,body}=>{
        line(out,d,format!("while ({}) {{",e(condition)));block(out,d+1,body,output);line(out,d,"}");
    },
    Stmt::If{condition,yes,no}=>{
        line(out,d,format!("if ({}) {{",e(condition)));block(out,d+1,yes,output);
        if no.statements.is_empty(){line(out,d,"}")}else{
            line(out,d,"} else {");block(out,d+1,no,output);line(out,d,"}");
        }
    },
    Stmt::Op(o)=>op(out,d,o,output),
}}}
/// Fixed data ABI declarations only; no algorithm or external kernel include.
pub fn abi_header()->String {
    ["#pragma once", "#include <stdint.h>",
     "struct RudaGm { uint64_t addr; uint64_t stride_outer; uint64_t stride_batch; };",
     "struct RudaEpilogue { float alpha; uint32_t pad0; uint64_t sfd; uint64_t stride; uint32_t n; uint32_t pad1; };",
     "static_assert(sizeof(RudaGm) == 24);", "static_assert(sizeof(RudaEpilogue) == 32);", ""].join("\n")
}
pub fn emit(p:&Program)->Result<String,String>{
    p.check()?;
    let mut out=String::new();
    line(&mut out,0,"// Generated from Rust device IR. No DeepGEMM C++ kernel is linked or included.");
    line(&mut out,0,"// CCE is a target intermediate, NOT a claim of rustc-to-Ascend-ISA compilation.");
    line(&mut out,0,"#include <c_api/asc_simd.h>");
    // Expand the header to make every generated translation unit self-contained.
    for s in abi_header().lines().filter(|s|*s!="#pragma once"){line(&mut out,0,s);}
    line(&mut out,0,format!("extern \"C\" __global__ __mix__(1, 0) void {}(",p.spec.entry()));
    line(&mut out,1,"RudaGm gm_a, RudaGm gm_b, RudaGm gm_d, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,");
    line(&mut out,1,"__gm__ int32_t* grouped_layout, uint32_t num_groups, RudaEpilogue epilogue) {");
    line(&mut out,1,"if ASCEND_IS_AIC {");
    for (name,field) in [("m","shape_m"),("n","shape_n"),("k","shape_k"),("groups","num_groups"),("core_id","block_idx"),
        ("a_addr","gm_a.addr"),("a_stride","gm_a.stride_outer"),("a_batch","gm_a.stride_batch"),
        ("b_addr","gm_b.addr"),("b_stride","gm_b.stride_outer"),("b_batch","gm_b.stride_batch"),
        ("d_addr","gm_d.addr"),("d_stride","gm_d.stride_outer"),("d_batch","gm_d.stride_batch")]{
        line(&mut out,2,format!("const uint64_t {name} = {field};"));
    }
    block(&mut out,2,&p.body,p.spec.output);line(&mut out,1,"}");line(&mut out,0,"}");
    Ok(out)
}
