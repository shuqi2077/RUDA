//! BF16 device algorithm translated to structured Rust; no external GEMM call.
//! The first port deliberately keeps the v37 direct-store/Identity contract.
use crate::{config::{Major,Spec},ir::{Block,Expr,Op,Operand,Pipe,Program,u,v},layout::*,scheduler};

fn addr(base:Expr,stride:Expr,mn:Expr,k:Expr,major:Major)->Expr{
    base+(match major{Major::K=>mn*stride+k,Major::Mn=>k*stride+mn})*u(2)
}
fn notify(b:&mut Block,src:Pipe,dst:Pipe,e:Expr){b.op(Op::Signal{src,dst,event:e});}
fn wait(b:&mut Block,src:Pipe,dst:Pipe,e:Expr){b.op(Op::Wait{src,dst,event:e});}

pub fn build(spec:Spec)->Result<Program,String>{
    spec.check()?;
    let t=TileLayout::new(spec)?;
    let mut b=Block::default();b.op(Op::Init);
    // Seed the two independent free-buffer rings.
    for i in 0..L1_STAGES {notify(&mut b,Pipe::Mte1,Pipe::Mte2,u(i));}
    for i in 0..L0_STAGES {notify(&mut b,Pipe::Matrix,Pipe::Mte1,u(i));}
    scheduler::persistent(spec,&mut b,|b|{
        b.let_("a_base",v("a_addr")+v("batch_id")*v("a_batch")*u(2));
        b.let_("b_base",v("b_addr")+v("batch_id")*v("b_batch")*u(2)+v("expert_id")*v("n")*v("k")*u(2));
        b.let_("d_base",v("d_addr")+v("batch_id")*v("d_batch")*u(spec.output_bytes()));
        b.for_("kb",u(0),v("k"),u(BLOCK_K),|b|{
            b.let_("l1_slot",(v("kb")/u(BLOCK_K))%u(L1_STAGES));
            b.let_("valid_k",(v("k")-v("kb")).min(u(BLOCK_K)));
            b.let_("l1_a",v("l1_slot")*u(t.a_stage_bytes));
            b.let_("l1_b",u(t.b_base)+v("l1_slot")*u(t.b_stage_bytes));
            wait(b,Pipe::Mte1,Pipe::Mte2,v("l1_slot"));
            b.op(Op::GlobalToL1{operand:Operand::A,major:spec.major_a(),
                global_byte:addr(v("a_base"),v("a_stride"),v("m_index"),v("kb"),spec.major_a()),
                local_byte:v("l1_a"),stride:v("a_stride"),valid_mn:v("valid_m"),valid_k:v("valid_k"),l1_mn:t.block_m,l1_k:BLOCK_K});
            b.op(Op::GlobalToL1{operand:Operand::B,major:spec.major_b(),
                global_byte:addr(v("b_base"),v("b_stride"),v("n_index"),v("kb"),spec.major_b()),
                local_byte:v("l1_b"),stride:v("b_stride"),valid_mn:v("valid_n"),valid_k:v("valid_k"),l1_mn:BLOCK_N,l1_k:BLOCK_K});
            notify(b,Pipe::Mte2,Pipe::Mte1,v("l1_slot"));
            wait(b,Pipe::Mte2,Pipe::Mte1,v("l1_slot"));
            b.for_("mi",u(0),v("valid_m"),u(MAD_M),|b|{
                b.for_("ni",u(0),v("valid_n"),u(MAD_N),|b|{
                    b.let_("em",(v("valid_m")-v("mi")).min(u(MAD_M)));
                    b.let_("en",(v("valid_n")-v("ni")).min(u(MAD_N)));
                    b.let_("acc_id",(v("mi")/u(MAD_M))*u(BLOCK_N/MAD_N)+v("ni")/u(MAD_N));
                    b.let_("l0_c",v("acc_id")*u(t.acc_stage_bytes));
                    b.for_("ki",u(0),v("valid_k"),u(MAD_K),|b|{
                        b.let_("ab_slot",(v("ki")/u(MAD_K))%u(L0_STAGES));
                        b.let_("l0_ab",v("ab_slot")*u(t.operand_stage_bytes));
                        b.let_("ek",(v("valid_k")-v("ki")).min(u(MAD_K)));
                        b.let_("first",v("kb").eq(u(0)).and(v("ki").eq(u(0))));
                        b.let_("last",(v("kb")+u(BLOCK_K)).lt(v("k")).eq(u(0))
                            .and((v("ki")+u(MAD_K)).lt(v("valid_k")).eq(u(0))));
                        wait(b,Pipe::Matrix,Pipe::Mte1,v("ab_slot"));
                        b.op(Op::L1ToL0{operand:Operand::A,major:spec.major_a(),src_byte:v("l1_a"),dst_byte:v("l0_ab"),
                            mn_index:v("mi"),k_index:v("ki"),rows:v("em"),cols:v("ek"),l1_mn:t.block_m,l1_k:BLOCK_K});
                        // With N=64 and K=128 this L0B stage remains valid across
                        // the four M subtiles of grouped GEMM, as in the reference.
                        b.if_(v("mi").eq(u(0)),|b|{
                            b.op(Op::L1ToL0{operand:Operand::B,major:spec.major_b(),src_byte:v("l1_b"),dst_byte:v("l0_ab"),
                                mn_index:v("ni"),k_index:v("ki"),rows:v("en"),cols:v("ek"),l1_mn:BLOCK_N,l1_k:BLOCK_K});
                        },|_|{});
                        notify(b,Pipe::Mte1,Pipe::Matrix,v("ab_slot"));
                        wait(b,Pipe::Mte1,Pipe::Matrix,v("ab_slot"));
                        b.op(Op::Mmad{a:v("l0_ab"),b:v("l0_ab"),c:v("l0_c"),m:v("em"),n:v("en"),k:v("ek"),first:v("first"),last:v("last")});
                        notify(b,Pipe::Matrix,Pipe::Mte1,v("ab_slot"));
                    });
                    b.if_((v("kb")+u(BLOCK_K)).lt(v("k")).eq(u(0)),|b|{
                        b.op(Op::Store{global_byte:v("d_base")+((v("m_index")+v("mi"))*v("d_stride")+v("n_index")+v("ni"))*u(spec.output_bytes()),
                            local_byte:v("l0_c"),m:v("em"),n:v("en"),stride:v("d_stride")});
                    },|_|{});
                });
            });
            notify(b,Pipe::Mte1,Pipe::Mte2,v("l1_slot"));
        });
    });
    // Drain free-buffer flags (one notify for every consumed initial token).
    for i in 0..L1_STAGES {wait(&mut b,Pipe::Mte1,Pipe::Mte2,u(i));}
    for i in 0..L0_STAGES {wait(&mut b,Pipe::Matrix,Pipe::Mte1,u(i));}
    let p=Program{spec,body:b};p.check()?;Ok(p)
}
