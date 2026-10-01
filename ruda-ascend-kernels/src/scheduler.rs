//! Persistent tile assignment and the reference's GROUP_M=4 XOR/snake swizzle.
//! The kernel scheduler is Rust IR, not a call into scheduler.hpp.
use crate::{config::{Kind,Spec},ir::{Block,Expr,u,v},layout::{BLOCK_N,CORES}};

pub fn persistent(spec:Spec,b:&mut Block,body:impl Fn(&mut Block)){
    let bm=spec.block_m();
    b.let_("m_tiles",(v("m")+u(bm-1))/u(bm));
    b.let_("n_tiles",(v("n")+u(BLOCK_N-1))/u(BLOCK_N));
    b.let_("matrix_tiles",v("m_tiles")*v("n_tiles"));
    if spec.kind==Kind::MGrouped {
        b.let_("current_group",u(0));b.let_("group_begin",u(0));
        b.let_("group_end",Expr::group_end(u(0)));
    }
    let total=if spec.kind==Kind::Batched {v("matrix_tiles")*v("groups")} else {v("matrix_tiles")};
    b.for_("tile_id",v("core_id"),total,u(CORES),|b|{
        match spec.kind {
            Kind::Dense=>{b.let_("batch_id",u(0));b.let_("expert_id",u(0));
                b.let_("region_m",v("m_tiles"));b.let_("local_tile",v("tile_id"));b.let_("row_base",u(0));},
            Kind::Batched=>{b.let_("batch_id",v("tile_id")/v("matrix_tiles"));b.let_("expert_id",u(0));
                b.let_("region_m",v("m_tiles"));b.let_("local_tile",v("tile_id")%v("matrix_tiles"));b.let_("row_base",u(0));},
            Kind::MGrouped=>{
                // Validated physical group ends are monotone and 256-aligned.
                // Repeated ends represent empty groups and are skipped here.
                b.while_((v("tile_id")/v("n_tiles")).lt(v("group_end")/u(bm)).eq(u(0)),|b|{
                    b.set("group_begin",v("group_end"));
                    b.set("current_group",v("current_group")+u(1));
                    b.set("group_end",Expr::group_end(v("current_group")));
                });
                b.let_("batch_id",u(0));b.let_("expert_id",v("current_group"));
                b.let_("region_m",(v("group_end")-v("group_begin"))/u(bm));
                b.let_("local_tile",v("tile_id")-(v("group_begin")/u(bm))*v("n_tiles"));
                b.let_("row_base",v("group_begin"));
            }
        }
        b.let_("mapped_m",u(0));b.let_("mapped_n",u(0));
        swizzle(b);
        b.let_("m_index",v("row_base")+v("mapped_m")*u(bm));
        b.let_("n_index",v("mapped_n")*u(BLOCK_N));
        b.let_("valid_m",(v("m")-v("m_index")).min(u(bm)));
        b.let_("valid_n",(v("n")-v("n_index")).min(u(BLOCK_N)));
        body(b);
    });
}

pub fn swizzle(b:&mut Block){
    b.let_("tail_m",v("region_m")-v("region_m")%u(4));
    b.let_("tail_tile",v("tail_m")*v("n_tiles"));
    b.if_(v("region_m").lt(u(4)),|b|{
        b.set("mapped_m",v("local_tile")/v("n_tiles"));
        b.set("mapped_n",v("local_tile")%v("n_tiles"));
    },|b|{
        b.if_(v("local_tile").lt(v("tail_tile")),|b|{
            b.let_("n_groups",v("n_tiles")/u(8));
            b.let_("per_superrow",u(4)*v("n_tiles"));
            b.let_("superrow",v("local_tile")/v("per_superrow"));
            b.let_("within",v("local_tile")%v("per_superrow"));
            b.let_("full",v("n_groups")*u(32));
            b.if_(v("within").lt(v("full")),|b|{
                b.let_("ng",v("within")/u(32));
                b.let_("local",v("within")%u(32));
                b.let_("ml",v("local")&u(3));
                b.let_("nl",(v("local")/u(4))^v("ml"));
                b.if_((v("superrow")&u(1)).ne(u(0)),|b|{
                    b.set("ng",v("n_groups")-u(1)-v("ng"));
                },|_|{});
                b.set("mapped_m",v("superrow")*u(4)+v("ml"));
                b.set("mapped_n",v("ng")*u(8)+v("nl"));
            },|b|{
                b.let_("rem",v("within")-v("full"));
                b.set("mapped_m",v("superrow")*u(4)+(v("rem")&u(3)));
                b.set("mapped_n",v("n_groups")*u(8)+v("rem")/u(4));
            });
        },|b|{
            b.let_("tail",v("local_tile")-v("tail_tile"));
            b.set("mapped_m",v("tail_m")+v("tail")/v("n_tiles"));
            b.set("mapped_n",v("tail")%v("n_tiles"));
        });
    });
}
