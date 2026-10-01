//! Ascend950DT tile/storage calculations. Every offset here is in BYTES.
//! The values match the uploaded reference's direct-store BF16 configuration.
use crate::config::Spec;
pub const BLOCK_N:u64=64;
pub const BLOCK_K:u64=128;
pub const MAD_M:u64=64;
pub const MAD_N:u64=64;
pub const MAD_K:u64=64;
pub const L1_STAGES:u64=2;
pub const L0_STAGES:u64=2;
pub const CORES:u64=32;
pub const FRACTAL:u64=16;
#[derive(Clone, Copy, Debug)]
pub struct TileLayout {
    pub block_m:u64,
    pub a_stage_bytes:u64,
    pub b_stage_bytes:u64,
    pub b_base:u64,
    pub l1_used:u64,
    pub operand_stage_bytes:u64,
    pub acc_stage_bytes:u64,
    pub acc_stages:u64,
}
impl TileLayout {
    pub fn new(spec:Spec)->Result<Self,String> {
        let block_m=spec.block_m();
        let a_stage_bytes=block_m*BLOCK_K*2;
        let b_stage_bytes=BLOCK_N*BLOCK_K*2;
        let b_base=a_stage_bytes*L1_STAGES;
        let s=Self {block_m,a_stage_bytes,b_stage_bytes,b_base,
            l1_used:b_base+b_stage_bytes*L1_STAGES,
            operand_stage_bytes:MAD_M*MAD_K*2,
            acc_stage_bytes:MAD_M*MAD_N*4,
            acc_stages:(block_m/MAD_M)*(BLOCK_N/MAD_N)};
        if s.l1_used>512*1024 || s.operand_stage_bytes*L0_STAGES>64*1024
            || s.acc_stage_bytes*s.acc_stages>256*1024 {
            return Err("Ascend950DT local memory budget exceeded".into());
        }
        Ok(s)
    }
}

pub fn ceil_div(n:u64,d:u64)->Result<u64,String> {
    if d==0 {return Err("zero divisor".into())}
    Ok(n/d+u64::from(n%d!=0))
}
