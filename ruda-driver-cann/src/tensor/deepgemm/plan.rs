//! Checked host-side dispatch contract; no device probing or allocation.
use super::super::{DType, TensorLayout};
use crate::CannError;
use super::super::layout::invalid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transpose { No, Yes }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmKind { Dense, Batched, MGrouped }

/// An *aligned physical* M-group layout, not an arbitrary token prefix sum.
/// All ends are multiples of 256. Repeated ends represent empty experts.
/// No device-to-host read is needed: the validated host ends are uploaded once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupEnds { ends: Vec<i32> }
impl GroupEnds {
    pub fn new(ends: &[i32]) -> Result<Self, CannError> {
        if ends.is_empty() || ends.len() > 4096 { return Err(invalid("group count must be 1..=4096")); }
        let mut previous = 0;
        for &end in ends {
            if end < previous || end % 256 != 0 { return Err(invalid("group ends must be nondecreasing multiples of 256")); }
            previous = end;
        }
        if previous == 0 { return Err(invalid("all-empty grouped GEMM is not implemented")); }
        Ok(Self { ends: ends.to_vec() })
    }
    pub fn ends(&self) -> &[i32] { &self.ends }
    pub fn total_rows(&self) -> i64 { i64::from(*self.ends.last().unwrap()) }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmSpec {
    pub(crate) kind: GemmKind,
    pub(crate) ta: Transpose,
    pub(crate) tb: Transpose,
    pub(crate) dtype: DType,
    pub(crate) m: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
    pub(crate) groups: u32,
    pub(crate) a: TensorLayout,
    pub(crate) b: TensorLayout,
    pub(crate) out: TensorLayout,
    pub(crate) ends: Option<GroupEnds>,
}
impl GemmSpec {
    /// Rust-authored device kernel contract; only schema v2 artifacts are loaded.
    /// Row-major Y = op(A) @ op(B), no broadcasting or bias.
    pub fn new(a: &TensorLayout, b: &TensorLayout, ta: Transpose, tb: Transpose, output: DType) -> Result<Self, CannError> {
        let r = a.shape().len();
        if !(r == 2 || r == 3) || b.shape().len() != r { return Err(invalid("GEMM requires matching rank 2 or rank 3")); }
        if r == 3 && a.shape()[0] != b.shape()[0] { return Err(invalid("batch broadcasting is not supported")); }
        let a2 = &a.shape()[r-2..]; let b2 = &b.shape()[r-2..];
        let (m,k) = if ta == Transpose::No {(a2[0],a2[1])} else {(a2[1],a2[0])};
        let (kb,n) = if tb == Transpose::No {(b2[0],b2[1])} else {(b2[1],b2[0])};
        if kb != k { return Err(invalid("GEMM contraction mismatch")); }
        let groups = if r == 3 { a.shape()[0] } else { 1 };
        Self::checked(a,b,ta,tb,output,m,n,k,groups,if r==2{GemmKind::Dense}else{GemmKind::Batched},None)
    }
    /// A=[M,K], weights=[G,N,K], output=[M,N]. Ends describe aligned physical rows.
    pub fn grouped_nt(a: &TensorLayout, weights: &TensorLayout, ends: GroupEnds, output: DType) -> Result<Self,CannError> {
        if a.shape().len()!=2 || weights.shape().len()!=3 { return Err(invalid("grouped NT requires [M,K] and [G,N,K]")); }
        if a.shape()[0]!=ends.total_rows() || weights.shape()[0]!=ends.ends.len() as i64 || a.shape()[1]!=weights.shape()[2] {
            return Err(invalid("grouped NT dimensions do not match the physical prefix sums"));
        }
        Self::checked(a,weights,Transpose::No,Transpose::Yes,output,a.shape()[0],weights.shape()[1],a.shape()[1],weights.shape()[0],GemmKind::MGrouped,Some(ends))
    }
    #[allow(clippy::too_many_arguments)]
    fn checked(a:&TensorLayout,b:&TensorLayout,ta:Transpose,tb:Transpose,dtype:DType,m:i64,n:i64,k:i64,groups:i64,kind:GemmKind,ends:Option<GroupEnds>) -> Result<Self,CannError> {
        if a.dtype()!=DType::BF16 || b.dtype()!=DType::BF16 || !matches!(dtype,DType::BF16|DType::F32) { return Err(invalid("native DeepGEMM accepts BF16 inputs and BF16/F32 output only")); }
        // Deliberately narrow first-port contract, not all upstream-supported tails.
        if [m,n,k].iter().any(|&x|x<=0 || x%16!=0 || x>i32::MAX as i64) || groups<=0 || groups>4096 {
            return Err(invalid("M/N/K must be positive multiples of 16 <= INT32_MAX; batches/groups 1..=4096"));
        }
        let block_m = if kind==GemmKind::MGrouped {256i64} else {64};
        let tiles = ((m+block_m-1)/block_m).checked_mul((n+63)/64).and_then(|x|x.checked_mul(if kind==GemmKind::Batched {groups}else{1}));
        if tiles.is_none_or(|x| x > (i32::MAX as i64)-32) { return Err(invalid("persistent scheduler tile range overflows")); }
        let shape = if kind==GemmKind::Batched {vec![groups,m,n]} else {vec![m,n]};
        Ok(Self {kind,ta,tb,dtype,m:m as u32,n:n as u32,k:k as u32,groups:groups as u32,a:a.clone(),b:b.clone(),out:TensorLayout::contiguous(&shape,dtype)?,ends})
    }
    pub fn output_layout(&self)->&TensorLayout {&self.out}
    pub fn dimensions(&self)->[u32;3] {[self.m,self.n,self.k]}
    pub fn kind(&self)->GemmKind {self.kind}
    pub fn key(&self)->String {
        format!("bf16_{}_{}{}_{}",match self.kind {GemmKind::Dense=>"dense",GemmKind::Batched=>"batched",GemmKind::MGrouped=>"mgrouped"},if self.ta==Transpose::No {"n"} else {"t"},if self.tb==Transpose::No {"n"} else {"t"},if self.dtype==DType::F32 {"f32"} else {"bf16"})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(s:&[i64])->TensorLayout {TensorLayout::contiguous(s,DType::BF16).unwrap()}
    #[test] fn four_transposes_share_the_logical_shape() {
        for ta in [Transpose::No,Transpose::Yes] { for tb in [Transpose::No,Transpose::Yes] {
            let a=t(if ta==Transpose::No {&[32,48]}else{&[48,32]});
            let b=t(if tb==Transpose::No {&[48,64]}else{&[64,48]});
            let s=GemmSpec::new(&a,&b,ta,tb,DType::F32).unwrap();assert_eq!(s.dimensions(),[32,64,48]); assert_eq!(s.out.byte_len(),32*64*4);
        }}
    }
    #[test] fn batch_broadcast_is_rejected() {assert!(GemmSpec::new(&t(&[2,16,16]),&t(&[1,16,16]),Transpose::No,Transpose::No,DType::F32).is_err());}
    #[test] fn tails_empty_rank_dtype_are_rejected() {
        for dims in [vec![0,16],vec![17,16],vec![16],vec![1,1,16,16]] {assert!(GemmSpec::new(&t(&dims),&t(&[16,16]),Transpose::No,Transpose::No,DType::F32).is_err());}
        assert!(GemmSpec::new(&TensorLayout::contiguous(&[16,16],DType::F16).unwrap(),&t(&[16,16]),Transpose::No,Transpose::No,DType::F32).is_err());
    }
    #[test] fn grouped_accepts_aligned_empty_experts() {let e=GroupEnds::new(&[0,256,256,768]).unwrap();let s=GemmSpec::grouped_nt(&t(&[768,64]),&t(&[4,128,64]),e,DType::BF16).unwrap();assert_eq!(s.out.shape(),&[768,128]);}
    #[test] fn grouped_rejects_unpadded_prefixes() {for e in [&[1,17][..],&[256,0][..],&[0,0][..],&[][..]] {assert!(GroupEnds::new(e).is_err());}}
    #[test] fn training_dimension_mapping() {let x=t(&[32,48]);let w=t(&[64,48]);let dy=t(&[32,64]);assert_eq!(GemmSpec::new(&dy,&w,Transpose::No,Transpose::No,DType::BF16).unwrap().out,x);assert_eq!(GemmSpec::new(&dy,&x,Transpose::Yes,Transpose::No,DType::F32).unwrap().out.shape(),w.shape());}
    #[test] fn scheduler_overflow_rejected() {let a=t(&[16*(1<<20),16]);let b=t(&[16,16*(1<<20)]);assert!(GemmSpec::new(&a,&b,Transpose::No,Transpose::No,DType::BF16).is_err());}
}
