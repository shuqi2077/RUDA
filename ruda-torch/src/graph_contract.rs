//! Host-only validation shared by the native static graph bridge.
//! No backend calls or GPU emulation. Kernels are fixed, contiguous, inference-only.

pub const MAX_NODES: usize = 256;
pub const MAX_TENSORS: usize = 512;
pub const NO_WEIGHT: u32 = u32::MAX;
pub const COPY: u32 = 0;
pub const ADD: u32 = 1;
pub const MUL: u32 = 2;
pub const SILU: u32 = 14;
pub const RMS_NORM: u32 = 100;
pub const SILU_MUL: u32 = 101;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NodeSpec {
    pub op: u32,
    pub a: u32,
    pub b: u32,
    pub scalar: f32,
}

#[derive(Debug)]
pub struct TensorSpec<'a> {
    pub shape: &'a [usize],
    pub strides: &'a [usize],
    pub dtype: u32,
}

pub fn validate(tensors: &[TensorSpec<'_>], nodes: &[NodeSpec], inputs: usize) -> Result<(), String> {
    if nodes.is_empty() || nodes.len() > MAX_NODES || inputs == 0
        || tensors.len() > MAX_TENSORS || inputs.checked_add(nodes.len()) != Some(tensors.len()) {
        return Err("static graph requires 1..256 nodes, nonempty inputs, and one logical output per node (max 512 tensors)".into());
    }
    for t in tensors {
        if t.dtype > 2 || t.shape.is_empty() || t.shape.len() > 8 || t.shape.len() != t.strides.len() {
            return Err("static graph accepts rank 1..8 FP32/FP16/BF16 tensors".into());
        }
        let mut count = 1usize;
        for (&dim, &stride) in t.shape.iter().zip(t.strides).rev() {
            if dim == 0 || (dim > 1 && stride != count) {
                return Err("static graph requires nonempty contiguous storage".into());
            }
            count = count.checked_mul(dim).ok_or("static graph shape overflow")?;
        }
        if count > u32::MAX as usize { return Err("static graph exceeds 32-bit kernel indexing".into()); }
    }
    for (i, n) in nodes.iter().enumerate() {
        let out_index = inputs + i;
        if n.a as usize >= out_index { return Err("static graph input must precede its output".into()); }
        let a = &tensors[n.a as usize];
        let out = &tensors[out_index];
        if a.shape != out.shape || a.dtype != out.dtype { return Err("static graph output shape/dtype mismatch".into()); }
        match n.op {
            COPY | SILU => {
                if n.b != n.a || n.scalar.to_bits() != 0f32.to_bits() {
                    return Err("unary graph node must use its input as dummy and canonical zero scalar".into());
                }
            }
            ADD | MUL | SILU_MUL => {
                if n.b as usize >= out_index { return Err("static graph second input must precede its output".into()); }
                let b = &tensors[n.b as usize];
                if a.shape != b.shape || a.dtype != b.dtype { return Err("static graph broadcasting/type promotion is not supported".into()); }
                if !n.scalar.is_finite() || (n.op != ADD && n.scalar.to_bits() != 0f32.to_bits()) {
                    return Err("invalid static graph pointwise scalar".into());
                }
            }
            RMS_NORM => {
                if !n.scalar.is_finite() || n.scalar <= 0.0 { return Err("RMSNorm epsilon must be finite and positive".into()); }
                let width = *a.shape.last().unwrap();
                let rows = a.shape.iter().product::<usize>() / width;
                if rows > u32::MAX as usize / 32 { return Err("RMSNorm row grid exceeds 32-bit indexing".into()); }
                if n.b != NO_WEIGHT {
                    if n.b as usize >= out_index { return Err("RMSNorm weight must precede its output".into()); }
                    let w = &tensors[n.b as usize];
                    if w.shape != [width] || w.dtype != a.dtype { return Err("RMSNorm requires a same-dtype last-axis weight".into()); }
                }
            }
            _ => return Err("unsupported static graph operator (no CPU fallback)".into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &'static [usize], strides: &'static [usize], dtype: u32) -> TensorSpec<'static> {
        TensorSpec {shape, strides, dtype}
    }
    fn add() -> NodeSpec { NodeSpec {op: ADD, a: 0, b: 1, scalar: 1.0} }
    #[test] fn valid_add() { assert!(validate(&[t(&[2,3], &[3,1],0),t(&[2,3], &[3,1],0),t(&[2,3], &[3,1],0)], &[add()],2).is_ok()); }
    #[test] fn empty_graph() { assert!(validate(&[], &[],0).is_err()); }
    #[test] fn wrong_slot_count() { assert!(validate(&[t(&[1],&[1],0)], &[add()],1).is_err()); }
    #[test] fn forward_reference() { let mut n=add(); n.b=2; assert!(validate(&[t(&[1],&[1],0),t(&[1],&[1],0),t(&[1],&[1],0)], &[n],2).is_err()); }
    #[test] fn zero_dimension() { assert!(validate(&[t(&[0],&[1],0),t(&[0],&[1],0)], &[NodeSpec{op:COPY,a:0,b:0,scalar:0.}],1).is_err()); }
    #[test] fn noncontiguous() { assert!(validate(&[t(&[2],&[2],0),t(&[2],&[1],0)], &[NodeSpec{op:COPY,a:0,b:0,scalar:0.}],1).is_err()); }
    #[test] fn no_promotion() { assert!(validate(&[t(&[2],&[1],1),t(&[2],&[1],2),t(&[2],&[1],1)], &[add()],2).is_err()); }
    #[test] fn no_broadcast() { assert!(validate(&[t(&[2],&[1],0),t(&[1],&[1],0),t(&[2],&[1],0)], &[add()],2).is_err()); }
    #[test] fn unknown_operator() { assert!(validate(&[t(&[2],&[1],0),t(&[2],&[1],0)], &[NodeSpec{op:7,a:0,b:0,scalar:0.}],1).is_err()); }
    #[test] fn rms_weight_shape() { assert!(validate(&[t(&[2,3],&[3,1],0),t(&[2],&[1],0),t(&[2,3],&[3,1],0)], &[NodeSpec{op:RMS_NORM,a:0,b:1,scalar:1e-5}],2).is_err()); }
    #[test] fn rms_without_weight() { assert!(validate(&[t(&[2,3],&[3,1],1),t(&[2,3],&[3,1],1)], &[NodeSpec{op:RMS_NORM,a:0,b:NO_WEIGHT,scalar:1e-5}],1).is_ok()); }
    #[test] fn rms_bad_epsilon() { for eps in [0.,-1.,f32::NAN,f32::INFINITY] { assert!(validate(&[t(&[3],&[1],0),t(&[3],&[1],0)], &[NodeSpec{op:RMS_NORM,a:0,b:NO_WEIGHT,scalar:eps}],1).is_err()); } }
    #[test] fn unary_negative_zero_rejected() { assert!(validate(&[t(&[1],&[1],0),t(&[1],&[1],0)], &[NodeSpec{op:SILU,a:0,b:0,scalar:-0.}],1).is_err()); }
    #[test] fn valid_silu_mul_storage_dtypes() { for dt in 0..=2 { assert!(validate(&[t(&[2,3],&[3,1],dt),t(&[2,3],&[3,1],dt),t(&[2,3],&[3,1],dt)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar:0.}],2).is_ok()); } }
    #[test] fn silu_mul_noncanonical_scalar() { for scalar in [-0.,1.,f32::NAN,f32::INFINITY] { assert!(validate(&[t(&[2],&[1],0),t(&[2],&[1],0),t(&[2],&[1],0)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar}],2).is_err()); } }
    #[test] fn silu_mul_future_input() { assert!(validate(&[t(&[2],&[1],0),t(&[2],&[1],0)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar:0.}],1).is_err()); }
    #[test] fn silu_mul_no_promotion() { assert!(validate(&[t(&[2],&[1],1),t(&[2],&[1],2),t(&[2],&[1],1)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar:0.}],2).is_err()); }
    #[test] fn silu_mul_no_broadcast() { assert!(validate(&[t(&[2],&[1],0),t(&[1],&[1],0),t(&[2],&[1],0)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar:0.}],2).is_err()); }
    #[test] fn silu_mul_code_is_extension_only() { assert_eq!(SILU_MUL,101); assert_ne!(SILU_MUL,SILU); }
    #[test] fn abi_layout() { assert_eq!(std::mem::size_of::<NodeSpec>(),16); assert_eq!(std::mem::align_of::<NodeSpec>(),4); }
}
