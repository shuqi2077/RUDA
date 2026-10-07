//! Host-only validation shared by the native static graph bridge.
//! No backend calls or GPU emulation. Kernels use fixed, contiguous buffers; AOTAutograd owns training differentiation.

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
        if t.dtype > 2 || t.shape.len() > 8 || t.shape.len() != t.strides.len() {
            return Err("static graph accepts rank 0..8 FP32/FP16/BF16 tensors".into());
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
        if (n.op != 114 && a.dtype != out.dtype) || (!matches!(n.op, 7 | 30 | 57 | 104 | 105 | 112..=124) && a.shape != out.shape) {
            return Err("static graph output shape/dtype mismatch".into());
        }
        match n.op {
            112 | 114 => {
                if n.b != n.a || n.scalar.to_bits()!=0 || (n.op==112 && a.shape.iter().product::<usize>()!=out.shape.iter().product::<usize>())
                    || (n.op==114 && a.shape!=out.shape) {
                    return Err("invalid graph reshape/cast".into());
                }
            }
            113 => {
                let code=n.scalar as u32;
                if !n.scalar.is_finite() || code as f32!=n.scalar || n.b!=n.a || a.shape.len()!=out.shape.len()
                    || code >= (1u32 << (3*a.shape.len())) {
                    return Err("invalid graph permutation encoding".into());
                }
                let mut seen=0u32;
                for (dim,&size) in out.shape.iter().enumerate() {
                    let axis=((code>>(3*dim))&7) as usize;
                    if axis>=a.shape.len() || seen&(1<<axis)!=0 || size!=a.shape[axis] {
                        return Err("invalid graph permutation axes/shape".into());
                    }
                    seen|=1<<axis;
                }
            }
            115 => {
                if n.b!=n.a || n.scalar.to_bits()!=0 || a.shape.len()>out.shape.len()
                    || a.shape.iter().rev().zip(out.shape.iter().rev()).any(|(&x,&y)|x!=1 && x!=y) {
                    return Err("invalid graph expansion".into());
                }
            }
            57 | 116..=118 | 121..=124 => {
                if n.b as usize>=out_index || !n.scalar.is_finite() || (n.op!=116 && n.scalar.to_bits()!=0) {
                    return Err("invalid graph broadcast inputs/scalar".into());
                }
                let b=&tensors[n.b as usize];
                let rank=a.shape.len().max(b.shape.len());
                if b.dtype!=a.dtype || out.shape.len()!=rank {return Err("invalid graph broadcast dtype/rank".into());}
                for dim in 0..rank {
                    let x=if dim+a.shape.len()<rank {1} else {a.shape[dim+a.shape.len()-rank]};
                    let y=if dim+b.shape.len()<rank {1} else {b.shape[dim+b.shape.len()-rank]};
                    if (x!=y && x!=1 && y!=1) || out.shape[dim]!=x.max(y) {
                        return Err("invalid graph broadcast output".into());
                    }
                }
            }
            119 | 120 => {
                let mask=n.scalar as u32;
                let expected:Vec<_>=a.shape.iter().enumerate().filter_map(|(d,&size)|
                    if mask&(1<<d)==0 {Some(size)} else {None}).collect();
                if !n.scalar.is_finite() || mask as f32!=n.scalar || mask==0 || mask>=(1<<a.shape.len())
                    || n.b!=n.a || out.shape!=expected.as_slice() {
                    return Err("invalid squeezed reduction output/mask".into());
                }
            }
            COPY | 3 | 9..=14 | 17 | 19..=25 | 27 | 35..=40 | 43 | 45 | 108 => {
                if n.b != n.a || n.scalar.to_bits() != 0f32.to_bits() {
                    return Err("unary graph node must use its input as dummy and canonical zero scalar".into());
                }
            }
            ADD | MUL | 8 | 15 | 16 | 18 | SILU_MUL => {
                if n.b as usize >= out_index { return Err("static graph second input must precede its output".into()); }
                let b = &tensors[n.b as usize];
                if a.shape != b.shape || a.dtype != b.dtype { return Err("static graph broadcasting/type promotion is not supported".into()); }
                if !n.scalar.is_finite() || (n.op != ADD && n.scalar.to_bits() != 0f32.to_bits()) {
                    return Err("invalid static graph pointwise scalar".into());
                }
            }
            109..=111 | 129..=133 => {
                if n.b != n.a || !n.scalar.is_finite() {
                    return Err("invalid unary scalar operation".into());
                }
            }
            7 | 30 => {
                if n.b as usize >= out_index { return Err("future matrix input".into()); }
                let b = &tensors[n.b as usize];
                let rank = if n.op == 7 { 2 } else { 3 };
                if a.shape.len() != rank || b.shape.len() != rank || out.shape.len() != rank
                    || b.dtype != a.dtype || n.scalar.to_bits() != 0
                    || a.shape[rank-1] != b.shape[rank-2]
                    || out.shape[rank-2] != a.shape[rank-2] || out.shape[rank-1] != b.shape[rank-1]
                    || a.shape[..rank-2] != b.shape[..rank-2] || a.shape[..rank-2] != out.shape[..rank-2] {
                    return Err("invalid static matrix multiply shapes/dtypes".into());
                }
            }
            102 | 103 | 106 | 107 => {
                if !n.scalar.is_finite() || n.scalar.fract() != 0.0 || n.scalar < 0.0
                    || n.scalar as usize >= a.shape.len() || n.b as usize >= out_index {
                    return Err("invalid softmax axis or input".into());
                }
                let b = &tensors[n.b as usize];
                if b.shape != a.shape || b.dtype != a.dtype || (n.op < 106 && n.a != n.b) {
                    return Err("softmax gradient/output mismatch".into());
                }
            }
            104 | 105 => {
                let mask = n.scalar as u32;
                if !n.scalar.is_finite() || mask as f32 != n.scalar || mask == 0
                    || mask >= (1 << a.shape.len()) || n.b != n.a || out.shape.len() != a.shape.len()
                    || a.shape.iter().enumerate().any(|(d, &v)| out.shape[d] != if mask & (1 << d) != 0 {1} else {v}) {
                    return Err("invalid keepdim reduction mask/output".into());
                }
            }
            RMS_NORM => {
                if !n.scalar.is_finite() || n.scalar <= 0.0 { return Err("RMSNorm epsilon must be finite and positive".into()); }
                if a.shape.is_empty() {return Err("RMSNorm requires a feature axis".into());}
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

#[cfg(test)]
mod api3_tests {
    use super::*;
    #[test]
    fn changed_matrix_output_shape_is_validated() {
        fn spec(shape: &'static [usize], strides: &'static [usize]) -> TensorSpec<'static> { TensorSpec{shape,strides,dtype:0} }
        let values=[spec(&[2,3],&[3,1]),spec(&[3,4],&[4,1]),spec(&[2,4],&[4,1])];
        assert!(validate(&values,&[NodeSpec{op:7,a:0,b:1,scalar:0.}],2).is_ok());
    }
    #[test]
    fn reduction_bitmask_and_softmax_axis_checked() {
        let values=[TensorSpec{shape:&[2,3,4],strides:&[12,4,1],dtype:1},
            TensorSpec{shape:&[1,3,1],strides:&[3,1,1],dtype:1}];
        assert!(validate(&values,&[NodeSpec{op:104,a:0,b:0,scalar:5.}],1).is_ok());
        assert!(validate(&values,&[NodeSpec{op:104,a:0,b:0,scalar:2.}],1).is_err());
    }
}
