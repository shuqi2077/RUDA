//! Stable composition of normalized attention values and log-sum-exp states.
use super::{InferenceError, alloc, check, float, plane};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogBase {
    Natural,
    Two,
}
pub struct AttentionState<R: Runtime> {
    pub values: RudaTensor<R>,
    pub lse: RudaTensor<R>,
}
/// Ownership-preserving in-place merge. Optional U8/U32 row mask [rows]
/// leaves both values and LSE unchanged for disabled rows.
pub fn merge_state_in_place<R: Runtime>(
    a: &mut AttentionState<R>,
    b: &AttentionState<R>,
    mask: Option<&RudaTensor<R>>,
    base: LogBase,
) -> Result<(), InferenceError> {
    let values_mutable = a.values.can_mut();
    let lse_mutable = a.lse.can_mut();
    if !values_mutable {
        a.values = crate::paged_attention::copy_contiguous(&a.values);
    }
    if !lse_mutable {
        a.lse = crate::paged_attention::copy_contiguous(&a.lse);
    }
    unsafe { merge_into(a, b, a, mask, base) }
}
/// Fixed-address masked merge for graph capture. `destination` may equal `a`;
/// all other aliasing and concurrent writes are the caller's responsibility.
pub unsafe fn merge_into<R: Runtime>(
    a: &AttentionState<R>,
    b: &AttentionState<R>,
    destination: &AttentionState<R>,
    mask: Option<&RudaTensor<R>>,
    base: LogBase,
) -> Result<(), InferenceError> {
    for t in [
        &a.values,
        &a.lse,
        &b.values,
        &b.lse,
        &destination.values,
        &destination.lse,
    ] {
        check(t, &a.values)?;
    }
    float(&a.values)?;
    if a.values.meta.num_dims() != 3
        || b.values.meta.shape() != a.values.meta.shape()
        || destination.values.meta.shape() != a.values.meta.shape()
        || b.values.dtype != a.values.dtype
        || destination.values.dtype != a.values.dtype
    {
        return Err(InferenceError("masked merge value shapes/dtypes mismatch"));
    }
    let s = a.values.meta.shape();
    let (rows, heads, width) = (s[0], s[1], s[2]);
    for t in [&a.lse, &b.lse, &destination.lse] {
        if t.dtype != DType::F32 || t.meta.shape()[..] != [rows, heads] {
            return Err(InferenceError("masked merge LSE mismatch"));
        }
    }
    let md = mask.map_or(DType::U32, |t| t.dtype);
    if let Some(m) = mask {
        check(m, &a.values)?;
        if !matches!(m.dtype, DType::U8 | DType::U32) || m.meta.shape()[..] != [rows] {
            return Err(InferenceError("merge mask must be U8/U32 [rows]"));
        }
    }
    if heads == 0 || width == 0 {
        return Err(InferenceError("merge head axes must be nonempty"));
    }
    let lanes = plane(&a.values, rows, heads)?;
    if rows > 0 {
        masked_pair::launch::<R>(
            &a.values.client,
            RudaCount::Static(rows as u32, heads as u32, 1),
            RudaDim::new_1d(lanes),
            a.values.clone().into_array_arg(),
            a.lse.clone().into_array_arg(),
            b.values.clone().into_array_arg(),
            b.lse.clone().into_array_arg(),
            mask.unwrap_or(&a.lse).clone().into_array_arg(),
            destination.values.clone().into_array_arg(),
            destination.lse.clone().into_array_arg(),
            heads as u32,
            width as u32,
            mask.is_some(),
            base == LogBase::Two,
            a.values.dtype.into(),
            md.into(),
        );
    }
    Ok(())
}
pub struct CascadePlan<R: Runtime> {
    pub levels: Vec<super::attention::AttentionPlan<R>>,
}
impl<R: Runtime> CascadePlan<R> {
    /// Shared-prefix and arbitrary multi-level cascades are lists of disjoint
    /// attention intervals, merged in log space. Each level may reuse K/V pages.
    pub fn run(
        &self,
        query: &RudaTensor<R>,
        keys: &[&RudaTensor<R>],
        values: &[&RudaTensor<R>],
        options: &[super::attention::Options],
    ) -> Result<AttentionState<R>, InferenceError> {
        if self.levels.is_empty()
            || keys.len() != self.levels.len()
            || values.len() != keys.len()
            || options.len() != keys.len()
            || options.iter().any(|o| o.lse_base != options[0].lse_base)
        {
            return Err(InferenceError(
                "cascade requires nonempty matching levels with one LSE base",
            ));
        }
        let mut result =
            self.levels[0].run(query, keys[0], values[0], options[0], Default::default())?;
        for level in 1..self.levels.len() {
            let state = self.levels[level].run(
                query,
                keys[level],
                values[level],
                options[level],
                Default::default(),
            )?;
            result = merge_state(&result, &state, options[0].lse_base)?;
        }
        Ok(result)
    }
}

/// Prepared multi-level/shared-prefix cascade. Every level keeps its decoder
/// buffers and the accumulator is reset from level zero on every invocation.
pub struct PreparedCascade<'a, R: Runtime> {
    pub levels: Vec<super::attention::PreparedAttention<'a, R>>,
    pub output: AttentionState<R>,
}
impl<'a, R: Runtime> PreparedCascade<'a, R> {
    pub fn new(
        levels: Vec<super::attention::PreparedAttention<'a, R>>,
    ) -> Result<Self, InferenceError> {
        let first = levels
            .first()
            .ok_or(InferenceError("cascade requires at least one level"))?;
        let v = &first.output.values;
        let lse = &first.output.lse;
        for p in &levels {
            check(&p.output.values, v)?;
            if p.output.values.meta.shape() != v.meta.shape()
                || p.output.values.dtype != v.dtype
                || p.output.lse.meta.shape() != lse.meta.shape()
            {
                return Err(InferenceError(
                    "prepared cascade levels need identical output geometry",
                ));
            }
        }
        let output = AttentionState {
            values: alloc(v, v.meta.shape().clone(), v.dtype),
            lse: alloc(lse, lse.meta.shape().clone(), DType::F32),
        };
        Ok(Self { levels, output })
    }
    pub unsafe fn run(
        &self,
        options: &[super::attention::Options],
    ) -> Result<&AttentionState<R>, InferenceError> {
        if options.len() != self.levels.len()
            || options.iter().any(|p| p.lse_base != options[0].lse_base)
        {
            return Err(InferenceError(
                "cascade options must match all levels and LSE bases",
            ));
        }
        for (i, p) in self.levels.iter().enumerate() {
            let state = unsafe { p.run(options[i], Default::default())? };
            if i == 0 {
                ruda_kernel::library::tensor::copy_into(
                    &state.values.client,
                    state.values.clone().binding(),
                    self.output.values.clone().binding(),
                    state.values.dtype.into(),
                );
                ruda_kernel::library::tensor::copy_into(
                    &state.lse.client,
                    state.lse.clone().binding(),
                    self.output.lse.clone().binding(),
                    DType::F32.into(),
                );
            } else {
                unsafe {
                    merge_into(&self.output, state, &self.output, None, options[0].lse_base)?;
                }
            }
        }
        Ok(&self.output)
    }
}
/// Merge [rows, states, heads, width] values and F32 [rows, states, heads]
/// log-sum-exp. Empty states have LSE=-infinity. An entirely empty row returns
/// zeros/-infinity. NaN/+infinity states propagate NaN, rather than being lost.
pub fn merge_states<R: Runtime>(
    values: &RudaTensor<R>,
    lse: &RudaTensor<R>,
    base: LogBase,
) -> Result<AttentionState<R>, InferenceError> {
    check(values, values)?;
    check(lse, values)?;
    float(values)?;
    if values.meta.num_dims() != 4 || lse.dtype != DType::F32 {
        return Err(InferenceError("expected rank-four values and F32 LSE"));
    }
    let s = values.meta.shape();
    let (rows, states, heads, width) = (s[0], s[1], s[2], s[3]);
    if heads == 0
        || width == 0
        || lse.meta.shape()[..] != [rows, states, heads]
        || rows
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(width))
            .filter(|&n| n <= u32::MAX as usize)
            .is_none()
    {
        return Err(InferenceError("incompatible attention states"));
    }
    let lanes = plane(values, rows, heads)?;
    let out = alloc(values, [rows, heads, width], values.dtype);
    let out_lse = alloc(values, [rows, heads], DType::F32);
    if rows != 0 {
        merge::launch::<R>(
            &values.client,
            RudaCount::Static(rows as u32, heads as u32, 1),
            RudaDim::new_1d(lanes),
            values.clone().into_array_arg(),
            lse.clone().into_array_arg(),
            out.clone().into_array_arg(),
            out_lse.clone().into_array_arg(),
            states as u32,
            heads as u32,
            width as u32,
            base == LogBase::Two,
            values.dtype.into(),
        );
    }
    Ok(AttentionState {
        values: out,
        lse: out_lse,
    })
}

/// Pairwise merge. Consumes no input buffers; snapshots remain unchanged.
pub fn merge_state<R: Runtime>(
    a: &AttentionState<R>,
    b: &AttentionState<R>,
    base: LogBase,
) -> Result<AttentionState<R>, InferenceError> {
    for t in [&a.values, &a.lse, &b.values, &b.lse] {
        check(t, &a.values)?;
    }
    float(&a.values)?;
    if a.values.meta.num_dims() != 3
        || a.values.meta.shape() != b.values.meta.shape()
        || a.values.dtype != b.values.dtype
        || a.lse.dtype != DType::F32
        || b.lse.dtype != DType::F32
    {
        return Err(InferenceError("incompatible pairwise attention states"));
    }
    let s = a.values.meta.shape();
    let (rows, heads, width) = (s[0], s[1], s[2]);
    if heads == 0
        || width == 0
        || a.lse.meta.shape()[..] != [rows, heads]
        || b.lse.meta.shape() != a.lse.meta.shape()
    {
        return Err(InferenceError("incompatible pairwise LSE shapes"));
    }
    let lanes = plane(&a.values, rows, heads)?;
    let out = alloc(&a.values, [rows, heads, width], a.values.dtype);
    let lse = alloc(&a.values, [rows, heads], DType::F32);
    if rows != 0 {
        pair::launch::<R>(
            &a.values.client,
            RudaCount::Static(rows as u32, heads as u32, 1),
            RudaDim::new_1d(lanes),
            a.values.clone().into_array_arg(),
            a.lse.clone().into_array_arg(),
            b.values.clone().into_array_arg(),
            b.lse.clone().into_array_arg(),
            out.clone().into_array_arg(),
            lse.clone().into_array_arg(),
            heads as u32,
            width as u32,
            base == LogBase::Two,
            a.values.dtype.into(),
        );
    }
    Ok(AttentionState { values: out, lse })
}

#[ruda]
fn to_natural(x: f32, #[comptime] base_two: bool) -> f32 {
    let mut result = x;
    if comptime!(base_two) {
        result *= core::f32::consts::LN_2;
    }
    result
}
#[ruda]
fn from_natural(x: f32, #[comptime] base_two: bool) -> f32 {
    let mut result = x;
    if comptime!(base_two) {
        result *= core::f32::consts::LOG2_E;
    }
    result
}
#[ruda(launch)]
pub fn merge<F: Float>(
    values: &Array<F>,
    lse: &Array<f32>,
    out: &mut Array<F>,
    out_lse: &mut Array<f32>,
    states: u32,
    heads: u32,
    width: u32,
    #[comptime] base_two: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let head = RUDA_POS_Y as usize;
    let lane = UNIT_POS as usize;
    let mut maximum = f32::NEG_INFINITY;
    let mut bad = false;
    for state in 0..states as usize {
        let s = to_natural(
            lse[(row * states as usize + state) * heads as usize + head],
            base_two,
        );
        if s != s || s == f32::INFINITY {
            bad = true;
        }
        maximum = f32::max(maximum, s);
    }
    if bad {
        maximum = f32::NAN;
    }
    let mut denominator: f32 = 0.0f32;
    for state in 0..states as usize {
        let s = to_natural(
            lse[(row * states as usize + state) * heads as usize + head],
            base_two,
        );
        if s != f32::NEG_INFINITY {
            denominator += (s - maximum).exp();
        }
    }
    let mut d = lane;
    while d < width as usize {
        let mut value: f32 = 0.0f32;
        for state in 0..states as usize {
            let index = (row * states as usize + state) * heads as usize + head;
            let s = to_natural(lse[index], base_two);
            if s != f32::NEG_INFINITY {
                let weight = (s - maximum).exp();
                if weight != 0.0 {
                    value += weight * f32::cast_from(values[index * width as usize + d]);
                }
            }
        }
        if denominator != 0.0 {
            value /= denominator;
        }
        if bad {
            value = f32::NAN;
        }
        out[(row * heads as usize + head) * width as usize + d] = F::cast_from(value);
        d += PLANE_DIM as usize;
    }
    if lane == 0 {
        let mut result = f32::NEG_INFINITY;
        if denominator != 0.0 {
            result = maximum + denominator.ln();
        }
        if bad {
            result = f32::NAN;
        }
        out_lse[row * heads as usize + head] = from_natural(result, base_two);
    }
}

#[ruda(launch)]
pub fn pair<F: Float>(
    a: &Array<F>,
    sa: &Array<f32>,
    b: &Array<F>,
    sb: &Array<f32>,
    out: &mut Array<F>,
    lse: &mut Array<f32>,
    heads: u32,
    width: u32,
    #[comptime] base_two: bool,
    #[define(F)] _dtype: StorageType,
) {
    let index = RUDA_POS_X as usize * heads as usize + RUDA_POS_Y as usize;
    let x = to_natural(sa[index], base_two);
    let y = to_natural(sb[index], base_two);
    let maximum = f32::max(x, y);
    let bad = x != x || y != y || x == f32::INFINITY || y == f32::INFINITY;
    let mut wa: f32 = 0.0f32;
    let mut wb: f32 = 0.0f32;
    if x != f32::NEG_INFINITY {
        wa = (x - maximum).exp();
    }
    if y != f32::NEG_INFINITY {
        wb = (y - maximum).exp();
    }
    let denominator = wa + wb;
    let mut d = UNIT_POS as usize;
    while d < width as usize {
        let mut value: f32 = 0.0f32;
        // Do not multiply the undefined value of an empty state by zero.
        if wa != 0.0 {
            value += wa * f32::cast_from(a[index * width as usize + d]);
        }
        if wb != 0.0 {
            value += wb * f32::cast_from(b[index * width as usize + d]);
        }
        if denominator != 0.0 {
            value /= denominator;
        }
        if bad {
            value = f32::NAN;
        }
        out[index * width as usize + d] = F::cast_from(value);
        d += PLANE_DIM as usize;
    }
    if UNIT_POS == 0 {
        let mut result = f32::NEG_INFINITY;
        if denominator != 0.0 {
            result = maximum + denominator.ln();
        }
        if bad {
            result = f32::NAN;
        }
        lse[index] = from_natural(result, base_two);
    }
}
#[ruda(launch)]
pub fn masked_pair<F: Float, M: Int>(
    a: &Array<F>,
    sa: &Array<f32>,
    b: &Array<F>,
    sb: &Array<f32>,
    mask: &Array<M>,
    out: &mut Array<F>,
    lse: &mut Array<f32>,
    heads: u32,
    width: u32,
    #[comptime] masked: bool,
    #[comptime] base_two: bool,
    #[define(F)] _f: StorageType,
    #[define(M)] _m: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let index = row * heads as usize + RUDA_POS_Y as usize;
    let mut enabled = true;
    if comptime!(masked) {
        enabled = u32::cast_from(mask[row]) != 0;
    }
    let x = to_natural(sa[index], base_two);
    let y = to_natural(sb[index], base_two);
    let maximum = f32::max(x, y);
    let bad = x != x || y != y || x == f32::INFINITY || y == f32::INFINITY;
    let mut wa: f32 = 0.0f32;
    let mut wb: f32 = 0.0f32;
    if x != f32::NEG_INFINITY {
        wa = (x - maximum).exp();
    }
    if y != f32::NEG_INFINITY {
        wb = (y - maximum).exp();
    }
    let denominator = wa + wb;
    let mut d = UNIT_POS as usize;
    while d < width as usize {
        let address = index * width as usize + d;
        let mut value: f32 = 0.0f32;
        if enabled {
            if wa != 0.0 {
                value += wa * f32::cast_from(a[address]);
            }
            if wb != 0.0 {
                value += wb * f32::cast_from(b[address]);
            }
            if denominator != 0.0 {
                value /= denominator;
            }
            if bad {
                value = f32::NAN;
            }
            out[address] = F::cast_from(value);
        } else {
            out[address] = a[address];
        }
        d += PLANE_DIM as usize;
    }
    if UNIT_POS == 0 {
        if enabled {
            let mut result = f32::NEG_INFINITY;
            if denominator != 0.0 {
                result = maximum + denominator.ln();
            }
            if bad {
                result = f32::NAN;
            }
            lse[index] = from_natural(result, base_two);
        } else {
            lse[index] = sa[index];
        }
    }
}
