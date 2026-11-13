//! Rotary embeddings: explicit position IDs, ragged offsets, interleaving,
//! Llama 3.1 wavelength scaling and reusable cos/sin tables.
use super::{InferenceError, alloc, check, float};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};

#[derive(Clone, Copy, Debug)]
pub struct Llama31Scaling {
    pub low_frequency_factor: f32,
    pub high_frequency_factor: f32,
    pub original_context_length: f32,
}
#[derive(Clone, Copy, Debug)]
pub struct RopeOptions {
    pub rotary_dim: usize,
    pub interleaved: bool,
    pub theta: f32,
    pub scale: f32,
    pub llama31: Option<Llama31Scaling>,
}
impl RopeOptions {
    pub fn validate(self, width: usize) -> Result<(), InferenceError> {
        if self.rotary_dim == 0
            || self.rotary_dim % 2 != 0
            || self.rotary_dim > width
            || !self.theta.is_finite()
            || self.theta <= 0.0
            || !self.scale.is_finite()
            || self.scale <= 0.0
        {
            return Err(InferenceError(
                "invalid rotary dimension, theta or scaling factor",
            ));
        }
        if let Some(s) = self.llama31 {
            if !s.low_frequency_factor.is_finite()
                || !s.high_frequency_factor.is_finite()
                || !s.original_context_length.is_finite()
                || s.low_frequency_factor <= 0.0
                || s.high_frequency_factor <= s.low_frequency_factor
                || s.original_context_length <= 0.0
            {
                return Err(InferenceError("invalid Llama 3.1 wavelength scaling"));
            }
        }
        Ok(())
    }
}
pub struct RotaryPair<R: Runtime> {
    pub query: RudaTensor<R>,
    pub key: RudaTensor<R>,
}

/// Build device position IDs from CPU scheduler metadata. No activation data is
/// downloaded. Offsets are absolute starts, not the lengths of earlier chunks.
pub fn positions_from_ragged<R: Runtime>(
    like: &RudaTensor<R>,
    indptr: &[u32],
    offsets: &[i64],
) -> Result<RudaTensor<R>, InferenceError> {
    if indptr.len() != offsets.len() + 1
        || indptr.first() != Some(&0)
        || indptr.windows(2).any(|p| p[0] > p[1])
    {
        return Err(InferenceError("invalid ragged rotary positions"));
    }
    let mut positions = Vec::with_capacity(*indptr.last().unwrap() as usize);
    for (i, range) in indptr.windows(2).enumerate() {
        for j in 0..range[1] - range[0] {
            positions.push(
                offsets[i]
                    .checked_add(j as i64)
                    .ok_or(InferenceError("position overflow"))?,
            );
        }
    }
    Ok(RudaTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        Shape::from([positions.len()]),
        like.client
            .create_from_slice(bytemuck::cast_slice(&positions)),
        DType::I64,
    ))
}

pub fn apply_rope<R: Runtime>(
    q: &RudaTensor<R>,
    k: &RudaTensor<R>,
    positions: &RudaTensor<R>,
    options: RopeOptions,
) -> Result<RotaryPair<R>, InferenceError> {
    Ok(RotaryPair {
        query: apply_one(q, positions, options, None)?,
        key: apply_one(k, positions, options, None)?,
    })
}
/// Tables are F32 [max_position, rotary_dim]: cosine half then sine half.
/// Negative/out-of-table device IDs produce NaN outputs, without an OOB load.
pub fn apply_rope_with_cos_sin_cache<R: Runtime>(
    q: &RudaTensor<R>,
    k: &RudaTensor<R>,
    positions: &RudaTensor<R>,
    table: &RudaTensor<R>,
    rotary_dim: usize,
    interleaved: bool,
) -> Result<RotaryPair<R>, InferenceError> {
    let options = RopeOptions {
        rotary_dim,
        interleaved,
        theta: 10000.0,
        scale: 1.0,
        llama31: None,
    };
    Ok(RotaryPair {
        query: apply_one(q, positions, options, Some(table))?,
        key: apply_one(k, positions, options, Some(table))?,
    })
}
pub fn apply_one<R: Runtime>(
    x: &RudaTensor<R>,
    positions: &RudaTensor<R>,
    options: RopeOptions,
    table: Option<&RudaTensor<R>>,
) -> Result<RudaTensor<R>, InferenceError> {
    execute(x, positions, options, table, None)
}
/// Preallocated/native in-place variant. Each unit owns both halves of a
/// rotary pair, so an output that aliases x does not race with another unit.
/// The caller owns exclusivity of the output and any in-flight launches.
pub unsafe fn apply_one_into<R: Runtime>(
    x: &RudaTensor<R>,
    positions: &RudaTensor<R>,
    options: RopeOptions,
    table: Option<&RudaTensor<R>>,
    out: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    execute(x, positions, options, table, Some(out)).map(|_| ())
}
fn execute<R: Runtime>(
    x: &RudaTensor<R>,
    positions: &RudaTensor<R>,
    options: RopeOptions,
    table: Option<&RudaTensor<R>>,
    destination: Option<&RudaTensor<R>>,
) -> Result<RudaTensor<R>, InferenceError> {
    check(x, x)?;
    check(positions, x)?;
    float(x)?;
    if x.meta.num_dims() != 3
        || x.meta.shape()[1] == 0
        || !matches!(positions.dtype, DType::I32 | DType::I64)
        || positions.meta.shape()[..] != [x.meta.shape()[0]]
    {
        return Err(InferenceError(
            "RoPE requires [tokens,heads,width] and I32/I64 [tokens] positions",
        ));
    }
    let width = x.meta.shape()[2];
    options.validate(width)?;
    if table.is_some_and(|t| t.meta.num_dims() != 2) {
        return Err(InferenceError("cos/sin table must have rank two"));
    }
    let table_tensor = table.unwrap_or(x);
    let table_rows = table.map_or(0, |t| t.meta.shape()[0]);
    if let Some(table) = table {
        check(table, x)?;
        if table.dtype != DType::F32 || table.meta.shape()[..] != [table_rows, options.rotary_dim] {
            return Err(InferenceError(
                "cos/sin table must be F32 [positions,rotary_dim]",
            ));
        }
    }
    let out = if let Some(out) = destination {
        check(out, x)?;
        if out.dtype != x.dtype || out.meta.shape() != x.meta.shape() {
            return Err(InferenceError("rotary destination mismatch"));
        }
        out.clone()
    } else {
        alloc(x, x.meta.shape().clone(), x.dtype)
    };
    let n = x.meta.num_elements();
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        let s = options.llama31.unwrap_or(Llama31Scaling {
            low_frequency_factor: 1.0,
            high_frequency_factor: 4.0,
            original_context_length: 8192.0,
        });
        rotate::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            positions.clone().into_array_arg(),
            table_tensor.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            x.meta.shape()[1] as u32,
            width as u32,
            options.theta,
            options.scale,
            s.low_frequency_factor,
            s.high_frequency_factor,
            s.original_context_length,
            table_rows as u32,
            options.rotary_dim,
            options.interleaved,
            options.llama31.is_some(),
            table.is_some(),
            x.dtype.into(),
            positions.dtype.into(),
        );
    }
    Ok(out)
}
/// Rust ownership-preserving in-place update: aliases retain their snapshot.
pub fn apply_rope_inplace<R: Runtime>(
    q: &mut RudaTensor<R>,
    k: &mut RudaTensor<R>,
    positions: &RudaTensor<R>,
    options: RopeOptions,
) -> Result<(), InferenceError> {
    let qm = q.can_mut();
    let km = k.can_mut();
    if !qm {
        *q = crate::paged_attention::copy_contiguous(q);
    }
    if !km {
        *k = crate::paged_attention::copy_contiguous(k);
    }
    execute(q, positions, options, None, Some(q))?;
    execute(k, positions, options, None, Some(k))?;
    Ok(())
}

#[ruda(launch)]
pub fn rotate<F: Float, P: Int>(
    x: &Array<F>,
    positions: &Array<P>,
    table: &Array<f32>,
    out: &mut Array<F>,
    n: u32,
    heads: u32,
    width: u32,
    theta: f32,
    scale: f32,
    low: f32,
    high: f32,
    old_context: f32,
    table_rows: u32,
    #[comptime] rotary: usize,
    #[comptime] interleaved: bool,
    #[comptime] llama31: bool,
    #[comptime] cached: bool,
    #[define(F)] _dtype: StorageType,
    #[define(P)] _position_dtype: StorageType,
) {
    let unit = ABSOLUTE_POS as usize;
    let units = width as usize - rotary / 2;
    if unit < (n as usize / width as usize) * units {
        let head_row = unit / units;
        let local = unit % units;
        let row = head_row / heads as usize;
        let base = head_row * width as usize;
        if local >= rotary / 2 {
            let d = rotary + local - rotary / 2;
            out[base + d] = x[base + d];
        } else {
            let pair = local;
            let mut first = pair;
            let mut second = pair + rotary / 2;
            if comptime!(interleaved) {
                first = pair * 2;
                second = pair * 2 + 1;
            }
            let position = positions[row];
            let mut c: f32 = 0.0f32;
            let mut s: f32 = 0.0f32;
            let mut valid = true;
            if comptime!(cached) {
                if position < P::cast_from(0i32) || position >= P::cast_from(table_rows) {
                    valid = false;
                } else {
                    let offset = usize::cast_from(position) * rotary;
                    c = table[offset + pair];
                    s = table[offset + rotary / 2 + pair];
                }
            } else {
                let mut frequency = f32::powf(theta, -2.0 * pair as f32 / rotary as f32);
                if comptime!(llama31) {
                    let wavelength = core::f32::consts::TAU / frequency;
                    if wavelength > old_context / low {
                        frequency /= scale;
                    } else {
                        if wavelength >= old_context / high {
                            let blend = (old_context / wavelength - low) / (high - low);
                            frequency = frequency / scale * (1.0 - blend) + frequency * blend;
                        }
                    }
                } else {
                    frequency /= scale;
                }
                let angle = f32::cast_from(position) * frequency;
                c = angle.cos();
                s = angle.sin();
            }
            let a = f32::cast_from(x[base + first]);
            let b = f32::cast_from(x[base + second]);
            let mut va = f32::NAN;
            let mut vb = f32::NAN;
            if valid {
                va = a * c - b * s;
                vb = b * c + a * s;
            }
            out[base + first] = F::cast_from(va);
            out[base + second] = F::cast_from(vb);
        }
    }
}
