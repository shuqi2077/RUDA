//! Packed query sparse attention from on-device blocks or token IDs.
use super::{InferenceError, alloc, check};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
pub struct SparseOutput<R: Runtime> {
    pub values: RudaTensor<R>,
    pub values_f32: RudaTensor<R>,
    pub lse: RudaTensor<R>,
}
pub struct QsaPlan<R: Runtime> {
    pub schedule: RudaTensor<R>,
    pub indices: RudaTensor<R>,
    pub attention: super::attention::AttentionPlan<R>,
    queries: usize,
    keys: usize,
    width: usize,
}
impl<R: Runtime> QsaPlan<R> {
    pub fn new(
        like: &RudaTensor<R>,
        queries: usize,
        keys: usize,
        selection_width: usize,
    ) -> Result<Self, InferenceError> {
        if queries
            .checked_mul(selection_width)
            .filter(|n| *n <= u32::MAX as usize)
            .is_none()
            || keys > i32::MAX as usize
            || queries > i32::MAX as usize
            || selection_width == 0
        {
            return Err(InferenceError("QSA capacities invalid"));
        }
        let words: Vec<u32> = (0..queries)
            .flat_map(|row| [0, keys as u32, row as u32, 0, selection_width as u32])
            .collect();
        let schedule = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            ruda_core::tensor::Shape::from([queries, 5]),
            like.client.create_from_slice(bytemuck::cast_slice(&words)),
            DType::U32,
        );
        let indices = alloc(like, [queries, selection_width], DType::I32);
        let attention = super::attention::AttentionPlan::device(&schedule, keys, keys)?;
        Ok(Self {
            schedule,
            indices,
            attention,
            queries,
            keys,
            width: selection_width,
        })
    }
    pub fn indexed(
        &self,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        indices: &RudaTensor<R>,
        scale: f32,
    ) -> Result<SparseOutput<R>, InferenceError> {
        check(indices, q)?;
        if indices.dtype != DType::I32 || indices.meta.shape()[..] != [self.queries, self.width] {
            return Err(InferenceError("QSA indices shape mismatch"));
        }
        let qf = super::norm::cast(q, DType::F32)?;
        let kf = super::norm::cast(k, DType::F32)?;
        let vf = super::norm::cast(v, DType::F32)?;
        let state = self.attention.run(
            &qf,
            &kf,
            &vf,
            super::attention::Options {
                scale,
                causal: false,
                window_left: -1,
                window_right: -1,
                soft_cap: 0.0,
                lse_base: super::cascade::LogBase::Natural,
            },
            super::attention::RuntimeInputs {
                indexed_keys: Some(indices),
                ..Default::default()
            },
        )?;
        let values = super::norm::cast(&state.values, q.dtype)?;
        let heads = state.lse.meta.shape()[1];
        let lse = alloc(&state.lse, [heads, self.queries], DType::F32);
        let n = heads * self.queries;
        if n > 0 {
            let dim = RudaDim::new(lse.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&lse.client, n, dim);
            transpose_lse::launch::<R>(
                &lse.client,
                count,
                dim,
                state.lse.into_array_arg(),
                lse.clone().into_array_arg(),
                self.queries as u32,
                heads as u32,
            );
        }
        Ok(SparseOutput {
            values,
            values_f32: state.values,
            lse,
        })
    }
    /// Expand a selection bitmap using inclusive token spans. A token appears
    /// once even if input block spans overlap; lo/hi clip selected blocks.
    pub fn blocks(
        &self,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        bitmap: &RudaTensor<R>,
        lo: &RudaTensor<R>,
        hi: &RudaTensor<R>,
        block_first: &RudaTensor<R>,
        block_last: &RudaTensor<R>,
        scale: f32,
    ) -> Result<SparseOutput<R>, InferenceError> {
        for t in [bitmap, lo, hi, block_first, block_last] {
            check(t, q)?;
        }
        if bitmap.dtype != DType::U8
            || bitmap.meta.num_dims() != 2
            || bitmap.meta.shape()[0] != self.queries
            || lo.dtype != DType::I32
            || hi.dtype != DType::I32
            || lo.meta.shape()[..] != [self.queries]
            || hi.meta.shape() != lo.meta.shape()
            || block_first.dtype != DType::I32
            || block_last.dtype != DType::I32
            || block_first.meta.shape()[..] != [bitmap.meta.shape()[1]]
            || block_last.meta.shape() != block_first.meta.shape()
            || self.width < self.keys
        {
            return Err(InferenceError("QSA block bitmap/bounds/capacity mismatch"));
        }
        if self.queries > 0 {
            let dim = RudaDim::new(q.client.properties(), self.queries);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, self.queries, dim);
            expand_blocks::launch::<R>(
                &q.client,
                count,
                dim,
                bitmap.clone().into_array_arg(),
                lo.clone().into_array_arg(),
                hi.clone().into_array_arg(),
                block_first.clone().into_array_arg(),
                block_last.clone().into_array_arg(),
                self.indices.clone().into_array_arg(),
                self.queries as u32,
                self.keys as u32,
                self.width as u32,
                bitmap.meta.shape()[1] as u32,
            );
        }
        self.indexed(q, k, v, &self.indices, scale)
    }
}
/// Allocation-free indexed/bitmap QSA with retained dense or FP4/FP8 format
/// decoders and both typed/FP32 values plus head-major LSE destinations.
pub struct PreparedQsa<'a, R: Runtime> {
    plan: &'a QsaPlan<R>,
    attention: super::attention::PreparedAttention<'a, R>,
    pub output: SparseOutput<R>,
}
impl<'a, R: Runtime> PreparedQsa<'a, R> {
    pub fn prepare(
        plan: &'a QsaPlan<R>,
        q: super::linear::Matrix<'a, R>,
        k: super::linear::Matrix<'a, R>,
        v: super::linear::Matrix<'a, R>,
        dtype: DType,
    ) -> Result<Self, InferenceError> {
        let attention =
            super::attention::PreparedAttention::prepare(&plan.attention, q, k, v, DType::F32)?;
        let values_f32 = attention.output.values.clone();
        let values = alloc(&values_f32, values_f32.meta.shape().clone(), dtype);
        super::float(&values)?;
        let heads = values_f32.meta.shape()[1];
        let lse = alloc(&values_f32, [heads, plan.queries], DType::F32);
        Ok(Self {
            plan,
            attention,
            output: SparseOutput {
                values,
                values_f32,
                lse,
            },
        })
    }
    pub unsafe fn indexed(
        &self,
        indices: &RudaTensor<R>,
        scale: f32,
    ) -> Result<&SparseOutput<R>, InferenceError> {
        check(indices, &self.output.values)?;
        if indices.dtype != DType::I32
            || indices.meta.shape()[..] != [self.plan.queries, self.plan.width]
        {
            return Err(InferenceError("prepared QSA index dimensions mismatch"));
        }
        let options = super::attention::Options {
            scale,
            causal: false,
            window_left: -1,
            window_right: -1,
            soft_cap: 0.0,
            lse_base: super::cascade::LogBase::Natural,
        };
        let state = unsafe {
            self.attention.run(
                options,
                super::attention::RuntimeInputs {
                    indexed_keys: Some(indices),
                    ..Default::default()
                },
            )?
        };
        unsafe {
            super::norm::cast_into(&state.values, &self.output.values)?;
        }
        let heads = state.lse.meta.shape()[1];
        let n = heads * self.plan.queries;
        if n > 0 {
            let dim = RudaDim::new(state.lse.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&state.lse.client, n, dim);
            transpose_lse::launch::<R>(
                &state.lse.client,
                count,
                dim,
                state.lse.clone().into_array_arg(),
                self.output.lse.clone().into_array_arg(),
                self.plan.queries as u32,
                heads as u32,
            );
        }
        Ok(&self.output)
    }
    pub unsafe fn blocks(
        &self,
        bitmap: &RudaTensor<R>,
        lo: &RudaTensor<R>,
        hi: &RudaTensor<R>,
        first: &RudaTensor<R>,
        last: &RudaTensor<R>,
        scale: f32,
    ) -> Result<&SparseOutput<R>, InferenceError> {
        for t in [bitmap, lo, hi, first, last] {
            check(t, &self.output.values)?;
        }
        if bitmap.dtype != DType::U8
            || bitmap.meta.num_dims() != 2
            || bitmap.meta.shape()[0] != self.plan.queries
            || lo.dtype != DType::I32
            || hi.dtype != DType::I32
            || lo.meta.shape()[..] != [self.plan.queries]
            || hi.meta.shape() != lo.meta.shape()
            || first.dtype != DType::I32
            || last.dtype != DType::I32
            || first.meta.shape()[..] != [bitmap.meta.shape()[1]]
            || last.meta.shape() != first.meta.shape()
            || self.plan.width < self.plan.keys
        {
            return Err(InferenceError("prepared QSA bitmap/bounds mismatch"));
        }
        if self.plan.queries > 0 {
            let dim = RudaDim::new(bitmap.client.properties(), self.plan.queries);
            let count =
                kernel_dsl::calculate_ruda_count_elemwise(&bitmap.client, self.plan.queries, dim);
            expand_blocks::launch::<R>(
                &bitmap.client,
                count,
                dim,
                bitmap.clone().into_array_arg(),
                lo.clone().into_array_arg(),
                hi.clone().into_array_arg(),
                first.clone().into_array_arg(),
                last.clone().into_array_arg(),
                self.plan.indices.clone().into_array_arg(),
                self.plan.queries as u32,
                self.plan.keys as u32,
                self.plan.width as u32,
                bitmap.meta.shape()[1] as u32,
            );
        }
        unsafe { self.indexed(&self.plan.indices, scale) }
    }
}
#[ruda(launch)]
pub fn expand_blocks(
    bitmap: &Array<u8>,
    lo: &Array<i32>,
    hi: &Array<i32>,
    first: &Array<i32>,
    last: &Array<i32>,
    indices: &mut Array<i32>,
    queries: u32,
    keys: u32,
    width: u32,
    blocks: u32,
) {
    let row = ABSOLUTE_POS as usize;
    if row < queries as usize {
        let mut position: usize = 0usize;
        for token in 0..keys as usize {
            let mut live = false;
            if token as i32 >= lo[row] && token as i32 <= hi[row] {
                for b in 0..blocks as usize {
                    live = live
                        || (bitmap[row * blocks as usize + b] != 0
                            && token as i32 >= first[b]
                            && token as i32 <= last[b]);
                }
            }
            if live {
                indices[row * width as usize + position] = token as i32;
                position += 1;
            }
        }
        while position < width as usize {
            indices[row * width as usize + position] = -1;
            position += 1;
        }
    }
}
#[ruda(launch)]
pub fn transpose_lse(x: &Array<f32>, y: &mut Array<f32>, rows: u32, heads: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < rows as usize * heads as usize {
        let row = i / heads as usize;
        let head = i % heads as usize;
        y[head * rows as usize + row] = x[i];
    }
}
