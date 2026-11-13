//! GPU-resident CSR page metadata, cache transfers and graph-replay schedules.
use super::{InferenceError, alloc, check, float, paged::CacheLayout};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, reshape::reshape},
};
pub struct CsrPages<R: Runtime> {
    pub indptr: RudaTensor<R>,
    pub indices: RudaTensor<R>,
    pub last_page_len: RudaTensor<R>,
    pub page_size: usize,
    pub sequences: usize,
    pub pages: usize,
}
impl<R: Runtime> CsrPages<R> {
    /// Physical row map for packed/quantized storage. Negative entries mask
    /// padded requests. Layout controls the row order before the final D axis.
    pub fn token_rows(
        &self,
        batch: &RudaTensor<R>,
        positions: &RudaTensor<R>,
        heads: usize,
        layout: CacheLayout,
    ) -> Result<RudaTensor<R>, InferenceError> {
        query_metadata(batch, positions, &self.indptr)?;
        let n = batch
            .meta
            .num_elements()
            .checked_mul(heads)
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("cache row map exceeds U32"))?;
        if heads == 0
            || self
                .pages
                .checked_mul(self.page_size)
                .and_then(|n| n.checked_mul(heads))
                .filter(|n| *n <= i32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError("cache physical rows exceed I32"));
        }
        let out = alloc(batch, [n], DType::I32);
        if n > 0 {
            let dim = RudaDim::new(batch.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&batch.client, n, dim);
            physical_rows::launch::<R>(
                &batch.client,
                count,
                dim,
                batch.clone().into_array_arg(),
                positions.clone().into_array_arg(),
                self.indptr.clone().into_array_arg(),
                self.indices.clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                heads as u32,
                self.sequences as u32,
                self.pages as u32,
                self.page_size as u32,
                self.indices.meta.num_elements() as u32,
                layout == CacheLayout::Hnd,
            );
        }
        Ok(out)
    }
    /// NVFP4/MXFP4/FP8 append updates payload and block-scale rows only; all
    /// untouched cache bytes are preserved. MLA [pages,P,D] uses one head.
    pub fn append_block_float(
        &self,
        tokens: &RudaTensor<R>,
        cache: &rublas::tensor_float_quant::BlockFloatTensor<R>,
        batch: &RudaTensor<R>,
        positions: &RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<rublas::tensor_float_quant::BlockFloatTensor<R>, InferenceError> {
        let shape = &cache.logical_shape;
        let (heads, width) =
            if shape.len() == 3 && shape[0] == self.pages && shape[1] == self.page_size {
                (1, shape[2])
            } else if shape.len() == 4
                && shape[0] == self.pages
                && shape[if layout == CacheLayout::Nhd { 1 } else { 2 }] == self.page_size
            {
                (
                    shape[if layout == CacheLayout::Nhd { 2 } else { 1 }],
                    shape[3],
                )
            } else {
                return Err(InferenceError("quantized cache geometry mismatch"));
            };
        let q = batch.meta.num_elements();
        if tokens.meta.shape()[..] != [q, heads, width]
            && !(heads == 1 && tokens.meta.shape()[..] == [q, width])
        {
            return Err(InferenceError("quantized append tokens shape mismatch"));
        }
        let rows = self.token_rows(batch, positions, heads, layout)?;
        let values = ruda_kernel::tensor::contiguous::into_contiguous(tokens.clone());
        let values = reshape(values, Shape::from([q * heads, width]));
        cache
            .append_rows(&values, &rows)
            .map_err(|e| InferenceError(e.0))
    }
    pub fn new(
        indptr: &RudaTensor<R>,
        indices: &RudaTensor<R>,
        last_page_len: &RudaTensor<R>,
        pages: usize,
        page_size: usize,
    ) -> Result<Self, InferenceError> {
        for t in [indptr, indices, last_page_len] {
            check(t, indptr)?;
            if t.dtype != DType::I32 || t.meta.num_dims() != 1 {
                return Err(InferenceError("CSR page metadata must be I32 vectors"));
            }
        }
        if indptr.meta.shape()[0] < 1
            || indptr.meta.shape()[0] != last_page_len.meta.shape()[0] + 1
            || page_size == 0
            || pages
                .checked_mul(page_size)
                .filter(|n| *n <= i32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError("CSR page geometry exceeds I32"));
        }
        Ok(Self {
            indptr: indptr.clone(),
            indices: indices.clone(),
            last_page_len: last_page_len.clone(),
            page_size,
            sequences: last_page_len.meta.shape()[0],
            pages,
        })
    }
    pub fn sequence_lengths(&self) -> Result<RudaTensor<R>, InferenceError> {
        let out = alloc(&self.indptr, [self.sequences], DType::I32);
        self.sequence_lengths_into(&out)?;
        Ok(out)
    }
    pub fn sequence_lengths_into(&self, out: &RudaTensor<R>) -> Result<(), InferenceError> {
        check(out, &self.indptr)?;
        if out.dtype != DType::I32 || out.meta.shape()[..] != [self.sequences] {
            return Err(InferenceError("page lengths output must be I32 [N]"));
        }
        if self.sequences > 0 {
            let dim = RudaDim::new(out.client.properties(), self.sequences);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, self.sequences, dim);
            lengths::launch::<R>(
                &out.client,
                count,
                dim,
                self.indptr.clone().into_array_arg(),
                self.last_page_len.clone().into_array_arg(),
                out.clone().into_array_arg(),
                self.sequences as u32,
                self.page_size as u32,
            );
        }
        Ok(())
    }
    /// Query batch IDs and positions may change in-place on replay. Allocating
    /// preparation fixes only capacities, not metadata values.
    pub fn prepare_attention(
        &self,
        batch: &RudaTensor<R>,
        positions: &RudaTensor<R>,
        max_history: usize,
    ) -> Result<PageAttention<R>, InferenceError> {
        query_metadata(batch, positions, &self.indptr)?;
        let queries = batch.meta.shape()[0];
        if max_history > i32::MAX as usize
            || queries
                .checked_mul(max_history)
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError("page selection capacity exceeds U32"));
        }
        let schedule = alloc(batch, [queries, 5], DType::U32);
        let selected = alloc(batch, [queries, max_history], DType::I32);
        let logical_positions = alloc(batch, [queries, max_history], DType::I32);
        let lengths = alloc(batch, [self.sequences], DType::I32);
        let plan = super::attention::AttentionPlan::device(
            &schedule,
            self.pages * self.page_size,
            max_history,
        )?;
        Ok(PageAttention {
            schedule,
            selected,
            logical_positions,
            lengths,
            plan,
            batch: batch.clone(),
            positions: positions.clone(),
            queries,
            max_history,
        })
    }
    /// Raw component transfer, including MLA's separate latent/positional cache.
    /// Flat-head cache form is [pages,P,H,D] or [pages,H,P,D].
    pub fn append(
        &self,
        tokens: &RudaTensor<R>,
        cache: &RudaTensor<R>,
        batch: &RudaTensor<R>,
        positions: &RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<RudaTensor<R>, InferenceError> {
        let out = crate::paged_attention::copy_contiguous(cache);
        unsafe {
            self.transfer_into(tokens, &out, batch, positions, layout, true)?;
        }
        Ok(out)
    }
    pub fn gather(
        &self,
        cache: &RudaTensor<R>,
        batch: &RudaTensor<R>,
        positions: &RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<RudaTensor<R>, InferenceError> {
        let (heads, width) = self.geometry(cache, layout)?;
        query_metadata(batch, positions, cache)?;
        let out = alloc(cache, [batch.meta.shape()[0], heads, width], cache.dtype);
        unsafe {
            self.transfer_into(&out, cache, batch, positions, layout, false)?;
        }
        Ok(out)
    }
    /// Cache mutation requires unique physical destinations and exclusive access.
    /// For reads, tokens is the exclusive output. All other buffers are read-only.
    pub unsafe fn transfer_into(
        &self,
        tokens: &RudaTensor<R>,
        cache: &RudaTensor<R>,
        batch: &RudaTensor<R>,
        positions: &RudaTensor<R>,
        layout: CacheLayout,
        append: bool,
    ) -> Result<(), InferenceError> {
        let (heads, width) = self.geometry(cache, layout)?;
        query_metadata(batch, positions, cache)?;
        check(tokens, cache)?;
        if tokens.dtype != cache.dtype
            || tokens.meta.shape()[..] != [batch.meta.shape()[0], heads, width]
        {
            return Err(InferenceError("cache transfer tokens must be [Q,H,D]"));
        }
        let n = tokens.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(cache.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&cache.client, n, dim);
            transfer::launch::<R>(
                &cache.client,
                count,
                dim,
                tokens.clone().into_array_arg(),
                cache.clone().into_array_arg(),
                batch.clone().into_array_arg(),
                positions.clone().into_array_arg(),
                self.indptr.clone().into_array_arg(),
                self.indices.clone().into_array_arg(),
                n as u32,
                heads as u32,
                width as u32,
                self.sequences as u32,
                self.pages as u32,
                self.page_size as u32,
                self.indices.meta.num_elements() as u32,
                layout == CacheLayout::Hnd,
                append,
                cache.dtype.into(),
            );
        }
        Ok(())
    }
    fn geometry(
        &self,
        cache: &RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<(usize, usize), InferenceError> {
        check(cache, &self.indptr)?;
        if !matches!(
            cache.dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::U8
        ) {
            return Err(InferenceError("cache needs float or raw FP8 bytes"));
        }
        if cache.meta.num_dims() != 4
            || cache.meta.shape()[0] != self.pages
            || cache.meta.shape()[if layout == CacheLayout::Nhd { 1 } else { 2 }] != self.page_size
        {
            return Err(InferenceError("CSR cache geometry mismatch"));
        }
        let heads = cache.meta.shape()[if layout == CacheLayout::Nhd { 2 } else { 1 }];
        let width = cache.meta.shape()[3];
        if heads == 0 || width == 0 {
            return Err(InferenceError("cache head dimensions must be nonempty"));
        }
        Ok((heads, width))
    }
    pub fn flatten_cache(
        &self,
        cache: &RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<RudaTensor<R>, InferenceError> {
        let (heads, width) = self.geometry(cache, layout)?;
        if layout == CacheLayout::Nhd {
            return Ok(reshape(
                cache.clone(),
                Shape::from([self.pages * self.page_size, heads, width]),
            ));
        }
        let out = alloc(
            cache,
            [self.pages * self.page_size, heads, width],
            cache.dtype,
        );
        let n = out.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(cache.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&cache.client, n, dim);
            flatten::launch::<R>(
                &cache.client,
                count,
                dim,
                cache.clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                heads as u32,
                width as u32,
                self.page_size as u32,
                cache.dtype.into(),
            );
        }
        Ok(out)
    }
}
pub struct PageAttention<R: Runtime> {
    pub schedule: RudaTensor<R>,
    pub selected: RudaTensor<R>,
    pub logical_positions: RudaTensor<R>,
    pub lengths: RudaTensor<R>,
    pub plan: super::attention::AttentionPlan<R>,
    pub batch: RudaTensor<R>,
    pub positions: RudaTensor<R>,
    queries: usize,
    max_history: usize,
}
pub struct PreparedPagedAttention<R: Runtime> {
    pages: CsrPages<R>,
    schedule: PageAttention<R>,
    query: super::PreparedPack<R>,
    key: super::PreparedPack<R>,
    value: super::PreparedPack<R>,
    flat_key: RudaTensor<R>,
    flat_value: RudaTensor<R>,
    pub output: super::cascade::AttentionState<R>,
    layout: CacheLayout,
}
impl<R: Runtime> PreparedPagedAttention<R> {
    pub fn prepare(
        pages: &CsrPages<R>,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        layout: CacheLayout,
        batch: &RudaTensor<R>,
        positions: &RudaTensor<R>,
        max_history: usize,
    ) -> Result<Self, InferenceError> {
        let query = super::PreparedPack::prepare(q)?;
        let key = super::PreparedPack::prepare(k)?;
        let value = super::PreparedPack::prepare(v)?;
        let (heads, width) = pages.geometry(&key.output, layout)?;
        let (vheads, vwidth) = pages.geometry(&value.output, layout)?;
        if q.meta.num_dims() != 3
            || q.meta.shape()[0] != batch.meta.num_elements()
            || q.meta.shape()[2] != width
            || q.meta.shape()[1] == 0
            || q.meta.shape()[1] % heads != 0
            || vheads != heads
            || q.dtype != k.dtype
            || q.dtype != v.dtype
        {
            return Err(InferenceError(
                "prepared paged attention heads/dtypes mismatch",
            ));
        }
        let schedule = pages.prepare_attention(batch, positions, max_history)?;
        let flat_key = if layout == CacheLayout::Nhd {
            reshape(
                key.output.clone(),
                Shape::from([pages.pages * pages.page_size, heads, width]),
            )
        } else {
            alloc(q, [pages.pages * pages.page_size, heads, width], q.dtype)
        };
        let flat_value = if layout == CacheLayout::Nhd {
            reshape(
                value.output.clone(),
                Shape::from([pages.pages * pages.page_size, heads, vwidth]),
            )
        } else {
            alloc(q, [pages.pages * pages.page_size, heads, vwidth], q.dtype)
        };
        let output = super::cascade::AttentionState {
            values: alloc(
                q,
                [batch.meta.num_elements(), q.meta.shape()[1], vwidth],
                q.dtype,
            ),
            lse: alloc(
                q,
                [batch.meta.num_elements(), q.meta.shape()[1]],
                DType::F32,
            ),
        };
        let pages = CsrPages {
            indptr: pages.indptr.clone(),
            indices: pages.indices.clone(),
            last_page_len: pages.last_page_len.clone(),
            page_size: pages.page_size,
            sequences: pages.sequences,
            pages: pages.pages,
        };
        Ok(Self {
            pages,
            schedule,
            query,
            key,
            value,
            flat_key,
            flat_value,
            output,
            layout,
        })
    }
    pub unsafe fn run(
        &self,
        options: super::attention::Options,
        extra: super::attention::RuntimeInputs<'_, R>,
    ) -> Result<&super::cascade::AttentionState<R>, InferenceError> {
        unsafe {
            self.query.run();
            self.key.run();
            self.value.run();
            if self.layout == CacheLayout::Hnd {
                self.pages
                    .flatten_into(&self.key.output, &self.flat_key, self.layout)?;
                self.pages
                    .flatten_into(&self.value.output, &self.flat_value, self.layout)?;
            }
        }
        self.schedule.refresh(&self.pages)?;
        let extra = super::attention::RuntimeInputs {
            byte_mask: extra.byte_mask,
            packed_mask: extra.packed_mask,
            indexed_keys: Some(&self.schedule.selected),
            indexed_positions: Some(&self.schedule.logical_positions),
            alibi_slopes: extra.alibi_slopes,
            sinks: extra.sinks,
            bidirectional_ranges: extra.bidirectional_ranges,
        };
        unsafe {
            self.schedule.plan.run_into(
                &self.query.output,
                &self.flat_key,
                &self.flat_value,
                options,
                extra,
                &self.output,
            )?;
        }
        Ok(&self.output)
    }
}
impl<R: Runtime> CsrPages<R> {
    pub unsafe fn flatten_into(
        &self,
        cache: &RudaTensor<R>,
        out: &RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<(), InferenceError> {
        let (heads, width) = self.geometry(cache, layout)?;
        check(out, cache)?;
        if out.dtype != cache.dtype
            || out.meta.shape()[..] != [self.pages * self.page_size, heads, width]
        {
            return Err(InferenceError("preallocated cache flatten shape mismatch"));
        }
        let n = out.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(cache.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&cache.client, n, dim);
            flatten_prepared::launch::<R>(
                &cache.client,
                count,
                dim,
                cache.clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                heads as u32,
                width as u32,
                self.page_size as u32,
                layout == CacheLayout::Hnd,
                cache.dtype.into(),
            );
        }
        Ok(())
    }
}
impl<R: Runtime> PageAttention<R> {
    /// No allocation or host/device synchronization. Schedule values are rebuilt
    /// from the current CSR tables and query positions on every replay.
    pub fn refresh(&self, pages: &CsrPages<R>) -> Result<(), InferenceError> {
        query_metadata(&self.batch, &self.positions, &pages.indptr)?;
        if self.lengths.meta.shape()[..] != [pages.sequences] {
            return Err(InferenceError("page sequence capacity changed"));
        }
        pages.sequence_lengths_into(&self.lengths)?;
        let n = self.queries * self.max_history;
        if self.queries > 0 {
            let work = n.max(self.queries);
            let dim = RudaDim::new(self.batch.client.properties(), work);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.batch.client, work, dim);
            attention_schedule::launch::<R>(
                &self.batch.client,
                count,
                dim,
                self.batch.clone().into_array_arg(),
                self.positions.clone().into_array_arg(),
                pages.indptr.clone().into_array_arg(),
                pages.indices.clone().into_array_arg(),
                self.lengths.clone().into_array_arg(),
                self.schedule.clone().into_array_arg(),
                self.selected.clone().into_array_arg(),
                self.logical_positions.clone().into_array_arg(),
                self.queries as u32,
                self.max_history as u32,
                pages.sequences as u32,
                pages.pages as u32,
                pages.page_size as u32,
                pages.indices.meta.num_elements() as u32,
            );
        }
        Ok(())
    }
    pub fn run(
        &self,
        pages: &CsrPages<R>,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        layout: CacheLayout,
        options: super::attention::Options,
        extra: super::attention::RuntimeInputs<'_, R>,
    ) -> Result<super::cascade::AttentionState<R>, InferenceError> {
        self.refresh(pages)?;
        let k = pages.flatten_cache(k, layout)?;
        let v = pages.flatten_cache(v, layout)?;
        let extra = super::attention::RuntimeInputs {
            byte_mask: extra.byte_mask,
            packed_mask: extra.packed_mask,
            indexed_keys: Some(&self.selected),
            indexed_positions: Some(&self.logical_positions),
            alibi_slopes: extra.alibi_slopes,
            sinks: extra.sinks,
            bidirectional_ranges: extra.bidirectional_ranges,
        };
        self.plan.run(q, &k, &v, options, extra)
    }
}
fn query_metadata<R: Runtime>(
    batch: &RudaTensor<R>,
    positions: &RudaTensor<R>,
    like: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    for t in [batch, positions] {
        check(t, like)?;
        if t.dtype != DType::I32 || t.meta.num_dims() != 1 {
            return Err(InferenceError("query batch/positions must be I32 vectors"));
        }
    }
    if batch.meta.shape() != positions.meta.shape() {
        return Err(InferenceError("query batch/positions shape mismatch"));
    }
    Ok(())
}
#[ruda(launch)]
pub fn lengths(cu: &Array<i32>, last: &Array<i32>, out: &mut Array<i32>, n: u32, page: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        out[i] = i32::max(cu[i + 1] - cu[i] - 1, 0) * page as i32 + last[i];
    }
}
#[ruda(launch)]
pub fn attention_schedule(
    batch: &Array<i32>,
    positions: &Array<i32>,
    cu: &Array<i32>,
    indices: &Array<i32>,
    lengths: &Array<i32>,
    schedule: &mut Array<u32>,
    selected: &mut Array<i32>,
    logical: &mut Array<i32>,
    queries: u32,
    history: u32,
    sequences: u32,
    pages: u32,
    page_size: u32,
    index_count: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < queries as usize {
        schedule[i * 5] = 0;
        schedule[i * 5 + 1] = pages * page_size;
        schedule[i * 5 + 2] = positions[i] as u32;
        schedule[i * 5 + 3] = 0;
        schedule[i * 5 + 4] = history;
    }
    if i < queries as usize * history as usize {
        let row = i / history as usize;
        let token = i % history as usize;
        let seq = batch[row];
        let mut slot: i32 = -1i32;
        if seq >= 0 && seq < sequences as i32 && token < lengths[seq as usize] as usize {
            let table = cu[seq as usize] + (token / page_size as usize) as i32;
            if table >= 0 && table < cu[seq as usize + 1] && table < index_count as i32 {
                let page = indices[table as usize];
                if page >= 0 && page < pages as i32 {
                    slot = page * page_size as i32 + (token % page_size as usize) as i32;
                }
            }
        }
        selected[i] = slot;
        logical[i] = token as i32;
    }
}
#[ruda(launch)]
pub fn transfer<F: Numeric>(
    tokens: &mut Array<F>,
    cache: &mut Array<F>,
    batch: &Array<i32>,
    positions: &Array<i32>,
    cu: &Array<i32>,
    indices: &Array<i32>,
    n: u32,
    heads: u32,
    width: u32,
    sequences: u32,
    pages: u32,
    page_size: u32,
    index_count: u32,
    #[comptime] hnd: bool,
    #[comptime] append: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / (heads as usize * width as usize);
        let head = i / width as usize % heads as usize;
        let d = i % width as usize;
        let seq = batch[row];
        let pos = positions[row];
        let mut valid = seq >= 0 && seq < sequences as i32 && pos >= 0;
        let mut address: usize = 0usize;
        if valid {
            let table = cu[seq as usize] + pos / page_size as i32;
            valid = table >= 0 && table < cu[seq as usize + 1] && table < index_count as i32;
            if valid {
                let page = indices[table as usize];
                valid = page >= 0 && page < pages as i32;
                let offset = pos as usize % page_size as usize;
                address = ((page as usize * page_size as usize + offset) * heads as usize + head)
                    * width as usize
                    + d;
                if comptime!(hnd) {
                    address = ((page as usize * heads as usize + head) * page_size as usize
                        + offset)
                        * width as usize
                        + d;
                }
            }
        }
        if valid {
            if comptime!(append) {
                cache[address] = tokens[i];
            } else {
                tokens[i] = cache[address];
            }
        } else {
            if comptime!(!append) {
                tokens[i] = F::cast_from(0.0f32);
            }
        }
    }
}
#[ruda(launch)]
pub fn flatten<F: Numeric>(
    x: &Array<F>,
    out: &mut Array<F>,
    n: u32,
    heads: u32,
    width: u32,
    page_size: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let d = i % width as usize;
        let head = i / width as usize % heads as usize;
        let token = i / (width as usize * heads as usize);
        let page = token / page_size as usize;
        let offset = token % page_size as usize;
        out[i] =
            x[((page * heads as usize + head) * page_size as usize + offset) * width as usize + d];
    }
}
#[ruda(launch)]
pub fn flatten_prepared<F: Numeric>(
    x: &Array<F>,
    out: &mut Array<F>,
    n: u32,
    heads: u32,
    width: u32,
    page_size: u32,
    #[comptime] hnd: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut address = i;
        if comptime!(hnd) {
            let d = i % width as usize;
            let head = i / width as usize % heads as usize;
            let token = i / (width as usize * heads as usize);
            let page = token / page_size as usize;
            let offset = token % page_size as usize;
            address =
                ((page * heads as usize + head) * page_size as usize + offset) * width as usize + d;
        }
        out[i] = x[address];
    }
}

pub struct BatchPositions<R: Runtime> {
    pub batch: RudaTensor<R>,
    pub positions: RudaTensor<R>,
}
/// Dequantize/gather a whole page table to a dense batch. Rows beyond each
/// sequence length are left untouched, matching caller-owned cache workspaces.
pub unsafe fn gather_dense_into<R: Runtime>(
    cache: &RudaTensor<R>,
    table: &RudaTensor<R>,
    sequence_lengths: &RudaTensor<R>,
    out: &RudaTensor<R>,
    layout: CacheLayout,
) -> Result<(), InferenceError> {
    for t in [cache, table, sequence_lengths, out] {
        check(t, cache)?;
    }
    float(cache)?;
    float(out)?;
    if cache.meta.num_dims() != 4
        || out.meta.num_dims() != 4
        || table.meta.num_dims() != 2
        || !matches!(table.dtype, DType::I32 | DType::I64)
        || sequence_lengths.dtype != DType::I32
        || sequence_lengths.meta.shape()[..] != [out.meta.shape()[0]]
        || table.meta.shape()[0] != out.meta.shape()[0]
    {
        return Err(InferenceError(
            "dense page gather needs cache rank four, table[B,L], lengths[B], out[B,S,H,D]",
        ));
    }
    let heads = cache.meta.shape()[if layout == CacheLayout::Nhd { 2 } else { 1 }];
    let page_size = cache.meta.shape()[if layout == CacheLayout::Nhd { 1 } else { 2 }];
    let width = cache.meta.shape()[3];
    if page_size == 0 || heads == 0 || width == 0 || out.meta.shape()[2..] != [heads, width] {
        return Err(InferenceError("dense page gather head dimensions mismatch"));
    }
    let n = out.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(cache.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&cache.client, n, dim);
        dense_pages::launch::<R>(
            &cache.client,
            count,
            dim,
            cache.clone().into_array_arg(),
            table.clone().into_array_arg(),
            sequence_lengths.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            out.meta.shape()[1] as u32,
            heads as u32,
            width as u32,
            page_size as u32,
            cache.meta.shape()[0] as u32,
            table.meta.shape()[1] as u32,
            layout == CacheLayout::Hnd,
            cache.dtype.into(),
            table.dtype.into(),
            out.dtype.into(),
        );
    }
    Ok(())
}
pub unsafe fn dequantize_paged_into<R: Runtime>(
    cache: &rublas::tensor_float_quant::BlockFloatTensor<R>,
    table: &RudaTensor<R>,
    lengths: &RudaTensor<R>,
    out: &RudaTensor<R>,
    layout: CacheLayout,
) -> Result<(), InferenceError> {
    let decoded = cache
        .dequantize(DType::F32)
        .map_err(|e| InferenceError(e.0))?;
    unsafe { gather_dense_into(&decoded, table, lengths, out, layout) }
}
pub fn batch_indices_positions<R: Runtime>(
    append_indptr: &RudaTensor<R>,
    sequence_lengths: &RudaTensor<R>,
    tokens: usize,
) -> Result<BatchPositions<R>, InferenceError> {
    check(append_indptr, sequence_lengths)?;
    if append_indptr.dtype != DType::I32
        || sequence_lengths.dtype != DType::I32
        || append_indptr.meta.num_dims() != 1
        || sequence_lengths.meta.num_dims() != 1
        || append_indptr.meta.shape()[0] != sequence_lengths.meta.shape()[0] + 1
        || tokens > i32::MAX as usize
    {
        return Err(InferenceError(
            "batch positions need I32 append indptr/lengths",
        ));
    }
    let batch = alloc(append_indptr, [tokens], DType::I32);
    let positions = alloc(append_indptr, [tokens], DType::I32);
    if tokens > 0 {
        let dim = RudaDim::new(batch.client.properties(), tokens);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&batch.client, tokens, dim);
        batch_positions::launch::<R>(
            &batch.client,
            count,
            dim,
            append_indptr.clone().into_array_arg(),
            sequence_lengths.clone().into_array_arg(),
            batch.clone().into_array_arg(),
            positions.clone().into_array_arg(),
            tokens as u32,
            sequence_lengths.meta.num_elements() as u32,
        );
    }
    Ok(BatchPositions { batch, positions })
}
#[ruda(launch)]
pub fn batch_positions(
    cu: &Array<i32>,
    lengths: &Array<i32>,
    batch: &mut Array<i32>,
    positions: &mut Array<i32>,
    tokens: u32,
    sequences: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize {
        let mut lo: usize = 0usize;
        let mut hi = sequences as usize;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if cu[mid] <= i as i32 {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let mut b: i32 = -1i32;
        let mut p: i32 = -1i32;
        if lo < sequences as usize && (i as i32) < cu[lo + 1] {
            b = lo as i32;
            p = lengths[lo] - (cu[lo + 1] - i as i32);
        }
        batch[i] = b;
        positions[i] = p;
    }
}
#[ruda(launch)]
pub fn physical_rows(
    batch: &Array<i32>,
    positions: &Array<i32>,
    cu: &Array<i32>,
    table: &Array<i32>,
    out: &mut Array<i32>,
    n: u32,
    heads: u32,
    sequences: u32,
    pages: u32,
    page_size: u32,
    index_count: u32,
    #[comptime] hnd: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / heads as usize;
        let head = i % heads as usize;
        let seq = batch[row];
        let pos = positions[row];
        let mut result: i32 = -1i32;
        if seq >= 0 && seq < sequences as i32 && pos >= 0 {
            let entry = cu[seq as usize] + pos / page_size as i32;
            if entry >= 0 && entry < cu[seq as usize + 1] && entry < index_count as i32 {
                let page = table[entry as usize];
                if page >= 0 && page < pages as i32 {
                    result = (page * page_size as i32 + pos % page_size as i32) * heads as i32
                        + head as i32;
                    if comptime!(hnd) {
                        result = (page * heads as i32 + head as i32) * page_size as i32
                            + pos % page_size as i32;
                    }
                }
            }
        }
        out[i] = result;
    }
}
#[ruda(launch)]
pub fn dense_pages<F: Float, I: Int, O: Float>(
    cache: &Array<F>,
    table: &Array<I>,
    lengths: &Array<i32>,
    out: &mut Array<O>,
    n: u32,
    max_length: u32,
    heads: u32,
    width: u32,
    page_size: u32,
    pages: u32,
    table_width: u32,
    #[comptime] hnd: bool,
    #[define(F)] _f: StorageType,
    #[define(I)] _i: StorageType,
    #[define(O)] _o: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let d = i % width as usize;
        let head = i / width as usize % heads as usize;
        let token = i / (width as usize * heads as usize) % max_length as usize;
        let seq = i / (width as usize * heads as usize * max_length as usize);
        if token < i32::max(0, lengths[seq]) as usize
            && token / (page_size as usize) < table_width as usize
        {
            let page =
                i64::cast_from(table[seq * table_width as usize + token / page_size as usize]);
            if page >= 0 && page < pages as i64 {
                let offset = token % page_size as usize;
                let mut address = ((page as usize * page_size as usize + offset) * heads as usize
                    + head)
                    * width as usize
                    + d;
                if comptime!(hnd) {
                    address = ((page as usize * heads as usize + head) * page_size as usize
                        + offset)
                        * width as usize
                        + d;
                }
                out[i] = O::cast_from(cache[address]);
            }
        }
    }
}

/// Retained native FP4/FP8 paged append. Physical slots are regenerated from
/// caller-owned device metadata every replay; no full-cache COW is performed.
pub struct PreparedQuantizedAppend<'a, R: Runtime> {
    pages: &'a CsrPages<R>,
    tokens: super::PreparedPack<R>,
    batch: super::metadata::PreparedIndices<R>,
    positions: super::metadata::PreparedIndices<R>,
    rows: RudaTensor<R>,
    append: rublas::tensor_float_quant::PreparedRowAppend<'a, R>,
    heads: usize,
    layout: CacheLayout,
}
impl<'a, R: Runtime> PreparedQuantizedAppend<'a, R> {
    pub fn prepare(
        pages: &'a CsrPages<R>,
        tokens: &RudaTensor<R>,
        cache: &'a rublas::tensor_float_quant::BlockFloatTensor<R>,
        batch: &RudaTensor<R>,
        positions: &RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<Self, InferenceError> {
        let s = &cache.logical_shape;
        let (heads, width) = if s.len() == 3 && s[0] == pages.pages && s[1] == pages.page_size {
            (1, s[2])
        } else if s.len() == 4
            && s[0] == pages.pages
            && s[if layout == CacheLayout::Nhd { 1 } else { 2 }] == pages.page_size
        {
            (s[if layout == CacheLayout::Nhd { 2 } else { 1 }], s[3])
        } else {
            return Err(InferenceError(
                "prepared quantized paged cache geometry mismatch",
            ));
        };
        let tokens = super::PreparedPack::prepare(tokens)?;
        let batch = super::metadata::PreparedIndices::prepare(batch, DType::I32)?;
        let positions = super::metadata::PreparedIndices::prepare(positions, DType::I32)?;
        query_metadata(&batch.output, &positions.output, &pages.indptr)?;
        let q = batch.output.meta.num_elements();
        if tokens.output.meta.shape()[..] != [q, heads, width]
            && !(heads == 1 && tokens.output.meta.shape()[..] == [q, width])
            || heads == 0
            || q.checked_mul(heads)
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
            || pages
                .pages
                .checked_mul(pages.page_size)
                .and_then(|n| n.checked_mul(heads))
                .filter(|n| *n <= i32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError(
                "prepared quantized append input/physical rows invalid",
            ));
        }
        let rows = alloc(&tokens.output, [q * heads], DType::I32);
        let flat = reshape(tokens.output.clone(), Shape::from([q * heads, width]));
        let append = rublas::tensor_float_quant::PreparedRowAppend::prepare(cache, &flat, &rows)
            .map_err(|e| InferenceError(e.0))?;
        Ok(Self {
            pages,
            tokens,
            batch,
            positions,
            rows,
            append,
            heads,
            layout,
        })
    }
    pub unsafe fn run(&self) -> Result<(), InferenceError> {
        unsafe {
            self.tokens.run();
            self.batch.run();
            self.positions.run();
        }
        let n = self.rows.meta.num_elements();
        if n > 0 {
            let p = self.pages;
            let dim = RudaDim::new(self.rows.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.rows.client, n, dim);
            physical_rows::launch::<R>(
                &self.rows.client,
                count,
                dim,
                self.batch.output.clone().into_array_arg(),
                self.positions.output.clone().into_array_arg(),
                p.indptr.clone().into_array_arg(),
                p.indices.clone().into_array_arg(),
                self.rows.clone().into_array_arg(),
                n as u32,
                self.heads as u32,
                p.sequences as u32,
                p.pages as u32,
                p.page_size as u32,
                p.indices.meta.num_elements() as u32,
                self.layout == CacheLayout::Hnd,
            );
            unsafe {
                self.append.run().map_err(|e| InferenceError(e.0))?;
            }
        }
        Ok(())
    }
}
