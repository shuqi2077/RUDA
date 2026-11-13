#![allow(unsafe_code)]
//! Exact native top-k for dense/ragged rows, with explicit tie ordering and
//! on-device page/ragged index transforms. No sorting or logits on the host.
use ruda_core::{
    device::Device,
    tensor::{DType, Shape},
};
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieBreak {
    SmallestIndex,
    LargestIndex,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopKError(pub &'static str);
impl core::fmt::Display for TopKError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for TopKError {}
pub struct TopKOutput<R: Runtime> {
    pub values: RudaTensor<R>,
    pub indices: RudaTensor<R>,
    pub counts: RudaTensor<R>,
}
pub struct TopKPlan<R: Runtime> {
    metadata: RudaTensor<R>,
    rows: usize,
    k: usize,
    input_elements: usize,
}
fn alloc<R: Runtime>(x: &RudaTensor<R>, s: impl Into<Shape>, dtype: DType) -> RudaTensor<R> {
    empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), s.into(), dtype)
}
fn check<R: Runtime>(x: &RudaTensor<R>, like: &RudaTensor<R>) -> Result<(), TopKError> {
    if x.qparams.is_some()
        || !x.is_contiguous()
        || x.meta.num_elements() > u32::MAX as usize
        || x.device.to_id() != like.device.to_id()
        || !x.client.same_execution_queue(&like.client)
    {
        return Err(TopKError(
            "expected same-queue contiguous U32-indexed tensors",
        ));
    }
    Ok(())
}
impl<R: Runtime> TopKPlan<R> {
    /// [rows,2] U32 pairs (flat start, length), refreshed by a producer kernel.
    /// Each row must stay inside the input allocation and have length <= I32_MAX.
    pub fn from_device(
        input: &RudaTensor<R>,
        metadata: &RudaTensor<R>,
        k: usize,
    ) -> Result<Self, TopKError> {
        check(input, input)?;
        check(metadata, input)?;
        if !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
            || metadata.dtype != DType::U32
            || metadata.meta.num_dims() != 2
            || metadata.meta.shape()[1] != 2
            || k > i32::MAX as usize
            || metadata.meta.shape()[0]
                .checked_mul(k)
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
        {
            return Err(TopKError("invalid device top-k schedule"));
        }
        Ok(Self {
            metadata: metadata.clone(),
            rows: metadata.meta.shape()[0],
            k,
            input_elements: input.meta.num_elements(),
        })
    }
    pub fn dense(input: &RudaTensor<R>, k: usize) -> Result<Self, TopKError> {
        if input.meta.num_dims() != 2 {
            return Err(TopKError("dense top-k requires [rows,width]"));
        }
        let rows = input.meta.shape()[0];
        let width = input.meta.shape()[1];
        let mut offsets = Vec::with_capacity(rows + 1);
        for i in 0..=rows {
            offsets.push(
                i.checked_mul(width)
                    .filter(|&n| n <= u32::MAX as usize)
                    .ok_or(TopKError("dense top-k indexing overflow"))? as u32,
            );
        }
        Self::ragged(input, &offsets, k)
    }
    pub fn ragged(input: &RudaTensor<R>, indptr: &[u32], k: usize) -> Result<Self, TopKError> {
        check(input, input)?;
        if !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
            || indptr.is_empty()
            || indptr[0] != 0
            || indptr.last().copied() != Some(input.meta.num_elements() as u32)
            || indptr
                .windows(2)
                .any(|x| x[0] > x[1] || x[1] - x[0] > i32::MAX as u32)
            || k > i32::MAX as usize
        {
            return Err(TopKError("invalid ragged top-k input"));
        }
        let rows = indptr.len() - 1;
        if rows
            .checked_mul(k)
            .filter(|&n| n <= u32::MAX as usize)
            .is_none()
            || rows > input.client.properties().hardware.max_ruda_count.0 as usize
        {
            return Err(TopKError("top-k output/grid exceeds U32"));
        }
        let mut words = Vec::with_capacity(rows * 2);
        for p in indptr.windows(2) {
            words.extend([p[0], p[1] - p[0]]);
        }
        let metadata = RudaTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::from([rows, 2]),
            input.client.create_from_slice(bytemuck::cast_slice(&words)),
            DType::U32,
        );
        Ok(Self {
            metadata,
            rows,
            k,
            input_elements: input.meta.num_elements(),
        })
    }
    pub fn allocate_output(&self, input: &RudaTensor<R>) -> Result<TopKOutput<R>, TopKError> {
        check(input, &self.metadata)?;
        if input.meta.num_elements() != self.input_elements
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return Err(TopKError("top-k workspace input mismatch"));
        }
        Ok(TopKOutput {
            values: alloc(input, [self.rows, self.k], input.dtype),
            indices: alloc(input, [self.rows, self.k], DType::I32),
            counts: alloc(input, [self.rows], DType::U32),
        })
    }
    pub fn run(
        &self,
        input: &RudaTensor<R>,
        lengths: Option<&RudaTensor<R>>,
        per_row_k: Option<&RudaTensor<R>>,
        tie: TieBreak,
    ) -> Result<TopKOutput<R>, TopKError> {
        self.execute(input, lengths, per_row_k, tie, None)
    }
    /// Retained outputs are exclusive; input and outputs must not alias.
    pub unsafe fn run_into(
        &self,
        input: &RudaTensor<R>,
        lengths: Option<&RudaTensor<R>>,
        per_row_k: Option<&RudaTensor<R>>,
        tie: TieBreak,
        output: TopKOutput<R>,
    ) -> Result<TopKOutput<R>, TopKError> {
        self.execute(input, lengths, per_row_k, tie, Some(output))
    }
    fn execute(
        &self,
        input: &RudaTensor<R>,
        lengths: Option<&RudaTensor<R>>,
        per_row_k: Option<&RudaTensor<R>>,
        tie: TieBreak,
        destination: Option<TopKOutput<R>>,
    ) -> Result<TopKOutput<R>, TopKError> {
        let input = if destination.is_some() {
            input.clone()
        } else {
            into_contiguous(input.clone())
        };
        check(&input, &self.metadata)?;
        if !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
            || input.meta.num_elements() != self.input_elements
        {
            return Err(TopKError("input no longer matches top-k plan"));
        }
        for t in lengths.iter().chain(per_row_k.iter()) {
            check(t, &input)?;
            if !matches!(t.dtype, DType::U32 | DType::I32) || t.meta.shape()[..] != [self.rows] {
                return Err(TopKError("top-k lengths/k must be U32 [rows]"));
            }
        }
        let output = destination.unwrap_or_else(|| TopKOutput {
            values: alloc(&input, [self.rows, self.k], input.dtype),
            indices: alloc(&input, [self.rows, self.k], DType::I32),
            counts: alloc(&input, [self.rows], DType::U32),
        });
        for (t, dtype, shape) in [
            (&output.values, input.dtype, vec![self.rows, self.k]),
            (&output.indices, DType::I32, vec![self.rows, self.k]),
            (&output.counts, DType::U32, vec![self.rows]),
        ] {
            check(t, &input)?;
            if t.dtype != dtype || t.meta.shape()[..] != shape {
                return Err(TopKError("preallocated top-k outputs mismatch"));
            }
        }
        let values = output.values;
        let indices = output.indices;
        let counts = output.counts;
        let lanes = input.client.properties().hardware.plane_size_max;
        if !matches!(lanes, 32 | 64) {
            return Err(TopKError("top-k needs a full plane"));
        }
        if self.rows != 0 {
            select::launch::<R>(
                &input.client,
                RudaCount::Static(self.rows as u32, 1, 1),
                RudaDim::new_1d(lanes),
                input.clone().into_array_arg(),
                self.metadata.clone().into_array_arg(),
                lengths.unwrap_or(&self.metadata).clone().into_array_arg(),
                per_row_k.unwrap_or(&self.metadata).clone().into_array_arg(),
                values.clone().into_array_arg(),
                indices.clone().into_array_arg(),
                counts.clone().into_array_arg(),
                self.k as u32,
                lengths.is_some(),
                per_row_k.is_some(),
                tie == TieBreak::LargestIndex,
                input.dtype.into(),
            );
        }
        Ok(TopKOutput {
            values,
            indices,
            counts,
        })
    }
}
pub struct VarlenOptions {
    pub next_n: usize,
    pub compression: usize,
    pub absolute_indices: bool,
    pub max_window: Option<usize>,
}
impl Default for VarlenOptions {
    fn default() -> Self {
        Self {
            next_n: 1,
            compression: 1,
            absolute_indices: false,
            max_window: None,
        }
    }
}
/// Device lengths are measured in uncompressed tokens. Window starts switch to
/// packed-prefill semantics. A preexisting candidate hint is unnecessary for
/// correctness: this native path always ranks the entire eligible window.
pub fn top_k_varlen<R: Runtime>(
    input: &RudaTensor<R>,
    sequence_lengths: &RudaTensor<R>,
    row_starts: Option<&RudaTensor<R>>,
    k: usize,
    tie: TieBreak,
    options: VarlenOptions,
) -> Result<TopKOutput<R>, TopKError> {
    check(input, input)?;
    check(sequence_lengths, input)?;
    if input.meta.num_dims() != 2
        || sequence_lengths.dtype != DType::I32
        || sequence_lengths.meta.num_dims() != 1
        || options.next_n == 0
        || options.compression == 0
    {
        return Err(TopKError(
            "varlen top-k expects [rows,width], I32 lengths, positive next_n/compression",
        ));
    }
    let rows = input.meta.shape()[0];
    let width = input.meta.shape()[1];
    if width > i32::MAX as usize
        || rows != sequence_lengths.meta.num_elements() * options.next_n
        || options.next_n > i32::MAX as usize
        || options.compression > i32::MAX as usize
        || row_starts.is_none() && (options.absolute_indices || options.max_window.is_some())
    {
        return Err(TopKError(
            "varlen top-k row grouping/window options mismatch",
        ));
    }
    if let Some(starts) = row_starts {
        check(starts, input)?;
        if starts.dtype != DType::I32
            || starts.meta.shape()[..] != [rows]
            || options.next_n != 1
            || options.compression != 1
        {
            return Err(TopKError(
                "window starts require per-row lengths without compression/speculation",
            ));
        }
    }
    let metadata = alloc(input, [rows, 2], DType::U32);
    if rows > 0 {
        let dim = RudaDim::new(input.client.properties(), rows);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, rows, dim);
        varlen_schedule::launch::<R>(
            &input.client,
            count,
            dim,
            sequence_lengths.clone().into_array_arg(),
            row_starts
                .unwrap_or(sequence_lengths)
                .clone()
                .into_array_arg(),
            metadata.clone().into_array_arg(),
            rows as u32,
            width as u32,
            options.next_n as u32,
            options.compression as u32,
            options.max_window.unwrap_or(width).min(width) as u32,
            row_starts.is_some(),
        );
    }
    let mut output = TopKPlan::from_device(input, &metadata, k)?.run(input, None, None, tie)?;
    if options.absolute_indices {
        let out = alloc(input, [rows, k], DType::I32);
        let n = rows * k;
        if n > 0 {
            let dim = RudaDim::new(input.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
            absolute_indices::launch::<R>(
                &input.client,
                count,
                dim,
                output.indices.clone().into_array_arg(),
                row_starts.unwrap().clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                k as u32,
            );
        }
        output.indices = out;
    }
    Ok(output)
}
#[ruda(launch)]
pub fn varlen_schedule(
    lengths: &Array<i32>,
    starts: &Array<i32>,
    out: &mut Array<u32>,
    rows: u32,
    width: u32,
    next_n: u32,
    compression: u32,
    max_window: u32,
    #[comptime] window: bool,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows as usize {
        let request = row / next_n as usize;
        let mut start: i32 = 0i32;
        let mut length = i32::max(
            0,
            lengths[request] - next_n as i32 + (row % next_n as usize) as i32 + 1,
        ) / compression as i32;
        if comptime!(window) {
            start = i32::min(i32::max(starts[row], 0), width as i32);
            length = i32::max(0, lengths[row]);
        }
        length = i32::min(length, i32::min(width as i32 - start, max_window as i32));
        out[row * 2] = row as u32 * width + start as u32;
        out[row * 2 + 1] = length as u32;
    }
}
#[ruda(launch)]
pub fn absolute_indices(
    ids: &Array<i32>,
    starts: &Array<i32>,
    out: &mut Array<i32>,
    n: u32,
    k: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let id = ids[i];
        let mut value: i32 = -1i32;
        if id >= 0 {
            value = id + starts[i / k as usize];
        }
        out[i] = value;
    }
}
pub fn top_k<R: Runtime>(
    input: &RudaTensor<R>,
    k: usize,
    tie: TieBreak,
) -> Result<TopKOutput<R>, TopKError> {
    let input = into_contiguous(input.clone());
    TopKPlan::dense(&input, k)?.run(&input, None, None, tie)
}
/// Convert selected local token IDs to a flattened ragged address. Offsets are
/// U32 [rows]; invalid selections stay -1.
pub fn ragged_transform<R: Runtime>(
    selected: &RudaTensor<R>,
    offsets: &RudaTensor<R>,
) -> Result<RudaTensor<R>, TopKError> {
    transform(selected, offsets, None, 1)
}
/// src_page_table is I32 [requests,table_width], row_to_batch is optional U32
/// [rows]. Output is physical token slots: page*page_size + intra-page offset.
pub fn page_table_transform<R: Runtime>(
    selected: &RudaTensor<R>,
    table: &RudaTensor<R>,
    row_to_batch: Option<&RudaTensor<R>>,
    page_size: usize,
) -> Result<RudaTensor<R>, TopKError> {
    transform(selected, table, row_to_batch, page_size)
}
/// Sliding score/page windows have independent units. Selected IDs are local
/// to the score window; page_table_row_starts is measured in table entries.
pub fn page_table_transform_windows<R: Runtime>(
    selected: &RudaTensor<R>,
    table: &RudaTensor<R>,
    row_to_batch: Option<&RudaTensor<R>>,
    page_table_row_starts: &RudaTensor<R>,
    page_size: usize,
) -> Result<RudaTensor<R>, TopKError> {
    for t in [selected, table, page_table_row_starts] {
        check(t, selected)?;
    }
    if selected.dtype != DType::I32
        || selected.meta.num_dims() != 2
        || table.dtype != DType::I32
        || table.meta.num_dims() != 2
        || page_table_row_starts.dtype != DType::I32
        || page_table_row_starts.meta.shape()[..] != [selected.meta.shape()[0]]
        || page_size == 0
        || page_size > i32::MAX as usize
    {
        return Err(TopKError("invalid windowed page-table transform"));
    }
    let rows = selected.meta.shape()[0];
    if let Some(map) = row_to_batch {
        check(map, selected)?;
        if !matches!(map.dtype, DType::I32 | DType::U32) || map.meta.shape()[..] != [rows] {
            return Err(TopKError("row map must be I32/U32 [rows]"));
        }
    } else if table.meta.shape()[0] != rows {
        return Err(TopKError("window table needs one row per selection row"));
    }
    let k = selected.meta.shape()[1];
    let out = alloc(selected, selected.meta.shape().clone(), DType::I32);
    let n = out.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(out.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
        window_transform::launch::<R>(
            &out.client,
            count,
            dim,
            selected.clone().into_array_arg(),
            table.clone().into_array_arg(),
            row_to_batch
                .unwrap_or(page_table_row_starts)
                .clone()
                .into_array_arg(),
            page_table_row_starts.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            k as u32,
            table.meta.shape()[0] as u32,
            table.meta.shape()[1] as u32,
            page_size as u32,
            row_to_batch.is_some(),
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn window_transform(
    ids: &Array<i32>,
    table: &Array<i32>,
    row_map: &Array<i32>,
    starts: &Array<i32>,
    out: &mut Array<i32>,
    n: u32,
    k: u32,
    requests: u32,
    table_width: u32,
    page_size: u32,
    #[comptime] remapped: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / k as usize;
        let id = ids[i];
        let mut batch = row as i32;
        if comptime!(remapped) {
            batch = row_map[row];
        }
        let mut value: i32 = -1i32;
        if id >= 0 && batch >= 0 && batch < requests as i32 {
            let page = starts[row] as i64 + id as i64 / page_size as i64;
            if page >= 0 && page < table_width as i64 {
                let physical = table[batch as usize * table_width as usize + page as usize];
                if physical >= 0 {
                    let address = physical as i64 * page_size as i64 + id as i64 % page_size as i64;
                    if address <= i32::MAX as i64 {
                        value = address as i32;
                    }
                }
            }
        }
        out[i] = value;
    }
}
fn transform<R: Runtime>(
    ids: &RudaTensor<R>,
    mapping: &RudaTensor<R>,
    row_map: Option<&RudaTensor<R>>,
    page_size: usize,
) -> Result<RudaTensor<R>, TopKError> {
    check(ids, ids)?;
    check(mapping, ids)?;
    if ids.dtype != DType::I32
        || ids.meta.num_dims() != 2
        || page_size == 0
        || page_size > i32::MAX as usize
    {
        return Err(TopKError("invalid top-k transform IDs/page size"));
    }
    let rows = ids.meta.shape()[0];
    let k = ids.meta.shape()[1];
    let paged = mapping.meta.num_dims() == 2;
    if (paged && mapping.dtype != DType::I32)
        || (!paged
            && (!matches!(mapping.dtype, DType::U32 | DType::I32)
                || mapping.meta.shape()[..] != [rows]))
    {
        return Err(TopKError("invalid top-k transform mapping"));
    }
    if let Some(map) = row_map {
        check(map, ids)?;
        if !matches!(map.dtype, DType::U32 | DType::I32) || map.meta.shape()[..] != [rows] {
            return Err(TopKError("row-to-batch must be U32 [rows]"));
        }
    }
    if paged && row_map.is_none() && mapping.meta.shape()[0] != rows {
        return Err(TopKError("page table needs one request per selected row"));
    }
    let out = alloc(ids, ids.meta.shape().clone(), DType::I32);
    let n = rows * k;
    if n != 0 {
        let dim = RudaDim::new(ids.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&ids.client, n, dim);
        transform_indices::launch::<R>(
            &ids.client,
            count,
            dim,
            ids.clone().into_array_arg(),
            mapping.clone().into_array_arg(),
            row_map.unwrap_or(mapping).clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            k as u32,
            if paged {
                mapping.meta.shape()[1] as u32
            } else {
                0
            },
            mapping.meta.shape()[0] as u32,
            page_size as u32,
            paged,
            row_map.is_some(),
        );
    }
    Ok(out)
}

#[ruda]
fn ordered_key(x: f32) -> u32 {
    let mut bits = u32::reinterpret(x);
    if x == 0.0 {
        bits = 0u32;
    }
    let mut key = bits ^ 0x80000000u32;
    if bits & 0x80000000u32 != 0 {
        key = !bits;
    }
    if x != x {
        key = u32::MAX;
    }
    key
}
#[ruda(launch)]
pub fn select<F: Float>(
    input: &Array<F>,
    metadata: &Array<u32>,
    lengths: &Array<u32>,
    per_row_k: &Array<u32>,
    out: &mut Array<F>,
    ids: &mut Array<i32>,
    counts: &mut Array<u32>,
    k: u32,
    #[comptime] limited: bool,
    #[comptime] variable_k: bool,
    #[comptime] largest: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let lanes = PLANE_DIM as usize;
    let start = metadata[row * 2] as usize;
    let mut width = metadata[row * 2 + 1] as usize;
    if comptime!(limited) {
        width = usize::min(width, lengths[row] as usize);
    }
    let mut n = usize::min(k as usize, width);
    if comptime!(variable_k) {
        n = usize::min(n, per_row_k[row] as usize);
    }
    let mut previous_key = u32::MAX;
    let mut previous_index: i32 = -1i32;
    if comptime!(largest) {
        previous_index = i32::MAX;
    }
    for position in 0..k as usize {
        let mut key: u32 = 0u32;
        let mut any: u32 = 0u32;
        let mut j = lane;
        while j < width {
            let candidate = ordered_key(f32::cast_from(input[start + j]));
            let mut eligible = candidate < previous_key
                || (candidate == previous_key && j as i32 > previous_index);
            if comptime!(largest) {
                eligible = candidate < previous_key
                    || (candidate == previous_key && (j as i32) < previous_index);
            }
            if eligible {
                key = u32::max(key, candidate);
                any = 1u32;
            }
            j += lanes;
        }
        key = plane_max(key);
        any = plane_max(any);
        let mut index = u32::MAX;
        if comptime!(largest) {
            index = 0;
        }
        j = lane;
        while j < width {
            let candidate = ordered_key(f32::cast_from(input[start + j]));
            let mut eligible = candidate < previous_key
                || (candidate == previous_key && j as i32 > previous_index);
            if comptime!(largest) {
                eligible = candidate < previous_key
                    || (candidate == previous_key && (j as i32) < previous_index);
            }
            if eligible && candidate == key {
                if comptime!(largest) {
                    index = u32::max(index, j as u32);
                } else {
                    index = u32::min(index, j as u32);
                }
            }
            j += lanes;
        }
        if comptime!(largest) {
            index = plane_max(index);
        } else {
            index = plane_min(index);
        }
        if lane == 0 {
            let mut id: i32 = -1i32;
            let mut value = f32::NEG_INFINITY;
            if position < n && any != 0 {
                id = index as i32;
                value = f32::cast_from(input[start + index as usize]);
            }
            ids[row * k as usize + position] = id;
            out[row * k as usize + position] = F::cast_from(value);
        }
        previous_key = key;
        previous_index = index as i32;
    }
    if lane == 0 {
        counts[row] = n as u32;
    }
}
#[ruda(launch)]
pub fn transform_indices(
    ids: &Array<i32>,
    mapping: &Array<u32>,
    row_map: &Array<u32>,
    out: &mut Array<i32>,
    n: u32,
    k: u32,
    table_width: u32,
    requests: u32,
    page_size: u32,
    #[comptime] paged: bool,
    #[comptime] remapped: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / k as usize;
        let id = ids[i];
        let mut result: i32 = -1i32;
        if id >= 0 {
            if comptime!(paged) {
                let mut batch = row;
                if comptime!(remapped) {
                    batch = row_map[row] as usize;
                }
                let page = id as usize / page_size as usize;
                if batch < requests as usize && page < table_width as usize {
                    let physical = mapping[batch * table_width as usize + page] as i32;
                    if physical >= 0 {
                        let slot =
                            physical as u64 * page_size as u64 + id as u64 % page_size as u64;
                        if slot <= i32::MAX as u64 {
                            result = slot as i32;
                        }
                    }
                }
            } else {
                let slot = mapping[row] as u64 + id as u64;
                if slot <= i32::MAX as u64 {
                    result = slot as i32;
                }
            }
        }
        out[i] = result;
    }
}
