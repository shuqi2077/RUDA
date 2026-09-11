use super::*;
use crate::{
    CausalModelLimits, ContinuousBatchScheduler, DeviceBatchModel, KvPageId, PagedKvCacheConfig,
    RequestId, ScheduledBatch,
};
use ruda_tensor::api::activation::sigmoid;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
struct KvPage<B: Backend> {
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
}

/// Physical device pages per full-attention layer, plus request-local hybrid state.
/// Attention gathers these pages on the device into a padded batch; this is not
/// a fused paged-attention kernel. No KV or recurrent tensor is read back to CPU.
#[derive(Clone)]
pub struct Qwen35BatchCache<B: Backend> {
    config: PagedKvCacheConfig,
    pages: Vec<BTreeMap<KvPageId, KvPage<B>>>,
    requests: BTreeMap<RequestId, Qwen35Cache<B>>,
    tables: BTreeMap<RequestId, Vec<KvPageId>>,
}

impl<B: Backend> Qwen35BatchCache<B> {
    pub fn resident_requests(&self) -> usize {
        self.requests.len()
    }
    /// Count layer-specific physical K/V page pairs, not scheduler page IDs.
    pub fn resident_layer_pages(&self) -> usize {
        self.pages.iter().map(BTreeMap::len).sum()
    }
    pub fn sequence_length(&self, id: RequestId) -> Option<usize> {
        self.requests.get(&id).map(|cache| cache.position)
    }
}

impl<R, F, I, BT> DeviceBatchModel<DeviceBackend<R, F, I, BT>>
    for Qwen35TextModel<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type Cache = Qwen35BatchCache<DeviceBackend<R, F, I, BT>>;
    fn batch_limits(&self) -> CausalModelLimits {
        CausalModelLimits {
            vocab_size: self.config.vocab_size,
            max_sequence_length: self.config.max_position_embeddings,
        }
    }
    fn new_batch_cache(&self, config: PagedKvCacheConfig) -> Self::Cache {
        Qwen35BatchCache {
            config,
            pages: (0..self.layers.len()).map(|_| BTreeMap::new()).collect(),
            requests: BTreeMap::new(),
            tables: BTreeMap::new(),
        }
    }
    fn fork_batch_cache(&self, cache: &Self::Cache) -> Self::Cache {
        cache.clone()
    }
    fn retain_batch_cache(&self, cache: &mut Self::Cache, scheduler: &ContinuousBatchScheduler) {
        cache
            .requests
            .retain(|id, _| scheduler.kv_cache().contains_sequence(*id));
        cache.tables.retain(|id, _| cache.requests.contains_key(id));
        let live: BTreeSet<_> = cache.tables.values().flatten().copied().collect();
        for pages in &mut cache.pages {
            pages.retain(|id, _| live.contains(id));
        }
    }
    fn forward_batch(
        &self,
        batch: &ScheduledBatch,
        cache: &mut Self::Cache,
        device: &R::Device,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let (b, s) = (batch.batch_size(), batch.next_n());
        cache
            .config
            .validate()
            .map_err(|e| GenerationError(e.to_string()))?;
        if b == 0 || s == 0 || cache.pages.len() != self.layers.len() {
            return Err(GenerationError(
                "invalid Qwen batch/cache dimensions".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        let mut owners = BTreeMap::new();
        for (id, table) in &cache.tables {
            for page in table {
                owners.insert(*page, *id);
            }
        }
        for row in &batch.sequences {
            let valid = ids.insert(row.request_id)
                && row.token_ids.len() == s
                && row.start_position.checked_add(s) == Some(row.context_length)
                && row.context_length <= cache.config.max_sequence_length
                && row.context_length <= self.config.max_position_embeddings
                && cache
                    .requests
                    .get(&row.request_id)
                    .map_or(0, |c| c.position)
                    == row.start_position
                && row.block_table.len() == row.context_length.div_ceil(cache.config.block_size)
                && row
                    .token_ids
                    .iter()
                    .all(|&t| t >= 0 && (t as usize) < self.config.vocab_size)
                && cache
                    .tables
                    .get(&row.request_id)
                    .is_none_or(|old| row.block_table.starts_with(old));
            if !valid {
                return Err(GenerationError(
                    "invalid Qwen scheduled row or cache position".into(),
                ));
            }
            let mut unique = BTreeSet::new();
            for &page in &row.block_table {
                if page.0 as usize >= cache.config.num_pages
                    || !unique.insert(page)
                    || owners
                        .insert(page, row.request_id)
                        .is_some_and(|id| id != row.request_id)
                {
                    return Err(GenerationError(
                        "Qwen scheduled pages overlap or exceed capacity".into(),
                    ));
                }
            }
        }
        for row in &batch.sequences {
            cache
                .requests
                .entry(row.request_id)
                .or_insert_with(|| self.new_cache());
            cache.tables.insert(row.request_id, row.block_table.clone());
        }
        let tokens = Tensor::<DeviceBackend<R, F, I, BT>, 2, Int>::from_data(
            TensorData::new(batch.token_matrix(), [b, s]),
            device,
        );
        let mut hidden = self.embedding.forward(tokens);
        let dtype = hidden.dtype();
        let mut cos_rows = Vec::with_capacity(b);
        let mut sin_rows = Vec::with_capacity(b);
        for row in &batch.sequences {
            let (cos, sin) =
                attention::text_rope(&self.config, row.start_position, s, dtype, device);
            cos_rows.push(cos);
            sin_rows.push(sin);
        }
        let cos = Tensor::cat(cos_rows, 0);
        let sin = Tensor::cat(sin_rows, 0);
        for (index, layer) in self.layers.iter().enumerate() {
            let normalized = layer.input_norm.forward(hidden.clone())?;
            let update = match &layer.mixer {
                Mixer::Full(attn) => attn.forward_pages(
                    normalized,
                    &self.config,
                    cos.clone(),
                    sin.clone(),
                    batch,
                    cache.config.block_size,
                    &mut cache.pages[index],
                )?,
                Mixer::Delta(delta) => {
                    let mut convolutions = Vec::new();
                    let mut states = Vec::new();
                    let mut initialized = None;
                    for row in &batch.sequences {
                        let LayerCache::Delta { convolution, state } =
                            &cache.requests[&row.request_id].layers[index]
                        else {
                            return Err(GenerationError(
                                "Qwen batch hybrid cache type mismatch".into(),
                            ));
                        };
                        if convolution.is_some() != state.is_some()
                            || initialized.is_some_and(|v| v != state.is_some())
                        {
                            return Err(GenerationError(
                                "mixed initialized and empty Qwen recurrent rows".into(),
                            ));
                        }
                        initialized = Some(state.is_some());
                        if let Some(c) = convolution {
                            convolutions.push(c.clone());
                        }
                        if let Some(s) = state {
                            states.push(s.clone());
                        }
                    }
                    let mut convolution =
                        (!convolutions.is_empty()).then(|| Tensor::cat(convolutions, 0));
                    let mut state = (!states.is_empty()).then(|| Tensor::cat(states, 0));
                    let out =
                        delta.forward(normalized, &self.config, &mut convolution, &mut state)?;
                    let convolution = convolution.expect("delta forward returns convolution state");
                    let state = state.expect("delta forward returns recurrent state");
                    let [_, channels, width] = convolution.dims();
                    let [_, heads, kd, vd] = state.dims();
                    for (i, row) in batch.sequences.iter().enumerate() {
                        cache.requests.get_mut(&row.request_id).unwrap().layers[index] =
                            LayerCache::Delta {
                                convolution: Some(convolution.clone().slice([
                                    i..i + 1,
                                    0..channels,
                                    0..width,
                                ])),
                                state: Some(state.clone().slice([
                                    i..i + 1,
                                    0..heads,
                                    0..kd,
                                    0..vd,
                                ])),
                            };
                    }
                    out
                }
            };
            hidden = hidden + update;
            hidden = hidden.clone() + layer.mlp.forward(layer.post_norm.forward(hidden)?);
        }
        for row in &batch.sequences {
            let state = cache.requests.get_mut(&row.request_id).unwrap();
            state.position = row.context_length;
            state.batch = Some(1);
        }
        let width = hidden.dims()[2];
        Ok(self.head.forward(
            self.norm
                .forward(hidden.slice([0..b, s - 1..s, 0..width]))?,
        ))
    }
}

impl<R, F, I, BT> attention::Attention<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn forward_pages(
        &self,
        x: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        c: &Qwen35TextConfig,
        cos: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        sin: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        batch: &ScheduledBatch,
        block: usize,
        pages: &mut BTreeMap<KvPageId, KvPage<DeviceBackend<R, F, I, BT>>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let [b, s, _] = x.dims();
        let (h, k, d) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let qgate = self.q.forward(x.clone()).reshape([b, s, h, 2 * d]);
        let gate = qgate
            .clone()
            .slice([0..b, 0..s, 0..h, d..2 * d])
            .reshape([b, s, h * d]);
        let q = self
            .q_norm
            .forward(qgate.slice([0..b, 0..s, 0..h, 0..d]))?
            .swap_dims(1, 2);
        let key = self
            .k_norm
            .forward(self.k.forward(x.clone()).reshape([b, s, k, d]))?
            .swap_dims(1, 2);
        let value = self.v.forward(x).reshape([b, s, k, d]).swap_dims(1, 2);
        let q = attention::rotate(q, cos.clone(), sin.clone());
        let key = attention::rotate(key, cos, sin);
        write_pages(pages, batch, block, key, value)?;
        let length = batch
            .sequences
            .iter()
            .map(|r| r.context_length)
            .max()
            .unwrap();
        let (keys, values) = read_pages(pages, batch, block, length)?;
        let floor = attention::mask_floor(q.dtype());
        let mut mask = Vec::with_capacity(b * s * length);
        for row in &batch.sequences {
            for i in 0..s {
                for j in 0..length {
                    mask.push(if j <= row.start_position + i {
                        0.0
                    } else {
                        floor
                    });
                }
            }
        }
        let mask = Tensor::<DeviceBackend<R, F, I, BT>, 4>::from_data(
            TensorData::new(mask, [b, 1, s, length]),
            (&q.device(), q.dtype()),
        );
        let scores = q.matmul(attention::repeat_heads(keys, h / k).swap_dims(2, 3))
            * (d as f64).sqrt().recip()
            + mask;
        let probabilities = Self::probabilities(scores)?;
        let out = Self::value_product(probabilities, attention::repeat_heads(values, h / k))?
            .swap_dims(1, 2)
            .reshape([b, s, h * d]);
        Ok(self.out.forward(out * sigmoid(gate)))
    }
}

fn write_pages<B: Backend>(
    pages: &mut BTreeMap<KvPageId, KvPage<B>>,
    batch: &ScheduledBatch,
    block: usize,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
) -> Result<(), GenerationError> {
    let [_, k, _, d] = key.dims();
    for (i, row) in batch.sequences.iter().enumerate() {
        let mut offset = 0;
        while offset < row.token_ids.len() {
            let position = row.start_position + offset;
            let slot = position % block;
            let count = (block - slot).min(row.token_ids.len() - offset);
            let id = row.block_table[position / block];
            if !pages.contains_key(&id) {
                if slot != 0 {
                    return Err(GenerationError("missing committed physical KV page".into()));
                }
                pages.insert(
                    id,
                    KvPage {
                        key: Tensor::zeros([1, k, block, d], (&key.device(), key.dtype())),
                        value: Tensor::zeros([1, k, block, d], (&value.device(), value.dtype())),
                    },
                );
            }
            let page = pages.get_mut(&id).unwrap();
            page.key = page.key.clone().slice_assign(
                [0..1, 0..k, slot..slot + count, 0..d],
                key.clone()
                    .slice([i..i + 1, 0..k, offset..offset + count, 0..d]),
            );
            page.value = page.value.clone().slice_assign(
                [0..1, 0..k, slot..slot + count, 0..d],
                value
                    .clone()
                    .slice([i..i + 1, 0..k, offset..offset + count, 0..d]),
            );
            offset += count;
        }
    }
    Ok(())
}

fn read_pages<B: Backend>(
    pages: &BTreeMap<KvPageId, KvPage<B>>,
    batch: &ScheduledBatch,
    block: usize,
    length: usize,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>), GenerationError> {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for row in &batch.sequences {
        let mut row_keys = Vec::new();
        let mut row_values = Vec::new();
        for (i, id) in row.block_table.iter().enumerate() {
            let page = pages
                .get(id)
                .ok_or_else(|| GenerationError("missing physical KV page".into()))?;
            let [_, k, _, d] = page.key.dims();
            let count = block.min(row.context_length - i * block);
            row_keys.push(page.key.clone().slice([0..1, 0..k, 0..count, 0..d]));
            row_values.push(page.value.clone().slice([0..1, 0..k, 0..count, 0..d]));
        }
        if length > row.context_length {
            let first = &row_keys[0];
            let [_, k, _, d] = first.dims();
            row_keys.push(Tensor::zeros(
                [1, k, length - row.context_length, d],
                (&first.device(), first.dtype()),
            ));
            let first = &row_values[0];
            row_values.push(Tensor::zeros(
                [1, k, length - row.context_length, d],
                (&first.device(), first.dtype()),
            ));
        }
        keys.push(Tensor::cat(row_keys, 2));
        values.push(Tensor::cat(row_values, 2));
    }
    Ok((Tensor::cat(keys, 0), Tensor::cat(values, 0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KvReservationId, ScheduledBatchKind, ScheduledSequence};
    use ruda_tensor_host::{Host, HostDevice};
    fn row(id: u64, start: usize, count: usize, table: &[u32]) -> ScheduledSequence {
        ScheduledSequence {
            request_id: RequestId(id),
            start_position: start,
            context_length: start + count,
            token_ids: vec![1; count],
            block_table: table.iter().map(|&p| KvPageId(p)).collect(),
            reservation_id: KvReservationId(id),
        }
    }
    fn tensor(v: Vec<f32>, b: usize, s: usize) -> Tensor<Host, 4> {
        Tensor::from_data(TensorData::new(v, [b, 1, s, 1]), &HostDevice)
    }
    #[test]
    fn physical_pages_cross_boundaries_fork_and_gather_in_row_order() {
        let mut pages = BTreeMap::new();
        let first = ScheduledBatch {
            id: 1,
            kind: ScheduledBatchKind::Prefill,
            sequences: vec![row(1, 0, 3, &[3, 0])],
        };
        write_pages(
            &mut pages,
            &first,
            2,
            tensor(vec![1., 2., 3.], 1, 3),
            tensor(vec![11., 12., 13.], 1, 3),
        )
        .unwrap();
        let saved = pages.clone();
        let second = ScheduledBatch {
            id: 2,
            kind: ScheduledBatchKind::Decode,
            sequences: vec![row(1, 3, 1, &[3, 0]), row(2, 0, 1, &[2])],
        };
        write_pages(
            &mut pages,
            &second,
            2,
            tensor(vec![4., 9.], 2, 1),
            tensor(vec![14., 19.], 2, 1),
        )
        .unwrap();
        let (k, v) = read_pages(&pages, &second, 2, 4).unwrap();
        assert_eq!(
            k.into_data().to_vec::<f32>().unwrap(),
            [1., 2., 3., 4., 9., 0., 0., 0.]
        );
        assert_eq!(
            v.into_data().to_vec::<f32>().unwrap(),
            [11., 12., 13., 14., 19., 0., 0., 0.]
        );
        assert_eq!(
            saved[&KvPageId(0)]
                .key
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            [3., 0.]
        );
        let (k, _) = read_pages(&saved, &first, 2, 3).unwrap();
        assert_eq!(k.into_data().to_vec::<f32>().unwrap(), [1., 2., 3.]);
    }
}
