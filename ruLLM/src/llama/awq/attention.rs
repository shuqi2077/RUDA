use super::*;

#[derive(Debug)]
pub(super) struct AwqAttention<R: DeviceRuntime>
where
    R::Device: DeviceOps,
{
    pub(super) q_proj: AwqProjection<R>,
    pub(super) k_proj: AwqProjection<R>,
    pub(super) v_proj: AwqProjection<R>,
    pub(super) o_proj: AwqProjection<R>,
    pub(super) rope: RotaryEncoding<AwqBackend<R>>,
    pub(super) rotary_layout: RotaryLayout,
    pub(super) num_query_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dimension: usize,
}

impl<R: DeviceRuntime> AwqAttention<R>
where
    R::Device: DeviceOps,
{
    fn project(
        &self,
        input: Tensor<AwqBackend<R>, 3>,
        position: usize,
    ) -> (
        Tensor<AwqBackend<R>, 4>,
        Tensor<AwqBackend<R>, 4>,
        Tensor<AwqBackend<R>, 4>,
    ) {
        let [batch, sequence, _] = input.dims();
        let query = self
            .q_proj
            .forward(input.clone())
            .reshape([batch, sequence, self.num_query_heads, self.head_dimension])
            .swap_dims(1, 2);
        let key = self
            .k_proj
            .forward(input.clone())
            .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
            .swap_dims(1, 2);
        let value = self
            .v_proj
            .forward(input)
            .reshape([batch, sequence, self.num_kv_heads, self.head_dimension])
            .swap_dims(1, 2);
        (
            self.apply_rope(query, position),
            self.apply_rope(key, position),
            value,
        )
    }

    pub(super) fn apply_rope(
        &self,
        input: Tensor<AwqBackend<R>, 4>,
        start: usize,
    ) -> Tensor<AwqBackend<R>, 4> {
        match self.rotary_layout {
            RotaryLayout::Interleaved => self.rope.apply(input, start),
            RotaryLayout::HalfSplit => apply_half_split_rope(&self.rope, input, start),
        }
    }

    pub(super) fn finish(&self, context: Tensor<AwqBackend<R>, 4>) -> Tensor<AwqBackend<R>, 3> {
        let [batch, _, sequence, _] = context.dims();
        let context = if sequence == 1 {
            // `[B, Hq, 1, D]` already has the same physical element order as
            // `[B, 1, Hq * D]`; swapping the two singleton-adjacent axes first
            // makes Ruda materialize an otherwise unnecessary copy.
            context.reshape([batch, sequence, self.num_query_heads * self.head_dimension])
        } else {
            context.swap_dims(1, 2).reshape([
                batch,
                sequence,
                self.num_query_heads * self.head_dimension,
            ])
        };
        self.o_proj.forward(context)
    }

    pub(super) fn forward(&self, input: Tensor<AwqBackend<R>, 3>) -> Tensor<AwqBackend<R>, 3> {
        let (query, key, value) = self.project(input, 0);
        let key = repeat_kv(key, self.num_query_heads / self.num_kv_heads);
        let value = repeat_kv(value, self.num_query_heads / self.num_kv_heads);
        self.finish(causal_attention(query, key, value, 0))
    }

    pub(super) fn forward_cached(
        &self,
        input: Tensor<AwqBackend<R>, 3>,
        cache: &mut LlamaLayerCache<AwqBackend<R>>,
        position: usize,
    ) -> Tensor<AwqBackend<R>, 3> {
        assert_eq!(
            cache.sequence_length(),
            position,
            "all Llama layer caches must advance in lockstep"
        );
        let (query, key, value) = self.project(input, position);
        let (key, value) = cache.append(key, value);
        let key = repeat_kv(key, self.num_query_heads / self.num_kv_heads);
        let value = repeat_kv(value, self.num_query_heads / self.num_kv_heads);
        self.finish(causal_attention(query, key, value, position))
    }
}
