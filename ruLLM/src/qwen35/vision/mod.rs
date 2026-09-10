mod config;
mod loading;
mod position;
mod processor;
mod image_files;
mod resize;
#[cfg(all(test, feature = "nvidia"))]
mod tests;

use crate::GenerationError;
pub use config::Qwen35VisionConfig;
pub use processor::{Qwen35ImageProcessor, Qwen35PreparedImages, Qwen35RgbImage};
pub use loading::{LoadedQwen35Vision, load_huggingface_qwen35_vision};
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use ruda_nn::{Embedding, LayerNorm, Linear};
use ruda_tensor::TensorPrimitive;
use ruda_tensor::api::{
    DType, Int, Tensor, TensorData,
    activation::{gelu, gelu_approximate},
    backend::Backend,
    module::conv3d,
    ops::ConvOptions,
};

struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    qkv: Linear<B>,
    proj: Linear<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

pub struct Qwen35VisionModel<B: Backend> {
    config: Qwen35VisionConfig,
    patch_weight: Tensor<B, 5>,
    patch_bias: Tensor<B, 1>,
    position: Embedding<B>,
    blocks: Vec<Block<B>>,
    merger_norm: LayerNorm<B>,
    merger_fc1: Linear<B>,
    merger_fc2: Linear<B>,
}

pub struct Qwen35VisionOutput<B: Backend> {
    pub hidden_states: Tensor<B, 2>,
    pub merged_states: Tensor<B, 2>,
}

fn normalize<B, R>(norm: &LayerNorm<B>, x: Tensor<B, 2>) -> Result<Tensor<B, 2>, GenerationError>
where
    R: Runtime,
    B: Backend<FloatTensorPrimitive = RudaTensor<R>>,
{
    let out = rudnn::normalization::layer_norm(
        x.into_primitive().tensor(),
        norm.gamma.val().into_primitive().tensor(),
        norm.beta
            .as_ref()
            .map(|b| b.val().into_primitive().tensor()),
        1e-6,
    )
    .map_err(|e| GenerationError(e.to_string()))?;
    Ok(Tensor::from_primitive(TensorPrimitive::Float(out)))
}

fn linear<B, R>(layer: &Linear<B>, x: Tensor<B, 2>) -> Result<Tensor<B, 2>, GenerationError>
where
    R: Runtime,
    B: Backend<FloatTensorPrimitive = RudaTensor<R>>,
{
    let dtype = x.dtype();
    let product = rublas::tensor_matmul::matmul(
        x.into_primitive().tensor(),
        layer.weight.val().into_primitive().tensor(),
        None,
        Default::default(),
        DType::F32,
    )
    .map_err(|e| GenerationError(e.to_string()))?;
    let mut out = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(product));
    if let Some(bias) = &layer.bias {
        out = out + bias.val().cast(DType::F32).unsqueeze::<2>();
    }
    Ok(out.cast(dtype))
}

fn rotate<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let dtype = x.dtype();
    let [b, s, h, d] = x.dims();
    let x = x.cast(DType::F32);
    let first = x.clone().slice([0..b, 0..s, 0..h, 0..d / 2]);
    let second = x.clone().slice([0..b, 0..s, 0..h, d / 2..d]);
    (x * cos + Tensor::cat(vec![-second, first], 3) * sin).cast(dtype)
}

fn rotary<B: Backend>(
    coordinates: Vec<f32>,
    tokens: usize,
    d: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let coords = Tensor::<B, 3>::from_data(
        TensorData::new(coordinates, [tokens, 2, 1]),
        (device, DType::F32),
    );
    let inv = (0..d / 4)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / (d / 2) as f32))
        .collect::<Vec<_>>();
    let inv = Tensor::<B, 3>::from_data(
        TensorData::new(inv, [1, 1, d / 4]),
        (device, DType::F32),
    );
    let freq = (coords * inv).reshape([tokens, d / 2]);
    let freq = Tensor::cat(vec![freq.clone(), freq], 1).reshape([1, tokens, 1, d]);
    (freq.clone().cos(), freq.sin())
}

impl<B, R> Block<B>
where
    R: Runtime,
    B: Backend<FloatTensorPrimitive = RudaTensor<R>>,
{
    fn forward(
        &self,
        x: Tensor<B, 2>,
        heads: usize,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        segments: &[usize],
    ) -> Result<Tensor<B, 2>, GenerationError> {
        let qkv = linear(&self.qkv, normalize(&self.norm1, x.clone())?)?;
        let context = Self::attention(qkv, heads, cos, sin, segments, |_, _, _| {})?;
        let x = x + linear(&self.proj, context)?;
        let mlp = linear(&self.fc1, normalize(&self.norm2, x.clone())?)?;
        let dtype = mlp.dtype();
        Ok(x + linear(
            &self.fc2,
            gelu_approximate(mlp.cast(DType::F32)).cast(dtype),
        )?)
    }

    fn attention(
        x: Tensor<B, 2>,
        heads: usize,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        segments: &[usize],
        mut observe: impl FnMut(usize, &str, &Tensor<B, 4>),
    ) -> Result<Tensor<B, 2>, GenerationError> {
        let [s, width] = x.dims();
        let width = width / 3;
        let d = width / heads;
        let qkv = x.reshape([s, 3, heads, d]);
        let q = rotate(
            qkv.clone()
                .slice([0..s, 0..1, 0..heads, 0..d])
                .reshape([1, s, heads, d]),
            cos.clone(),
            sin.clone(),
        )
        .swap_dims(1, 2);
        let k = rotate(
            qkv.clone()
                .slice([0..s, 1..2, 0..heads, 0..d])
                .reshape([1, s, heads, d]),
            cos,
            sin,
        )
        .swap_dims(1, 2);
        let v = qkv
            .slice([0..s, 2..3, 0..heads, 0..d])
            .reshape([1, s, heads, d])
            .swap_dims(1, 2);
        let dtype = q.dtype();
        let mut offset = 0;
        let mut contexts = Vec::with_capacity(segments.len());
        for (segment, &length) in segments.iter().enumerate() {
            let range = [0..1, 0..heads, offset..offset + length, 0..d];
            let q = q.clone().slice(range.clone());
            let k = k.clone().slice(range.clone());
            let v = v.clone().slice(range);
            observe(segment, "q", &q);
            observe(segment, "k", &k);
            observe(segment, "v", &v);
            let scores = q.matmul(k.swap_dims(2, 3)) * (d as f64).sqrt().recip();
            observe(segment, "scores", &scores);
            let probs = rudnn::normalization::softmax_last_axis(scores.cast(DType::F32).into_primitive().tensor())
                .map_err(|e| GenerationError(e.to_string()))?;
            let probs = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(probs)).cast(dtype);
            observe(segment, "probs", &probs);
            let context = probs.matmul(v).swap_dims(1, 2);
            observe(segment, "context", &context);
            contexts.push(context.reshape([length, width]));
            offset += length;
        }
        Ok(Tensor::cat(contexts, 0))
    }
}

impl<B: Backend> Qwen35VisionModel<B> {
    pub fn config(&self) -> &Qwen35VisionConfig {
        &self.config
    }
}

impl<B, R> Qwen35VisionModel<B>
where
    R: Runtime,
    B: Backend<FloatTensorPrimitive = RudaTensor<R>>,
{
    /// Encodes processor-packed patches `[sum(T*H*W), C*temporal_patch*patch*patch]`.
    /// Grids are patch dimensions; rows are ordered by frame, merge block, then in-block row/column.
    pub fn forward(
        &self,
        patches: Tensor<B, 2>,
        grids: &[[usize; 3]],
    ) -> Result<Qwen35VisionOutput<B>, GenerationError> {
        self.forward_observed(patches, grids, |_, _| {})
    }

    fn forward_observed(
        &self,
        patches: Tensor<B, 2>,
        grids: &[[usize; 3]],
        mut observe: impl FnMut(usize, &Tensor<B, 2>),
    ) -> Result<Qwen35VisionOutput<B>, GenerationError> {
        let c = &self.config;
        let tokens = c.grid_tokens(grids)?;
        let patch_width = c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size;
        if patches.dims() != [tokens, patch_width]
            || tokens
                .checked_mul(c.hidden_size)
                .is_none_or(|n| n > u32::MAX as usize)
            || tokens
                .checked_mul(patch_width)
                .is_none_or(|n| n > u32::MAX as usize)
            || grids.iter().any(|&[_, h, w]| {
                (h * w)
                    .checked_mul(h * w)
                    .and_then(|n| n.checked_mul(c.num_heads))
                    .is_none_or(|n| n > u32::MAX as usize)
            })
        {
            return Err(GenerationError(
                "Qwen3.5 vision patch shape or attention indexing is invalid".into(),
            ));
        }
        let device = self.patch_weight.device();
        let dtype = self.patch_weight.dtype();
        let p = position::positions(
            grids,
            c.spatial_merge_size,
            c.num_position_embeddings.isqrt(),
        );
        let indices =
            Tensor::<B, 2, Int>::from_data(TensorData::new(p.indices, [tokens, 4]), &device);
        let weights = Tensor::<B, 3>::from_data(
            TensorData::new(p.weights, [tokens, 4, 1]),
            (&device, DType::F32),
        );
        let pos = (self.position.forward(indices).cast(DType::F32) * weights)
            .sum_dim(1)
            .reshape([tokens, c.hidden_size])
            .cast(dtype);
        let hidden = conv3d(
            patches.cast(dtype).reshape([
                tokens,
                c.in_channels,
                c.temporal_patch_size,
                c.patch_size,
                c.patch_size,
            ]),
            self.patch_weight.clone(),
            None,
            ConvOptions::new(
                [c.temporal_patch_size, c.patch_size, c.patch_size],
                [0; 3],
                [1; 3],
                1,
            ),
        )
        .reshape([tokens, c.hidden_size])
            + self.patch_bias.clone().unsqueeze::<2>();
        observe(0, &hidden);
        let mut hidden = hidden + pos;
        observe(1, &hidden);
        let d = c.hidden_size / c.num_heads;
        let (cos, sin) = rotary(p.coordinates, tokens, d, &device);
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(hidden, c.num_heads, cos.clone(), sin.clone(), &p.segments)?;
            observe(index + 2, &hidden);
        }
        let width = c.hidden_size * c.spatial_merge_size * c.spatial_merge_size;
        let merged = linear(
            &self.merger_fc1,
            normalize(&self.merger_norm, hidden.clone())?.reshape([
                tokens / (c.spatial_merge_size * c.spatial_merge_size),
                width,
            ]),
        )?;
        let merged_dtype = merged.dtype();
        let merged = linear(
            &self.merger_fc2,
            gelu(merged.cast(DType::F32)).cast(merged_dtype),
        )?;
        Ok(Qwen35VisionOutput {
            hidden_states: hidden,
            merged_states: merged,
        })
    }
}
