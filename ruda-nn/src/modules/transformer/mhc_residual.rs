use alloc::vec::Vec;
use core::fmt::Debug;
use ruda_model::{module::{Module, ModuleDisplay, Param},
    tensor::{Bool, DType, FloatDType, Int, MoeDispatchOps, Tensor, backend::Backend}};
use crate::{Mhc, MhcMappings, Linear, Nf4MoeLayer, FrozenExpertGeometry, FrozenSelectedExperts, FrozenNf4SwiGluExperts,
    attention::{CompressedAttention, CompressedAttentionProjection, CompressedAttentionOutput, CompressedAttentionSession}};
use super::{ProjectedFeedForward, NativeMoeFeedForward, NativeMoeTransformerError, Nf4MoeTransformerError,
    TransformerProjection, TransformerProjectionShape, MhcTransformerOutput, DenseFeedForward, AdaptedFeedForward};
use super::compressed::{normalized, visible};

/// Actual residual-free branch geometry, independent of dense/packed expert storage.
pub trait MhcResidualBranchShape<B: Backend>: Module<B> + ModuleDisplay {
    fn width(&self) -> usize;
    fn validate_branch(&self);
}

/// Real native branch execution and its original projection/dispatch/expert failure.
pub trait MhcResidualBranch<B: Backend>: MhcResidualBranchShape<B> {
    type Error: Debug;
    fn forward_branch(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, Self::Error>;
}

/// Original packed routed experts and explicitly present shared FFN.
#[derive(Module, Debug)]
pub struct PackedMhcFeedForward<B: Backend, Q: Module<B>, E: Module<B> = FrozenNf4SwiGluExperts<B>> {
    pub routed: Nf4MoeLayer<B, Q, E>,
    pub shared: Option<ProjectedFeedForward<B, Q>>,
}

impl<B: Backend, Q: TransformerProjectionShape<B>, E: FrozenExpertGeometry<B>> PackedMhcFeedForward<B, Q, E> {
    pub fn from_parts(routed: Nf4MoeLayer<B, Q, E>, shared: Option<ProjectedFeedForward<B, Q>>) -> Self {
        let branch = Self { routed, shared };
        branch.validate_branch();
        branch
    }
}

fn shared_width<B: Backend, Q: TransformerProjectionShape<B>>(shared: Option<&ProjectedFeedForward<B, Q>>, width: usize) {
    if let Some(shared) = shared {
        assert_eq!(shared.up.dimensions()[0], width, "mHC shared FFN input width differs");
        assert_eq!(shared.down.dimensions()[1], width, "mHC shared FFN output width differs");
    }
}

impl<B: Backend, Q: TransformerProjectionShape<B>, E: FrozenExpertGeometry<B>> MhcResidualBranchShape<B> for PackedMhcFeedForward<B, Q, E> {
    fn width(&self) -> usize { self.routed.width() }
    fn validate_branch(&self) { self.routed.validate(); shared_width(self.shared.as_ref(), self.width()); }
}

impl<B: MoeDispatchOps, Q: TransformerProjection<B>, E: FrozenSelectedExperts<B>> MhcResidualBranch<B> for PackedMhcFeedForward<B, Q, E> {
    type Error = Nf4MoeTransformerError<Q::Error, B::MoeError, E::Error>;
    fn forward_branch(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, Self::Error> {
        let routed = self.routed.forward(input.clone()).map_err(Nf4MoeTransformerError::Packed)?;
        if let Some(shared) = &self.shared {
            let shared = shared.forward(input).map_err(Nf4MoeTransformerError::Projection)?;
            assert_eq!(shared.dims(), routed.dims(), "mHC shared/routed actual row geometry differs");
            Ok(routed + shared)
        } else { Ok(routed) }
    }
}

/// Caller-selected dense, floating-expert or native packed-expert branch in each layer.
#[derive(Module, Debug)]
pub enum MhcFeedForward<B: Backend, Q: Module<B> = Linear<B>, E: Module<B> = FrozenNf4SwiGluExperts<B>> {
    Dense(ProjectedFeedForward<B, Q>),
    Floating(NativeMoeFeedForward<B, Q>),
    Packed(PackedMhcFeedForward<B, Q, E>),
}

impl<B: Backend, Q: TransformerProjectionShape<B>, E: FrozenExpertGeometry<B>> MhcResidualBranchShape<B> for MhcFeedForward<B, Q, E> {
    fn width(&self) -> usize {
        match self { Self::Dense(layer) => layer.up.dimensions()[0], Self::Floating(layer) => layer.routed.width(),
            Self::Packed(layer) => layer.width() }
    }
    fn validate_branch(&self) {
        match self {
            Self::Dense(layer) => {
                let [width, hidden] = layer.up.dimensions();
                assert_eq!(layer.down.dimensions(), [hidden, width], "mHC dense FFN geometry differs");
                if let Some(gate) = &layer.gate { assert_eq!(gate.dimensions(), [width, hidden], "mHC dense gate geometry differs"); }
            }
            Self::Floating(layer) => { layer.routed.validate(); shared_width(layer.shared.as_ref(), layer.routed.width()); }
            Self::Packed(layer) => layer.validate_branch(),
        }
    }
}

impl<B: MoeDispatchOps, Q: TransformerProjection<B>, E: FrozenSelectedExperts<B>> MhcResidualBranch<B> for MhcFeedForward<B, Q, E> {
    type Error = Nf4MoeTransformerError<Q::Error, B::MoeError, E::Error>;
    fn forward_branch(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, Self::Error> {
        match self {
            Self::Dense(layer) => layer.forward(input).map_err(Nf4MoeTransformerError::Projection),
            Self::Floating(layer) => layer.forward(input).map_err(|error| match error {
                NativeMoeTransformerError::Projection(error) => Nf4MoeTransformerError::Projection(error),
                NativeMoeTransformerError::Routed(error) => Nf4MoeTransformerError::Floating(error),
            }),
            Self::Packed(layer) => layer.forward_branch(input),
        }
    }
}

impl<B: Backend, Q: TransformerProjectionShape<B>> MhcResidualBranchShape<B> for ProjectedFeedForward<B, Q> {
    fn width(&self) -> usize { self.up.dimensions()[0] }
    fn validate_branch(&self) {
        let [width, hidden] = self.up.dimensions();
        assert_eq!(self.down.dimensions(), [hidden, width], "mHC projected FFN geometry differs");
        if let Some(gate) = &self.gate { assert_eq!(gate.dimensions(), [width, hidden], "mHC projected gate geometry differs"); }
    }
}
impl<B: Backend, Q: TransformerProjection<B>> MhcResidualBranch<B> for ProjectedFeedForward<B, Q> {
    type Error = Q::Error;
    fn forward_branch(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, Self::Error> { self.forward(input) }
}

impl<B: Backend> MhcResidualBranchShape<B> for DenseFeedForward<B> {
    fn width(&self) -> usize { self.up.weight.val().dims()[0] }
    fn validate_branch(&self) {
        let [width, hidden] = self.up.weight.val().dims();
        assert_eq!(self.down.weight.val().dims(), [hidden, width], "mHC original dense FFN geometry differs");
        if let Some(gate) = &self.gate { assert_eq!(gate.weight.val().dims(), [width, hidden], "mHC original dense gate geometry differs"); }
    }
}
impl<B: Backend> MhcResidualBranch<B> for DenseFeedForward<B> {
    type Error = core::convert::Infallible;
    fn forward_branch(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, Self::Error> { Ok(self.forward(input)) }
}
impl<B: Backend> MhcResidualBranchShape<B> for AdaptedFeedForward<B> {
    fn width(&self) -> usize { TransformerProjectionShape::dimensions(&self.up)[0] }
    fn validate_branch(&self) {
        let [width, hidden] = TransformerProjectionShape::dimensions(&self.up);
        assert_eq!(TransformerProjectionShape::dimensions(&self.down), [hidden, width], "mHC actual adapted FFN geometry differs");
        if let Some(gate) = &self.gate { assert_eq!(TransformerProjectionShape::dimensions(gate), [width, hidden], "mHC actual adapted gate geometry differs"); }
    }
}
impl<B: Backend> MhcResidualBranch<B> for AdaptedFeedForward<B> {
    type Error = core::convert::Infallible;
    fn forward_branch(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, Self::Error> { Ok(self.forward(input)) }
}

impl<B: Backend, Q: TransformerProjectionShape<B>> MhcResidualBranchShape<B> for NativeMoeFeedForward<B, Q> {
    fn width(&self) -> usize { self.routed.width() }
    fn validate_branch(&self) { self.routed.validate(); shared_width(self.shared.as_ref(), self.width()); }
}
impl<B: ruda_model::tensor::MoeOps, Q: TransformerProjection<B>> MhcResidualBranch<B> for NativeMoeFeedForward<B, Q> {
    type Error = NativeMoeTransformerError<Q::Error, B::MoeError>;
    fn forward_branch(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, Self::Error> { self.forward(input) }
}

/// Two actual mHC connections around native CSA/HCA and a caller-owned residual-free branch.
#[derive(Module, Debug)]
pub struct MhcResidualBlock<B: Backend, P: Module<B>, F: Module<B>> {
    pub attention_connection: Mhc<B>,
    pub ffn_connection: Mhc<B>,
    pub attention: CompressedAttention<B, P>,
    pub attention_norm: Param<Tensor<B, 1>>,
    pub ffn_norm: Param<Tensor<B, 1>>,
    pub feed_forward: F,
    pub epsilon: f64,
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualBlock<B, P, F> {
    pub fn from_parts(attention_connection: Mhc<B>, ffn_connection: Mhc<B>, attention: CompressedAttention<B, P>,
        attention_norm: Param<Tensor<B, 1>>, ffn_norm: Param<Tensor<B, 1>>, feed_forward: F, epsilon: f64) -> Self {
        feed_forward.validate_branch();
        let width = attention.width;
        assert_eq!((attention_connection.width, ffn_connection.width, feed_forward.width()), (width, width, width), "mHC residual widths differ");
        assert_eq!(attention_connection.streams, ffn_connection.streams, "mHC residual stream counts differ");
        assert_eq!(attention_norm.val().dims(), [width], "mHC residual attention norm width differs");
        assert_eq!(ffn_norm.val().dims(), [width], "mHC residual FFN norm width differs");
        assert!(epsilon.is_finite() && epsilon > 0.0, "invalid mHC residual epsilon");
        let device = attention.parts.query_down.device();
        assert!(attention_norm.val().device() == device && ffn_norm.val().device() == device
            && attention_connection.mapping.val().device() == device && ffn_connection.mapping.val().device() == device,
            "mHC residual parameter devices differ");
        Self { attention_connection, ffn_connection, attention, attention_norm, ffn_norm, feed_forward, epsilon }
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualBlock<B, P, F> {
    pub(super) fn finish_with<R, G>(&self, state: Tensor<B, 4>, attention: Tensor<B, 3>, mappings: MhcMappings<B>,
        valid: Tensor<B, 2, Bool>, branch: G) -> Result<Tensor<B, 4>, R>
    where G: FnOnce(&F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let state = self.attention_connection.post(state, attention, mappings);
        self.ffn_connection.try_forward(state, |merged| {
            let update = branch(&self.feed_forward, normalized(merged, &self.ffn_norm, self.epsilon))?;
            let [batch, tokens, _] = update.dims();
            let storage = update.dtype();
            Ok(update * valid.cast::<FloatDType>(storage.into()).reshape([batch, tokens, 1]))
        })
    }

    /// Execute the original branch with caller-owned runtime context, never stored in a Module record.
    pub fn try_forward_with<R, G>(&self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>, branch: G)
        -> Result<Tensor<B, 4>, R>
    where G: FnOnce(&F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.forward_options_with(state, valid, false, false, branch).map(|result| result.state)
    }

    pub fn try_forward_with_aux<R, G>(&self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>, indexer_warmup: bool, branch: G)
        -> Result<MhcTransformerOutput<B>, R>
    where G: FnOnce(&F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.forward_options_with(state, valid, true, indexer_warmup, branch)
    }

    fn forward_options_with<R, G>(&self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>, auxiliary: bool, warmup: bool, branch: G)
        -> Result<MhcTransformerOutput<B>, R>
    where G: FnOnce(&F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let (query, mappings) = self.attention_connection.pre(state.clone());
        let query = normalized(query, &self.attention_norm, self.epsilon);
        let valid = visible(&query, valid);
        let result = if auxiliary { self.attention.forward_with_aux(query, Some(valid.clone()), warmup) }
        else { CompressedAttentionOutput { output: self.attention.forward(query, Some(valid.clone())),
            indexer_loss: Tensor::zeros([1], (&state.device(), if state.dtype() == DType::F64 { DType::F64 } else { DType::F32 })) } };
        Ok(MhcTransformerOutput { state: self.finish_with(state, result.output, mappings, valid, branch)?, indexer_loss: result.indexer_loss })
    }

    pub fn inference_session(&self) -> MhcResidualSession<'_, B, P, F> {
        MhcResidualSession { block: self, attention: self.attention.inference_session() }
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranch<B>> MhcResidualBlock<B, P, F> {
    pub fn forward(&self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>) -> Result<Tensor<B, 4>, F::Error> {
        self.try_forward_with(state, valid, MhcResidualBranch::forward_branch)
    }
    pub fn forward_with_aux(&self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>, indexer_warmup: bool)
        -> Result<MhcTransformerOutput<B>, F::Error> {
        self.try_forward_with_aux(state, valid, indexer_warmup, MhcResidualBranch::forward_branch)
    }
}

#[derive(Debug)]
pub struct MhcResidualSession<'a, B: Backend, P: Module<B>, F: Module<B>> {
    block: &'a MhcResidualBlock<B, P, F>,
    attention: CompressedAttentionSession<'a, B, P>,
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualSession<'a, B, P, F> {
    pub fn position(&self) -> usize { self.attention.position() }
    pub fn clear(&mut self) { self.attention.clear(); }
    pub fn tensor_bytes(&self) -> usize { self.attention.tensor_bytes() }
    pub fn reorder(&mut self, parents: Tensor<B, 1, Int>) { self.attention.reorder(parents); }
    pub fn fork(&self) -> Self { Self { block: self.block, attention: self.attention.fork() } }
    pub fn restore(&mut self, snapshot: Self) {
        assert!(core::ptr::eq(self.block, snapshot.block), "mHC branch snapshot belongs to another actual block");
        self.attention.restore(snapshot.attention);
    }

    /// Publish new cache state only after the actual fallible branch succeeds.
    pub fn try_forward_with<R, G>(&mut self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>, branch: G)
        -> Result<Tensor<B, 4>, R>
    where G: FnOnce(&F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let (query, mappings) = self.block.attention_connection.pre(state.clone());
        let query = normalized(query, &self.block.attention_norm, self.block.epsilon);
        let valid = visible(&query, valid);
        let mut pending = self.attention.fork();
        let attention = pending.forward(query, Some(valid.clone()));
        let result = self.block.finish_with(state, attention, mappings, valid, branch)?;
        self.attention = pending;
        Ok(result.detach())
    }
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranch<B>> MhcResidualSession<'a, B, P, F> {
    pub fn forward(&mut self, state: Tensor<B, 4>, valid: Option<Tensor<B, 2, Bool>>) -> Result<Tensor<B, 4>, F::Error> {
        self.try_forward_with(state, valid, MhcResidualBranch::forward_branch)
    }
}

/// Actual homogeneous branch type may itself be a dense/floating/packed enum per layer.
#[derive(Module, Debug)]
pub struct MhcResidualStack<B: Backend, P: Module<B>, F: Module<B>> {
    pub layers: Vec<MhcResidualBlock<B, P, F>>,
    pub final_norm: Param<Tensor<B, 1>>,
    pub epsilon: f64,
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualStack<B, P, F> {
    pub fn from_parts(layers: Vec<MhcResidualBlock<B, P, F>>, final_norm: Param<Tensor<B, 1>>, epsilon: f64) -> Self {
        assert!(!layers.is_empty() && epsilon.is_finite() && epsilon > 0.0, "mHC residual stack must be nonempty with a positive epsilon");
        let width = layers[0].attention.width;
        let streams = layers[0].attention_connection.streams;
        let device = layers[0].attention.parts.query_down.device();
        assert_eq!(final_norm.val().dims(), [width], "mHC residual stack final norm width differs");
        assert_eq!(final_norm.val().device(), device, "mHC residual stack final norm device differs");
        for layer in &layers {
            assert_eq!((layer.attention.width, layer.attention_connection.streams, layer.ffn_connection.streams),
                (width, streams, streams), "mHC residual stack layer geometry differs");
            assert_eq!(layer.attention.parts.query_down.device(), device, "mHC residual stack layer devices differ");
        }
        Self { layers, final_norm, epsilon }
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualStack<B, P, F> {
    /// Layer index and the original branch are supplied without inferring process groups.
    pub fn try_forward_with<R, G>(&self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>, mut branch: G)
        -> Result<Tensor<B, 3>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let valid = visible(&input, valid);
        let [batch, tokens, _] = input.dims();
        let storage = input.dtype();
        let input = input * valid.clone().cast::<FloatDType>(storage.into()).reshape([batch, tokens, 1]);
        let mut state = self.layers[0].attention_connection.expand(input);
        for (index, layer) in self.layers.iter().enumerate() {
            state = layer.try_forward_with(state, Some(valid.clone()), |feed, input| branch(index, feed, input))?;
        }
        Ok(normalized(self.layers.last().unwrap().ffn_connection.reduce(state), &self.final_norm, self.epsilon))
    }

    pub fn try_forward_with_aux<R, G>(&self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>, indexer_warmup: bool, mut branch: G)
        -> Result<CompressedAttentionOutput<B>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let valid = visible(&input, valid);
        let [batch, tokens, _] = input.dims();
        let storage = input.dtype();
        let input = input * valid.clone().cast::<FloatDType>(storage.into()).reshape([batch, tokens, 1]);
        let mut state = self.layers[0].attention_connection.expand(input);
        let mut losses = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let result = layer.try_forward_with_aux(state, Some(valid.clone()), indexer_warmup, |feed, input| branch(index, feed, input))?;
            state = result.state;
            losses.push(result.indexer_loss);
        }
        Ok(CompressedAttentionOutput { output: normalized(self.layers.last().unwrap().ffn_connection.reduce(state), &self.final_norm, self.epsilon),
            indexer_loss: Tensor::cat(losses, 0).sum() })
    }

    pub fn inference_session(&self) -> MhcResidualStackSession<'_, B, P, F> {
        MhcResidualStackSession { stack: self, layers: self.layers.iter().map(MhcResidualBlock::inference_session).collect() }
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranch<B>> MhcResidualStack<B, P, F> {
    pub fn forward(&self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>) -> Result<Tensor<B, 3>, F::Error> {
        self.try_forward_with(input, valid, |_, feed, input| feed.forward_branch(input))
    }
    pub fn forward_with_aux(&self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>, indexer_warmup: bool)
        -> Result<CompressedAttentionOutput<B>, F::Error> {
        self.try_forward_with_aux(input, valid, indexer_warmup, |_, feed, input| feed.forward_branch(input))
    }
}

#[derive(Debug)]
pub struct MhcResidualStackSession<'a, B: Backend, P: Module<B>, F: Module<B>> {
    stack: &'a MhcResidualStack<B, P, F>,
    layers: Vec<MhcResidualSession<'a, B, P, F>>,
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualStackSession<'a, B, P, F> {
    pub fn position(&self) -> usize {
        let position = self.layers[0].position();
        assert!(self.layers.iter().all(|layer| layer.position() == position), "mHC residual cache positions differ");
        position
    }
    pub fn clear(&mut self) { for layer in &mut self.layers { layer.clear(); } }
    pub fn reorder(&mut self, parents: Tensor<B, 1, Int>) { for layer in &mut self.layers { layer.reorder(parents.clone()); } }
    pub fn tensor_bytes(&self) -> usize { self.layers.iter().map(MhcResidualSession::tensor_bytes).sum() }
    pub fn fork(&self) -> Self { Self { stack: self.stack, layers: self.layers.iter().map(MhcResidualSession::fork).collect() } }
    pub fn restore(&mut self, snapshot: Self) {
        assert!(core::ptr::eq(self.stack, snapshot.stack), "mHC residual stack snapshot belongs to another module");
        self.layers = snapshot.layers;
    }

    pub fn try_forward_with<R, G>(&mut self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>, mut branch: G)
        -> Result<Tensor<B, 3>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let valid = visible(&input, valid);
        let [batch, tokens, _] = input.dims();
        let storage = input.dtype();
        let input = input * valid.clone().cast::<FloatDType>(storage.into()).reshape([batch, tokens, 1]);
        let mut state = self.stack.layers[0].attention_connection.expand(input);
        let mut pending: Vec<_> = self.layers.iter().map(MhcResidualSession::fork).collect();
        for (index, layer) in pending.iter_mut().enumerate() {
            state = layer.try_forward_with(state, Some(valid.clone()), |feed, input| branch(index, feed, input))?;
        }
        let output = normalized(self.stack.layers.last().unwrap().ffn_connection.reduce(state), &self.stack.final_norm, self.stack.epsilon);
        self.layers = pending;
        Ok(output.detach())
    }
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranch<B>> MhcResidualStackSession<'a, B, P, F> {
    pub fn forward(&mut self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>) -> Result<Tensor<B, 3>, F::Error> {
        self.try_forward_with(input, valid, |_, feed, input| feed.forward_branch(input))
    }
}
