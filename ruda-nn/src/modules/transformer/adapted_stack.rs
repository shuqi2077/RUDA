use alloc::{collections::{BTreeMap,BTreeSet},format,string::String,vec::Vec};
use ruda_model::{config::Config,module::{Module,ModuleVisitor,Param},
    record::{PrecisionSettings,Record,Recorder,RecorderError},tensor::{Tensor,backend::Backend}};
use crate::attention::{DenseAttentionMask,DenseAttentionOptions};
use super::{DenseTransformerBlock,DenseTransformerStack,AdaptedTransformerBlock,TransformerAdapterConfig,
    AttentionAdapterTarget,FeedForwardAdapterTarget,TransformerAdapterRecord};

/// Exact zero-based layer and projection targets, with independently chosen adapter options.
#[derive(Config,Debug)]
pub struct LayerAdapterConfig {
    /// Actual index in the supplied native stack, not a model-name pattern.
    pub layer: usize,
    /// Rank/alpha/dtype/dropout convention for this explicitly selected layer.
    pub adapter: TransformerAdapterConfig,
    /// Exact attention projections to adapt; an empty vector leaves attention dense.
    pub attention: Vec<AttentionAdapterTarget>,
    /// Exact FFN projections to adapt; an empty vector leaves the FFN dense.
    pub feed_forward: Vec<FeedForwardAdapterTarget>,
}

impl LayerAdapterConfig {
    fn validate_for<B: Backend>(&self,block: &DenseTransformerBlock<B>) {
        assert!(!self.attention.is_empty() || !self.feed_forward.is_empty(),"selected layer needs an actual adapter target");
        for (i,target) in self.attention.iter().enumerate() {
            assert!(!self.attention[..i].contains(target),"duplicate attention adapter target");
        }
        for (i,target) in self.feed_forward.iter().enumerate() {
            assert!(!self.feed_forward[..i].contains(target),"duplicate feed-forward adapter target");
        }
        assert!(block.feed_forward.gate.is_some() || !self.feed_forward.contains(&FeedForwardAdapterTarget::Gate),
            "selected layer has no gate projection");
        let config = &self.adapter.lora;
        assert!(config.rank > 0 && config.alpha.is_finite(),"invalid layer adapter rank/alpha");
        assert!(config.dropout.is_finite() && (0.0..1.0).contains(&config.dropout),"invalid layer adapter dropout");
        assert!(self.adapter.adapter_dtype.is_none_or(|dtype|dtype.is_float()),"layer adapter dtype must be floating");
    }
}

/// Actual unmodified dense layer or the same layer with explicit native adapters.
#[derive(Module,Debug)]
pub enum AdaptedStackLayer<B: Backend> {
    /// Original module/parameter identities and trainable flags.
    Dense(DenseTransformerBlock<B>),
    /// Actual adapter parameters, original base, normalization and residual rules.
    Adapted(AdaptedTransformerBlock<B>),
}

impl<B: Backend> AdaptedStackLayer<B> {
    /// Actual self-attention stage, before inserting an encoder-memory stage.
    pub fn forward_attention_with_positions<F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        match self {
            Self::Dense(block)=>block.forward_attention_with_positions(input,masks,options,positions),
            Self::Adapted(block)=>block.forward_attention_with_positions(input,masks,options,positions),
        }
    }

    /// Actual final FFN stage, with its original normalization and residual order.
    pub fn forward_feed_forward(&self,input: Tensor<B,3>) -> Tensor<B,3> {
        match self {Self::Dense(block)=>block.forward_feed_forward(input),Self::Adapted(block)=>block.forward_feed_forward(input)}
    }

    /// Run the actual layer without selecting a model family.
    pub fn forward(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        match self {Self::Dense(block)=>block.forward(input,masks,options),Self::Adapted(block)=>block.forward(input,masks,options)}
    }

    /// Use caller-owned projected Q/K position transforms on either actual layer type.
    pub fn forward_with_positions<F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        match self {
            Self::Dense(block)=>block.forward_with_positions(input,masks,options,positions),
            Self::Adapted(block)=>block.forward_with_positions(input,masks,options,positions),
        }
    }
}

/// Native layerwise fine-tuning with explicitly selected layers/projections/ranks.
#[derive(Module,Debug)]
pub struct AdaptedTransformerStack<B: Backend> {
    /// Every actual layer in its original order, including unselected dense layers.
    pub layers: Vec<AdaptedStackLayer<B>>,
}

impl<B: Backend> AdaptedTransformerStack<B> {
    /// Convert selected actual layers only, checking all selections before allocation.
    /// Unselected layers retain their exact parameters/flags. Freeze the base explicitly
    /// before calling this when the intended training regime is adapter-only.
    pub fn from_dense(base: DenseTransformerStack<B>,targets: &[LayerAdapterConfig]) -> Self {
        let mut selected = BTreeMap::new();
        for target in targets {
            assert!(target.layer < base.blocks.len(),"adapter layer index is outside the actual stack");
            assert!(selected.insert(target.layer,target).is_none(),"duplicate adapter layer index");
            target.validate_for(&base.blocks[target.layer]);
        }
        let layers = base.blocks.into_iter().enumerate().map(|(index,block)| {
            if let Some(target) = selected.get(&index) {
                AdaptedStackLayer::Adapted(AdaptedTransformerBlock::from_dense(block,&target.adapter,&target.attention,&target.feed_forward))
            } else { AdaptedStackLayer::Dense(block) }
        }).collect();
        Self {layers}
    }

    /// Connect actual already-prepared layers without replacing their parameters.
    pub fn new(layers: Vec<AdaptedStackLayer<B>>) -> Self { Self {layers} }

    /// Shared explicit visibility/window rules for every original/adapted layer.
    pub fn forward(&self,mut input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        for layer in &self.layers { input = layer.forward(input,masks.clone(),options); }
        input
    }

    /// Caller-owned per-layer positions/visibility/options, retaining original order.
    pub fn forward_with<F>(&self,mut input: Tensor<B,3>,mut layer: F) -> Tensor<B,3>
    where F: FnMut(usize,&AdaptedStackLayer<B>,Tensor<B,3>)->Tensor<B,3> {
        for (index,block) in self.layers.iter().enumerate() { input = layer(index,block,input); }
        input
    }

    /// Capture actual adapters only; no frozen stack weights are included.
    pub fn adapter_record(&self,base_id: &str) -> Result<StackAdapterRecord<B>,RecorderError> {
        StackAdapterRecord::capture(self,base_id)
    }
}

/// A/B-only stack record bound to exact original layer indices and projection paths.
pub struct StackAdapterRecord<B: Backend> {
    version: u32,
    base_id: String,
    layers: usize,
    entries: Vec<(usize,TransformerAdapterRecord<B>)>,
}

impl<B: Backend> Record<B> for StackAdapterRecord<B> {
    type Item<S: PrecisionSettings> = (u32,String,usize,Vec<(usize,<TransformerAdapterRecord<B> as Record<B>>::Item<S>)>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.base_id,self.layers,self.entries.into_iter().map(|(index,record)|(index,record.into_item::<S>())).collect())
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {version:item.0,base_id:item.1,layers:item.2,entries:item.3.into_iter().map(|(index,record)|
            (index,TransformerAdapterRecord::<B>::from_item::<S>(record,device))).collect()}
    }
}

fn invalid(reason: &str) -> RecorderError {
    RecorderError::Unknown(format!("Invalid native stack adapter record: {reason}"))
}

struct FrozenCheck { trainable: bool }
impl<B: Backend> ModuleVisitor<B> for FrozenCheck {
    fn visit_float<const D: usize>(&mut self,param: &Param<Tensor<B,D>>) {
        self.trainable |= param.val().is_require_grad();
    }
}

fn check_dense_frozen<B: Backend>(block: &DenseTransformerBlock<B>) -> Result<(),RecorderError> {
    let mut visitor = FrozenCheck {trainable:false};
    block.visit(&mut visitor);
    if visitor.trainable { return Err(invalid("unselected layer has trainable parameters; use a full training record")); }
    Ok(())
}

impl<B: Backend> StackAdapterRecord<B> {
    /// Capture only actually adapted layers, validating that nothing trainable is omitted.
    pub fn capture(stack: &AdaptedTransformerStack<B>,base_id: &str) -> Result<Self,RecorderError> {
        if base_id.is_empty() { return Err(invalid("complete frozen-base identity is required")); }
        let mut entries = Vec::new();
        for (index,layer) in stack.layers.iter().enumerate() {
            match layer {
                AdaptedStackLayer::Dense(block)=>check_dense_frozen(block)?,
                AdaptedStackLayer::Adapted(block)=>entries.push((index,block.adapter_record(base_id)?)),
            }
        }
        if entries.is_empty() { return Err(invalid("stack has no actual adapters")); }
        Ok(Self {version:1,base_id:base_id.into(),layers:stack.layers.len(),entries})
    }

    /// Actual stored layer indices; no guessed count or tensor work occurs.
    pub fn layer_indices(&self) -> impl Iterator<Item=usize> + '_ { self.entries.iter().map(|(index,_)|*index) }

    /// Check all layer/target/base contracts before any adapter replacement.
    pub fn validate_for(&self,stack: &AdaptedTransformerStack<B>,base_id: &str) -> Result<(),RecorderError> {
        if self.version != 1 || self.base_id != base_id || base_id.is_empty() || self.layers != stack.layers.len() {
            return Err(invalid("version, complete frozen base or actual layer count differs"));
        }
        let mut expected = BTreeSet::new();
        for (index,layer) in stack.layers.iter().enumerate() {
            match layer {
                AdaptedStackLayer::Dense(block)=>check_dense_frozen(block)?,
                AdaptedStackLayer::Adapted(_)=>{ expected.insert(index); },
            }
        }
        if expected.len() != self.entries.len() || expected.is_empty() { return Err(invalid("adapted layer set differs")); }
        let mut seen = BTreeSet::new();
        for (index,record) in &self.entries {
            if !expected.contains(index) || !seen.insert(*index) { return Err(invalid("unknown, dense or duplicated layer index")); }
            let AdaptedStackLayer::Adapted(block) = &stack.layers[*index] else { return Err(invalid("layer is no longer adapted")); };
            record.validate_for(block,base_id)?;
        }
        Ok(())
    }

    /// Save actual A/B and exact indices through a native recorder.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {
        recorder.record(self,args)
    }

    /// Load A/B on the explicit backend/device without reconstructing a frozen base.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {
        recorder.load(args,device)
    }

    /// Restore the exact selected A/B state, retaining all unselected modules.
    pub fn restore_into(self,stack: AdaptedTransformerStack<B>,base_id: &str)
        -> Result<AdaptedTransformerStack<B>,RecorderError> {
        self.validate_for(&stack,base_id)?;
        let mut records: BTreeMap<_,_> = self.entries.into_iter().collect();
        let mut layers = Vec::with_capacity(stack.layers.len());
        for (index,layer) in stack.layers.into_iter().enumerate() {
            layers.push(match layer {
                AdaptedStackLayer::Dense(block)=>AdaptedStackLayer::Dense(block),
                AdaptedStackLayer::Adapted(block)=>{
                    let record = records.remove(&index).ok_or_else(||invalid("missing actual layer record"))?;
                    AdaptedStackLayer::Adapted(record.restore_into(block,base_id)?)
                },
            });
        }
        if !records.is_empty() { return Err(invalid("unconsumed stored layers")); }
        Ok(AdaptedTransformerStack {layers})
    }
}
