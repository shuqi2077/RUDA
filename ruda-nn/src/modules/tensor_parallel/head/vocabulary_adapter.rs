use super::*;
use super::super::VocabParallelProjection;
use crate::{LinearConfig,LoRAAdapterSchema,LoRAAdapterRecord,loss::LossTerms,
    transformer::{DenseTransformerNorm,TransformerAdapterConfig}};
use ruda_model::module::Initializer;
use ruda_autodiff::tensor_parallel as region;

#[cfg(not(feature="std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

mod record;
pub use record::{VocabParallelLoRAAdapterRecord,VocabParallelHeadAdapterRecord};

/// Frozen native row-major vocabulary projection with explicit replicated A and local B adapters.
/// The base stays `[local_vocabulary,hidden]`, retaining the original embedding/head parameter tie.
#[derive(Module,Debug)]
pub struct VocabParallelLoRAProjection<B:Backend> {
    /// Actual frozen vocabulary rows/bias, never transposed into a new parameter.
    pub base:VocabParallelProjection<B>,
    /// Actual replicated `[hidden,rank]` native adapter, preserving its own ID/dtype/flags.
    pub adapter_a:Linear<B>,
    /// Actual local `[rank,local_vocabulary]` native adapter columns.
    pub adapter_b:Linear<B>,
    /// Original adapter-input dropout, separate from head-input dropout.
    pub dropout:Dropout,
    /// Caller-selected native alpha/rank or alpha/sqrt(rank) multiplier.
    pub scale:f64,
}

fn geometry<B:Backend>(layer:&VocabParallelLoRAProjection<B>) -> [usize;2] {
    let weight = layer.base.weight.val();let [classes,hidden] = weight.dims();
    let a = layer.adapter_a.weight.val();let b = layer.adapter_b.weight.val();let rank = a.dims()[1];
    assert!(classes > 0 && hidden > 0 && rank > 0,"vocabulary adapter dimensions must be positive");
    assert!(!weight.is_require_grad(),"freeze the complete shared embedding/head base before attaching vocabulary adapters");
    assert!(weight.dtype().is_float() || matches!(weight.dtype(),DType::QFloat(_)),"vocabulary adapter base must use native floating or packed storage");
    assert_eq!(a.dims(),[hidden,rank],"vocabulary adapter A hidden/rank geometry differs");
    assert_eq!(b.dims(),[rank,classes],"vocabulary adapter B rank/class geometry differs");
    assert!(layer.adapter_a.bias.is_none() && layer.adapter_b.bias.is_none(),"native vocabulary adapters must be bias-free");
    assert!(matches!(a.dtype(),DType::F16|DType::BF16|DType::F32|DType::Flex32|DType::F64)
        && matches!(b.dtype(),DType::F16|DType::BF16|DType::F32|DType::Flex32|DType::F64),"vocabulary adapter A/B must use native floating storage");
    assert!(a.device() == weight.device() && b.device() == weight.device(),"vocabulary adapter/base devices differ");
    if let Some(bias) = &layer.base.bias {
        let value = bias.val();
        assert!(value.dims() == [classes] && value.device() == weight.device() && !value.is_require_grad(),"vocabulary base bias geometry/device/frozen settings differ");
    }
    assert!(layer.scale.is_finite() && layer.dropout.prob.is_finite() && (0.0..1.0).contains(&layer.dropout.prob),"invalid vocabulary adapter scale/dropout");
    [hidden,classes]
}

impl<B:Backend> VocabParallelLoRAProjection<B> {
    /// Attach loaded corresponding A/B adapters without replacing their values or the base tie.
    /// Freeze the whole shared base explicitly first. A replicas/configuration must match on ranks.
    pub fn from_adapters(base:VocabParallelProjection<B>,adapter_a:Linear<B>,adapter_b:Linear<B>,dropout:Dropout,scale:f64) -> Self {
        let layer = Self {base,adapter_a,adapter_b,dropout,scale};geometry(&layer);layer
    }

    /// Initialize only real A/B parameters with RUDA's existing LoRA initializers/options.
    /// Does not broadcast independently initialized replicas or infer a different adapter dtype.
    pub fn init(base:VocabParallelProjection<B>,config:&TransformerAdapterConfig) -> Self {
        let [classes,hidden] = base.weight.val().dims();let rank = config.lora.rank;
        assert!(rank > 0 && config.lora.alpha.is_finite(),"invalid vocabulary adapter rank/alpha");
        assert!(config.lora.dropout.is_finite() && (0.0..1.0).contains(&config.lora.dropout),"invalid vocabulary adapter dropout");
        assert!(!base.weight.val().is_require_grad() && base.bias.as_ref().is_none_or(|bias|!bias.val().is_require_grad()),"freeze the complete shared base before vocabulary adapter initialization");
        let dtype = config.adapter_dtype.unwrap_or_else(||base.weight.val().dtype());let device = base.weight.val().device();
        assert!(matches!(dtype,DType::F16|DType::BF16|DType::F32|DType::Flex32|DType::F64),"declare native floating adapter storage for a packed vocabulary base");
        let mut a:Linear<B> = LinearConfig::new(hidden,rank).with_bias(false).init(&device);
        let mut b:Linear<B> = LinearConfig::new(rank,classes).with_bias(false).with_initializer(Initializer::Zeros).init(&device);
        a.weight = a.weight.map(|value|value.cast(dtype).detach().require_grad());
        b.weight = b.weight.map(|value|value.cast(dtype).detach().require_grad());
        let denominator = if config.use_rslora {(rank as f64).sqrt()} else {rank as f64};
        Self::from_adapters(base,a,b,crate::DropoutConfig::new(config.lora.dropout).init(),config.lora.alpha/denominator)
    }

    /// Native local base plus adapter inference, retaining original output storage and dropout mode.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,D>,C::Error> {
        geometry(self);
        let base = self.base.forward_inference_with_layout(input.clone(),communicator.clone(),layout,false)?;let storage = base.dtype();
        let adapted = self.dropout.forward(input.cast(self.adapter_a.weight.val().dtype()));
        let hidden = self.adapter_a.forward(adapted).cast(self.adapter_b.weight.val().dtype());
        let b = mask_padding(self.adapter_b.clone(),layout,communicator.rank() as usize);
        let logits = base+b.forward(hidden).mul_scalar(self.scale).cast(storage);
        if gather_output {layout.gather_logits_inference(logits,communicator)} else {Ok(logits)}
    }

    /// Merge actual floating head rows into an explicitly untied new frozen parameter for inference.
    /// The original embedding is left unchanged; merging a head-only adapter back into a shared
    /// embedding would alter the backbone. Packed bases require explicit caller requantization.
    pub fn merge_untied(self) -> VocabParallelProjection<B> {
        geometry(&self);
        assert!(!matches!(self.base.weight.val().dtype(),DType::QFloat(_)),"merge vocabulary adapters into floating base storage before explicit quantization");
        let update = self.adapter_a.weight.val().cast(self.adapter_b.weight.val().dtype())
            .matmul(self.adapter_b.weight.val()).mul_scalar(self.scale).transpose().detach();
        let mut base = self.base;
        base.weight = base.weight.map(|weight| {
            let dtype = weight.dtype();
            let merged = if dtype == update.dtype() {weight+update} else {
                let work = if dtype == DType::F64 || update.dtype() == DType::F64 {DType::F64} else {DType::F32};
                (weight.cast(work)+update.cast(work)).cast(dtype)
            };
            merged.detach().set_require_grad(false)
        });
        base.weight.id = ruda_model::module::ParamId::new();
        base
    }
}

impl<B:Backend,S:CheckpointStrategy> VocabParallelLoRAProjection<Autodiff<B,S>> {
    /// Native shard logits; hidden and replicated A gradients are SUMs for one logical TP loss.
    /// The original frozen base is not copied through a trainable float-only parameter region.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with_dropout(input,communicator,layout,gather_output,|dropout,input|dropout.forward(input))
    }

    /// Explicit corresponding adapter-input dropout, using actual A storage before projection.
    pub fn forward_with_dropout<C,F,const D:usize>(&self,input:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool,dropout:F) -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
        geometry(self);
        let source = transformed(&self.dropout,input.clone().cast(self.adapter_a.weight.val().dtype()),dropout);
        let base = self.base.forward_with_layout(input,communicator.clone(),layout,false)?;let storage = base.dtype();
        let source = region::copy_to_region(source,communicator.clone())?;
        let a = if self.adapter_a.weight.val().is_require_grad() {region::copy_to_region(self.adapter_a.weight.val(),communicator.clone())?} else {self.adapter_a.weight.val()};
        let hidden = ruda_model::tensor::module::linear(source,a,None).cast(self.adapter_b.weight.val().dtype());
        let b = mask_padding(self.adapter_b.clone(),layout,communicator.rank() as usize);
        let logits = base+b.forward(hidden).mul_scalar(self.scale).cast(storage);
        if gather_output {layout.gather_logits(logits,communicator)} else {Ok(logits)}
    }
}

/// Native tied-vocabulary head with explicit local output adapters and original norm/head dropout.
#[derive(Module,Debug)]
pub struct VocabParallelAdaptedTransformerHead<B:Backend> {
    /// Actual frozen shared vocabulary base and its independent A/B adapters.
    pub projection:VocabParallelLoRAProjection<B>,
    /// Original optional replicated head normalization, preserving flags/IDs.
    pub normalization:Option<DenseTransformerNorm<B>>,
    /// Original head-input dropout, preceding the adapter's own input dropout.
    pub dropout:Dropout,
}

impl<B:Backend> VocabParallelAdaptedTransformerHead<B> {
    /// Attach adapters to the explicitly prepared frozen tied head, without freezing other modules.
    pub fn from_dense(head:VocabParallelTransformerHead<B>,config:&TransformerAdapterConfig) -> Self {
        Self {projection:VocabParallelLoRAProjection::init(head.projection,config),normalization:head.normalization,dropout:head.dropout}
    }

    /// Consume the adapters into an explicitly untied floating inference head, preserving norm/dropout.
    pub fn merge_untied(self) -> VocabParallelTransformerHead<B> {
        VocabParallelTransformerHead {projection:self.projection.merge_untied(),normalization:self.normalization,dropout:self.dropout}
    }

    /// Connect loaded native projection adapters to the existing norm/dropout and class placement.
    pub fn from_projection(projection:VocabParallelLoRAProjection<B>,normalization:Option<DenseTransformerNorm<B>>,dropout:Dropout,
        layout:&VocabParallelLossLayout,rank:usize) -> Self {
        let [hidden,classes] = geometry(&projection);
        assert_eq!(classes,layout.interval(rank).len(),"adapted native vocabulary head classes differ from actual rank storage");
        if let Some(norm) = &normalization {assert_eq!(norm.width(),hidden,"adapted vocabulary head norm/hidden width differs");}
        assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob),"invalid adapted vocabulary head dropout");
        Self {projection,normalization,dropout}
    }

    /// Native local inference with the original norm/head/adapter dropout order.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<B,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,D>,C::Error> {
        let hidden = if let Some(norm) = &self.normalization {norm.forward(hidden)} else {hidden};
        self.projection.forward_inference(self.dropout.forward(hidden),communicator,layout,gather_output)
    }
}

impl<B:Backend,S:CheckpointStrategy> VocabParallelAdaptedTransformerHead<Autodiff<B,S>> {
    /// Native tied-base adapter training, with optional gathering only when explicitly requested.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with_dropouts(hidden,communicator,layout,gather_output,|dropout,input|dropout.forward(input),|dropout,input|dropout.forward(input))
    }

    /// Use explicit shared head-input and adapter-input dropout without inferring RNG synchronization.
    pub fn forward_with_dropouts<C,F,A,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool,head_dropout:F,adapter_dropout:A) -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D>,
            A:FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
        let hidden = if let Some(norm) = &self.normalization {norm.forward(hidden)} else {hidden};
        self.projection.forward_with_dropout(transformed(&self.dropout,hidden,head_dropout),communicator,layout,gather_output,adapter_dropout)
    }
}

super::training::head_objectives!(VocabParallelAdaptedTransformerHead);
