use alloc::format;
use ruda_model::{module::{Module,ModuleVisitor,Param},record::{PrecisionSettings,Record,Recorder,RecorderError},
    tensor::{Bool,Tensor,backend::Backend}};
use crate::{LoRALinear,LoRAAdapterRecord,Dropout,pool::{pool_sequence,pool_packed_sequences,SequencePooling,SequencePoolOutput},
    attention::PackedSequenceLayout};
use super::{TransformerHead,TransformerAdapterConfig,DenseTransformerNorm,SequenceHeadOutput};

/// Actual native vocabulary/classification head with an explicitly selected low-rank update.
#[derive(Module,Debug)]
pub struct AdaptedTransformerHead<B: Backend> {
    /// Original output-class geometry and frozen base, plus real trainable A/B matrices.
    pub projection: LoRALinear<B>,
    /// Optional original normalization; its parameter identities and flags remain intact.
    pub normalization: Option<DenseTransformerNorm<B>>,
    /// Original head-input dropout, separate from adapter-input dropout.
    pub dropout: Dropout,
}

impl<B: Backend> AdaptedTransformerHead<B> {
    /// Consume the loaded head and adapt its actual output projection only.
    /// Freeze the complete base explicitly before this conversion for adapter-only training.
    pub fn from_dense(base: TransformerHead<B>,config: &TransformerAdapterConfig) -> Self {
        let dtype = config.adapter_dtype.unwrap_or_else(||base.projection.weight.val().dtype());
        Self {projection:config.lora.init_with_options(base.projection,dtype,config.use_rslora),
            normalization:base.normalization,dropout:base.dropout}
    }

    /// Project native token/sequence states with the original normalization/dropout order.
    /// Vocabulary/class count and leading dimensions are never inferred or resampled.
    pub fn forward<const D: usize>(&self,hidden: Tensor<B,D>) -> Tensor<B,D> {
        let hidden = if let Some(norm) = &self.normalization { norm.forward(hidden) } else { hidden };
        self.projection.forward(self.dropout.forward(hidden))
    }

    /// Classify caller-declared visible tokens with explicit pooling and A/B gradients.
    pub fn forward_sequence(&self,hidden: Tensor<B,3>,visible: Tensor<B,2,Bool>,pooling: SequencePooling)
        -> SequenceHeadOutput<B> {
        self.forward_pooled(pool_sequence(hidden,visible,pooling))
    }

    /// Classify independent actual packed documents without cross-document pooling.
    pub fn forward_packed_sequences(&self,hidden: Tensor<B,2>,layout: &PackedSequenceLayout,
        visible: Option<Tensor<B,1,Bool>>,pooling: SequencePooling) -> SequenceHeadOutput<B> {
        self.forward_pooled(pool_packed_sequences(hidden,layout,visible,pooling))
    }

    /// Project actual pooled values while retaining empty-row/real-token metadata.
    pub fn forward_pooled(&self,pooled: SequencePoolOutput<B>) -> SequenceHeadOutput<B> {
        SequenceHeadOutput {logits:self.forward(pooled.values),valid_rows:pooled.valid_rows,token_counts:pooled.token_counts}
    }

    /// Capture only this head's A/B weights, not its frozen output/norm parameters.
    pub fn adapter_record(&self,base_id: &str) -> Result<TransformerHeadAdapterRecord<B>,RecorderError> {
        TransformerHeadAdapterRecord::capture(self,base_id)
    }
}

/// Native A/B-only head record, composable with backbone/decoder adapter records.
pub struct TransformerHeadAdapterRecord<B: Backend> {
    projection: LoRAAdapterRecord<B>,
}

impl<B: Backend> Record<B> for TransformerHeadAdapterRecord<B> {
    type Item<S: PrecisionSettings> = <LoRAAdapterRecord<B> as Record<B>>::Item<S>;
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> { self.projection.into_item::<S>() }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {projection:LoRAAdapterRecord::<B>::from_item::<S>(item,device)}
    }
}

fn invalid(reason: &str) -> RecorderError {
    RecorderError::Unknown(format!("Invalid native head adapter record: {reason}"))
}

fn check_adapter_only<B: Backend>(head: &AdaptedTransformerHead<B>) -> Result<(),RecorderError> {
    struct FrozenCheck { trainable: bool }
    impl<B: Backend> ModuleVisitor<B> for FrozenCheck {
        fn visit_float<const D: usize>(&mut self,param: &Param<Tensor<B,D>>) {
            self.trainable |= param.val().is_require_grad();
        }
    }
    if let Some(norm) = &head.normalization {
        let mut visitor = FrozenCheck {trainable:false};
        norm.visit(&mut visitor);
        if visitor.trainable { return Err(invalid("trainable head normalization requires a full model checkpoint")); }
    }
    Ok(())
}

impl<B: Backend> TransformerHeadAdapterRecord<B> {
    /// Capture actual head adapters, rejecting omitted trainable normalization state.
    pub fn capture(head: &AdaptedTransformerHead<B>,base_id: &str) -> Result<Self,RecorderError> {
        check_adapter_only(head)?;
        Ok(Self {projection:head.projection.adapter_record(base_id)?})
    }

    /// Check the prepared frozen head and exact native adapter continuation contract.
    pub fn validate_for(&self,head: &AdaptedTransformerHead<B>,base_id: &str) -> Result<(),RecorderError> {
        check_adapter_only(head)?;
        self.projection.schema.validate_for(&head.projection,base_id)
    }

    /// Save A/B state through an existing native recorder without frozen head values.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {
        recorder.record(self,args)
    }

    /// Load the actual native head A/B state on the explicit device.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {
        recorder.load(args,device)
    }

    /// Restore actual A/B IDs/dtypes without replacing the output base, norm or dropout.
    pub fn restore_into(self,mut head: AdaptedTransformerHead<B>,base_id: &str)
        -> Result<AdaptedTransformerHead<B>,RecorderError> {
        self.validate_for(&head,base_id)?;
        head.projection = self.projection.restore_into(head.projection,base_id)?;
        Ok(head)
    }
}
