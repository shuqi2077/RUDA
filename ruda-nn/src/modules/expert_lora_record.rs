use super::{ExpertLinear,ExpertLoRABase,PackedExpertLoRA,FrozenPackedExpertProjection};
use alloc::{format,string::String};
use ruda_model::{module::{Module,ModuleDTypeRecord},record::{Record,PrecisionSettings,Recorder,RecorderError},
    serde::{Serialize,Deserialize},tensor::{DType,MoeExpertStrategy,backend::Backend}};

/// Recorded original grouped execution policy, independent of recorder tensor precision.
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize,Deserialize)]
#[serde(crate="ruda_model::serde")]
pub enum ExpertExecutionSchema {Scalar,Auto,TensorCore}
impl From<MoeExpertStrategy> for ExpertExecutionSchema {
    fn from(value:MoeExpertStrategy) -> Self {match value {
        MoeExpertStrategy::Scalar=>Self::Scalar,MoeExpertStrategy::Auto=>Self::Auto,MoeExpertStrategy::TensorCore=>Self::TensorCore}}
}
/// Source representation/execution metadata only, never a packed or floating base tensor.
#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
#[serde(crate="ruda_model::serde")]
pub enum ExpertAdapterBaseSchema {
    /// Actual original floating cube storage and independent native forward/backward policies.
    Floating {dtype:DType,execution:[ExpertExecutionSchema;2]},
    /// Original flat NF4 blocks and source native projection policy; bytes/scales/book stay outside the adapter record.
    Nf4 {block_size:usize,tile_rows:usize,use_tensor_core:bool},
    /// Actual original AWQ groups and independently stored scales/optional bias; words/zeros stay outside the record.
    Awq {group_size:usize,scale_dtype:DType,bias_dtype:Option<DType>},
}
/// Actual source-base metadata needed by an expert A/B-only record.
pub trait ExpertAdapterRecordBase<B:Backend>:ExpertLoRABase<B> {
    /// Capture the source representation contract, rejecting a base that is no longer frozen.
    fn adapter_base_schema(&self) -> Result<ExpertAdapterBaseSchema,RecorderError>;
}
fn invalid(reason:&str) -> RecorderError {RecorderError::Unknown(format!("Invalid native expert adapter record: {reason}"))}
impl<B:Backend> ExpertAdapterRecordBase<B> for ExpertLinear<B> {
    fn adapter_base_schema(&self) -> Result<ExpertAdapterBaseSchema,RecorderError> {
        if self.weight.val().is_require_grad() {return Err(invalid("original floating expert base is not frozen"));}
        self.validate();Ok(ExpertAdapterBaseSchema::Floating {dtype:self.weight.val().dtype(),
            execution:[self.forward_strategy.into(),self.backward_strategy.into()]})
    }
}
impl<B:Backend> ExpertAdapterRecordBase<B> for FrozenPackedExpertProjection<B> {
    fn adapter_base_schema(&self) -> Result<ExpertAdapterBaseSchema,RecorderError> {
        match self {
            Self::Nf4(value)=>{
                if value.payload.scales.val().is_require_grad() || value.payload.codebook.val().is_require_grad() {
                    return Err(invalid("original NF4 scale/codebook is not frozen"));}
                value.validate();Ok(ExpertAdapterBaseSchema::Nf4 {block_size:value.payload.block_size,
                    tile_rows:value.payload.tile_rows,use_tensor_core:value.payload.use_tensor_core})
            },
            Self::Awq(value)=>{
                if value.scales.val().is_require_grad() || value.bias.as_ref().is_some_and(|bias|bias.val().is_require_grad()) {
                    return Err(invalid("original AWQ scale/bias is not frozen"));}
                value.validate();Ok(ExpertAdapterBaseSchema::Awq {group_size:value.group_size,scale_dtype:value.scales.val().dtype(),
                    bias_dtype:value.bias.as_ref().map(|bias|bias.val().dtype())})
            },
        }
    }
}
/// Caller-identified original expert base and actual A/B-only continuation contract.
#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
#[serde(crate="ruda_model::serde")]
pub struct ExpertLoRAAdapterSchema {
    /// Adapter-only format version.
    pub version:u32,
    /// Caller identity for exact original weights, quantization metadata and architecture.
    pub base_id:String,
    /// Actual original `[experts,input,output]` widths.
    pub base_shape:[usize;3],
    /// Original floating/NF4/AWQ representation/execution contract.
    pub base:ExpertAdapterBaseSchema,
    /// Actual rank of each expert's A/B pair.
    pub rank:usize,
    /// Actual LoRA/rsLoRA multiplier, not a newly inferred alpha.
    pub scale:f64,
    /// Actual adapter-only input dropout.
    pub dropout:f64,
    /// Actual independent A/B storage dtypes.
    pub adapter_dtypes:[DType;2],
    /// Original independent forward/backward policies for A, then B.
    pub adapter_execution:[[ExpertExecutionSchema;2];2],
    /// Actual independent A/B training flags.
    pub trainable:[bool;2],
}
impl<B:Backend> Record<B> for ExpertLoRAAdapterSchema {
    type Item<S:PrecisionSettings> = Self;
    fn into_item<S:PrecisionSettings>(self) -> Self {self}
    fn from_item<S:PrecisionSettings>(item:Self,_device:&B::Device) -> Self {item}
}
impl ExpertLoRAAdapterSchema {
    /// Capture only original metadata, without retaining or reading any base values.
    pub fn capture<B:Backend,Base:ExpertAdapterRecordBase<B>>(layer:&PackedExpertLoRA<B,Base>,base_id:&str) -> Result<Self,RecorderError> {
        if base_id.is_empty() {return Err(invalid("exact original base identity must be supplied"));}
        let base=layer.base.adapter_base_schema()?;let [e,k,n]=layer.base.dimensions();
        let a=layer.adapter_a.weight.val();let b=layer.adapter_b.weight.val();let [ae,rank,ak]=a.dims();
        if k==0 || n==0 || rank==0 || [ae,ak]!=[e,k] || b.dims()!=[e,n,rank] {
            return Err(invalid("original base and actual expert A/B geometry differs"));}
        if !matches!(a.dtype(),DType::F16|DType::BF16|DType::F32) || !matches!(b.dtype(),DType::F16|DType::BF16|DType::F32)
            || a.device()!=layer.base.device() || b.device()!=layer.base.device() {
            return Err(invalid("actual A/B storage or resident device differs"));}
        if !layer.scale.is_finite() || !layer.dropout.prob.is_finite() || !(0.0..1.0).contains(&layer.dropout.prob) {
            return Err(invalid("original expert adapter scale/dropout is invalid"));}
        Ok(Self {version:1,base_id:base_id.into(),base_shape:[e,k,n],base,rank,scale:layer.scale,dropout:layer.dropout.prob,
            adapter_dtypes:[a.dtype(),b.dtype()],trainable:[a.is_require_grad(),b.is_require_grad()],
            adapter_execution:[[layer.adapter_a.forward_strategy.into(),layer.adapter_a.backward_strategy.into()],
                [layer.adapter_b.forward_strategy.into(),layer.adapter_b.backward_strategy.into()]]})
    }
    /// Match the caller-prepared original layer before replacing any adapter leaves.
    /// Base identity is caller-owned and is not substituted with an invented content hash.
    pub fn validate_for<B:Backend,Base:ExpertAdapterRecordBase<B>>(&self,layer:&PackedExpertLoRA<B,Base>,base_id:&str) -> Result<(),RecorderError> {
        if self.version!=1 {return Err(invalid("unsupported expert adapter format version"));}
        let actual=Self::capture(layer,base_id)?;
        if self.base_id!=actual.base_id || self.base_shape!=actual.base_shape || self.base!=actual.base || self.rank!=actual.rank
            || self.adapter_dtypes!=actual.adapter_dtypes || self.adapter_execution!=actual.adapter_execution || self.trainable!=actual.trainable
            || self.scale.to_bits()!=actual.scale.to_bits() || self.dropout.to_bits()!=actual.dropout.to_bits() {
            return Err(invalid("original base, rank, storage, execution, training flags or forward configuration differs"));}Ok(())
    }
}
/// Native rank-three expert A/B-only record; excludes original base payloads and optimizer/transport state.
/// Recorder precision controls saved values; use full precision for exact A/B value continuation.
pub struct ExpertLoRAAdapterRecord<B:Backend> {
    /// Exact original caller-bound source and adapter continuation contract.
    pub schema:ExpertLoRAAdapterSchema,
    adapter_a:<ExpertLinear<B> as Module<B>>::Record,
    adapter_b:<ExpertLinear<B> as Module<B>>::Record,
    dtypes:ModuleDTypeRecord,
}
impl<B:Backend> Record<B> for ExpertLoRAAdapterRecord<B> {
    type Item<S:PrecisionSettings> = (
        <ExpertLoRAAdapterSchema as Record<B>>::Item<S>,
        <<ExpertLinear<B> as Module<B>>::Record as Record<B>>::Item<S>,
        <<ExpertLinear<B> as Module<B>>::Record as Record<B>>::Item<S>,
        <ModuleDTypeRecord as Record<B>>::Item<S>,
    );
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {
        (<ExpertLoRAAdapterSchema as Record<B>>::into_item::<S>(self.schema),self.adapter_a.into_item::<S>(),self.adapter_b.into_item::<S>(),
            <ModuleDTypeRecord as Record<B>>::into_item::<S>(self.dtypes))
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {schema:<ExpertLoRAAdapterSchema as Record<B>>::from_item::<S>(item.0,device),
            adapter_a:<<ExpertLinear<B> as Module<B>>::Record as Record<B>>::from_item::<S>(item.1,device),
            adapter_b:<<ExpertLinear<B> as Module<B>>::Record as Record<B>>::from_item::<S>(item.2,device),
            dtypes:<ModuleDTypeRecord as Record<B>>::from_item::<S>(item.3,device)}
    }
}
impl<B:Backend> ExpertLoRAAdapterRecord<B> {
    /// Capture actual A/B leaves, including their IDs/dtypes, without pinning frozen expert payloads.
    pub fn capture<Base:ExpertAdapterRecordBase<B>>(layer:&PackedExpertLoRA<B,Base>,base_id:&str) -> Result<Self,RecorderError> {
        let schema=ExpertLoRAAdapterSchema::capture(layer,base_id)?;
        let snapshot=|mut adapter:ExpertLinear<B>| {
            adapter.weight=adapter.weight.map(|value| {let trainable=value.is_require_grad();value.detach().set_require_grad(trainable)});adapter
        };
        let adapters=(snapshot(layer.adapter_a.clone()),snapshot(layer.adapter_b.clone()));let dtypes=ModuleDTypeRecord::capture(&adapters)?;
        Ok(Self {schema,adapter_a:adapters.0.into_record(),adapter_b:adapters.1.into_record(),dtypes})
    }
    /// Save only actual expert A/B and source metadata using an existing native recorder.
    pub fn save<R:Recorder<B>>(self,recorder:&R,args:R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load actual expert A/B onto the caller-selected device without constructing a base or a model replica.
    pub fn load<R:Recorder<B>>(recorder:&R,args:R::LoadArgs,device:&B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}
    /// Restore original A/B identities and dtype leaves on a prepared source, leaving every original base tensor unchanged.
    /// Restore optimizer/pending-gradient state after obtaining the restored layer.
    pub fn restore_into<Base:ExpertAdapterRecordBase<B>>(self,mut layer:PackedExpertLoRA<B,Base>,base_id:&str)
        -> Result<PackedExpertLoRA<B,Base>,RecorderError> {
        self.schema.validate_for(&layer,base_id)?;let schema=self.schema;let device=layer.base.device();
        let a=layer.adapter_a.load_record(self.adapter_a).fork(&device);let b=layer.adapter_b.load_record(self.adapter_b).fork(&device);
        let (a,b)=self.dtypes.apply((a,b))?;layer.adapter_a=a;layer.adapter_b=b;
        schema.validate_for(&layer,base_id)?;Ok(layer)
    }
}
impl<B:Backend,Base:ExpertAdapterRecordBase<B>> PackedExpertLoRA<B,Base> {
    /// Export only this original expert projection's actual A/B, with explicit original base identity.
    pub fn adapter_record(&self,base_id:&str) -> Result<ExpertLoRAAdapterRecord<B>,RecorderError> {ExpertLoRAAdapterRecord::capture(self,base_id)}
}
