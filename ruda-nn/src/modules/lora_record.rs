use alloc::{format,string::String};
use ruda_model::{
    module::{Module,ModuleDTypeRecord},
    record::{PrecisionSettings,Record,Recorder,RecorderError},
    tensor::{DType,backend::Backend},
};
use crate::{Linear,LoRALinear,QuantizedLoRALinear};

/// Caller-identified frozen base and the actual adapter continuation contract.
#[derive(Clone,Debug,PartialEq)]
pub struct LoRAAdapterSchema {
    /// Adapter-only format version, independent of the full module record.
    pub version: u32,
    /// Caller-provided identity of the exact frozen weights/configuration.
    pub base_id: String,
    /// Actual logical [input,output] dimensions of the base projection.
    pub base_shape: [usize;2],
    /// Original frozen weight storage, including its precision.
    pub base_dtype: DType,
    /// Original optional frozen bias storage; its width equals base_shape[1].
    pub base_bias_dtype: Option<DType>,
    /// Actual intermediate adapter rank.
    pub rank: usize,
    /// Actual forward multiplier, including the selected LoRA/rsLoRA convention.
    pub scale: f64,
    /// Actual adapter-input dropout probability.
    pub dropout: f64,
    /// Actual A/B trainable settings, retained independently from recorder precision.
    pub trainable: [bool;2],
}

impl<B: Backend> Record<B> for LoRAAdapterSchema {
    type Item<S: PrecisionSettings> = (u32,String,([usize;2],DType,Option<DType>),(usize,f64,f64),[bool;2]);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.base_id,(self.base_shape,self.base_dtype,self.base_bias_dtype),
            (self.rank,self.scale,self.dropout),self.trainable)
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,_device: &B::Device) -> Self {
        Self {version:item.0,base_id:item.1,base_shape:item.2.0,base_dtype:item.2.1,base_bias_dtype:item.2.2,
            rank:item.3.0,scale:item.3.1,dropout:item.3.2,trainable:item.4}
    }
}

/// Native A/B-only record; no base tensors, optimizer state or input-source state.
/// Use a full-precision recorder when exact optimizer continuation is required.
pub struct LoRAAdapterRecord<B: Backend> {
    /// Actual immutable frozen-base/configuration contract.
    pub schema: LoRAAdapterSchema,
    adapter_a: <Linear<B> as Module<B>>::Record,
    adapter_b: <Linear<B> as Module<B>>::Record,
    dtypes: ModuleDTypeRecord,
}

impl<B: Backend> Record<B> for LoRAAdapterRecord<B> {
    type Item<S: PrecisionSettings> = (
        <LoRAAdapterSchema as Record<B>>::Item<S>,
        <<Linear<B> as Module<B>>::Record as Record<B>>::Item<S>,
        <<Linear<B> as Module<B>>::Record as Record<B>>::Item<S>,
        <ModuleDTypeRecord as Record<B>>::Item<S>,
    );
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (<LoRAAdapterSchema as Record<B>>::into_item::<S>(self.schema),
            self.adapter_a.into_item::<S>(),self.adapter_b.into_item::<S>(),
            <ModuleDTypeRecord as Record<B>>::into_item::<S>(self.dtypes))
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {schema:<LoRAAdapterSchema as Record<B>>::from_item::<S>(item.0,device),
            adapter_a:<<Linear<B> as Module<B>>::Record as Record<B>>::from_item::<S>(item.1,device),
            adapter_b:<<Linear<B> as Module<B>>::Record as Record<B>>::from_item::<S>(item.2,device),
            dtypes:<ModuleDTypeRecord as Record<B>>::from_item::<S>(item.3,device)}
    }
}

fn invalid(reason: &str) -> RecorderError { RecorderError::Unknown(format!("Invalid native adapter record: {reason}")) }

impl LoRAAdapterSchema {
    /// Read actual projection metadata, without downloading frozen base values.
    pub fn capture<B: Backend>(layer: &LoRALinear<B>,base_id: &str) -> Result<Self,RecorderError> {
        if base_id.is_empty() { return Err(invalid("base identity must be supplied explicitly")); }
        let base = layer.base.weight.val();
        let [input,output] = base.dims();
        if input == 0 || output == 0 || base.is_require_grad() || (!base.dtype().is_float() && !matches!(base.dtype(),DType::QFloat(_))) {
            return Err(invalid("the declared base must be nonempty, floating/quantized and frozen"));
        }
        let base_bias_dtype = if let Some(bias) = &layer.base.bias {
            let bias = bias.val();
            if bias.dims() != [output] || bias.device() != base.device() || bias.is_require_grad() {
                return Err(invalid("base bias geometry/device or frozen setting differs"));
            }
            Some(bias.dtype())
        } else { None };
        if layer.adapter_a.bias.is_some() || layer.adapter_b.bias.is_some() {
            return Err(invalid("native adapter A/B projections must be bias-free"));
        }
        let a = layer.adapter_a.weight.val();
        let b = layer.adapter_b.weight.val();
        let [a_input,rank] = a.dims();
        if rank == 0 || a_input != input || b.dims() != [rank,output]
            || a.device() != base.device() || b.device() != base.device()
            || !a.dtype().is_float() || !b.dtype().is_float() {
            return Err(invalid("actual adapter A/B geometry, storage or device differs"));
        }
        if !layer.scale.is_finite() || !layer.dropout.prob.is_finite() || !(0.0..1.0).contains(&layer.dropout.prob) {
            return Err(invalid("invalid adapter multiplier/dropout"));
        }
        Ok(Self {version:1,base_id:base_id.into(),base_shape:[input,output],base_dtype:base.dtype(),base_bias_dtype,
            rank,scale:layer.scale,dropout:layer.dropout.prob,trainable:[a.is_require_grad(),b.is_require_grad()]})
    }

    /// Match an already-prepared layer before replacing any adapter parameters.
    /// Base identity is caller-owned; this method does not invent a content hash.
    pub fn validate_for<B: Backend>(&self,layer: &LoRALinear<B>,base_id: &str) -> Result<(),RecorderError> {
        if self.version != 1 { return Err(invalid("unsupported format version")); }
        let actual = Self::capture(layer,base_id)?;
        if self.base_id != actual.base_id || self.base_shape != actual.base_shape || self.base_dtype != actual.base_dtype
            || self.base_bias_dtype != actual.base_bias_dtype || self.rank != actual.rank || self.trainable != actual.trainable
            || self.scale.to_bits() != actual.scale.to_bits() || self.dropout.to_bits() != actual.dropout.to_bits() {
            return Err(invalid("frozen base, rank, precision, training settings or forward configuration differs"));
        }
        Ok(())
    }
}

impl<B: Backend> LoRAAdapterRecord<B> {
    /// Capture actual A/B leaves over a generic packed base; full QuantScheme is retained in dtype metadata.
    pub fn capture_quantized(layer:&QuantizedLoRALinear<B>,base_id:&str) -> Result<Self,RecorderError> {
        let schema = LoRAAdapterSchema::capture_quantized(layer,base_id)?;
        Self::capture_adapters(schema,&layer.adapter_a,&layer.adapter_b)
    }

    /// Restore only actual adapters; the original packed codes/scales and base IDs are unchanged.
    pub fn restore_into_quantized(self,layer:QuantizedLoRALinear<B>,base_id:&str) -> Result<QuantizedLoRALinear<B>,RecorderError> {
        self.schema.validate_quantized(&layer,base_id)?;
        let schema = self.schema.clone();
        let device = layer.base.weight.val().device();
        let (a,b) = self.restore_adapters(layer.adapter_a,layer.adapter_b,&device)?;
        let layer = QuantizedLoRALinear {base:layer.base,adapter_a:a,adapter_b:b,dropout:layer.dropout,scale:layer.scale};
        schema.validate_quantized(&layer,base_id)?;
        Ok(layer)
    }

    /// Pin only actual A/B values and per-parameter dtypes at a caller-owned boundary.
    /// Frozen base handles/values are never retained in this record.
    pub fn capture(layer: &LoRALinear<B>,base_id: &str) -> Result<Self,RecorderError> {
        let schema = LoRAAdapterSchema::capture(layer,base_id)?;
        Self::capture_adapters(schema,&layer.adapter_a,&layer.adapter_b)
    }

    pub(crate) fn capture_adapters(schema:LoRAAdapterSchema,adapter_a:&Linear<B>,adapter_b:&Linear<B>) -> Result<Self,RecorderError> {
        let adapters = (adapter_a.clone(),adapter_b.clone());
        let dtypes = ModuleDTypeRecord::capture(&adapters)?;
        Ok(Self {schema,adapter_a:adapters.0.into_record(),adapter_b:adapters.1.into_record(),dtypes})
    }

    /// Save A/B and continuation metadata through an existing recorder.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {
        recorder.record(self,args)
    }

    /// Read A/B onto the caller's device, without loading or creating a base model.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {
        recorder.load(args,device)
    }

    /// Restore original adapter IDs/dtypes on a prepared layer, leaving the base unchanged.
    /// Restore optimizer/pending-gradient records only after obtaining this new layer.
    pub fn restore_into(self,layer: LoRALinear<B>,base_id: &str) -> Result<LoRALinear<B>,RecorderError> {
        self.schema.validate_for(&layer,base_id)?;
        let schema = self.schema.clone();
        let device = layer.base.weight.val().device();
        let (a,b) = self.restore_adapters(layer.adapter_a,layer.adapter_b,&device)?;
        let restored = LoRALinear {base:layer.base,adapter_a:a,adapter_b:b,dropout:layer.dropout,scale:layer.scale};
        schema.validate_for(&restored,base_id)?;
        Ok(restored)
    }

    pub(crate) fn restore_adapters(self,adapter_a:Linear<B>,adapter_b:Linear<B>,device:&B::Device) -> Result<(Linear<B>,Linear<B>),RecorderError> {
        let a = adapter_a.load_record(self.adapter_a).fork(device);
        let b = adapter_b.load_record(self.adapter_b).fork(device);
        let (a,b) = self.dtypes.apply((a,b))?;
        Ok((a,b))
    }
}

impl LoRAAdapterSchema {
    /// Original packed-base and A/B continuation contract without downloading or decoding the base.
    pub fn capture_quantized<B:Backend>(layer:&QuantizedLoRALinear<B>,base_id:&str) -> Result<Self,RecorderError> {
        if base_id.is_empty() {return Err(invalid("nonempty exact frozen base identity required"));}
        let base = layer.base.weight.val();
        let [output,input] = base.dims();
        if input == 0 || output == 0 || !matches!(base.dtype(),DType::QFloat(_)) || base.is_require_grad() {
            return Err(invalid("generic packed base geometry/storage/frozen setting differs"));
        }
        let bias_dtype = if let Some(bias) = &layer.base.bias {
            let bias = bias.val();
            if bias.dims() != [output] || bias.device() != base.device() || bias.is_require_grad()
                || !matches!(bias.dtype(),DType::F16|DType::BF16|DType::F32) {
                return Err(invalid("packed base bias geometry/device/frozen storage differs"));
            }
            Some(bias.dtype())
        } else {None};
        let a = layer.adapter_a.weight.val(); let b = layer.adapter_b.weight.val();
        let [a_input,rank] = a.dims();
        if rank == 0 || a_input != input || b.dims() != [rank,output] || a.device() != base.device() || b.device() != base.device()
            || layer.adapter_a.bias.is_some() || layer.adapter_b.bias.is_some()
            || !matches!(a.dtype(),DType::F16|DType::BF16|DType::F32) || !matches!(b.dtype(),DType::F16|DType::BF16|DType::F32) {
            return Err(invalid("packed adapter A/B geometry/device/storage differs"));
        }
        if !layer.scale.is_finite() || !layer.dropout.prob.is_finite() || !(0.0..1.0).contains(&layer.dropout.prob) {
            return Err(invalid("invalid packed adapter multiplier/dropout"));
        }
        Ok(Self {version:1,base_id:base_id.into(),base_shape:[input,output],base_dtype:base.dtype(),base_bias_dtype:bias_dtype,
            rank,scale:layer.scale,dropout:layer.dropout.prob,trainable:[a.is_require_grad(),b.is_require_grad()]})
    }

    /// Check exact packed format/block/scale metadata and actual continuation semantics before replacing A/B.
    pub fn validate_quantized<B:Backend>(&self,layer:&QuantizedLoRALinear<B>,base_id:&str) -> Result<(),RecorderError> {
        let actual = Self::capture_quantized(layer,base_id)?;
        if self.version != 1 || self.base_id != actual.base_id || self.base_shape != actual.base_shape
            || self.base_dtype != actual.base_dtype || self.base_bias_dtype != actual.base_bias_dtype
            || self.rank != actual.rank || self.trainable != actual.trainable
            || self.scale.to_bits() != actual.scale.to_bits() || self.dropout.to_bits() != actual.dropout.to_bits() {
            return Err(invalid("packed base, format, geometry or adapter continuation configuration differs"));
        }
        Ok(())
    }
}

impl<B: Backend> LoRALinear<B> {
    /// Export only this layer's actual A/B state, with explicit frozen-base identity.
    pub fn adapter_record(&self,base_id: &str) -> Result<LoRAAdapterRecord<B>,RecorderError> {
        LoRAAdapterRecord::capture(self,base_id)
    }
}
