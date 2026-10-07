use super::*;
use alloc::{format,string::String,vec::Vec};
use ruda_model::{module::{ModuleVisitor,Param},record::{PrecisionSettings,Record,Recorder}};

fn invalid(reason:&str) -> RecorderError {RecorderError::Unknown(format!("Invalid native vocabulary adapter record: {reason}"))}

fn schema<B:Backend>(layer:&VocabParallelLoRAProjection<B>,base_id:&str) -> Result<LoRAAdapterSchema,RecorderError> {
    if base_id.is_empty() {return Err(invalid("explicit frozen base identity is required"));}
    let weight = layer.base.weight.val();let [classes,hidden] = weight.dims();
    if classes == 0 || hidden == 0 || weight.is_require_grad() || (!weight.dtype().is_float() && !matches!(weight.dtype(),DType::QFloat(_))) {
        return Err(invalid("actual row-major vocabulary base must be nonempty, floating/quantized and frozen"));
    }
    let bias_dtype = if let Some(bias) = &layer.base.bias {
        let value = bias.val();
        if value.dims() != [classes] || value.device() != weight.device() || value.is_require_grad() {return Err(invalid("base bias geometry/device/frozen setting differs"));}
        Some(value.dtype())
    } else {None};
    let a = layer.adapter_a.weight.val();let b = layer.adapter_b.weight.val();let rank = a.dims()[1];
    if rank == 0 || a.dims() != [hidden,rank] || b.dims() != [rank,classes]
        || a.device() != weight.device() || b.device() != weight.device() || !a.dtype().is_float() || !b.dtype().is_float()
        || matches!(a.dtype(),DType::QFloat(_)) || matches!(b.dtype(),DType::QFloat(_))
        || layer.adapter_a.bias.is_some() || layer.adapter_b.bias.is_some() {
        return Err(invalid("actual native adapter dimensions/storage/devices differ"));
    }
    if !layer.scale.is_finite() || !layer.dropout.prob.is_finite() || !(0.0..1.0).contains(&layer.dropout.prob) {
        return Err(invalid("native multiplier/dropout differs"));
    }
    Ok(LoRAAdapterSchema {version:1,base_id:String::from(base_id),base_shape:[hidden,classes],base_dtype:weight.dtype(),base_bias_dtype:bias_dtype,
        rank,scale:layer.scale,dropout:layer.dropout.prob,trainable:[a.is_require_grad(),b.is_require_grad()]})
}

fn check_schema(expected:&LoRAAdapterSchema,actual:&LoRAAdapterSchema) -> Result<(),RecorderError> {
    if expected.version != 1 || expected.version != actual.version || expected.base_id != actual.base_id || expected.base_shape != actual.base_shape
        || expected.base_dtype != actual.base_dtype || expected.base_bias_dtype != actual.base_bias_dtype || expected.rank != actual.rank
        || expected.trainable != actual.trainable || expected.scale.to_bits() != actual.scale.to_bits() || expected.dropout.to_bits() != actual.dropout.to_bits() {
        return Err(invalid("frozen base identity, actual geometry/precision/flags or forward configuration differs"));
    }
    Ok(())
}

fn widths(layout:&VocabParallelLossLayout) -> Vec<usize> {(0..layout.world_size()).map(|rank|layout.interval(rank).len()).collect()}

/// Exact rank-local A/B-only continuation record, reusing RUDA's native LoRA tensor/dtype format.
/// The row-major frozen vocabulary base is never transposed, retained or serialized in this record.
pub struct VocabParallelLoRAAdapterRecord<B:Backend> {
    version:u32,widths:Vec<usize>,vocabulary_size:usize,rank:usize,adapters:LoRAAdapterRecord<B>,
}

impl<B:Backend> Record<B> for VocabParallelLoRAAdapterRecord<B> {
    type Item<S:PrecisionSettings> = (u32,Vec<usize>,usize,usize,<LoRAAdapterRecord<B> as Record<B>>::Item<S>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {(self.version,self.widths,self.vocabulary_size,self.rank,self.adapters.into_item::<S>())}
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {version:item.0,widths:item.1,vocabulary_size:item.2,rank:item.3,adapters:LoRAAdapterRecord::<B>::from_item::<S>(item.4,device)}
    }
}

impl<B:Backend> VocabParallelLoRAAdapterRecord<B> {
    /// Actual owning rank under the saved explicit vocabulary layout.
    pub fn rank(&self) -> usize {self.rank}
    /// Actual rank-ordered stored vocabulary widths, including declared trailing padding.
    pub fn widths(&self) -> &[usize] {&self.widths}
    /// Number of logical real output classes in the saved layout.
    pub fn vocabulary_size(&self) -> usize {self.vocabulary_size}
    /// Native logical base/A/B continuation metadata without frozen base tensor handles.
    pub fn schema(&self) -> &LoRAAdapterSchema {&self.adapters.schema}

    /// Capture real A/B values and exact rank/layout, without materializing a transposed base view.
    pub fn capture(layer:&VocabParallelLoRAProjection<B>,layout:&VocabParallelLossLayout,rank:usize,base_id:&str) -> Result<Self,RecorderError> {
        if rank >= layout.world_size() || layer.base.weight.val().dims()[0] != layout.interval(rank).len() {return Err(invalid("projection/rank vocabulary placement differs"));}
        let adapters = LoRAAdapterRecord::capture_adapters(schema(layer,base_id)?,&layer.adapter_a,&layer.adapter_b)?;
        Ok(Self {version:1,widths:widths(layout),vocabulary_size:layout.vocabulary_size(),rank,adapters})
    }

    /// Check the exact native frozen base, physical rank/layout and A/B continuation contract.
    pub fn validate_for(&self,layer:&VocabParallelLoRAProjection<B>,layout:&VocabParallelLossLayout,rank:usize,base_id:&str) -> Result<(),RecorderError> {
        if self.version != 1 || rank >= layout.world_size() || self.rank != rank || self.widths != widths(layout)
            || self.vocabulary_size != layout.vocabulary_size() || layer.base.weight.val().dims()[0] != layout.interval(rank).len() {
            return Err(invalid("record schema or actual vocabulary rank/layout differs"));
        }
        check_schema(&self.adapters.schema,&schema(layer,base_id)?)
    }

    /// Save only native adapter values and placement metadata through the supplied recorder.
    pub fn save<R:Recorder<B>>(self,recorder:&R,args:R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load native adapter values onto the explicit device, not a frozen vocabulary base.
    pub fn load<R:Recorder<B>>(recorder:&R,args:R::LoadArgs,device:&B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}

    /// Restore original A/B IDs/dtypes/flags without replacing the actual row-major base or tie.
    pub fn restore_into(self,layer:VocabParallelLoRAProjection<B>,layout:&VocabParallelLossLayout,rank:usize,base_id:&str)
        -> Result<VocabParallelLoRAProjection<B>,RecorderError> {
        self.validate_for(&layer,layout,rank,base_id)?;
        let expected = self.adapters.schema.clone();let device = layer.base.weight.val().device();
        let (adapter_a,adapter_b) = self.adapters.restore_adapters(layer.adapter_a,layer.adapter_b,&device)?;
        let restored = VocabParallelLoRAProjection {base:layer.base,adapter_a,adapter_b,dropout:layer.dropout,scale:layer.scale};
        check_schema(&expected,&schema(&restored,base_id)?)?;Ok(restored)
    }
}

fn frozen_norm<B:Backend>(head:&VocabParallelAdaptedTransformerHead<B>) -> Result<(),RecorderError> {
    struct Inspect {trainable:bool}
    impl<B:Backend> ModuleVisitor<B> for Inspect {
        fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<B,D>>) {self.trainable |= param.val().is_require_grad();}
    }
    if let Some(norm) = &head.normalization {
        let mut inspect = Inspect {trainable:false};norm.visit(&mut inspect);
        if inspect.trainable {return Err(invalid("trainable head normalization requires a full model checkpoint"));}
    }
    Ok(())
}

/// Native vocabulary-head A/B-only record with exact original physical placement.
pub struct VocabParallelHeadAdapterRecord<B:Backend> {projection:VocabParallelLoRAAdapterRecord<B>}
impl<B:Backend> Record<B> for VocabParallelHeadAdapterRecord<B> {
    type Item<S:PrecisionSettings> = <VocabParallelLoRAAdapterRecord<B> as Record<B>>::Item<S>;
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {self.projection.into_item::<S>()}
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {Self {projection:VocabParallelLoRAAdapterRecord::from_item::<S>(item,device)}}
}
impl<B:Backend> VocabParallelHeadAdapterRecord<B> {
    /// Capture native head adapters only, requiring the omitted normalization to be frozen.
    pub fn capture(head:&VocabParallelAdaptedTransformerHead<B>,layout:&VocabParallelLossLayout,rank:usize,base_id:&str) -> Result<Self,RecorderError> {
        frozen_norm(head)?;Ok(Self {projection:VocabParallelLoRAAdapterRecord::capture(&head.projection,layout,rank,base_id)?})
    }
    /// Validate exact vocabulary placement and frozen native head continuation before restoration.
    pub fn validate_for(&self,head:&VocabParallelAdaptedTransformerHead<B>,layout:&VocabParallelLossLayout,rank:usize,base_id:&str) -> Result<(),RecorderError> {
        frozen_norm(head)?;self.projection.validate_for(&head.projection,layout,rank,base_id)
    }
    /// Save only actual head adapters through an existing native recorder.
    pub fn save<R:Recorder<B>>(self,recorder:&R,args:R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load this native rank-local head adapter record on the explicit device.
    pub fn load<R:Recorder<B>>(recorder:&R,args:R::LoadArgs,device:&B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}
    /// Restore real A/B state while leaving the shared base, normalization and dropout untouched.
    pub fn restore_into(self,mut head:VocabParallelAdaptedTransformerHead<B>,layout:&VocabParallelLossLayout,rank:usize,base_id:&str)
        -> Result<VocabParallelAdaptedTransformerHead<B>,RecorderError> {
        self.validate_for(&head,layout,rank,base_id)?;head.projection = self.projection.restore_into(head.projection,layout,rank,base_id)?;Ok(head)
    }
}

impl<B:Backend> VocabParallelLoRAProjection<B> {
    /// Capture this actual rank-local native adapter and physical vocabulary placement.
    pub fn adapter_record(&self,layout:&VocabParallelLossLayout,rank:usize,base_id:&str) -> Result<VocabParallelLoRAAdapterRecord<B>,RecorderError> {
        VocabParallelLoRAAdapterRecord::capture(self,layout,rank,base_id)
    }
}
impl<B:Backend> VocabParallelAdaptedTransformerHead<B> {
    /// Capture actual head A/B state without retaining the frozen tied vocabulary base.
    pub fn adapter_record(&self,layout:&VocabParallelLossLayout,rank:usize,base_id:&str) -> Result<VocabParallelHeadAdapterRecord<B>,RecorderError> {
        VocabParallelHeadAdapterRecord::capture(self,layout,rank,base_id)
    }
}
