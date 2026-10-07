use super::{Backend,Qwen35CacheRecord,Qwen35MultimodalCache};
use crate::GenerationCacheContinuation;
use ruda_model::record::{PrecisionSettings,Record,Recorder,RecorderError};

/// Original mixed text cache and each row's actual next three-axis RoPE position.
/// Image-compressed positions are saved directly, not inferred from physical length.
pub struct Qwen35MultimodalCacheRecord<B: Backend> {
    version: u32,
    model_id: String,
    text: Qwen35CacheRecord<B>,
    next_positions: Vec<usize>,
}

impl<B: Backend> Record<B> for Qwen35MultimodalCacheRecord<B> {
    type Item<S: PrecisionSettings> = (u32,String,<Qwen35CacheRecord<B> as Record<B>>::Item<S>,Vec<usize>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.model_id,self.text.into_item::<S>(),self.next_positions)
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {version:item.0,model_id:item.1,text:Qwen35CacheRecord::from_item::<S>(item.2,device),next_positions:item.3}
    }
}

fn invalid(reason: &str) -> RecorderError {RecorderError::Unknown(format!("invalid multimodal generation continuation: {reason}"))}

fn check_cache<B: Backend>(cache: &Qwen35MultimodalCache<B>) -> Result<(),RecorderError> {
    if cache.sequence_length() == 0 {
        if !cache.next_positions.is_empty() {return Err(invalid("empty text cache contains image-prefill positions"));}
    } else if cache.text.batch != Some(cache.next_positions.len()) || cache.next_positions.is_empty() || cache.next_positions.contains(&0) {
        return Err(invalid("actual source rows and image-compressed positions differ"));
    }
    Ok(())
}

impl<B: Backend> Qwen35MultimodalCacheRecord<B> {
    /// Capture completed actual image-conditioned history and exact position cursors.
    /// Does not save original images/vision parameters or encode images again.
    pub fn capture(cache: &Qwen35MultimodalCache<B>,model_id: &str) -> Result<Self,RecorderError> {
        if model_id.is_empty() {return Err(invalid("exact text/vision/source identity is required"));}
        check_cache(cache)?;
        Ok(Self {version:1,model_id:model_id.into(),text:cache.text.record(model_id)?,next_positions:cache.next_positions.clone()})
    }
    /// Save the original text cache and virtual position cursors as one native record.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load exact host state without allocating decoder cache tensors.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}
    /// Restore against an actual fresh model cache, retaining original MRoPE cursors.
    pub fn restore(self,model_id: &str,template: &Qwen35MultimodalCache<B>,device: &B::Device)
        -> Result<Qwen35MultimodalCache<B>,RecorderError> {
        if self.version != 1 || model_id.is_empty() || self.model_id != model_id || !template.next_positions.is_empty() {
            return Err(invalid("record version, exact source/model identity or fresh cache differs"));
        }
        let cache = Qwen35MultimodalCache {text:self.text.restore(model_id,&template.text,device)?,next_positions:self.next_positions};
        check_cache(&cache)?;
        Ok(cache)
    }
}

impl<B: Backend> Qwen35MultimodalCache<B> {
    /// Snapshot original mixed text state and actual image-prefill position cursors.
    pub fn record(&self,model_id: &str) -> Result<Qwen35MultimodalCacheRecord<B>,RecorderError> {Qwen35MultimodalCacheRecord::capture(self,model_id)}
}

impl<B: Backend> GenerationCacheContinuation<B> for Qwen35MultimodalCache<B> {
    type Record = Qwen35MultimodalCacheRecord<B>;
    fn generation_position(&self) -> usize {self.sequence_length()}
    fn capture_generation(&self,model_id: &str,_device: &B::Device) -> Result<Self::Record,RecorderError> {self.record(model_id)}
    fn restore_generation(record: Self::Record,model_id: &str,template: &Self,position: usize,device: &B::Device)
        -> Result<Self,RecorderError> {
        let cache = record.restore(model_id,template,device)?;
        if cache.sequence_length() != position {return Err(invalid("text cache and actual generation input positions differ"));}
        Ok(cache)
    }
}
