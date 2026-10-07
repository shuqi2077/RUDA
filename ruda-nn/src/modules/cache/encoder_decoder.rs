use alloc::{format,string::String,vec::Vec};
use ruda_model::{record::{PrecisionSettings,Record,Recorder,RecorderError},tensor::{Int,Tensor,backend::Backend}};
use super::{ProjectedKvCache,ProjectedKvCacheRecord,TransformerKvCache,TransformerKvCacheRecord};

/// Paired native decoder history and per-layer immutable projected encoder memory.
#[derive(Clone,Debug)]
pub struct EncoderDecoderKvCache<B: Backend> {
    decoder: TransformerKvCache<B>,
    memory: Vec<ProjectedKvCache<B>>,
}

impl<B: Backend> EncoderDecoderKvCache<B> {
    /// Connect actual prepared per-layer memory and original decoder cache ordering.
    pub fn new(decoder: TransformerKvCache<B>,memory: Vec<ProjectedKvCache<B>>) -> Self {
        let cache = Self {decoder,memory};
        assert!(cache.is_consistent(),"encoder-decoder cache layer counts, prepared rows or completed positions differ");
        cache
    }

    /// Completed decoder physical slot position, independent of encoder source length.
    pub fn position(&self) -> usize { self.decoder.position() }

    /// Actual original decoder layer state without discarding any source memory.
    pub fn decoder(&self) -> &TransformerKvCache<B> { &self.decoder }

    /// Actual per-layer projected encoder memory in original architecture order.
    pub fn memory(&self) -> &[ProjectedKvCache<B>] { &self.memory }

    /// Split actual mutable decoder state and immutable encoder state for native forward.
    pub fn parts_mut(&mut self) -> (&mut TransformerKvCache<B>,&[ProjectedKvCache<B>]) {
        (&mut self.decoder,&self.memory)
    }

    /// Metadata consistency of actual paired rows/layers and completed decoder positions.
    pub fn is_consistent(&self) -> bool {
        if !self.decoder.is_synchronised() || self.memory.len() != self.decoder.layers().len() { return false; }
        let rows = self.memory.first().and_then(ProjectedKvCache::batch_size);
        self.memory.iter().zip(self.decoder.layers()).all(|(memory,decoder)|
            memory.is_initialized() && memory.batch_size() == rows
                && decoder.batch_size().is_none_or(|batch|Some(batch) == memory.batch_size()))
    }

    /// Reorder/duplicate both source and decoder rows together for actual beam parents.
    pub fn reordered(&self,parents: Tensor<B,1,Int>) -> Self {
        Self::new(self.decoder.reordered(parents.clone()),self.memory.iter().map(|memory|memory.reordered(parents.clone())).collect())
    }

    /// Explicit speculative decoder rollback; immutable encoder memory remains unchanged.
    pub fn rollback_to(&mut self,position: usize) { self.decoder.rollback_to(position); }

    /// Restart decoder generation on the same actual prepared encoder memory.
    pub fn clear_decoder(&mut self) { self.decoder.clear(); }

    /// Move both sides' actual retained tensors without copying unused reserve slots.
    pub fn to_device(self,device: &B::Device) -> Self {
        Self::new(self.decoder.to_device(device),self.memory.into_iter().map(|memory|memory.to_device(device)).collect())
    }

    /// Capture both actual cache sides with an explicit exact model/adapter identity.
    pub fn record(&self,model_id: &str) -> Result<EncoderDecoderKvCacheRecord<B>,RecorderError> {
        EncoderDecoderKvCacheRecord::capture(self,model_id)
    }
}

/// Exact-storage native encoder/decoder cache continuation record, without model parameters.
pub struct EncoderDecoderKvCacheRecord<B: Backend> {
    version: u32,
    model_id: String,
    decoder: TransformerKvCacheRecord<B>,
    memory: Vec<ProjectedKvCacheRecord<B>>,
}

impl<B: Backend> Record<B> for EncoderDecoderKvCacheRecord<B> {
    type Item<S: PrecisionSettings> = (u32,String,<TransformerKvCacheRecord<B> as Record<B>>::Item<S>,
        Vec<<ProjectedKvCacheRecord<B> as Record<B>>::Item<S>>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.model_id,self.decoder.into_item::<S>(),self.memory.into_iter().map(|memory|memory.into_item::<S>()).collect())
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {version:item.0,model_id:item.1,decoder:TransformerKvCacheRecord::<B>::from_item::<S>(item.2,device),
            memory:item.3.into_iter().map(|memory|ProjectedKvCacheRecord::<B>::from_item::<S>(memory,device)).collect()}
    }
}

fn invalid(reason: &str) -> RecorderError { RecorderError::Unknown(format!("Invalid native encoder-decoder KV record: {reason}")) }

impl<B: Backend> EncoderDecoderKvCacheRecord<B> {
    /// Capture complete actual source/decoder inference state, without inferred source IDs.
    pub fn capture(cache: &EncoderDecoderKvCache<B>,model_id: &str) -> Result<Self,RecorderError> {
        if !cache.is_consistent() || model_id.is_empty() { return Err(invalid("complete paired cache and explicit model identity are required")); }
        let decoder = cache.decoder.record(model_id)?;
        let memory = cache.memory.iter().map(|memory|memory.record(model_id)).collect::<Result<Vec<_>,_>>()?;
        Ok(Self {version:1,model_id:model_id.into(),decoder,memory})
    }

    /// Save exact native retained source and decoder payloads through an existing recorder.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {
        recorder.record(self,args)
    }

    /// Read raw native paired state; checked restoration performs the device allocation.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {
        recorder.load(args,device)
    }

    /// Restore exact actual layer/model identity, original projected values and paired rows.
    pub fn restore(self,model_id: &str,layers: usize,device: &B::Device) -> Result<EncoderDecoderKvCache<B>,RecorderError> {
        if self.version != 1 || self.model_id != model_id || model_id.is_empty() || self.memory.len() != layers {
            return Err(invalid("version, exact model/adapter identity or actual source layer count differs"));
        }
        let decoder = self.decoder.restore(model_id,layers,device)?;
        let memory = self.memory.into_iter().map(|memory|memory.restore(model_id,device)).collect::<Result<Vec<_>,_>>()?;
        let cache = EncoderDecoderKvCache {decoder,memory};
        if !cache.is_consistent() { return Err(invalid("restored actual source/decoder rows or completed layers differ")); }
        Ok(cache)
    }
}
