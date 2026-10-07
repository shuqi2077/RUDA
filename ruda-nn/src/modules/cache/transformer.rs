use alloc::{format,string::String,vec::Vec};
use ruda_model::{record::{PrecisionSettings,Record,Recorder,RecorderError},tensor::{Int,Tensor,backend::Backend}};
use super::{ProjectedKvCache,ProjectedKvCacheRecord};

/// Actual per-layer projected inference state and completed physical token position.
#[derive(Clone,Debug)]
pub struct TransformerKvCache<B: Backend> {
    layers: Vec<ProjectedKvCache<B>>,
    position: usize,
}

impl<B: Backend> TransformerKvCache<B> {
    /// Prepare exactly the caller's actual layer count with no tensor allocations.
    pub fn new(layers: usize,initial_capacity: usize) -> Self {
        Self::new_at_position(layers,initial_capacity,0)
    }

    /// Explicit initial absolute slot position; no model context/window rule is inferred.
    pub fn new_at_position(layers: usize,initial_capacity: usize,position: usize) -> Self {
        Self {layers:(0..layers).map(|_|ProjectedKvCache::new(initial_capacity).with_start_position(position)).collect(),position}
    }

    /// Next actual physical input position after the last complete stack chunk.
    pub fn position(&self) -> usize { self.position }

    /// Actual original layer caches, in unchanged architecture order.
    pub fn layers(&self) -> &[ProjectedKvCache<B>] { &self.layers }

    /// Explicit caller-owned per-layer prefix/window operations and projected state.
    pub fn layers_mut(&mut self) -> &mut [ProjectedKvCache<B>] { &mut self.layers }

    /// True when every actual layer ends at the completed stack position.
    pub fn is_synchronised(&self) -> bool { self.layers.iter().all(|layer|layer.position() == self.position) }

    /// Check exact layer count and completed positions before processing another chunk.
    pub fn validate_layers(&self,layers: usize) {
        assert_eq!(self.layers.len(),layers,"native cache and actual transformer layer counts differ");
        assert!(self.is_synchronised(),"native cache layers do not share a completed stack position");
    }

    pub(crate) fn finish_chunk(&mut self,position: usize) {
        assert!(position >= self.position,"completed cached forward cannot move position backwards");
        assert!(self.layers.iter().all(|layer|layer.position() == position),"cached layer did not append the actual complete input chunk");
        self.position = position;
    }

    /// Explicitly roll back completed speculative slots still retained in every layer.
    /// Validate all retained ranges before dropping any suffix; discarded prefixes cannot be recovered.
    pub fn rollback_to(&mut self,position: usize) {
        assert!(position <= self.position,"cannot recover unknown future cache history");
        for layer in &self.layers {
            assert!(position >= layer.start_position() && position <= layer.position(),"rollback target is outside a retained layer range");
        }
        for layer in &mut self.layers { layer.truncate(position-layer.start_position()); }
        self.position = position;
    }

    /// Select/duplicate actual parent rows in every layer for caller-controlled beam search.
    pub fn reordered(&self,parents: Tensor<B,1,Int>) -> Self {
        Self {layers:self.layers.iter().map(|layer|layer.reordered(parents.clone())).collect(),position:self.position}
    }

    /// Move only actual retained per-layer inference tensors to the selected device.
    pub fn to_device(self,device: &B::Device) -> Self {
        Self {layers:self.layers.into_iter().map(|layer|layer.to_device(device)).collect(),position:self.position}
    }

    /// Clear all history and completed positions without changing original model parameters.
    pub fn clear(&mut self) {
        for layer in &mut self.layers { layer.clear(); }
        self.position = 0;
    }

    /// Capture actual completed history; partially advanced layer sets cannot be snapshotted.
    pub fn record(&self,model_id: &str) -> Result<TransformerKvCacheRecord<B>,RecorderError> {
        TransformerKvCacheRecord::capture(self,model_id)
    }
}

/// Native exact-storage snapshot of every actual layer and its completed stack position.
pub struct TransformerKvCacheRecord<B: Backend> {
    version: u32,
    model_id: String,
    position: usize,
    layers: Vec<ProjectedKvCacheRecord<B>>,
}

impl<B: Backend> Record<B> for TransformerKvCacheRecord<B> {
    type Item<S: PrecisionSettings> = (u32,String,usize,Vec<<ProjectedKvCacheRecord<B> as Record<B>>::Item<S>>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.model_id,self.position,self.layers.into_iter().map(|layer|layer.into_item::<S>()).collect())
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {version:item.0,model_id:item.1,position:item.2,
            layers:item.3.into_iter().map(|layer|ProjectedKvCacheRecord::<B>::from_item::<S>(layer,device)).collect()}
    }
}

fn invalid(reason: &str) -> RecorderError { RecorderError::Unknown(format!("Invalid native transformer KV record: {reason}")) }

impl<B: Backend> TransformerKvCacheRecord<B> {
    /// Capture only actual retained projected history, with an explicit complete model identity.
    pub fn capture(cache: &TransformerKvCache<B>,model_id: &str) -> Result<Self,RecorderError> {
        if model_id.is_empty() || !cache.is_synchronised() { return Err(invalid("exact model identity and a completed layer set are required")); }
        let layers = cache.layers.iter().map(|layer|layer.record(model_id)).collect::<Result<Vec<_>,_>>()?;
        Ok(Self {version:1,model_id:model_id.into(),position:cache.position,layers})
    }

    /// Save native retained tensors and completed positions, without model weight copies.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {
        recorder.record(self,args)
    }

    /// Read raw native cache state; tensors allocate only during checked restoration.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {
        recorder.load(args,device)
    }

    /// Restore exact original layer count/model identity and completed slot positions.
    pub fn restore(self,model_id: &str,layers: usize,device: &B::Device) -> Result<TransformerKvCache<B>,RecorderError> {
        if self.version != 1 || self.model_id != model_id || model_id.is_empty() || self.layers.len() != layers {
            return Err(invalid("version, exact model/adapter identity or actual layer count differs"));
        }
        let layers = self.layers.into_iter().map(|layer|layer.restore(model_id,device)).collect::<Result<Vec<_>,_>>()?;
        let cache = TransformerKvCache {layers,position:self.position};
        if !cache.is_synchronised() { return Err(invalid("restored layers have inconsistent completed positions")); }
        Ok(cache)
    }
}
