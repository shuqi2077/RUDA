use super::{Backend,DType,LayerCache,Qwen35Cache,Tensor,TensorData};
use crate::GenerationCacheContinuation;
use ruda_model::record::{PrecisionSettings,Record,Recorder,RecorderError};
use std::marker::PhantomData;

/// Actual heterogeneous layer history, including recurrent and convolution state.
/// Each tensor retains its own native storage; FP32 recurrent state is not narrowed
/// to the FP16/BF16 convolution or full-attention cache dtype.
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
pub enum Qwen35LayerCacheState {
    /// Native full-attention K/V in their original [batch,heads,position,width] axes.
    Full {key: Option<TensorData>,value: Option<TensorData>},
    /// Native [batch,channels,window] history and FP32 [batch,heads,key,value] state.
    Delta {convolution: Option<TensorData>,state: Option<TensorData>},
}

/// Frozen exact-storage Qwen3.5 cache, original layer order and input position.
pub struct Qwen35CacheRecord<B: Backend> {
    version: u32,
    model_id: String,
    position: usize,
    batch: Option<usize>,
    layers: Vec<Qwen35LayerCacheState>,
    backend: PhantomData<B>,
}

impl<B: Backend> Record<B> for Qwen35CacheRecord<B> {
    type Item<S: PrecisionSettings> = (u32,String,usize,Option<usize>,Vec<Qwen35LayerCacheState>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {(self.version,self.model_id,self.position,self.batch,self.layers)}
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,_device: &B::Device) -> Self {
        Self {version:item.0,model_id:item.1,position:item.2,batch:item.3,layers:item.4,backend:PhantomData}
    }
}

fn invalid(reason: &str) -> RecorderError {RecorderError::Unknown(format!("invalid hybrid decoder cache continuation: {reason}"))}

fn check_data(data: &TensorData,rank: usize,batch: usize) -> Result<(),RecorderError> {
    if data.rank() != rank || data.shape[0] != batch || data.shape.iter().any(|&size|size == 0)
        || !matches!(data.dtype,DType::F16|DType::BF16|DType::F32|DType::Flex32|DType::F64) {
        return Err(invalid("layer payload has invalid native storage, rank or batch geometry"));
    }
    let bytes = data.shape.iter().try_fold(1usize,|count,&dimension|count.checked_mul(dimension))
        .and_then(|count|count.checked_mul(data.dtype.size())).ok_or_else(||invalid("native cache byte count overflow"))?;
    if data.as_bytes().len() != bytes {return Err(invalid("layer payload byte count differs from actual native storage"));}
    Ok(())
}

fn check_layer(layer: &Qwen35LayerCacheState,position: usize,batch: Option<usize>) -> Result<(),RecorderError> {
    match layer {
        Qwen35LayerCacheState::Full {key:None,value:None} | Qwen35LayerCacheState::Delta {convolution:None,state:None}
            if position == 0 && batch.is_none() => Ok(()),
        Qwen35LayerCacheState::Full {key:Some(key),value:Some(value)} => {
            let batch = batch.filter(|&batch|batch > 0).ok_or_else(||invalid("completed cache is missing its batch"))?;
            check_data(key,4,batch)?; check_data(value,4,batch)?;
            if key.shape != value.shape || key.dtype != value.dtype || key.shape[2] != position {
                return Err(invalid("full-attention K/V do not share the actual completed geometry/dtype"));
            }
            Ok(())
        }
        Qwen35LayerCacheState::Delta {convolution:Some(convolution),state:Some(state)} => {
            let batch = batch.filter(|&batch|batch > 0).ok_or_else(||invalid("completed recurrent cache is missing its batch"))?;
            check_data(convolution,3,batch)?; check_data(state,4,batch)?;
            if position == 0 || state.dtype != DType::F32 {return Err(invalid("recurrent history requires a completed position and original FP32 state"));}
            let values = state.shape[1].checked_mul(state.shape[3]).ok_or_else(||invalid("recurrent value channel overflow"))?;
            let keys = convolution.shape[1].checked_sub(values).ok_or_else(||invalid("convolution channels are smaller than actual value channels"))?;
            let divisor = state.shape[2].checked_mul(2).ok_or_else(||invalid("recurrent key channel overflow"))?;
            if keys == 0 || keys % divisor != 0 || state.shape[1] % (keys/divisor) != 0 {
                return Err(invalid("convolution and recurrent key/value head geometry differs"));
            }
            Ok(())
        }
        _ => Err(invalid("layer has incomplete or inconsistent native history")),
    }
}

impl<B: Backend> Qwen35CacheRecord<B> {
    /// Capture the caller's completed full/recurrent cache with native readback.
    /// A model failure must be recovered from its previous completed record.
    /// Model identity includes the actual projections, normalization and position setup.
    pub fn capture(cache: &Qwen35Cache<B>,model_id: &str) -> Result<Self,RecorderError> {
        if model_id.is_empty() || cache.layers.is_empty() {return Err(invalid("original model identity and actual layers are required"));}
        let mut layers = Vec::with_capacity(cache.layers.len());
        for layer in &cache.layers {
            let layer = match layer {
                LayerCache::Full {key,value} => Qwen35LayerCacheState::Full {
                    key:key.as_ref().map(|value|value.clone().try_into_data()).transpose().map_err(|error|invalid(&error.to_string()))?,
                    value:value.as_ref().map(|value|value.clone().try_into_data()).transpose().map_err(|error|invalid(&error.to_string()))?,
                },
                LayerCache::Delta {convolution,state} => Qwen35LayerCacheState::Delta {
                    convolution:convolution.as_ref().map(|value|value.clone().try_into_data()).transpose().map_err(|error|invalid(&error.to_string()))?,
                    state:state.as_ref().map(|value|value.clone().try_into_data()).transpose().map_err(|error|invalid(&error.to_string()))?,
                },
            };
            check_layer(&layer,cache.position,cache.batch)?;
            layers.push(layer);
        }
        Ok(Self {version:1,model_id:model_id.into(),position:cache.position,batch:cache.batch,layers,backend:PhantomData})
    }

    /// Save exact raw layer tensors, not recorder-narrowed homogeneous weights.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load frozen host state without allocating any decoder cache tensors.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}

    /// Restore original layer variants against this actual model's fresh cache.
    /// No full-attention history is substituted for recurrent state or vice versa.
    pub fn restore(self,model_id: &str,template: &Qwen35Cache<B>,device: &B::Device) -> Result<Qwen35Cache<B>,RecorderError> {
        if self.version != 1 || model_id.is_empty() || self.model_id != model_id || self.layers.len() != template.layers.len()
            || self.layers.is_empty() || template.position != 0 || template.batch.is_some() {
            return Err(invalid("record version, exact model identity or fresh layer layout differs"));
        }
        for (layer,expected) in self.layers.iter().zip(&template.layers) {
            if !matches!((layer,expected),(Qwen35LayerCacheState::Full {..},LayerCache::Full {..})
                | (Qwen35LayerCacheState::Delta {..},LayerCache::Delta {..})) {
                return Err(invalid("saved full/recurrent layer order differs from the actual model"));
            }
            check_layer(layer,self.position,self.batch)?;
        }
        let layers = self.layers.into_iter().map(|layer|match layer {
            Qwen35LayerCacheState::Full {key,value} => LayerCache::Full {
                key:key.map(|data| {let dtype = data.dtype;Tensor::from_data(data,(device,dtype))}),
                value:value.map(|data| {let dtype = data.dtype;Tensor::from_data(data,(device,dtype))}),
            },
            Qwen35LayerCacheState::Delta {convolution,state} => LayerCache::Delta {
                convolution:convolution.map(|data| {let dtype = data.dtype;Tensor::from_data(data,(device,dtype))}),
                state:state.map(|data| {let dtype = data.dtype;Tensor::from_data(data,(device,dtype))}),
            },
        }).collect();
        Ok(Qwen35Cache {layers,position:self.position,batch:self.batch})
    }
}

impl<B: Backend> Qwen35Cache<B> {
    /// Freeze every actual full/recurrent layer at the current completed input position.
    pub fn record(&self,model_id: &str) -> Result<Qwen35CacheRecord<B>,RecorderError> {Qwen35CacheRecord::capture(self,model_id)}
}

impl<B: Backend> GenerationCacheContinuation<B> for Qwen35Cache<B> {
    type Record = Qwen35CacheRecord<B>;
    fn generation_position(&self) -> usize {self.position}
    fn capture_generation(&self,model_id: &str,_device: &B::Device) -> Result<Self::Record,RecorderError> {self.record(model_id)}
    fn restore_generation(record: Self::Record,model_id: &str,template: &Self,position: usize,device: &B::Device)
        -> Result<Self,RecorderError> {
        if record.position != position {return Err(invalid("hybrid cache and generation input counts differ"));}
        record.restore(model_id,template,device)
    }
}
