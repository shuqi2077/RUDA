use alloc::{format,string::String};
use ruda_model::{record::{PrecisionSettings,Record,Recorder,RecorderError},
    tensor::{Bool,DType,Element,Tensor,TensorData,backend::Backend}};
use super::ProjectedKvCache;

enum Payload<B: Backend> {
    Empty,
    Device(Tensor<B,4>,Tensor<B,4>,Tensor<B,2,Bool>),
    Stored(TensorData,TensorData,TensorData),
}

/// Exact-storage inference history, bound to an explicit original model/adapter identity.
/// Raw cache payload storage is preserved independently of recorder float precision.
pub struct ProjectedKvCacheRecord<B: Backend> {
    version: u32,
    model_id: String,
    start_position: usize,
    initial_capacity: usize,
    payload: Payload<B>,
}

impl<B: Backend> Record<B> for ProjectedKvCacheRecord<B> {
    type Item<S: PrecisionSettings> = (u32,String,usize,usize,Option<(TensorData,TensorData,TensorData)>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        let payload = match self.payload {
            Payload::Empty=>None,
            Payload::Device(key,value,visible)=>Some((key.into_data(),value.into_data(),visible.into_data())),
            Payload::Stored(key,value,visible)=>Some((key,value,visible)),
        };
        (self.version,self.model_id,self.start_position,self.initial_capacity,payload)
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,_device: &B::Device) -> Self {
        Self {version:item.0,model_id:item.1,start_position:item.2,initial_capacity:item.3,
            payload:match item.4 {None=>Payload::Empty,Some((key,value,visible))=>Payload::Stored(key,value,visible)}}
    }
}

fn invalid(reason: &str) -> RecorderError { RecorderError::Unknown(format!("Invalid native projected KV record: {reason}")) }

fn check_data(data: &TensorData) -> Result<(),RecorderError> {
    let elements = data.shape.iter().try_fold(1usize,|count,&dimension|count.checked_mul(dimension))
        .ok_or_else(||invalid("cache payload shape overflows element count"))?;
    let bytes = elements.checked_mul(data.dtype.size()).ok_or_else(||invalid("cache payload byte count overflow"))?;
    if data.as_bytes().len() != bytes { return Err(invalid("cache payload byte count differs from declared storage")); }
    if data.dtype == <bool as Element>::dtype() {
        data.as_slice::<bool>().map_err(|_|invalid("invalid native boolean cache visibility storage"))?;
    }
    Ok(())
}

impl<B: Backend> ProjectedKvCacheRecord<B> {
    /// Capture only actual retained slots; unused reserves and model weights are not saved.
    /// No payload readback occurs until a native recorder serializes the captured record.
    pub fn capture(cache: &ProjectedKvCache<B>,model_id: &str) -> Result<Self,RecorderError> {
        if model_id.is_empty() { return Err(invalid("exact model/adapter identity must be supplied explicitly")); }
        let payload = match cache.prefix() {Some((key,value,visible))=>Payload::Device(key,value,visible),None=>Payload::Empty};
        Ok(Self {version:1,model_id:model_id.into(),start_position:cache.start_position,
            initial_capacity:cache.initial_capacity,payload})
    }

    /// Save exact original FP16/BF16/FP32/FP64 cache values and actual visibility.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {
        recorder.record(self,args)
    }

    /// Read raw native history; device allocation is deferred until checked restoration.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {
        recorder.load(args,device)
    }

    /// Restore retained K/V dtype, validity and absolute positions on an explicit device.
    /// Cache data is neither requantized nor converted through the backend default float.
    pub fn restore(self,model_id: &str,device: &B::Device) -> Result<ProjectedKvCache<B>,RecorderError> {
        if self.version != 1 || self.model_id != model_id || model_id.is_empty() { return Err(invalid("version or exact model/adapter identity differs")); }
        let mut cache = match self.payload {
            Payload::Empty=>ProjectedKvCache::new(self.initial_capacity).with_start_position(self.start_position),
            Payload::Device(key,value,visible)=>{
                let length = key.dims()[2];
                self.start_position.checked_add(length).ok_or_else(||invalid("absolute cache position overflow"))?;
                ProjectedKvCache::from_projected(key.to_device(device),value.to_device(device),Some(visible.to_device(device)),self.start_position)
            },
            Payload::Stored(key,value,visible)=>{
                if key.rank() != 4 || value.rank() != 4 || visible.rank() != 2 { return Err(invalid("cache payload ranks differ")); }
                if !matches!(key.dtype,DType::F16|DType::BF16|DType::F32|DType::F64)
                    || value.dtype != key.dtype || !matches!(visible.dtype,DType::Bool(_)) {
                    return Err(invalid("cache payload storage differs from matching floating K/V and Bool visibility"));
                }
                let shape = &key.shape;
                if shape[1] == 0 || shape[3] == 0 || value.shape[3] == 0
                    || shape[..3] != value.shape[..3] || visible.shape.as_slice() != [shape[0],shape[2]] {
                    return Err(invalid("actual cache K/V/visibility geometry differs"));
                }
                self.start_position.checked_add(shape[2]).ok_or_else(||invalid("absolute cache position overflow"))?;
                check_data(&key)?; check_data(&value)?; check_data(&visible)?;
                let dtype = key.dtype;
                let visible_dtype = visible.dtype;
                ProjectedKvCache::from_projected(Tensor::from_data(key,(device,dtype)),Tensor::from_data(value,(device,dtype)),
                    Some(Tensor::from_data(visible,(device,visible_dtype))),self.start_position)
            },
        };
        cache.initial_capacity = self.initial_capacity;
        Ok(cache)
    }
}

impl<B: Backend> ProjectedKvCache<B> {
    /// Snapshot actual projected inference state with an explicit model/adapter identity.
    pub fn record(&self,model_id: &str) -> Result<ProjectedKvCacheRecord<B>,RecorderError> {
        ProjectedKvCacheRecord::capture(self,model_id)
    }
}
