use ruda_model::tensor::{Bool,DType,Int,Tensor,backend::Backend};

/// Native projected K/V history for explicit incremental inference.
/// Keys are stored after the caller's positional transform, with no model-family rule.
/// Growth uses native slice assignment; backend ownership still determines buffer reuse.
#[derive(Clone,Debug)]
pub struct ProjectedKvCache<B: Backend> {
    pub(super) key: Option<Tensor<B,4>>,
    pub(super) value: Option<Tensor<B,4>>,
    pub(super) visible: Option<Tensor<B,2,Bool>>,
    pub(super) length: usize,
    pub(super) capacity: usize,
    pub(super) start_position: usize,
    pub(super) initial_capacity: usize,
}

impl<B: Backend> ProjectedKvCache<B> {
    /// Empty cache with caller-selected first reserve; zero reserves exactly the first chunk.
    /// No tensor allocation, hidden context limit or model-specific position occurs here.
    pub fn new(initial_capacity: usize) -> Self {
        Self {key:None,value:None,visible:None,length:0,capacity:0,start_position:0,initial_capacity}
    }

    /// Actual retained physical token slots, including explicitly masked input slots.
    pub fn len(&self) -> usize { self.length }

    /// Whether the retained prefix has no token slots.
    pub fn is_empty(&self) -> bool { self.length == 0 }

    /// Whether actual K/V geometry has been supplied, including an initialized zero-length memory.
    pub fn is_initialized(&self) -> bool { self.key.is_some() }

    /// Actual batch-row count when payload geometry is known; no tensor readback occurs.
    pub fn batch_size(&self) -> Option<usize> { self.key.as_ref().map(|key|key.dims()[0]) }

    /// Actual allocated token-axis capacity, not the absolute model context length.
    pub fn capacity(&self) -> usize { self.capacity }

    /// Caller-owned absolute position of the first retained physical token slot.
    pub fn start_position(&self) -> usize { self.start_position }

    /// Absolute position of the next appended physical token slot.
    pub fn position(&self) -> usize { self.start_position.checked_add(self.length).expect("cache position overflow") }

    /// Set the absolute start on an empty cache, without modifying any positional formula.
    pub fn with_start_position(mut self,position: usize) -> Self {
        assert!(self.length == 0,"cannot change the position of retained projected history");
        self.start_position = position;
        self
    }

    /// Retained actual K/V and validity, excluding all unused physical reserve slots.
    pub fn prefix(&self) -> Option<(Tensor<B,4>,Tensor<B,4>,Tensor<B,2,Bool>)> {
        let key = self.key.as_ref()?;
        let value = self.value.as_ref().expect("cache value payload must accompany keys");
        let visible = self.visible.as_ref().expect("cache visibility must accompany keys");
        let [batch,heads,_,key_width] = key.dims();
        let value_width = value.dims()[3];
        Some((key.clone().slice([0..batch,0..heads,0..self.length,0..key_width]),
            value.clone().slice([0..batch,0..heads,0..self.length,0..value_width]),
            visible.clone().slice([0..batch,0..self.length])))
    }

    /// Check actual new projected geometry/storage/visibility before modifying history.
    pub fn validate_append(&self,key: &Tensor<B,4>,value: &Tensor<B,4>,visible: Option<&Tensor<B,2,Bool>>) {
        let [batch,heads,tokens,width] = key.dims();
        let value_shape = value.dims();
        assert!(heads > 0 && width > 0 && value_shape[3] > 0,"cache head/feature dimensions must be positive");
        assert_eq!((batch,heads,tokens),(value_shape[0],value_shape[1],value_shape[2]),"cache new key/value geometry differs");
        assert!(matches!(key.dtype(),DType::F16|DType::BF16|DType::F32|DType::F64)
            && key.dtype() == value.dtype(),"cache requires matching floating K/V storage");
        assert_eq!(key.device(),value.device(),"cache new K/V devices differ");
        if let Some(visible) = visible {
            assert_eq!(visible.dims(),[batch,tokens],"new cache visibility must describe only the actual appended slots");
            assert_eq!(visible.device(),key.device(),"cache new visibility device differs");
            if let Some(previous) = &self.visible {
                assert_eq!(visible.dtype(),previous.dtype(),"cache visibility storage cannot change during append");
            }
        }
        let end = self.length.checked_add(tokens).expect("cache length overflow");
        self.start_position.checked_add(end).expect("cache absolute position overflow");
        if let Some(previous) = &self.key {
            let previous_value = self.value.as_ref().expect("cache value payload missing");
            assert_eq!(previous.dims(),[batch,heads,self.capacity,width],"new cache key layout differs from retained history");
            assert_eq!(previous_value.dims(),[batch,heads,self.capacity,value_shape[3]],"new cache value layout differs from retained history");
            assert_eq!(previous.dtype(),key.dtype(),"cache key storage cannot change during append");
            assert_eq!(previous_value.dtype(),value.dtype(),"cache value storage cannot change during append");
            assert_eq!(previous.device(),key.device(),"cache append must use the retained device");
        }
    }

    /// Append actual already-positioned K/V chunks, never just the final token of a chunk.
    /// None declares every new slot visible. Cached K/V are detached inference state;
    /// this API does not claim full-history training derivatives or fused attention.
    pub fn append(&mut self,key: Tensor<B,4>,value: Tensor<B,4>,visible: Option<Tensor<B,2,Bool>>)
        -> (Tensor<B,4>,Tensor<B,4>,Tensor<B,2,Bool>) {
        self.validate_append(&key,&value,visible.as_ref());
        let [batch,heads,tokens,key_width] = key.dims();
        let value_width = value.dims()[3];
        let end = self.length+tokens;
        let device = key.device();
        let dtype = key.dtype();
        let visible = visible.unwrap_or_else(|| {
            if let Some(previous) = &self.visible { Tensor::<B,2,Bool>::zeros([batch,tokens],(&device,previous.dtype())).bool_not() }
            else { Tensor::<B,2,Bool>::zeros([batch,tokens],&device).bool_not() }
        });
        if self.key.is_none() && (end >= self.initial_capacity || tokens == 0) {
            self.key = Some(key.detach());
            self.value = Some(value.detach());
            self.visible = Some(visible);
            self.length = end;
            self.capacity = end;
            return self.prefix().unwrap();
        }
        if tokens == 0 { return self.prefix().unwrap(); }
        let (mut key_storage,mut value_storage,mut valid_storage) = if self.capacity < end {
            let mut capacity = self.capacity.max(self.initial_capacity).max(1);
            while capacity < end { capacity = capacity.checked_mul(2).unwrap_or(end); }
            let mut keys = Tensor::empty([batch,heads,capacity,key_width],(&device,dtype));
            let mut values = Tensor::empty([batch,heads,capacity,value_width],(&device,dtype));
            let mut validity = Tensor::<B,2,Bool>::zeros([batch,capacity],(&device,visible.dtype()));
            if self.length > 0 {
                let (previous_key,previous_value,previous_valid) = self.prefix().unwrap();
                keys.inplace(|storage|storage.slice_assign([0..batch,0..heads,0..self.length,0..key_width],previous_key));
                values.inplace(|storage|storage.slice_assign([0..batch,0..heads,0..self.length,0..value_width],previous_value));
                validity.inplace(|storage|storage.slice_assign([0..batch,0..self.length],previous_valid));
            }
            (keys,values,validity)
        } else {
            (self.key.take().unwrap(),self.value.take().unwrap(),self.visible.take().unwrap())
        };
        key_storage.inplace(|storage|storage.slice_assign([0..batch,0..heads,self.length..end,0..key_width],key.detach()));
        value_storage.inplace(|storage|storage.slice_assign([0..batch,0..heads,self.length..end,0..value_width],value.detach()));
        valid_storage.inplace(|storage|storage.slice_assign([0..batch,self.length..end],visible));
        self.capacity = key_storage.dims()[2];
        self.length = end;
        self.key = Some(key_storage);
        self.value = Some(value_storage);
        self.visible = Some(valid_storage);
        self.prefix().unwrap()
    }

    /// Connect caller-projected actual memory without reinitializing its values.
    pub fn from_projected(key: Tensor<B,4>,value: Tensor<B,4>,visible: Option<Tensor<B,2,Bool>>,start_position: usize) -> Self {
        let mut cache = Self::new(0).with_start_position(start_position);
        cache.append(key,value,visible);
        cache
    }

    /// Explicitly discard a retained suffix, for caller-controlled decode rollback.
    /// Future append overwrites discarded slots; no discarded value is exposed by prefix.
    pub fn truncate(&mut self,length: usize) {
        assert!(length <= self.length,"cannot restore token history that this cache does not retain");
        self.length = length;
    }

    /// Explicitly drop old prefix slots while keeping their absolute position offset.
    /// This does not infer a sliding-window/sink policy or silently discard history.
    pub fn drop_prefix(&mut self,tokens: usize) {
        assert!(tokens <= self.length,"cannot drop more cache slots than retained");
        if tokens == 0 { return; }
        let start = self.start_position.checked_add(tokens).expect("cache prefix position overflow");
        let key = self.key.take().unwrap();
        let value = self.value.take().unwrap();
        let visible = self.visible.take().unwrap();
        let [batch,heads,_,key_width] = key.dims();
        let value_width = value.dims()[3];
        self.key = Some(key.slice([0..batch,0..heads,tokens..self.length,0..key_width]));
        self.value = Some(value.slice([0..batch,0..heads,tokens..self.length,0..value_width]));
        self.visible = Some(visible.slice([0..batch,tokens..self.length]));
        self.length -= tokens;
        self.capacity = self.length;
        self.start_position = start;
    }

    /// Select/duplicate actual batch rows using same-device parent indices for beam search.
    /// Indices must refer to existing rows; validity and all retained K/V follow them.
    pub fn reordered(&self,parents: Tensor<B,1,Int>) -> Self {
        let Some((key,value,visible)) = self.prefix() else { return self.clone(); };
        assert_eq!(parents.device(),key.device(),"cache parent indices must use the retained device");
        let mut result = Self::from_projected(key.select(0,parents.clone()),value.select(0,parents.clone()),
            Some(visible.select(0,parents)),self.start_position);
        result.initial_capacity = self.initial_capacity;
        result
    }

    /// Move only actual retained payloads, not unused reserve slots, to the selected device.
    pub fn to_device(self,device: &B::Device) -> Self {
        let Some((key,value,visible)) = self.prefix() else { return self; };
        let mut result = Self::from_projected(key.to_device(device),value.to_device(device),Some(visible.to_device(device)),self.start_position);
        result.initial_capacity = self.initial_capacity;
        result
    }

    /// Clear inference history and position; no original model parameters are modified.
    pub fn clear(&mut self) { *self = Self::new(self.initial_capacity); }
}
