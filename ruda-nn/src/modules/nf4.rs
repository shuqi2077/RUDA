use alloc::vec::Vec;
use crate::{Dropout,DropoutConfig,Linear,LinearConfig,LoRALinearConfig};
use ruda_model::{module::{Initializer,Module,Param,ParamId},tensor::{Tensor,TensorData,TensorPrimitive,Int,DType,Element,ElementConversion,
    FrozenNf4Ops,Nf4ProjectionOptions,backend::Backend}};
#[cfg(not(feature="std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// The original RUDA NF4 codebook used by native PyTorch packing and decoding.
pub const NF4_CODEBOOK:[f32;16]=[-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,
    -0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,0.07958029955625534,
    0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,
    0.5626170039176941,0.7229568362236023,1.0];

/// Explicit CPU preprocessing result, retaining original row-major `[output,input]` geometry.
#[derive(Clone,Debug)]
pub struct PackedNf4Data {
    /// High-nibble-first original U8 packed codes, including odd final element padding.
    pub packed:Vec<u8>,
    /// Original FP32 absolute maximum per flat block.
    pub scales:Vec<f32>,
    /// Actual source input width.
    pub input_features:usize,
    /// Actual source output width.
    pub output_features:usize,
    /// Explicit original positive even block size.
    pub block_size:usize,
}
impl PackedNf4Data {
    /// Pack an explicitly supplied CPU slice. Input conversion/normalization,
    /// FP32 midpoint thresholds and lower-code tie breaking match the existing
    /// Python packer. No GPU download or complete FP32 source shadow is created.
    pub fn from_rows<F:Element>(values:&[F],output_features:usize,input_features:usize,block_size:usize) -> Self {
        assert!(matches!(F::dtype(),DType::F32|DType::F16|DType::BF16),"NF4 preprocessing requires floating CPU values");
        assert!(input_features>0 && output_features>0 && block_size>0 && block_size%2==0 && block_size<=u32::MAX as usize,"invalid NF4 packing geometry");
        let elements=input_features.checked_mul(output_features).expect("NF4 source size overflows");
        assert!(elements<=u32::MAX as usize,"NF4 source exceeds u32 indexing");assert_eq!(values.len(),elements,"NF4 source shape differs");
        let boundaries: [f32;15]=core::array::from_fn(|index|(NF4_CODEBOOK[index]+NF4_CODEBOOK[index+1])*0.5);
        let mut packed=alloc::vec![0u8;elements.div_ceil(2)];let mut scales=Vec::with_capacity(elements.div_ceil(block_size));
        for begin in (0..elements).step_by(block_size) {
            let end=begin.saturating_add(block_size).min(elements);let mut maximum=0.0f32;
            for value in &values[begin..end] {
                let value=value.elem::<f32>();assert!(value.is_finite(),"NF4 source weights must be finite");maximum=maximum.max(value.abs());
            }
            scales.push(maximum);let denominator=if maximum==0.0 {1.0} else {maximum};
            for index in begin..end {
                let normalized=values[index].elem::<f32>()/denominator;
                let code=boundaries.iter().take_while(|&&boundary|normalized>boundary).count() as u8;
                if index%2==0 {packed[index/2]=code<<4;} else {packed[index/2]|=code;}
            }
        }
        if elements%2!=0 {packed[elements/2]|=7;}
        Self {packed,scales,input_features,output_features,block_size}
    }
    /// Upload actual preprocessing payload to an explicitly selected native backend/device.
    /// No base bias, adapter values or parameter identities are fabricated.
    pub fn into_layer<B:Backend>(self,device:&B::Device,bias:Option<Param<Tensor<B,1>>>,tile_rows:usize,use_tensor_core:bool) -> FrozenNf4Linear<B> {
        assert!(self.input_features>0 && self.output_features>0 && self.block_size>0 && self.block_size%2==0 && self.block_size<=u32::MAX as usize && tile_rows>0,"invalid NF4 upload geometry");
        let size=self.input_features.checked_mul(self.output_features).expect("NF4 upload size overflows");assert!(size<=u32::MAX as usize,"NF4 upload indexing overflows");
        let packed=Param::initialized(ParamId::new(),Tensor::<B,1,Int>::from_data(TensorData::new(self.packed,[size.div_ceil(2)]),(device,DType::U8)));
        let scales=Param::from_tensor(Tensor::<B,1>::from_data(TensorData::new(self.scales,[size.div_ceil(self.block_size)]),(device,DType::F32)));
        let book=Param::from_tensor(Tensor::<B,1>::from_data(TensorData::new(NF4_CODEBOOK.to_vec(),[16]),(device,DType::F32)));
        FrozenNf4Linear::from_parameters(packed,scales,book,bias,self.input_features,self.output_features,self.block_size,tile_rows,use_tensor_core)
    }
}

/// Frozen original RUDA NF4 base with native first-order input gradients.
/// FP32 scales/codebook are independent of activation and adapter storage.
#[derive(Module,Debug)]
pub struct FrozenNf4Linear<B:Backend> {
    /// Original row-major packed byte vector, not AWQ words or a floating surrogate.
    pub packed:Param<Tensor<B,1,Int>>,
    /// Original frozen FP32 per-flat-block scales.
    pub scales:Param<Tensor<B,1>>,
    /// Original frozen FP32 sixteen-value codebook.
    pub codebook:Param<Tensor<B,1>>,
    /// Actual optional frozen bias in its original floating storage.
    pub bias:Option<Param<Tensor<B,1>>>,
    /// Actual original input width.
    pub input_features:usize,
    /// Actual original output width.
    pub output_features:usize,
    /// Original positive even flat block size.
    pub block_size:usize,
    /// Bounded decoded row tile on the explicitly tiled path.
    pub tile_rows:usize,
    /// Enable original fused half/BF16 path, without failed-call retries.
    pub use_tensor_core:bool,
}
impl<B:Backend> FrozenNf4Linear<B> {
    /// Connect actual caller-loaded payload, retaining original IDs and stored values.
    pub fn from_parameters(packed:Param<Tensor<B,1,Int>>,scales:Param<Tensor<B,1>>,codebook:Param<Tensor<B,1>>,bias:Option<Param<Tensor<B,1>>>,
        input_features:usize,output_features:usize,block_size:usize,tile_rows:usize,use_tensor_core:bool) -> Self {
        let layer=Self {packed,scales,codebook,bias,input_features,output_features,block_size,tile_rows,use_tensor_core}.no_grad();layer.validate();layer
    }
    /// Actual original geometry/execution choices for native packed operations.
    pub fn options(&self) -> Nf4ProjectionOptions {Nf4ProjectionOptions {input_features:self.input_features,output_features:self.output_features,
        block_size:self.block_size,tile_rows:self.tile_rows,use_tensor_core:self.use_tensor_core}}
    /// Validate original native payload metadata without downloading quantized values.
    pub fn validate(&self) {
        assert!(self.input_features>0 && self.output_features>0 && self.block_size>0 && self.block_size%2==0
            && self.block_size<=u32::MAX as usize && self.tile_rows>0,"invalid NF4 geometry/tile rows");
        let size=self.input_features.checked_mul(self.output_features).expect("NF4 matrix size overflows");assert!(size<=u32::MAX as usize,"NF4 indexing overflows");
        let packed=self.packed.val();let scales=self.scales.val();let book=self.codebook.val();
        assert_eq!(packed.dtype(),DType::U8,"NF4 packed storage must retain original bytes");assert_eq!(packed.dims(),[size.div_ceil(2)],"NF4 packed byte count differs");
        assert_eq!(scales.dtype(),DType::F32,"NF4 scales must retain FP32");assert_eq!(scales.dims(),[size.div_ceil(self.block_size)],"NF4 scale count differs");
        assert_eq!(book.dtype(),DType::F32,"NF4 codebook must retain FP32");assert_eq!(book.dims(),[16],"NF4 codebook length differs");
        for value in [&scales,&book] {assert_eq!(value.device(),packed.device(),"NF4 operand devices differ");assert!(!value.is_require_grad(),"NF4 quantization metadata is frozen");}
        if let Some(bias)=&self.bias {
            let bias=bias.val();assert_eq!(bias.dims(),[self.output_features],"NF4 bias width differs");
            assert!(matches!(bias.dtype(),DType::F32|DType::F16|DType::BF16),"NF4 bias storage is unsupported");
            assert_eq!(bias.device(),packed.device(),"NF4 bias device differs");assert!(!bias.is_require_grad(),"NF4 bias is frozen");
        }
    }
}
impl<B:FrozenNf4Ops> FrozenNf4Linear<B> {
    /// Select FP16/BF16/FP32 activation arithmetic; original NF4 bytes/scales/codebook are not cast.
    pub fn forward_with_dtype<const D:usize>(&self,input:Tensor<B,D>,dtype:ruda_model::tensor::FloatDType)
        -> Result<Tensor<B,D>,B::Nf4Error> {self.forward(input.cast(dtype))}

    /// Original packed native projection, retaining all actual leading activation axes.
    pub fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,B::Nf4Error> {
        self.validate();B::frozen_nf4_forward(input.into_primitive().tensor(),self.packed.val().into_primitive(),self.scales.val().into_primitive().tensor(),
            self.codebook.val().into_primitive().tensor(),self.bias.as_ref().map(|bias|bias.val().into_primitive().tensor()),self.options())
            .map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))
    }
}

/// Trainable native floating residual on original immutable packed NF4 storage.
#[derive(Module,Debug)]
pub struct Nf4LoRALinear<B:Backend> {
    /// Actual original frozen byte/scales/codebook/bias payload.
    pub base:FrozenNf4Linear<B>,
    /// Actual native bias-free input-to-rank matrix.
    pub adapter_a:Linear<B>,
    /// Actual native bias-free rank-to-output matrix.
    pub adapter_b:Linear<B>,
    /// Original adapter-only input dropout.
    pub dropout:Dropout,
    /// Original alpha/rank or explicit rsLoRA multiplier.
    pub scale:f64,
}
impl LoRALinearConfig {
    /// Attach actual new trainable A/B leaves with explicit dtype and rsLoRA selection.
    pub fn init_nf4<B:Backend>(&self,base:FrozenNf4Linear<B>,adapter_dtype:DType,use_rslora:bool) -> Nf4LoRALinear<B> {
        base.validate();assert!(self.rank>0 && self.alpha.is_finite(),"invalid NF4 adapter rank/alpha");
        assert!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),"NF4 adapter dropout must be in [0,1)");
        assert!(matches!(adapter_dtype,DType::F32|DType::F16|DType::BF16),"NF4 adapter storage must be floating");
        let device=base.packed.val().device();
        let mut a=LinearConfig::new(base.input_features,self.rank).with_bias(false).init(&device);
        let mut b=LinearConfig::new(self.rank,base.output_features).with_bias(false).with_initializer(Initializer::Zeros).init(&device);
        a.weight=a.weight.map(|value|value.cast(adapter_dtype).detach().require_grad());b.weight=b.weight.map(|value|value.cast(adapter_dtype).detach().require_grad());
        self.from_nf4_adapters(base,a,b,use_rslora)
    }
    /// Attach actual loaded floating adapter leaves, with no merge or base requantization.
    pub fn from_nf4_adapters<B:Backend>(&self,base:FrozenNf4Linear<B>,adapter_a:Linear<B>,adapter_b:Linear<B>,use_rslora:bool) -> Nf4LoRALinear<B> {
        base.validate();assert!(self.rank>0 && self.alpha.is_finite(),"invalid NF4 adapter rank/alpha");
        assert!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),"NF4 adapter dropout must be in [0,1)");
        let a=adapter_a.weight.val();let b=adapter_b.weight.val();let device=base.packed.val().device();
        assert_eq!(a.dims(),[base.input_features,self.rank],"NF4 adapter A shape differs");assert_eq!(b.dims(),[self.rank,base.output_features],"NF4 adapter B shape differs");
        assert!(adapter_a.bias.is_none() && adapter_b.bias.is_none(),"NF4 adapters must be bias-free");
        assert_eq!(a.device(),device,"NF4 adapter A device differs");assert_eq!(b.device(),device,"NF4 adapter B device differs");
        for dtype in [a.dtype(),b.dtype()] {assert!(matches!(dtype,DType::F32|DType::F16|DType::BF16),"unsupported NF4 adapter dtype");}
        assert!(!B::ad_enabled(&device) || (a.is_require_grad() && b.is_require_grad()),"NF4 adapters must be trainable on the AD backend");
        let denominator=if use_rslora {(self.rank as f64).sqrt()} else {self.rank as f64};
        Nf4LoRALinear {base,adapter_a,adapter_b,dropout:DropoutConfig::new(self.dropout).init(),scale:self.alpha/denominator}
    }
}
impl<B:FrozenNf4Ops> Nf4LoRALinear<B> {
    /// Explicit base activation precision, independent of A/B storage and packed NF4 metadata.
    pub fn forward_with_dtype<const D:usize>(&self,input:Tensor<B,D>,dtype:ruda_model::tensor::FloatDType)
        -> Result<Tensor<B,D>,B::Nf4Error> {self.forward(input.cast(dtype))}

    /// Original packed base and scaled native low-rank residual, with real first-order derivatives to input/A/B.
    pub fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,B::Nf4Error> {
        let base=self.base.forward(input.clone())?;
        let adapted=self.dropout.forward(input.cast(self.adapter_a.weight.val().dtype()));
        let hidden=self.adapter_a.forward(adapted).cast(self.adapter_b.weight.val().dtype());
        let update=self.adapter_b.forward(hidden).mul_scalar(self.scale);let dtype=base.dtype();Ok(base+update.cast(dtype))
    }
}
