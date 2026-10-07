//! RUDA high-nibble-first row-major NF4, shared with the native PyTorch bridge.
pub mod kernels;
use crate::tensor_matmul::{matmul_with_precision,MatmulStrategy,F32MathMode};
use crate::kernel_ir::definition::MatmulSetupError;
use ruda_core::{device::Device,tensor::{DType,Metadata,Shape},ir::{ElemType,FloatKind,features::MmaConfig}};
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::{RudaCount,RudaDim}},
    tensor::{RudaTensor,allocation::empty_device_contiguous_dtype,contiguous::into_contiguous}};
use half::{f16,bf16};
use std::fmt;

/// Native NF4 layout/launch setup error, without retrying through another policy.
#[derive(Debug)]
pub enum Nf4Error {
    /// Original packed layout, storage, device or capability contract failure.
    Layout(&'static str),
    /// Original ruBLAS tiled multiplication setup failure.
    Matmul(MatmulSetupError),
}
impl fmt::Display for Nf4Error {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Layout(message)=>f.write_str(message),Self::Matmul(error)=>write!(f,"NF4 tiled GEMM: {error}")}
    }
}
impl std::error::Error for Nf4Error {}

/// Original logical `[output,input]` matrix geometry, including partial final blocks.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Nf4Layout {
    /// Input feature width.
    pub input_features:usize,
    /// Output feature width.
    pub output_features:usize,
    /// Original positive even flat block size.
    pub block_size:usize,
}
impl Nf4Layout {
    /// Select actual checkpoint geometry without dimension rounding or padding its logical values.
    pub fn new(input_features:usize,output_features:usize,block_size:usize) -> Result<Self,Nf4Error> {
        if input_features==0 || output_features==0 || block_size==0 || block_size%2!=0 || block_size>u32::MAX as usize
            || input_features.checked_mul(output_features).is_none_or(|n|n>u32::MAX as usize) {
            return Err(Nf4Error::Layout("NF4 requires positive dimensions, an even block size and u32 weight indexing"));
        }
        Ok(Self {input_features,output_features,block_size})
    }
    /// Actual stored bytes, excluding allocator padding, activations and bounded decoded tiles.
    pub fn packed_payload_bytes(self,bias_dtype:Option<DType>) -> Result<usize,Nf4Error> {
        Self::new(self.input_features,self.output_features,self.block_size)?;
        if bias_dtype.is_some_and(|dtype|!matches!(dtype,DType::F32|DType::F16|DType::BF16)) {
            return Err(Nf4Error::Layout("NF4 bias requires FP32/FP16/BF16"));
        }
        let size=self.input_features*self.output_features;
        size.div_ceil(2).checked_add(size.div_ceil(self.block_size).checked_mul(4).ok_or(Nf4Error::Layout("NF4 scale size overflows"))?)
            .and_then(|n|n.checked_add(64)).and_then(|n|n.checked_add(self.output_features.checked_mul(bias_dtype.map_or(0,|dtype|dtype.size()))?))
            .ok_or(Nf4Error::Layout("NF4 payload size overflows"))
    }
}

/// Immutable original packed bytes and FP32 quantization metadata, without a dense shadow.
#[derive(Clone,Debug)]
pub struct Nf4Gemm<R:Runtime> {
    packed:RudaTensor<R>,
    scales:RudaTensor<R>,
    codebook:RudaTensor<R>,
    bias:Option<RudaTensor<R>>,
    layout:Nf4Layout,
}
impl<R:Runtime> Nf4Gemm<R> {
    /// Connect actual native packed payload. No weight quantization or format conversion occurs.
    pub fn new(packed:RudaTensor<R>,scales:RudaTensor<R>,codebook:RudaTensor<R>,bias:Option<RudaTensor<R>>,layout:Nf4Layout)
        -> Result<Self,Nf4Error> {
        let layout=Nf4Layout::new(layout.input_features,layout.output_features,layout.block_size)?;
        let size=layout.input_features*layout.output_features;
        if packed.dtype!=DType::U8 || packed.meta.shape()!=&Shape::from([size.div_ceil(2)])
            || scales.dtype!=DType::F32 || scales.meta.shape()!=&Shape::from([size.div_ceil(layout.block_size)])
            || codebook.dtype!=DType::F32 || codebook.meta.shape()!=&Shape::from([16]) {
            return Err(Nf4Error::Layout("NF4 requires original U8 bytes, FP32 flat-block scales and a 16-value FP32 codebook"));
        }
        for value in [&packed,&scales,&codebook].into_iter().chain(bias.iter()) {
            if value.device.to_id()!=packed.device.to_id() || !value.client.same_execution_queue(&packed.client) || value.qparams.is_some() {
                return Err(Nf4Error::Layout("NF4 operands must share a device/queue and retain plain native storage"));
            }
        }
        if let Some(bias)=&bias {
            if !matches!(bias.dtype,DType::F32|DType::F16|DType::BF16) || bias.meta.shape()!=&Shape::from([layout.output_features]) {
                return Err(Nf4Error::Layout("NF4 bias must be a floating output-feature vector"));
            }
        }
        Ok(Self {packed:into_contiguous(packed),scales:into_contiguous(scales),codebook:into_contiguous(codebook),bias:bias.map(into_contiguous),layout})
    }
    /// Actual original logical dimensions and block geometry.
    pub fn layout(&self) -> Nf4Layout {self.layout}
    /// Actual resident packed payload, not a peak memory or performance estimate.
    pub fn packed_payload_bytes(&self) -> usize {
        self.layout.packed_payload_bytes(self.bias.as_ref().map(|bias|bias.dtype)).expect("validated NF4 layout")
    }
    /// Original high/low nibble decoding into a bounded row interval, rounded to the requested activation storage.
    pub fn decode_rows(&self,begin:usize,rows:usize,dtype:DType) -> Result<RudaTensor<R>,Nf4Error> {
        let end=begin.checked_add(rows).ok_or(Nf4Error::Layout("NF4 decode interval overflows"))?;
        if end>self.layout.output_features || !matches!(dtype,DType::F32|DType::F16|DType::BF16) {
            return Err(Nf4Error::Layout("NF4 decode interval/dtype differs from original geometry"));
        }
        let output=empty_device_contiguous_dtype(self.packed.client.clone(),self.packed.device.clone(),Shape::from([rows,self.layout.input_features]),dtype);
        if rows==0 {return Ok(output);}
        let elements=rows*self.layout.input_features;
        let grid=RudaCount::Static(elements.div_ceil(128) as u32,1,1);
        macro_rules! run {
            ($f:ty)=>{unsafe {kernels::decode::launch::<$f,R>(&self.packed.client,grid,RudaDim::new_1d(128),
                self.packed.clone().into_tensor_arg(),self.scales.clone().into_tensor_arg(),self.codebook.clone().into_tensor_arg(),
                output.clone().into_tensor_arg(),(begin*self.layout.input_features) as u32,self.layout.block_size as u32)}};
        }
        match dtype {DType::F32=>run!(f32),DType::F16=>run!(f16),DType::BF16=>run!(bf16),_=>unreachable!()}
        Ok(output)
    }
    /// Native forward for any leading axes. Half/BF16 with `use_tensor_core` uses
    /// the shared original fused kernel; FP32 or an explicitly disabled fused path
    /// uses original bounded row decoding plus strict ruBLAS GEMM. Failed calls are not retried.
    pub fn forward(&self,input:RudaTensor<R>,tile_rows:usize,use_tensor_core:bool) -> Result<RudaTensor<R>,Nf4Error> {
        self.project(input,false,tile_rows,use_tensor_core)
    }
    /// Original packed-weight input VJP, with FP32 accumulation/output. Caller
    /// supplies gradients already cast to original activation storage, as in the Python bridge.
    pub fn input_backward_f32(&self,gradient:RudaTensor<R>,tile_rows:usize,use_tensor_core:bool) -> Result<RudaTensor<R>,Nf4Error> {
        self.project(gradient,true,tile_rows,use_tensor_core)
    }
    fn project(&self,input:RudaTensor<R>,backward:bool,tile_rows:usize,use_tensor_core:bool) -> Result<RudaTensor<R>,Nf4Error> {
        let rank=input.meta.num_dims();let (k,n)=(self.layout.input_features,self.layout.output_features);
        let (source,target)=if backward {(n,k)} else {(k,n)};
        if tile_rows==0 || rank==0 || input.meta.shape()[rank-1]!=source || !matches!(input.dtype,DType::F32|DType::F16|DType::BF16)
            || input.qparams.is_some() || input.device.to_id()!=self.packed.device.to_id() || !input.client.same_execution_queue(&self.packed.client) {
            return Err(Nf4Error::Layout("NF4 requires matching floating input geometry/device/queue and positive tile rows"));
        }
        let rows=input.meta.shape().iter().take(rank-1).try_fold(1usize,|n,&axis|n.checked_mul(axis)).ok_or(Nf4Error::Layout("NF4 batch size overflows"))?;
        let elements=rows.checked_mul(target).ok_or(Nf4Error::Layout("NF4 output size overflows"))?;
        if elements>u32::MAX as usize || rows.checked_mul(source).is_none_or(|n|n>u32::MAX as usize) {return Err(Nf4Error::Layout("NF4 batch exceeds u32 indexing"));}
        let mut output_shape=input.meta.shape().clone();output_shape[rank-1]=target;
        let input=into_contiguous(input);
        let input=RudaTensor::new(input.client,input.handle,Metadata::new([rows,source],[source,1]),input.device,input.dtype);
        let output=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([rows,target]),if backward {DType::F32} else {input.dtype});
        if elements>0 {
            if use_tensor_core && matches!(input.dtype,DType::F16|DType::BF16) {self.tensor_core(input.clone(),output.clone(),rows,backward)?;}
            else {self.tiled(input.clone(),output.clone(),rows,tile_rows,backward)?;}
            if !backward {if let Some(bias)=&self.bias {
                let dim=RudaDim::new(input.client.properties(),elements);
                kernels::add_bias::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,elements,dim),dim,
                    output.clone().into_array_arg(),bias.clone().into_array_arg(),n as u32,input.dtype.into(),bias.dtype.into());
            }}
        }
        let mut strides=vec![1usize;rank];for axis in (0..rank.saturating_sub(1)).rev() {
            strides[axis]=strides[axis+1].checked_mul(output_shape[axis+1]).ok_or(Nf4Error::Layout("NF4 leading output strides overflow"))?;
        }
        Ok(RudaTensor::new(output.client,output.handle,Metadata::new(output_shape,strides),output.device,output.dtype))
    }
    fn tensor_core(&self,input:RudaTensor<R>,output:RudaTensor<R>,rows:usize,backward:bool) -> Result<(),Nf4Error> {
        let columns=if backward {self.layout.input_features} else {self.layout.output_features};
        let cfg=MmaConfig {a_type:input.dtype.into(),b_type:input.dtype.into(),cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
        let props=&input.client.properties().hardware;
        if props.plane_size_min!=32 || props.plane_size_max!=32 || props.max_shared_memory_size<2048
            || !input.client.features().matmul.cmma.contains(&cfg) || columns.div_ceil(16)>props.max_ruda_count.0 as usize
            || rows.div_ceil(16)>props.max_ruda_count.1 as usize {return Err(Nf4Error::Layout("requested NF4 16x16x16 Tensor Core configuration is unavailable"));}
        let grid=RudaCount::Static(columns.div_ceil(16) as u32,rows.div_ceil(16) as u32,1);
        macro_rules! run {($f:ty,$o:ty)=>{unsafe {kernels::gemm::launch::<$f,$o,R>(&input.client,grid,RudaDim::new_1d(32),
            input.clone().into_tensor_arg(),self.packed.clone().into_tensor_arg(),self.scales.clone().into_tensor_arg(),self.codebook.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),rows as u32,self.layout.output_features as u32,self.layout.input_features as u32,self.layout.block_size as u32,backward)}};}
        match (input.dtype,backward) {(DType::F16,false)=>run!(f16,f16),(DType::BF16,false)=>run!(bf16,bf16),
            (DType::F16,true)=>run!(f16,f32),(DType::BF16,true)=>run!(bf16,f32),_=>unreachable!()}
        Ok(())
    }
    fn tiled(&self,input:RudaTensor<R>,output:RudaTensor<R>,rows:usize,tile_rows:usize,backward:bool) -> Result<(),Nf4Error> {
        let (k,n)=(self.layout.input_features,self.layout.output_features);
        if backward {
            let elements=rows*k;let dim=RudaDim::new(input.client.properties(),elements);
            kernels::zero_accumulator::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,elements,dim),dim,output.clone().into_array_arg());
        }
        for start in (0..n).step_by(tile_rows) {
            let size=tile_rows.min(n-start);let weight=self.decode_rows(start,size,input.dtype)?;
            let (lhs,rhs)=if backward {
                let slice=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([rows,size]),input.dtype);
                let elements=rows*size;let dim=RudaDim::new(input.client.properties(),elements);
                kernels::gather_columns::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,elements,dim),dim,
                    input.clone().into_array_arg(),slice.clone().into_array_arg(),n as u32,start as u32,size as u32,input.dtype.into());
                (slice,weight)
            } else {
                let weight=RudaTensor::new(weight.client,weight.handle,Metadata::new([k,size],[1,k]),weight.device,weight.dtype);(input.clone(),weight)
            };
            let partial=matmul_with_precision(lhs,rhs,None,MatmulStrategy::Ruda,output.dtype,F32MathMode::Strict).map_err(Nf4Error::Matmul)?;
            let elements=if backward {rows*k} else {rows*size};let dim=RudaDim::new(input.client.properties(),elements);
            if backward {kernels::add_partial::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,elements,dim),dim,
                partial.into_array_arg(),output.clone().into_array_arg());}
            else {kernels::store_columns::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,elements,dim),dim,
                partial.into_array_arg(),output.clone().into_array_arg(),n as u32,start as u32,size as u32,input.dtype.into());}
        }
        Ok(())
    }
}
