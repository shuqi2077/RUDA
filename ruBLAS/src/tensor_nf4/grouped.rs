use super::{Nf4Error,Nf4Gemm,Nf4Layout,grouped_kernel,kernels};
use crate::tensor_matmul::{matmul_with_precision,MatmulStrategy,F32MathMode};
use ruda_core::{device::Device,tensor::{DType,Metadata,Shape},ir::{ElemType,FloatKind,features::MmaConfig}};
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::{RudaCount,RudaDim}},
    tensor::{RudaTensor,allocation::empty_device_contiguous_dtype,contiguous::into_contiguous}};
use half::{f16,bf16};

/// Original NF4 payload interpreted as logical `[experts,output,input]`.
/// Flat blocks are shared across the complete cube, including expert boundaries.
/// Only selected expert segments are evaluated; there is no dense base shadow.
#[derive(Clone,Debug)]
pub struct Nf4GroupedGemm<R:Runtime> {
    weights:Nf4Gemm<R>,experts:usize,columns:usize,
}
impl<R:Runtime> Nf4GroupedGemm<R> {
    /// Connect actual high-nibble-first bytes and original flat-block FP32 metadata.
    pub fn new(packed:RudaTensor<R>,scales:RudaTensor<R>,codebook:RudaTensor<R>,experts:usize,layout:Nf4Layout) -> Result<Self,Nf4Error> {
        let layout=Nf4Layout::new(layout.input_features,layout.output_features,layout.block_size)?;
        if experts==0 || experts>=u32::MAX as usize {return Err(Nf4Error::Layout("grouped NF4 requires a positive expert count and U32 exclusive prefix"));}
        let columns=experts.checked_mul(layout.output_features).ok_or(Nf4Error::Layout("grouped NF4 expert geometry overflows"))?;
        let full=Nf4Layout::new(layout.input_features,columns,layout.block_size)?;
        Ok(Self {weights:Nf4Gemm::new(packed,scales,codebook,None,full)?,experts,columns:layout.output_features})
    }
    /// Actual expert count and per-expert original projection geometry.
    pub fn layout(&self) -> (usize,Nf4Layout) {
        (self.experts,Nf4Layout {input_features:self.weights.layout.input_features,output_features:self.columns,block_size:self.weights.layout.block_size})
    }
    /// Original resident payload bytes, excluding activations and bounded decoded tiles.
    pub fn packed_payload_bytes(&self) -> usize {self.weights.packed_payload_bytes()}
    /// Native selected expert forward, retaining activation storage.
    ///
    /// # Safety
    /// `offsets` is the immutable U32 exclusive prefix `[experts+1]`, starting
    /// at zero and ending at the input row count. Rows in segment e belong to e.
    #[allow(unsafe_code)]
    pub unsafe fn forward_segmented(&self,input:RudaTensor<R>,offsets:RudaTensor<R>,tile_rows:usize,use_tensor_core:bool) -> Result<RudaTensor<R>,Nf4Error> {
        self.project(input,offsets,tile_rows,use_tensor_core,false)
    }
    /// Original selected packed-weight input VJP with FP32 output/accumulation.
    /// Incoming seeds must already have original activation storage.
    ///
    /// # Safety
    /// The original forward's valid immutable expert prefix must be supplied.
    #[allow(unsafe_code)]
    pub unsafe fn input_backward_segmented_f32(&self,gradient:RudaTensor<R>,offsets:RudaTensor<R>,tile_rows:usize,use_tensor_core:bool) -> Result<RudaTensor<R>,Nf4Error> {
        self.project(gradient,offsets,tile_rows,use_tensor_core,true)
    }
    fn project(&self,input:RudaTensor<R>,offsets:RudaTensor<R>,tile_rows:usize,use_tensor_core:bool,backward:bool) -> Result<RudaTensor<R>,Nf4Error> {
        let k=self.weights.layout.input_features;let n=self.columns;
        let (source,target)=if backward {(n,k)} else {(k,n)};
        if input.meta.num_dims()!=2 || input.meta.shape()[1]!=source || !matches!(input.dtype,DType::F32|DType::F16|DType::BF16)
            || offsets.dtype!=DType::U32 || offsets.meta.shape()[..]!=[self.experts+1] || tile_rows==0 {
            return Err(Nf4Error::Layout("grouped NF4 requires original floating row geometry, U32 expert prefix and positive tile rows"));
        }
        for tensor in [&input,&offsets] {
            if tensor.qparams.is_some() || tensor.device.to_id()!=self.weights.packed.device.to_id()
                || !tensor.client.same_execution_queue(&self.weights.packed.client) {
                return Err(Nf4Error::Layout("grouped NF4 operands must retain native storage on the same device/queue"));
            }
        }
        let rows=input.meta.shape()[0];
        if rows.checked_mul(source).is_none_or(|size|size>u32::MAX as usize) || rows.checked_mul(target).is_none_or(|size|size>u32::MAX as usize) {
            return Err(Nf4Error::Layout("grouped NF4 activation geometry exceeds U32 indexing"));
        }
        let input=into_contiguous(input);let offsets=into_contiguous(offsets);
        let output=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([rows,target]),if backward {DType::F32} else {input.dtype});
        if rows==0 {return Ok(output);}
        if use_tensor_core && matches!(input.dtype,DType::F16|DType::BF16) {self.tensor_core(input,offsets,output.clone(),backward)?;}
        else {self.tiled(input,offsets,output.clone(),tile_rows,backward)?;}
        Ok(output)
    }
    fn tensor_core(&self,input:RudaTensor<R>,offsets:RudaTensor<R>,output:RudaTensor<R>,backward:bool) -> Result<(),Nf4Error> {
        let columns=if backward {self.weights.layout.input_features} else {self.columns};
        let cfg=MmaConfig {a_type:input.dtype.into(),b_type:input.dtype.into(),cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
        let props=&input.client.properties().hardware;
        if props.plane_size_min!=32 || props.plane_size_max!=32 || props.max_shared_memory_size<2048
            || !input.client.features().matmul.cmma.contains(&cfg) || columns.div_ceil(16)>props.max_ruda_count.0 as usize
            || self.experts>props.max_ruda_count.1 as usize {return Err(Nf4Error::Layout("requested grouped NF4 16x16x16 Tensor Core configuration is unavailable"));}
        let grid=RudaCount::Static(columns.div_ceil(16) as u32,self.experts as u32,1);
        macro_rules! run {($f:ty,$o:ty)=>{unsafe {grouped_kernel::segmented::launch::<$f,$o,R>(&input.client,grid,RudaDim::new_1d(32),
            input.clone().into_tensor_arg(),self.weights.packed.clone().into_tensor_arg(),self.weights.scales.clone().into_tensor_arg(),
            self.weights.codebook.clone().into_tensor_arg(),offsets.clone().into_tensor_arg(),output.clone().into_tensor_arg(),
            self.columns as u32,self.weights.layout.input_features as u32,self.weights.layout.block_size as u32,backward)}};}
        match (input.dtype,backward) {(DType::F16,false)=>run!(f16,f16),(DType::BF16,false)=>run!(bf16,bf16),
            (DType::F16,true)=>run!(f16,f32),(DType::BF16,true)=>run!(bf16,f32),_=>unreachable!()}
        Ok(())
    }
    fn tiled(&self,input:RudaTensor<R>,offsets:RudaTensor<R>,output:RudaTensor<R>,tile_rows:usize,backward:bool) -> Result<(),Nf4Error> {
        // The tiny dispatch prefix is scheduling metadata, never model/activation readback.
        let prefix=ruda_core::future::block_on(ruda_kernel::tensor::readback::into_data(offsets))
            .map_err(|_|Nf4Error::Layout("grouped NF4 prefix metadata read failed"))?.to_vec::<u32>()
            .map_err(|_|Nf4Error::Layout("grouped NF4 prefix storage differs"))?;
        if prefix.len()!=self.experts+1 || prefix.first()!=Some(&0) || prefix.last().copied().map(|n|n as usize)!=Some(input.meta.shape()[0])
            || prefix.windows(2).any(|pair|pair[0]>pair[1]) {return Err(Nf4Error::Layout("grouped NF4 expert prefix is not a complete monotone row partition"));}
        let (k,n)=(self.weights.layout.input_features,self.columns);
        let source=if backward {n} else {k};let target=if backward {k} else {n};
        for (expert,pair) in prefix.windows(2).enumerate() {
            let begin=pair[0] as usize;let rows=(pair[1]-pair[0]) as usize;if rows==0 {continue;}
            let segment=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([rows,source]),input.dtype);
            let size=rows*source;let dim=RudaDim::new(input.client.properties(),size);
            grouped_kernel::copy_rows::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,size,dim),dim,
                input.clone().into_array_arg(),segment.clone().into_array_arg(),(begin*source) as u32,input.dtype.into());
            let accumulator=backward.then(||empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([rows,k]),DType::F32));
            if let Some(acc)=&accumulator {let size=rows*k;let dim=RudaDim::new(input.client.properties(),size);
                kernels::zero_accumulator::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,size,dim),dim,acc.clone().into_array_arg());}
            for start in (0..n).step_by(tile_rows) {
                let tile=tile_rows.min(n-start);let decoded=self.weights.decode_rows(expert*n+start,tile,input.dtype)?;
                let (lhs,rhs)=if backward {
                    let slice=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([rows,tile]),input.dtype);
                    let size=rows*tile;let dim=RudaDim::new(input.client.properties(),size);
                    kernels::gather_columns::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,size,dim),dim,
                        segment.clone().into_array_arg(),slice.clone().into_array_arg(),n as u32,start as u32,tile as u32,input.dtype.into());(slice,decoded)
                } else {(segment.clone(),RudaTensor::new(decoded.client,decoded.handle,Metadata::new([k,tile],[1,k]),decoded.device,decoded.dtype))};
                let partial=matmul_with_precision(lhs,rhs,None,MatmulStrategy::Ruda,output.dtype,F32MathMode::Strict).map_err(Nf4Error::Matmul)?;
                let size=rows*if backward {k} else {tile};let dim=RudaDim::new(input.client.properties(),size);
                if let Some(acc)=&accumulator {kernels::add_partial::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,size,dim),dim,
                    partial.into_array_arg(),acc.clone().into_array_arg());}
                else {grouped_kernel::store_rows_columns::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,size,dim),dim,
                    partial.into_array_arg(),output.clone().into_array_arg(),begin as u32,n as u32,start as u32,tile as u32,input.dtype.into());}
            }
            if let Some(acc)=accumulator {let size=rows*target;let dim=RudaDim::new(input.client.properties(),size);
                grouped_kernel::store_rows_columns::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,size,dim),dim,
                    acc.into_array_arg(),output.clone().into_array_arg(),begin as u32,target as u32,0,target as u32,DType::F32.into());}
        }
        Ok(())
    }
}
