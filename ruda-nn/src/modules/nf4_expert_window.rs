use ruda_model::{module::{Module,Param},tensor::{Tensor,Int,DType,Nf4ExpertPayload,Nf4GroupedOptions,Nf4ProjectionOptions,
    PackedExpertPayload,backend::Backend}};

/// Original expert-owned NF4 window retaining source flat blocks across expert boundaries.
/// The leading partial block is stored unchanged; it is not a newly quantized local block.
#[derive(Module,Debug)]
pub struct FrozenNf4ExpertWindow<B:Backend> {
    /// Actual original U8 bytes beginning at a source scale-block boundary.
    pub packed:Param<Tensor<B,1,Int>>,
    /// Actual original FP32 scales for all blocks intersecting the owned interval.
    pub scales:Param<Tensor<B,1>>,
    /// Actual original FP32 sixteen-value codebook.
    pub codebook:Param<Tensor<B,1>>,
    /// Resident expert count, including zero on an empty owner.
    pub experts:usize,
    /// Original per-expert input width.
    pub input_features:usize,
    /// Original per-expert output width.
    pub output_features:usize,
    /// Original positive even flat-block width.
    pub block_size:usize,
    /// First owned coefficient's offset within the first stored original block.
    pub element_offset:usize,
    /// Original bounded decoded row tile.
    pub tile_rows:usize,
    /// Original explicit half/BF16 Tensor Core selection.
    pub use_tensor_core:bool,
}
impl<B:Backend> FrozenNf4ExpertWindow<B> {
    /// Actual `[resident_experts,input,output]` logical geometry.
    pub fn dimensions(&self) -> [usize;3] {[self.experts,self.input_features,self.output_features]}
    /// Validate only native source metadata; no packed bytes or numeric scales are read back.
    pub fn validate(&self) {
        assert!(self.experts<u32::MAX as usize && self.input_features>0 && self.output_features>0 && self.tile_rows>0,
            "invalid owned NF4 feature/expert geometry");
        assert!(self.block_size>0 && self.block_size%2==0 && self.block_size<=u32::MAX as usize
            && self.element_offset<self.block_size && (self.experts!=0 || self.element_offset==0),"invalid original NF4 block window offset");
        let matrix=self.input_features.checked_mul(self.output_features).expect("owned NF4 matrix overflows");
        assert!(matrix<=u32::MAX as usize,"owned NF4 matrix exceeds U32");
        let size=matrix.checked_mul(self.experts).and_then(|size|size.checked_add(self.element_offset)).expect("owned NF4 window overflows");
        assert!(size<=u32::MAX as usize,"owned NF4 window exceeds U32");
        let packed=self.packed.val();let scales=self.scales.val();let book=self.codebook.val();
        assert_eq!(packed.dtype(),DType::U8,"owned NF4 must retain actual U8 source bytes");assert_eq!(packed.dims(),[size.div_ceil(2)],"owned NF4 packed window length differs");
        assert_eq!(scales.dtype(),DType::F32,"owned NF4 scales must retain FP32");assert_eq!(scales.dims(),[size.div_ceil(self.block_size)],"owned NF4 source scale window differs");
        assert_eq!(book.dtype(),DType::F32,"owned NF4 codebook must retain FP32");assert_eq!(book.dims(),[16],"owned NF4 original codebook length differs");
        for value in [&scales,&book] {assert_eq!(value.device(),packed.device(),"owned NF4 devices differ");assert!(!value.is_require_grad(),"owned NF4 quantization metadata remains frozen");}
    }
    /// Original selected expert range and byte-window offset for native projection/input VJP.
    pub fn primitives(&self,expert_start:usize) -> PackedExpertPayload<B> {
        self.validate();assert!(expert_start.checked_add(self.experts).is_some_and(|end|end<=u32::MAX as usize),"owned NF4 global expert interval exceeds U32");
        PackedExpertPayload::Nf4Window {payload:Nf4ExpertPayload {packed:self.packed.val().into_primitive(),scales:self.scales.val().into_primitive().tensor(),
            codebook:self.codebook.val().into_primitive().tensor(),options:Nf4GroupedOptions {experts:self.experts,expert_start,
                projection:Nf4ProjectionOptions {input_features:self.input_features,output_features:self.output_features,block_size:self.block_size,
                    tile_rows:self.tile_rows,use_tensor_core:self.use_tensor_core}}},element_offset:self.element_offset}
    }
}
