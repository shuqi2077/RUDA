use alloc::vec::Vec;
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,tensor_parallel as region};
use ruda_model::tensor::{DType,Int,IntDType,Tensor,ElementConversion,backend::Backend,module::{embedding,linear}};
use region::BroadcastTensorCollective;
use super::{Embedding,VocabParallelEmbedding,VocabParallelLossLayout,VocabParallelProjection};

impl<B:Backend> VocabParallelProjection<B> {
    fn values_with_layout<const D:usize>(&self,input:&Tensor<B,D>,layout:&VocabParallelLossLayout,rank:u32,world:u32)
        -> (Tensor<B,2>,Option<Tensor<B,1>>) {
        assert!(D > 0,"vocabulary projection needs a feature axis");
        assert_eq!(world as usize,layout.world_size(),"projection topology and layout differ");
        let interval = layout.interval(rank as usize);let weight = self.weight.val();let [width,features] = weight.dims();
        assert_eq!(width,interval.len(),"projection rows differ from the actual rank interval");
        assert_eq!(input.dims()[D-1],features,"projection input width differs from local hidden width");
        assert_eq!(input.device(),weight.device(),"projection input and weight must share the device");
        let bias = self.bias.as_ref().map(|bias| {
            let value = bias.val();
            assert_eq!(value.dims(),[width],"projection bias differs from local vocabulary storage");
            assert_eq!(value.device(),weight.device(),"projection bias and weight must share the device");value
        });
        if interval.end <= layout.vocabulary_size() {return (weight,bias);}
        let rows = Tensor::<B,1,Int>::arange(interval.start as i64..interval.end as i64,(&weight.device(),DType::I64));
        let padding = rows.greater_equal_elem(layout.vocabulary_size() as i64);
        let weight = weight.mask_fill(padding.clone().reshape([width,1]).expand([width,features]),0);
        let bias = bias.map(|value|value.mask_fill(padding,0));
        (weight,bias)
    }

    /// Native-backend uneven vocabulary projection with explicitly optional real-logit gathering.
    /// Retains the original embedding/head storage and tie. No padding-mask transform is performed
    /// on an all-real shard, avoiding an unnecessary floating conversion of packed native weights.
    pub fn forward_inference_with_layout<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,D>,C::Error> {
        let (weight,bias) = self.values_with_layout(&input,layout,communicator.rank(),communicator.world_size());
        let logits = linear(input,weight.transpose(),bias);
        if gather_output {layout.gather_logits_inference(logits,communicator)} else {Ok(logits)}
    }
}

impl<B: Backend> VocabParallelEmbedding<B> {
    /// Retain a supplied local embedding and its parameter ID with an explicit
    /// uneven layout. No complete vocabulary weight is initialized or downloaded.
    pub fn from_shard_with_layout(local: Embedding<B>,layout: &VocabParallelLossLayout,rank: usize,
        padding_index: Option<usize>) -> Self {
        let interval = layout.interval(rank);
        assert_eq!(local.weight.val().dims()[0],interval.len(),"embedding rows differ from the rank's actual interval");
        Self::from_shard(local,interval.start,layout.vocabulary_size(),padding_index)
    }
}

impl<B: Backend,S: CheckpointStrategy> VocabParallelEmbedding<Autodiff<B,S>> {
    /// Replicated lookup from unequal rank-local vocabulary intervals.
    /// Tokens, shape and layout must match across the group. Trailing storage
    /// padding is not a token ID; a fully padded rank contributes zero values
    /// and gradients. The configured padding row retains its value but is detached.
    pub fn forward_with_layout<C: BroadcastTensorCollective<B>>(&self,tokens: Tensor<Autodiff<B,S>,2,Int>,
        communicator: C,layout: &VocabParallelLossLayout) -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!(communicator.world_size() as usize,layout.world_size(),"embedding topology and layout differ");
        let interval = layout.interval(communicator.rank() as usize);
        let [width,features] = self.local.weight.val().dims();
        assert_eq!(width,interval.len(),"embedding storage and rank interval differ");
        assert_eq!(self.vocabulary_start,interval.start,"embedding starts at a different global token ID");
        assert_eq!(self.vocabulary_size,layout.vocabulary_size(),"embedding logical vocabulary differs from layout");
        assert_eq!(tokens.device(),self.local.weight.val().device(),"token indices and local embedding must share the device");
        let [batch,length] = tokens.dims();
        let mut weight = self.local.weight.val();
        if batch == 0 || length == 0 {return Ok(weight.slice([0..0,0..features]).reshape([batch,length,features]));}
        let tokens = tokens.cast(IntDType::I64);
        let invalid = tokens.clone().lower_elem(0).bool_or(tokens.clone().greater_equal_elem(layout.vocabulary_size() as i64));
        assert!(!invalid.any().into_scalar().elem::<bool>(),"token lies outside the logical vocabulary");
        let outside = tokens.clone().lower_elem(interval.start as i64)
            .bool_or(tokens.clone().greater_equal_elem(interval.end.min(layout.vocabulary_size()) as i64));
        let indices = tokens.sub_scalar(interval.start as i64).mask_fill(outside.clone(),0);
        let rows = Tensor::<Autodiff<B,S>,1,Int>::arange(interval.start as i64..interval.end as i64,(&weight.device(),DType::I64));
        let padding = rows.clone().greater_equal_elem(layout.vocabulary_size() as i64).reshape([width,1]).expand([width,features]);
        weight = weight.mask_fill(padding,0);
        if let Some(padding) = self.padding_index {
            assert!(padding < layout.vocabulary_size(),"configured embedding padding is not a real class");
            let frozen = rows.equal_elem(padding as i64).reshape([width,1]).expand([width,features]);
            weight = weight.clone().mask_where(frozen,weight.detach());
        }
        let output = embedding(weight,indices).mask_fill(outside.reshape([batch,length,1]).expand([batch,length,features]),0);
        region::reduce_from_region(output,communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> VocabParallelProjection<Autodiff<B,S>> {
    /// Project this rank's actual uneven vocabulary rows, optionally gathering
    /// only real classes. Shared embedding/head IDs remain unchanged. Stored
    /// padding weights/bias are removed before linear, not merely masked afterward.
    /// Pass gather_output=false to feed VocabParallelCrossEntropy directly.
    pub fn forward_with_layout<C: BroadcastTensorCollective<B>,const D: usize>(&self,input: Tensor<Autodiff<B,S>,D>,
        communicator: C,layout: &VocabParallelLossLayout,gather_output: bool) -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        let (weight,bias) = self.values_with_layout(&input,layout,communicator.rank(),communicator.world_size());
        let input = region::copy_to_region(input,communicator.clone())?;
        let logits = linear(input,weight.transpose(),bias);
        if gather_output {layout.gather_logits(logits,communicator)} else {Ok(logits)}
    }
}

impl VocabParallelLossLayout {
    /// Explicitly gather unequal local logits into the real global vocabulary.
    /// Temporary transport padding to max(local_width) enables the existing equal
    /// gather primitive; it is removed in rank order before returning. Backward
    /// follows those slices and sends no gradients to either kind of padding.
    /// Other axes and native dtype must match across ranks. This materializes full
    /// logits and is not used by the sharded-loss path unless explicitly requested.
    pub fn gather_logits<B,S,C,const D: usize>(&self,logits: Tensor<Autodiff<B,S>,D>,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
        assert!(D > 0,"vocabulary gather requires a class axis");
        assert_eq!(communicator.world_size() as usize,self.world_size(),"gather topology and vocabulary layout differ");
        let rank = communicator.rank() as usize;
        let width = self.interval(rank).len();
        let mut shape = logits.dims();
        assert_eq!(shape[D-1],width,"gather logits differ from this rank's actual vocabulary interval");
        if shape[..D-1].contains(&0) {
            shape[D-1] = self.vocabulary_size();
            return Ok(logits.reshape(shape));
        }
        let maximum = (0..self.world_size()).map(|rank|self.interval(rank).len()).max().unwrap();
        maximum.checked_mul(self.world_size()).expect("padded vocabulary gather geometry overflow");
        let padded = if maximum == width {logits} else {logits.pad([(0,maximum-width)],0.)};
        let gathered = region::gather_from_region(padded,communicator,D-1)?;
        let mut parts = Vec::new();
        for rank in 0..self.world_size() {
            let interval = self.interval(rank);
            if interval.start >= self.vocabulary_size() {break;}
            let real = interval.end.min(self.vocabulary_size())-interval.start;
            parts.push(gathered.clone().slice_dim(D-1,rank*maximum..rank*maximum+real));
        }
        Ok(Tensor::cat(parts,D-1))
    }
}
