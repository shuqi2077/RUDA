use ruda_model::{module::Module,tensor::{Tensor,Int,FloatDType,backend::Backend}};
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy};
use crate::{Embedding,Dropout,transformer::{TransformerEmbeddings,DenseTransformerNorm}};
use super::{VocabParallelEmbedding,VocabParallelLossLayout,BroadcastTensorCollective,VocabParallelTransformerHead,VocabParallelAdaptedTransformerHead,head::transformed};

/// Original native transformer input tables with explicitly vocabulary-sharded token rows.
/// Learned position/type tables, normalization and combined-input dropout remain replicated.
#[derive(Module,Debug)]
pub struct TensorParallelTransformerEmbeddings<B:Backend> {
    /// Actual local token rows and original global padding-row policy.
    pub token:VocabParallelEmbedding<B>,
    /// Original replicated learned positions; absent when positions belong to the backbone.
    pub position:Option<Embedding<B>>,
    /// Original optional replicated token-type/segment table.
    pub token_type:Option<Embedding<B>>,
    /// Original normalization after adding actual table outputs.
    pub normalization:Option<DenseTransformerNorm<B>>,
    /// Original combined-input dropout, retaining native backend-mode behavior.
    pub dropout:Dropout,
}

fn independent_token<B:Backend>(base:&TransformerEmbeddings<B>,layout:&VocabParallelLossLayout,rank:usize) {
    if layout.interval(rank) == (0..base.token.weight.val().dims()[0]) {return;}
    for table in [&base.position,&base.token_type].into_iter().flatten() {
        assert_ne!(base.token.weight.id,table.weight.id,"token/position/type sharing across different shard roles needs explicit compatible local loading");
    }
}

impl<B:Backend> TensorParallelTransformerEmbeddings<B> {
    /// Connect actual local token rows and original replicated input tables/norm/dropout.
    /// No learned positions, type IDs, token scaling or full table initialization is inferred.
    pub fn from_tables(token:VocabParallelEmbedding<B>,position:Option<Embedding<B>>,token_type:Option<Embedding<B>>,
        normalization:Option<DenseTransformerNorm<B>>,dropout:Dropout) -> Self {
        let [rows,hidden] = token.local.weight.val().dims();assert!(rows > 0 && hidden > 0,"native parallel input token geometry must be positive");
        for table in [&position,&token_type].into_iter().flatten() {
            assert!(table.weight.val().dims()[0] > 0 && table.weight.val().dims()[1] == hidden,"parallel input table hidden width differs");
        }
        if let Some(norm) = &normalization {assert_eq!(norm.width(),hidden,"parallel input normalization width differs");}
        assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob),"invalid parallel input dropout");
        Self {token,position,token_type,normalization,dropout}
    }

    /// Partition original complete floating token rows, preserving all actual other input components.
    pub fn from_full(base:TransformerEmbeddings<B>,layout:&VocabParallelLossLayout,rank:usize,padding_index:Option<usize>) -> Self {
        independent_token(&base,layout,rank);
        Self::from_tables(VocabParallelEmbedding::from_full_with_layout(base.token,layout,rank,padding_index),
            base.position,base.token_type,base.normalization,base.dropout)
    }

    /// Jointly partition original input/head storage into one shared local native token/head leaf.
    pub fn from_full_tied_head(base:TransformerEmbeddings<B>,head:VocabParallelTransformerHead<B>,layout:&VocabParallelLossLayout,rank:usize,padding_index:Option<usize>)
        -> (Self,VocabParallelTransformerHead<B>) {
        independent_token(&base,layout,rank);
        let (token,head) = VocabParallelTransformerHead::from_full_tied(base.token,head,layout,rank,padding_index);
        (Self::from_tables(token,base.position,base.token_type,base.normalization,base.dropout),head)
    }

    /// Jointly partition frozen tied input/head weights and already-loaded native head adapters.
    pub fn from_full_tied_adapted_head(base:TransformerEmbeddings<B>,head:VocabParallelAdaptedTransformerHead<B>,layout:&VocabParallelLossLayout,rank:usize,padding_index:Option<usize>)
        -> (Self,VocabParallelAdaptedTransformerHead<B>) {
        independent_token(&base,layout,rank);
        let (token,head) = VocabParallelAdaptedTransformerHead::from_full_tied(base.token,head,layout,rank,padding_index);
        (Self::from_tables(token,base.position,base.token_type,base.normalization,base.dropout),head)
    }

    fn validate_inputs(&self,tokens:&Tensor<B,2,Int>,positions:&Option<Tensor<B,2,Int>>,types:&Option<Tensor<B,2,Int>>,compute:Option<FloatDType>) {
        let shape = tokens.dims();let device = tokens.device();let storage = self.token.local.weight.val().dtype();
        assert_eq!(self.token.local.weight.val().device(),device,"parallel token IDs/table must share the device");
        for (table,ids) in [(&self.position,positions),(&self.token_type,types)] {
            assert_eq!(table.is_some(),ids.is_some(),"parallel input table/ID presence differs");
            if let (Some(table),Some(ids)) = (table,ids) {
                assert_eq!(ids.dims(),shape,"parallel input metadata differs from actual token geometry");
                assert_eq!(ids.device(),device,"parallel input IDs must share the device");
                assert_eq!(table.weight.val().device(),device,"parallel input tables must share the token device");
                if compute.is_none() {assert_eq!(table.weight.val().dtype(),storage,"mixed table storage requires an explicit compute dtype");}
            }
        }
    }

    fn finish<F>(&self,mut hidden:Tensor<B,3>,positions:Option<Tensor<B,2,Int>>,types:Option<Tensor<B,2,Int>>,compute:Option<FloatDType>,dropout:F) -> Tensor<B,3>
        where F:FnOnce(&Dropout,Tensor<B,3>)->Tensor<B,3> {
        if let Some(dtype) = compute {hidden = hidden.cast(dtype);}
        for (table,ids) in [(&self.position,positions),(&self.token_type,types)] {
            if let (Some(table),Some(ids)) = (table,ids) {
                let rows = table.forward(ids);let rows = if let Some(dtype) = compute {rows.cast(dtype)} else {rows};hidden = hidden+rows;
            }
        }
        if let Some(norm) = &self.normalization {hidden = norm.forward(hidden);}
        transformed(&self.dropout,hidden,dropout)
    }

    /// Native inference with original token/position/type addition, normalization and dropout order.
    pub fn forward_inference<C:BroadcastTensorCollective<B>>(&self,tokens:Tensor<B,2,Int>,positions:Option<Tensor<B,2,Int>>,types:Option<Tensor<B,2,Int>>,
        communicator:C,layout:&VocabParallelLossLayout) -> Result<Tensor<B,3>,C::Error> {
        self.validate_inputs(&tokens,&positions,&types,None);
        self.token.forward_inference_with_layout(tokens,communicator,layout).map(|hidden|self.finish(hidden,positions,types,None,|dropout,input|dropout.forward(input)))
    }

    /// Explicit mixed-storage inference arithmetic/output, casting actual looked-up rows only.
    pub fn forward_inference_with_compute_dtype<C:BroadcastTensorCollective<B>>(&self,tokens:Tensor<B,2,Int>,positions:Option<Tensor<B,2,Int>>,types:Option<Tensor<B,2,Int>>,
        communicator:C,layout:&VocabParallelLossLayout,compute:FloatDType,output:FloatDType) -> Result<Tensor<B,3>,C::Error> {
        self.validate_inputs(&tokens,&positions,&types,Some(compute));
        self.token.forward_inference_with_layout(tokens,communicator,layout).map(|hidden|self.finish(hidden,positions,types,Some(compute),|dropout,input|dropout.forward(input)).cast(output))
    }

    /// Native packed input states from caller-supplied real/reset position and optional type IDs.
    pub fn forward_packed_inference<C:BroadcastTensorCollective<B>>(&self,tokens:Tensor<B,1,Int>,positions:Option<Tensor<B,1,Int>>,types:Option<Tensor<B,1,Int>>,
        communicator:C,layout:&VocabParallelLossLayout) -> Result<Tensor<B,2>,C::Error> {
        let count = tokens.dims()[0];let convert = |ids:Tensor<B,1,Int>| {assert_eq!(ids.dims(),[count],"packed parallel input metadata length differs");ids.reshape([1,count])};
        self.forward_inference(tokens.reshape([1,count]),positions.map(convert),types.map(convert),communicator,layout)
            .map(|hidden| {let features = hidden.dims()[2];hidden.reshape([count,features])})
    }
}

impl<B:Backend,S:CheckpointStrategy> TensorParallelTransformerEmbeddings<Autodiff<B,S>> {
    /// Native input training over unique vocabulary shards and explicitly replicated metadata tables.
    /// Replicated position/type/norm parameters use the full downstream hidden derivative, not another TP SUM.
    pub fn forward<C:BroadcastTensorCollective<B>>(&self,tokens:Tensor<Autodiff<B,S>,2,Int>,positions:Option<Tensor<Autodiff<B,S>,2,Int>>,types:Option<Tensor<Autodiff<B,S>,2,Int>>,
        communicator:C,layout:&VocabParallelLossLayout) -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        self.forward_with_dtype_and_dropout(tokens,positions,types,communicator,layout,None,None,|dropout,input|dropout.forward(input))
    }

    /// Explicit corresponding combined-input dropout, retaining the original addition/norm order.
    pub fn forward_with_dropout<C,F>(&self,tokens:Tensor<Autodiff<B,S>,2,Int>,positions:Option<Tensor<Autodiff<B,S>,2,Int>>,types:Option<Tensor<Autodiff<B,S>,2,Int>>,
        communicator:C,layout:&VocabParallelLossLayout,dropout:F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        self.forward_with_dtype_and_dropout(tokens,positions,types,communicator,layout,None,None,dropout)
    }

    /// Explicit arithmetic/output storage without creating full-table converted copies.
    pub fn forward_with_compute_dtype<C:BroadcastTensorCollective<B>>(&self,tokens:Tensor<Autodiff<B,S>,2,Int>,positions:Option<Tensor<Autodiff<B,S>,2,Int>>,types:Option<Tensor<Autodiff<B,S>,2,Int>>,
        communicator:C,layout:&VocabParallelLossLayout,compute:FloatDType,output:FloatDType) -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        self.forward_with_dtype_and_dropout(tokens,positions,types,communicator,layout,Some(compute),Some(output),|dropout,input|dropout.forward(input))
    }

    /// Explicit shared dropout and mixed-storage rows with the caller's declared native compute dtype.
    pub fn forward_with_compute_dtype_and_dropout<C,F>(&self,tokens:Tensor<Autodiff<B,S>,2,Int>,positions:Option<Tensor<Autodiff<B,S>,2,Int>>,types:Option<Tensor<Autodiff<B,S>,2,Int>>,
        communicator:C,layout:&VocabParallelLossLayout,compute:FloatDType,output:FloatDType,dropout:F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        self.forward_with_dtype_and_dropout(tokens,positions,types,communicator,layout,Some(compute),Some(output),dropout)
    }

    fn forward_with_dtype_and_dropout<C,F>(&self,tokens:Tensor<Autodiff<B,S>,2,Int>,positions:Option<Tensor<Autodiff<B,S>,2,Int>>,types:Option<Tensor<Autodiff<B,S>,2,Int>>,
        communicator:C,layout:&VocabParallelLossLayout,compute:Option<FloatDType>,output:Option<FloatDType>,dropout:F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        self.validate_inputs(&tokens,&positions,&types,compute);
        let hidden = self.token.forward_with_layout(tokens,communicator,layout)?;let hidden = self.finish(hidden,positions,types,compute,dropout);
        Ok(if let Some(dtype) = output {hidden.cast(dtype)} else {hidden})
    }

    /// Packed native input training with caller-owned document/reset position metadata.
    pub fn forward_packed<C:BroadcastTensorCollective<B>>(&self,tokens:Tensor<Autodiff<B,S>,1,Int>,positions:Option<Tensor<Autodiff<B,S>,1,Int>>,types:Option<Tensor<Autodiff<B,S>,1,Int>>,
        communicator:C,layout:&VocabParallelLossLayout) -> Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        let count = tokens.dims()[0];let convert = |ids:Tensor<Autodiff<B,S>,1,Int>| {assert_eq!(ids.dims(),[count],"packed parallel input metadata length differs");ids.reshape([1,count])};
        self.forward(tokens.reshape([1,count]),positions.map(convert),types.map(convert),communicator,layout)
            .map(|hidden| {let features = hidden.dims()[2];hidden.reshape([count,features])})
    }
}
