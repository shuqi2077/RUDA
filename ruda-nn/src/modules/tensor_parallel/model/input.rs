use super::*;

/// Actual token and optional learned-table metadata, with no inferred positions or type IDs.
/// Rank two represents batched sequences; rank one represents a flat packed token payload.
#[derive(Clone,Debug)]
pub struct TensorParallelTransformerInput<B:Backend,const D:usize=2> {
    /// Actual global token IDs under the separately supplied input vocabulary layout.
    pub tokens:Tensor<B,D,Int>,
    /// Caller-owned learned position IDs, present exactly when the input table is present.
    pub positions:Option<Tensor<B,D,Int>>,
    /// Caller-owned token-type/segment IDs for the actual optional input table.
    pub token_types:Option<Tensor<B,D,Int>>,
    /// Explicit embedding arithmetic/output dtype; None preserves original same-storage behavior.
    pub embedding_dtypes:Option<(FloatDType,FloatDType)>,
}

impl<B:Backend,const D:usize> TensorParallelTransformerInput<B,D> {
    /// Declare only real tokens. Add metadata explicitly when the prepared tables require it.
    pub fn new(tokens:Tensor<B,D,Int>) -> Self {Self {tokens,positions:None,token_types:None,embedding_dtypes:None}}

    /// Supply the actual learned/reset/absolute position IDs without generating any positions.
    pub fn with_positions(mut self,positions:Tensor<B,D,Int>) -> Self {self.positions = Some(positions);self}

    /// Supply the actual type IDs without guessing a default type.
    pub fn with_token_types(mut self,token_types:Tensor<B,D,Int>) -> Self {self.token_types = Some(token_types);self}

    /// Select explicit lookup-row arithmetic and output storage; full tables are not converted.
    pub fn with_embedding_dtypes(mut self,compute:FloatDType,output:FloatDType) -> Self {
        self.embedding_dtypes = Some((compute,output));self
    }
}

impl<B:Backend> TensorParallelTransformerInput<B,1> {
    pub(super) fn into_batched(self,packed:&PackedSequenceLayout) -> TensorParallelTransformerInput<B> {
        let count = self.tokens.dims()[0];assert_eq!(count,packed.tokens(),"parallel model packed input/document lengths differ");
        let convert = |ids:Tensor<B,1,Int>| {assert_eq!(ids.dims(),[count],"parallel packed table metadata length differs");ids.reshape([1,count])};
        TensorParallelTransformerInput {tokens:self.tokens.reshape([1,count]),positions:self.positions.map(convert),
            token_types:self.token_types.map(convert),embedding_dtypes:self.embedding_dtypes}
    }
}

impl<B:Backend> TensorParallelTransformerInput<B> {
    /// Native lookup with this input's exact table metadata and explicit mixed-dtype policy.
    pub fn embed_inference<C:BroadcastTensorCollective<B>>(self,embeddings:&TensorParallelTransformerEmbeddings<B>,
        communicator:C,layout:&VocabParallelLossLayout) -> Result<Tensor<B,3>,C::Error> {
        match self.embedding_dtypes {
            Some((compute,output))=>embeddings.forward_inference_with_compute_dtype(self.tokens,self.positions,self.token_types,communicator,layout,compute,output),
            None=>embeddings.forward_inference(self.tokens,self.positions,self.token_types,communicator,layout),
        }
    }
}

impl<B:Backend,S:CheckpointStrategy> TensorParallelTransformerInput<Autodiff<B,S>> {
    /// Native input-table graph using original backend dropout behavior.
    pub fn embed<C:BroadcastTensorCollective<B>>(self,embeddings:&TensorParallelTransformerEmbeddings<Autodiff<B,S>>,
        communicator:C,layout:&VocabParallelLossLayout) -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        self.embed_with_dropout(embeddings,communicator,layout,|dropout,input|dropout.forward(input))
    }

    /// Explicit corresponding input dropout while preserving original addition/norm/dtype order.
    pub fn embed_with_dropout<C,F>(self,embeddings:&TensorParallelTransformerEmbeddings<Autodiff<B,S>>,communicator:C,
        layout:&VocabParallelLossLayout,dropout:F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        match self.embedding_dtypes {
            Some((compute,output))=>embeddings.forward_with_compute_dtype_and_dropout(self.tokens,self.positions,self.token_types,communicator,layout,compute,output,dropout),
            None=>embeddings.forward_with_dropout(self.tokens,self.positions,self.token_types,communicator,layout,dropout),
        }
    }
}
