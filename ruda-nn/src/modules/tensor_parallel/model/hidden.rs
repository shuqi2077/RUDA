use super::*;

impl<B:Backend> TensorParallelTransformerModel<B> {
    /// Complete native input/backbone/final-norm path with actual per-layer architecture policy.
    /// Positions, masks, groups and any specialized layer arithmetic remain in the callback.
    pub fn forward_hidden_inference_with<C,F>(&self,input:TensorParallelTransformerInput<B>,input_group:C,
        input_layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        let hidden = input.embed_inference(&self.embeddings,input_group,input_layout)?;
        self.backbone.forward_inference_with(hidden,layer).map(|hidden|self.finish(hidden))
    }

    /// Flat packed input lookup and original layer sequence, without padding documents together.
    pub fn forward_packed_hidden_inference_with<C,F>(&self,input:TensorParallelTransformerInput<B,1>,packed:&PackedSequenceLayout,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<B,2>,C::Error>
        where C:BroadcastTensorCollective<B>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        let hidden = input.into_batched(packed).embed_inference(&self.embeddings,input_group,input_layout)?;
        let width = hidden.dims()[2];let hidden = hidden.reshape([packed.tokens(),width]);
        self.backbone.forward_packed_inference_with(hidden,layer).map(|hidden|self.finish(hidden))
    }

    /// Actual new input rows and native per-layer history, including explicit absolute positions.
    /// Cache completion follows the existing backbone contract; a failed layer can require record restoration.
    pub fn forward_cached_hidden_inference_with<C,F>(&self,input:TensorParallelTransformerInput<B>,cache:&mut TransformerKvCache<B>,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>)->Result<Tensor<B,3>,C::Error> {
        let hidden = input.embed_inference(&self.embeddings,input_group,input_layout)?;
        self.backbone.forward_cached_inference_with(hidden,cache,layer).map(|hidden|self.finish(hidden))
    }
}

impl<B:Backend,S:CheckpointStrategy> TensorParallelTransformerModel<Autodiff<B,S>> {
    /// Native end-to-end hidden graph, retaining the actual embedding/head leaves and layer choices.
    pub fn forward_hidden_with<C,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,input_group:C,
        input_layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        self.forward_hidden_with_dropout(input,input_group,input_layout,layer,|dropout,input|dropout.forward(input))
    }

    /// Caller-supplied combined-input dropout with complete original per-layer and final-norm graph.
    pub fn forward_hidden_with_dropout<C,F,I>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,input_group:C,
        input_layout:&VocabParallelLossLayout,layer:F,input_dropout:I) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            I:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let hidden = input.embed_with_dropout(&self.embeddings,input_group,input_layout,input_dropout)?;
        self.backbone.forward_with(hidden,layer).map(|hidden|self.finish(hidden))
    }

    /// Packed native input/backbone/final normalization with actual independent-document metadata.
    pub fn forward_packed_hidden_with<C,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,packed:&PackedSequenceLayout,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        self.forward_packed_hidden_with_dropout(input,packed,input_group,input_layout,layer,|dropout,input|dropout.forward(input))
    }

    /// Explicit shared packed-input dropout over actual lookup rows, without adding padding tokens.
    pub fn forward_packed_hidden_with_dropout<C,F,I>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,packed:&PackedSequenceLayout,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,input_dropout:I) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            I:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let hidden = input.into_batched(packed).embed_with_dropout(&self.embeddings,input_group,input_layout,input_dropout)?;
        let width = hidden.dims()[2];let hidden = hidden.reshape([packed.tokens(),width]);
        self.backbone.forward_packed_with(hidden,layer).map(|hidden|self.finish(hidden))
    }

    /// New-row cached graph with the native detached-history contract, not full-history training.
    pub fn forward_cached_hidden_with<C,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,cache:&mut TransformerKvCache<Autodiff<B,S>>,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,&mut ProjectedKvCache<Autodiff<B,S>>)
                ->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let hidden = input.embed(&self.embeddings,input_group,input_layout)?;
        self.backbone.forward_cached_with(hidden,cache,layer).map(|hidden|self.finish(hidden))
    }
}
