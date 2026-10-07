use super::*;
use crate::loss::LossTerms;

impl<B:Backend,S:CheckpointStrategy> TensorParallelTransformerHead<Autodiff<B,S>> {
    /// Project replicated hidden states into actual local class logits with native input gradients.
    /// Norm parameters receive the summed projection derivative, not a second replica SUM.
    /// Native head dropout must produce corresponding masks/values across this logical group.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with_dropout(hidden,communicator,layout,gather_output,|dropout,input|dropout.forward(input))
    }

    /// Use caller-owned shared head dropout without guessing seeds or changing native RNG behavior.
    pub fn forward_with_dropout<C,F,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool,dropout:F) -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
        check_input(&hidden,&self.local.projection,layout,communicator.rank(),communicator.world_size());
        let hidden = if let Some(norm) = &self.local.normalization {norm.forward(hidden)} else {hidden};
        let hidden = transformed(&self.local.dropout,hidden,dropout);
        let projection = ColumnParallelLinear::from_shard(mask_padding(self.local.projection.clone(),layout,communicator.rank() as usize));
        let logits = projection.forward(hidden,communicator.clone(),false)?;
        if gather_output {layout.gather_logits(logits,communicator)} else {Ok(logits)}
    }
}

impl<B:Backend,S:CheckpointStrategy> TensorParallelAdaptedTransformerHead<Autodiff<B,S>> {
    /// Native local base/B outputs with the full group derivative for replicated adapter A.
    /// Original adapter dtypes and scale are retained; no base merge or full-logit gather is implicit.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with_dropouts(hidden,communicator,layout,gather_output,|dropout,input|dropout.forward(input),|dropout,input|dropout.forward(input))
    }

    /// Explicit shared head-input and adapter-input dropout callbacks in their original order.
    /// The adapter callback receives the actual A storage dtype, not the base/output storage dtype.
    pub fn forward_with_dropouts<C,F,A,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool,head_dropout:F,adapter_dropout:A) -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D>,
            A:FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
        check_input(&hidden,&self.local.projection.base,layout,communicator.rank(),communicator.world_size());
        let hidden = if let Some(norm) = &self.local.normalization {norm.forward(hidden)} else {hidden};
        let hidden = transformed(&self.local.dropout,hidden,head_dropout);
        let projection = adapted_projection(self.local.projection.clone(),layout,communicator.rank() as usize);
        let logits = super::super::adapted::column::<B,S,C,C,_,D>(&projection,hidden,&communicator,None,adapter_dropout)?;
        if gather_output {layout.gather_logits(logits,communicator)} else {Ok(logits)}
    }
}

macro_rules! head_objectives {
    ($head:ident) => {
        impl<B:Backend,S:CheckpointStrategy> $head<Autodiff<B,S>> {
            /// Pool actual visible tokens, then classify with the declared class partition.
            /// The returned visibility/count metadata still describes real tokens, not label counts.
            pub fn forward_sequence<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,3>,visible:Tensor<Autodiff<B,S>,2,Bool>,
                pooling:SequencePooling,communicator:C,layout:&VocabParallelLossLayout,gather_output:bool)
                -> Result<SequenceHeadOutput<Autodiff<B,S>>,C::Error> {
                self.forward_pooled(pool_sequence(hidden,visible,pooling),communicator,layout,gather_output)
            }

            /// Preserve independent packed-document pooling, including actual empty rows.
            pub fn forward_packed_sequences<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,2>,packed:&PackedSequenceLayout,
                visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,pooling:SequencePooling,communicator:C,layout:&VocabParallelLossLayout,gather_output:bool)
                -> Result<SequenceHeadOutput<Autodiff<B,S>>,C::Error> {
                self.forward_pooled(pool_packed_sequences(hidden,packed,visible,pooling),communicator,layout,gather_output)
            }

            /// Project existing pooled hidden values without dropping caller-owned count/visibility.
            pub fn forward_pooled<C:BroadcastTensorCollective<B>>(&self,pooled:SequencePoolOutput<Autodiff<B,S>>,communicator:C,
                layout:&VocabParallelLossLayout,gather_output:bool) -> Result<SequenceHeadOutput<Autodiff<B,S>>,C::Error> {
                Ok(SequenceHeadOutput {logits:self.forward(pooled.values,communicator,layout,gather_output)?,valid_rows:pooled.valid_rows,token_counts:pooled.token_counts})
            }

            /// One actual replicated hard-label objective over local class logits, without gathering.
            /// Labels/masks/criterion must match on ranks; optional weights are local real class slices.
            pub fn forward_terms<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,2>,labels:Tensor<Autodiff<B,S>,1,Int>,
                criterion:&VocabParallelCrossEntropy,communicator:C,visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>)
                -> Result<LossTerms<Autodiff<B,S>>,C::Error> {
                let logits = self.forward(hidden,communicator.clone(),criterion.layout(),false)?;
                criterion.forward_terms(logits,labels,communicator,visible,weights)
            }

            /// Native soft-label loss using actual local slices, without detaching or renormalizing.
            pub fn forward_soft_terms<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,2>,targets:Tensor<Autodiff<B,S>,2>,
                criterion:&VocabParallelCrossEntropy,communicator:C,visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>)
                -> Result<LossTerms<Autodiff<B,S>>,C::Error> {
                let logits = self.forward(hidden,communicator.clone(),criterion.layout(),false)?;
                criterion.forward_soft_terms(logits,targets,communicator,visible,weights)
            }

            /// Per-token hard-label terms with explicit already-aligned targets; no causal shift.
            pub fn forward_token_terms<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,3>,labels:Tensor<Autodiff<B,S>,2,Int>,
                criterion:&VocabParallelCrossEntropy,communicator:C,visible:Option<Tensor<Autodiff<B,S>,2,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>)
                -> Result<LossTerms<Autodiff<B,S>,2>,C::Error> {
                let logits = self.forward(hidden,communicator.clone(),criterion.layout(),false)?;
                criterion.forward_token_terms(logits,labels,communicator,visible,weights)
            }

            /// Per-token soft-label terms using actual local target distributions and visibility.
            pub fn forward_soft_token_terms<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,3>,targets:Tensor<Autodiff<B,S>,3>,
                criterion:&VocabParallelCrossEntropy,communicator:C,visible:Option<Tensor<Autodiff<B,S>,2,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>)
                -> Result<LossTerms<Autodiff<B,S>,2>,C::Error> {
                let logits = self.forward(hidden,communicator.clone(),criterion.layout(),false)?;
                criterion.forward_soft_token_terms(logits,targets,communicator,visible,weights)
            }

            /// Use the native causal config's real target shift/sentinel and bounded projection chunks.
            /// Only local vocabulary logits are projected; the loss remains one replicated TP objective.
            pub fn forward_causal_loss<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,3>,labels:Tensor<Autodiff<B,S>,2,Int>,
                criterion:&crate::loss::CausalCrossEntropyConfig,communicator:C,layout:&VocabParallelLossLayout,label_smoothing:f64)
                -> Result<crate::loss::CausalLoss<Autodiff<B,S>>,C::Error> {
                criterion.forward_sharded_hidden(hidden,labels,layout,communicator,|rows,group|self.forward(rows,group.clone(),layout,false),label_smoothing)
            }

            /// Chunk actual packed document states without shifting supervision across their boundaries.
            pub fn forward_packed_causal_loss<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,2>,labels:Tensor<Autodiff<B,S>,1,Int>,
                packed:&PackedSequenceLayout,criterion:&crate::loss::CausalCrossEntropyConfig,communicator:C,layout:&VocabParallelLossLayout,label_smoothing:f64)
                -> Result<crate::loss::CausalLoss<Autodiff<B,S>>,C::Error> {
                criterion.forward_sharded_packed_hidden(hidden,labels,packed,layout,communicator,|rows,group|self.forward(rows,group.clone(),layout,false),label_smoothing)
            }
        }
    };
}
head_objectives!(TensorParallelTransformerHead);
head_objectives!(TensorParallelAdaptedTransformerHead);
pub(super) use head_objectives;
