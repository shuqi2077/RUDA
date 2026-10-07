use super::*;

macro_rules! inference_heads {
    ($head:ident) => {
        impl<B:Backend> $head<B> {
            /// Native-backend sequence classification using the caller's actual visible-token policy.
            /// Returns original valid-row/count metadata and local logits unless gathering is requested.
            pub fn forward_sequence_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,3>,visible:Tensor<B,2,Bool>,
                pooling:SequencePooling,communicator:C,layout:&VocabParallelLossLayout,gather_output:bool)
                -> Result<SequenceHeadOutput<B>,C::Error> {
                self.forward_pooled_inference(pool_sequence(hidden,visible,pooling),communicator,layout,gather_output)
            }

            /// Classify independent actual packed documents on the native inference backend.
            /// Empty document rows retain the original pool/head behavior, including projection bias.
            pub fn forward_packed_sequences_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,2>,packed:&PackedSequenceLayout,
                visible:Option<Tensor<B,1,Bool>>,pooling:SequencePooling,communicator:C,layout:&VocabParallelLossLayout,gather_output:bool)
                -> Result<SequenceHeadOutput<B>,C::Error> {
                self.forward_pooled_inference(pool_packed_sequences(hidden,packed,visible,pooling),communicator,layout,gather_output)
            }

            /// Native inference from already pooled hidden values without replacing their metadata.
            pub fn forward_pooled_inference<C:BroadcastTensorCollective<B>>(&self,pooled:SequencePoolOutput<B>,communicator:C,
                layout:&VocabParallelLossLayout,gather_output:bool) -> Result<SequenceHeadOutput<B>,C::Error> {
                Ok(SequenceHeadOutput {logits:self.forward_inference(pooled.values,communicator,layout,gather_output)?,
                    valid_rows:pooled.valid_rows,token_counts:pooled.token_counts})
            }

            /// Select native greedy global class IDs directly from rank-local head logits.
            /// Uses explicit real vocabulary layout/visibility without gathering complete outputs.
            pub fn forward_greedy_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,2>,communicator:C,
                layout:&VocabParallelLossLayout,visible:Option<Tensor<B,1,Bool>>) -> Result<super::super::VocabParallelGreedySelection<B>,C::Error> {
                let logits = self.forward_inference(hidden,communicator.clone(),layout,false)?;
                layout.greedy_indices_inference(logits,communicator,visible)
            }

            /// Project/select only the actual last hidden token of each nonempty cached sequence.
            pub fn forward_greedy_last_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,3>,communicator:C,
                layout:&VocabParallelLossLayout,visible:Option<Tensor<B,1,Bool>>) -> Result<super::super::VocabParallelGreedySelection<B>,C::Error> {
                let [batch,tokens,features] = hidden.dims();assert!(tokens > 0,"cached greedy head requires an actual last token");
                self.forward_greedy_inference(hidden.slice([0..batch,tokens-1..tokens,0..features]).reshape([batch,features]),communicator,layout,visible)
            }

            /// Native local teacher/inference log probabilities normalized over every real global class.
            pub fn forward_log_probabilities_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,2>,communicator:C,
                layout:&VocabParallelLossLayout,visible:Option<Tensor<B,1,Bool>>) -> Result<Tensor<B,2>,C::Error> {
                let logits = self.forward_inference(hidden,communicator.clone(),layout,false)?;
                layout.log_softmax_inference(logits,communicator,visible)
            }

            /// Native complete-class probability normalization without collecting full head logits.
            pub fn forward_probabilities_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,2>,communicator:C,
                layout:&VocabParallelLossLayout,visible:Option<Tensor<B,1,Bool>>) -> Result<Tensor<B,2>,C::Error> {
                let logits = self.forward_inference(hidden,communicator.clone(),layout,false)?;
                layout.softmax_inference(logits,communicator,visible)
            }
        }
    };
}
inference_heads!(TensorParallelTransformerHead);
inference_heads!(TensorParallelAdaptedTransformerHead);
inference_heads!(VocabParallelTransformerHead);
inference_heads!(VocabParallelAdaptedTransformerHead);
inference_heads!(TensorParallelOutputHead);
