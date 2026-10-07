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
        }
    };
}
inference_heads!(TensorParallelTransformerHead);
inference_heads!(TensorParallelAdaptedTransformerHead);
inference_heads!(VocabParallelTransformerHead);
