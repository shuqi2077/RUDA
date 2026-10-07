use super::*;

/// Complete actual paired model with independently declared source/target tables and hidden widths.
/// Native encoder memory remains in the AD graph; it is not detached for distributed training.
#[derive(Module,Debug)]
pub struct FullyShardedEncoderDecoderModel<B:Backend> {
    /// Actual source token/learned-position/type tables.
    pub source_embeddings:FullyShardedTransformerEmbeddings<B>,
    /// Original ordered encoder self-attention/FFN layers.
    pub encoder:FullyShardedTransformerStack<B>,
    /// Original explicit encoder-final norm before source-memory projections.
    pub encoder_normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Actual target tables, with only explicitly shared parameter identities.
    pub target_embeddings:FullyShardedTransformerEmbeddings<B>,
    /// Original target self -> source cross -> FFN layers.
    pub decoder:FullyShardedEncoderDecoderStack<B>,
    /// Original independently declared decoder-final norm.
    pub decoder_normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Actual target vocabulary/classification head.
    pub head:FullyShardedTransformerHead<B>,
}

impl<B:Backend> FullyShardedEncoderDecoderModel<B> {
    /// Connect actual prepared modules, without inferring weight ties or requiring equal source/target widths.
    pub fn from_parts(source_embeddings:FullyShardedTransformerEmbeddings<B>,encoder:FullyShardedTransformerStack<B>,
        encoder_normalization:Option<FullyShardedTransformerNorm<B>>,target_embeddings:FullyShardedTransformerEmbeddings<B>,
        decoder:FullyShardedEncoderDecoderStack<B>,decoder_normalization:Option<FullyShardedTransformerNorm<B>>,
        head:FullyShardedTransformerHead<B>) -> Self {
        let source=source_embeddings.hidden_width();let target=target_embeddings.hidden_width();
        assert_eq!(head.hidden_width(),target,"sharded paired target/head widths differ");
        if let Some(norm)=&encoder_normalization {assert_eq!(norm.width(),source,"sharded encoder final norm width differs");}
        if let Some(norm)=&decoder_normalization {assert_eq!(norm.width(),target,"sharded decoder final norm width differs");}
        for block in &encoder.blocks {validate_block_width(block,source);}
        for layer in &decoder.layers {
            validate_block_width(&layer.backbone,target);
            let cross=&layer.cross_attention;
            assert_eq!(cross.query_norm.width(),target,"sharded paired cross query norm width differs");
            if let Some(norm)=&cross.memory_norm {assert_eq!(norm.width(),source,"sharded paired memory norm width differs");}
            assert_eq!(projection_geometry(&cross.attention.query).0,target,"sharded paired cross query width differs");
            assert_eq!(projection_geometry(&cross.attention.key).0,source,"sharded paired cross key width differs");
            assert_eq!(projection_geometry(&cross.attention.value).0,source,"sharded paired cross value width differs");
            assert_eq!(projection_geometry(&cross.attention.output).1,target,"sharded paired cross output width differs");
        }
        Self {source_embeddings,encoder,encoder_normalization,target_embeddings,decoder,decoder_normalization,head}
    }
}

macro_rules! paired_construction {
    ($method:ident,$encoder:ty,$decoder:ty,$head:ty,$shard_encoder:ident,$shard_decoder:ident,$shard_head:ident) => {
        impl<B:Backend> ShardingContext<B> {
            /// Partition the actual complete loaded paired model through one cross-component alias context.
            /// Original source/target vocabulary, head and each actual native stage remain independent.
            pub fn $method(&mut self,source:crate::transformer::TransformerEmbeddings<B>,encoder:$encoder,
                source_norm:Option<crate::transformer::DenseTransformerNorm<B>>,target:crate::transformer::TransformerEmbeddings<B>,
                decoder:$decoder,target_norm:Option<crate::transformer::DenseTransformerNorm<B>>,head:$head)
                -> FullyShardedEncoderDecoderModel<B> {
                let source=self.transformer_embeddings(source);let encoder=self.$shard_encoder(encoder);
                let source_norm=source_norm.map(|norm|self.normalization(norm));
                let target=self.transformer_embeddings(target);let decoder=self.$shard_decoder(decoder);
                let target_norm=target_norm.map(|norm|self.normalization(norm));let head=self.$shard_head(head);
                FullyShardedEncoderDecoderModel::from_parts(source,encoder,source_norm,target,decoder,target_norm,head)
            }
        }
    };
}
paired_construction!(encoder_decoder_model,crate::transformer::DenseTransformerStack<B>,crate::transformer::DenseEncoderDecoderStack<B>,
    crate::transformer::TransformerHead<B>,transformer_stack,encoder_decoder_stack,transformer_head);
paired_construction!(adapted_encoder_decoder_model,crate::transformer::AdaptedTransformerStack<B>,crate::transformer::AdaptedEncoderDecoderStack<B>,
    crate::transformer::AdaptedTransformerHead<B>,adapted_transformer_stack,adapted_encoder_decoder_stack,adapted_transformer_head);

macro_rules! paired_execution {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$embed:ident,$encode:ident,$packed_encode:ident,$decode:ident,$packed_decode:ident,
        $hidden:ident,$packed_hidden:ident,$forward:ident,$packed_forward:ident,$sequence:ident,$packed_sequence:ident) => {
        impl<$($generics)*> FullyShardedEncoderDecoderModel<$backend> {
            /// Encode actual source rows once, retaining original source gradients and final norm.
            pub fn $encode<C,E>(&self,input:FullyShardedTransformerInput<$backend>,communicator:C,layer:E)
                -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error> {
                let [batch,tokens]=input.tokens.dims();
                let hidden=self.source_embeddings.$embed(input,communicator.clone())?;
                let hidden=self.encoder.forward_with(hidden,layer)?;
                assert_eq!(hidden.dims(),[batch,tokens,self.source_embeddings.hidden_width()],"sharded encoder changed actual source rows");
                Ok(if let Some(norm)=&self.encoder_normalization {norm.$gather(communicator)?.forward(hidden)} else {hidden})
            }
            /// Encode independent actual flat source documents with their separately supplied boundaries.
            pub fn $packed_encode<C,E>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,communicator:C,layer:E)
                -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error> {
                let hidden=self.source_embeddings.$embed(packed_input(input,layout),communicator.clone())?.reshape([layout.tokens(),self.source_embeddings.hidden_width()]);
                let hidden=self.encoder.forward_packed_with(hidden,layer)?;
                assert_eq!(hidden.dims(),[layout.tokens(),self.source_embeddings.hidden_width()],"sharded packed encoder changed source rows");
                Ok(if let Some(norm)=&self.encoder_normalization {norm.$gather(communicator)?.forward(hidden)} else {hidden})
            }
            /// Decode actual target inputs against the exact supplied encoder memory, not a detached shadow.
            pub fn $decode<C,F>(&self,input:FullyShardedTransformerInput<$backend>,memory:Tensor<$backend,3>,communicator:C,layer:F)
                -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedEncoderDecoderLayer<$backend>,Tensor<$backend,3>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error> {
                let [batch,tokens]=input.tokens.dims();
                assert_eq!(memory.dims()[0],batch,"sharded paired source/target batch rows differ");
                assert_eq!(memory.dims()[2],self.source_embeddings.hidden_width(),"sharded paired source memory width differs");
                assert_eq!(memory.device(),input.tokens.device(),"sharded paired source/target devices differ");
                let hidden=self.target_embeddings.$embed(input,communicator.clone())?;
                let hidden=self.decoder.forward_with(hidden,memory,layer)?;
                assert_eq!(hidden.dims(),[batch,tokens,self.head.hidden_width()],"sharded decoder changed actual target rows");
                Ok(if let Some(norm)=&self.decoder_normalization {norm.$gather(communicator)?.forward(hidden)} else {hidden})
            }
            /// Actual paired packed memory, with independent lengths and matching document identities/order.
            pub fn $packed_decode<C,F>(&self,input:FullyShardedTransformerInput<$backend,1>,memory:Tensor<$backend,2>,
                source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,communicator:C,layer:F)
                -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedEncoderDecoderLayer<$backend>,Tensor<$backend,2>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error> {
                assert_eq!(source_layout.documents(),target_layout.documents(),"sharded paired packed document counts differ");
                assert_eq!(memory.dims(),[source_layout.tokens(),self.source_embeddings.hidden_width()],"sharded paired packed source memory differs");
                assert_eq!(memory.device(),input.tokens.device(),"sharded paired packed source/target devices differ");
                let hidden=self.target_embeddings.$embed(packed_input(input,target_layout),communicator.clone())?.reshape([target_layout.tokens(),self.target_embeddings.hidden_width()]);
                let hidden=self.decoder.forward_packed_with(hidden,memory,layer)?;
                assert_eq!(hidden.dims(),[target_layout.tokens(),self.head.hidden_width()],"sharded packed decoder changed actual target rows");
                Ok(if let Some(norm)=&self.decoder_normalization {norm.$gather(communicator)?.forward(hidden)} else {hidden})
            }
            /// Complete real encoder and paired decoder graph with architecture-owned per-layer execution.
            pub fn $hidden<C,E,F>(&self,source:FullyShardedTransformerInput<$backend>,target:FullyShardedTransformerInput<$backend>,
                communicator:C,encoder:E,decoder:F) -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error>,
                    F:FnMut(usize,&FullyShardedEncoderDecoderLayer<$backend>,Tensor<$backend,3>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error> {
                let memory=self.$encode(source,communicator.clone(),encoder)?;self.$decode(target,memory,communicator,decoder)
            }
            /// Complete independent-document source and target graph, without padding or cross-document pooling.
            pub fn $packed_hidden<C,E,F>(&self,source:FullyShardedTransformerInput<$backend,1>,target:FullyShardedTransformerInput<$backend,1>,
                source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,communicator:C,encoder:E,decoder:F) -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error>,
                    F:FnMut(usize,&FullyShardedEncoderDecoderLayer<$backend>,Tensor<$backend,2>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error> {
                let memory=self.$packed_encode(source,source_layout,communicator.clone(),encoder)?;
                self.$packed_decode(target,memory,source_layout,target_layout,communicator,decoder)
            }
            /// Actual paired model target logits with complete original source-memory derivatives.
            pub fn $forward<C,E,F>(&self,source:FullyShardedTransformerInput<$backend>,target:FullyShardedTransformerInput<$backend>,
                communicator:C,encoder:E,decoder:F) -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error>,
                    F:FnMut(usize,&FullyShardedEncoderDecoderLayer<$backend>,Tensor<$backend,3>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error> {
                let hidden=self.$hidden(source,target,communicator.clone(),encoder,decoder)?;self.head.$embed(hidden,communicator)
            }
            /// Original complete paired packed token logits, preserving target document boundaries.
            pub fn $packed_forward<C,E,F>(&self,source:FullyShardedTransformerInput<$backend,1>,target:FullyShardedTransformerInput<$backend,1>,
                source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,communicator:C,encoder:E,decoder:F) -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error>,
                    F:FnMut(usize,&FullyShardedEncoderDecoderLayer<$backend>,Tensor<$backend,2>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error> {
                let hidden=self.$packed_hidden(source,target,source_layout,target_layout,communicator.clone(),encoder,decoder)?;self.head.$embed(hidden,communicator)
            }
            /// Explicit target-sequence classification after actual paired source/target execution.
            pub fn $sequence<C,E,F>(&self,source:FullyShardedTransformerInput<$backend>,target:FullyShardedTransformerInput<$backend>,
                visible:Tensor<$backend,2,Bool>,pooling:SequencePooling,communicator:C,encoder:E,decoder:F) -> Result<SequenceHeadOutput<$backend>,C::Error>
                where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error>,
                    F:FnMut(usize,&FullyShardedEncoderDecoderLayer<$backend>,Tensor<$backend,3>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error> {
                let hidden=self.$hidden(source,target,communicator.clone(),encoder,decoder)?;
                Ok(self.head.$gather(communicator)?.forward_sequence(hidden,visible,pooling))
            }
            /// Independent target-document classification and exact original visibility/I64 counts.
            pub fn $packed_sequence<C,E,F>(&self,source:FullyShardedTransformerInput<$backend,1>,target:FullyShardedTransformerInput<$backend,1>,
                source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,visible:Option<Tensor<$backend,1,Bool>>,pooling:SequencePooling,
                communicator:C,encoder:E,decoder:F) -> Result<SequenceHeadOutput<$backend>,C::Error>
                where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error>,
                    F:FnMut(usize,&FullyShardedEncoderDecoderLayer<$backend>,Tensor<$backend,2>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error> {
                let hidden=self.$packed_hidden(source,target,source_layout,target_layout,communicator.clone(),encoder,decoder)?;
                Ok(self.head.$gather(communicator)?.forward_packed_sequences(hidden,target_layout,visible,pooling))
            }
        }
    };
}
paired_execution!(B,[B:Backend],gather_inference,forward_inference,encode_inference_with,encode_packed_inference_with,
    decode_hidden_inference_with,decode_packed_hidden_inference_with,forward_hidden_inference_with,forward_packed_hidden_inference_with,
    forward_inference_with,forward_packed_inference_with,forward_sequence_inference_with,forward_packed_sequences_inference_with);
paired_execution!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather,forward,encode_with,encode_packed_with,
    decode_hidden_with,decode_packed_hidden_with,forward_hidden_with,forward_packed_hidden_with,
    forward_with,forward_packed_with,forward_sequence_with,forward_packed_sequences_with);
