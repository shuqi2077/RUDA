use super::*;
use super::super::super::VocabParallelCrossEntropy;
use crate::{loss::{LossTerms,KLDivLoss,CausalCrossEntropyConfig,CausalLoss},pool::SequencePooling,transformer::SequenceHeadOutput};
use ruda_model::tensor::Bool;

impl<B:Backend,S:CheckpointStrategy> TensorParallelEncoderDecoderModel<Autodiff<B,S>> {
    /// Complete paired logits with explicit corresponding source/target lookup and head/adapter dropout.
    /// Each actual architecture callback retains its own attention/FFN dropout and position policies.
    pub fn forward_with_dropouts<C,T,O,E,F,I,J,H,A>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>>,target:TensorParallelTransformerInput<Autodiff<B,S>>,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool,source_dropout:I,target_dropout:J,head_dropout:H,adapter_dropout:A)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            I:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,J:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,
            H:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,A:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let memory = self.encode_with_dropout(source,source_group,source_layout,encoder,source_dropout)?;
        let hidden = self.decode_hidden_with_dropout(target,memory,target_group,target_layout,decoder,target_dropout)?;
        self.head.forward_with_dropouts(hidden,output_group,output_layout,gather_output,head_dropout,adapter_dropout)
    }

    /// Native packed paired logits with explicit corresponding lookup/head dropout over real rows.
    pub fn forward_packed_with_dropouts<C,T,O,E,F,I,J,H,A>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>,1>,target:TensorParallelTransformerInput<Autodiff<B,S>,1>,
        source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool,
        source_dropout:I,target_dropout:J,head_dropout:H,adapter_dropout:A) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            I:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,J:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,
            H:FnOnce(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2>,A:FnOnce(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2> {
        assert_eq!(source_packed.documents(),target_packed.documents(),"packed paired dropout document counts differ");
        let memory = self.encode_packed_with_dropout(source,source_packed,source_group,source_layout,encoder,source_dropout)?;
        let hidden = self.decode_packed_hidden_with_dropout(target,memory,source_packed,target_packed,target_group,target_layout,decoder,target_dropout)?;
        self.head.forward_with_dropouts(hidden,output_group,output_layout,gather_output,head_dropout,adapter_dropout)
    }

    /// Complete aligned teacher forcing with separate explicit source/target lookup dropout and
    /// native bounded head projection chunks; criterion shift remains disabled for aligned labels.
    pub fn forward_aligned_loss_with_dropouts<C,T,O,E,F,I,J,H,A>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>>,target:TensorParallelTransformerInput<Autodiff<B,S>>,
        labels:Tensor<Autodiff<B,S>,2,Int>,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,source_dropout:I,target_dropout:J,mut head_dropout:H,mut adapter_dropout:A)
        -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            I:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,J:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,
            H:FnMut(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2>,A:FnMut(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2> {
        assert!(!criterion.shift,"aligned paired dropout labels must not be shifted again");
        assert!(labels.dims() == target.tokens.dims() && labels.device() == target.tokens.device(),"paired dropout target/label rows or devices differ");
        let memory = self.encode_with_dropout(source,source_group,source_layout,encoder,source_dropout)?;
        let hidden = self.decode_hidden_with_dropout(target,memory,target_group,target_layout,decoder,target_dropout)?;
        criterion.forward_sharded_hidden(hidden,labels,output_layout,output_group,|rows,group|
            self.head.forward_with_dropouts(rows,group.clone(),output_layout,false,&mut head_dropout,&mut adapter_dropout),label_smoothing)
    }

    /// Actual packed paired teacher forcing and corresponding chunk-wise dropout, without a second
    /// target shift, padded hidden rows or detached encoder memory.
    pub fn forward_packed_aligned_loss_with_dropouts<C,T,O,E,F,I,J,H,A>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>,1>,target:TensorParallelTransformerInput<Autodiff<B,S>,1>,
        labels:Tensor<Autodiff<B,S>,1,Int>,source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,
        source_dropout:I,target_dropout:J,mut head_dropout:H,mut adapter_dropout:A) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            I:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,J:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,
            H:FnMut(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2>,A:FnMut(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2> {
        assert!(!criterion.shift,"aligned packed paired dropout labels must not be shifted again");
        assert!(labels.dims() == target.tokens.dims() && labels.device() == target.tokens.device(),"packed paired dropout target/label rows or devices differ");
        assert_eq!(source_packed.documents(),target_packed.documents(),"packed paired dropout document counts differ");
        let memory = self.encode_packed_with_dropout(source,source_packed,source_group,source_layout,encoder,source_dropout)?;
        let hidden = self.decode_packed_hidden_with_dropout(target,memory,source_packed,target_packed,target_group,target_layout,decoder,target_dropout)?;
        criterion.forward_sharded_packed_hidden(hidden,labels,target_packed,output_layout,output_group,|rows,group|
            self.head.forward_with_dropouts(rows,group.clone(),output_layout,false,&mut head_dropout,&mut adapter_dropout),label_smoothing)
    }

    /// Complete paired hidden graph before the original output head, retaining source gradients.
    pub fn forward_hidden_with<C,T,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>>,target:TensorParallelTransformerInput<Autodiff<B,S>>,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let memory = self.encode_with(source,source_group,source_layout,encoder)?;
        self.decode_hidden_with(target,memory,target_group,target_layout,decoder)
    }

    /// Complete paired flat-document graph with independent real source and target boundaries.
    pub fn forward_packed_hidden_with<C,T,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>,1>,target:TensorParallelTransformerInput<Autodiff<B,S>,1>,
        source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(source_packed.documents(),target_packed.documents(),"packed paired objective document counts differ");
        let memory = self.encode_packed_with(source,source_packed,source_group,source_layout,encoder)?;
        self.decode_packed_hidden_with(target,memory,source_packed,target_packed,target_group,target_layout,decoder)
    }

    /// Native aligned paired hard-label token terms with explicit visibility and local class weights.
    /// Labels are already in decoder positions; no shift, global-logit gather or mask is inferred.
    pub fn forward_token_terms_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>>,target:TensorParallelTransformerInput<Autodiff<B,S>>,
        labels:Tensor<Autodiff<B,S>,2,Int>,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,criterion:&VocabParallelCrossEntropy,
        visible:Option<Tensor<Autodiff<B,S>,2,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>) -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!(labels.dims(),target.tokens.dims(),"paired token labels/decoder rows differ");
        let hidden = self.forward_hidden_with(source,target,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_token_terms(hidden,labels,criterion,output_group,visible,weights)
    }

    /// Flat paired hard-label terms over real aligned target rows, preserving criterion sentinel/counts.
    pub fn forward_packed_token_terms_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>,1>,target:TensorParallelTransformerInput<Autodiff<B,S>,1>,
        labels:Tensor<Autodiff<B,S>,1,Int>,source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,criterion:&VocabParallelCrossEntropy,visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>)
        -> Result<LossTerms<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(labels.dims(),target.tokens.dims(),"packed paired token labels/decoder rows differ");
        let hidden = self.forward_packed_hidden_with(source,target,source_packed,target_packed,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_terms(hidden,labels,criterion,output_group,visible,weights)
    }

    /// Complete native paired soft-label objective over actual local teacher class slices.
    /// Original teacher gradients, probability mass and class-weight semantics are retained.
    pub fn forward_soft_token_terms_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>>,target:TensorParallelTransformerInput<Autodiff<B,S>>,
        targets:Tensor<Autodiff<B,S>,3>,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,criterion:&VocabParallelCrossEntropy,
        visible:Option<Tensor<Autodiff<B,S>,2,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>) -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!([targets.dims()[0],targets.dims()[1]],target.tokens.dims(),"paired soft targets/decoder rows differ");
        let hidden = self.forward_hidden_with(source,target,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_soft_token_terms(hidden,targets,criterion,output_group,visible,weights)
    }

    /// Real flat paired soft targets without projecting padded document rows or shifting distributions.
    pub fn forward_packed_soft_token_terms_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>,1>,target:TensorParallelTransformerInput<Autodiff<B,S>,1>,
        targets:Tensor<Autodiff<B,S>,2>,source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,criterion:&VocabParallelCrossEntropy,visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>)
        -> Result<LossTerms<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(targets.dims()[0],target.tokens.dims()[0],"packed paired soft targets/decoder rows differ");
        let hidden = self.forward_packed_hidden_with(source,target,source_packed,target_packed,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_soft_terms(hidden,targets,criterion,output_group,visible,weights)
    }

    /// End-to-end paired KL terms in the caller's actual probability/log-target space.
    pub fn forward_token_kl_terms_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>>,target:TensorParallelTransformerInput<Autodiff<B,S>>,
        targets:Tensor<Autodiff<B,S>,3>,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,
        criterion:&KLDivLoss,visible:Option<Tensor<Autodiff<B,S>,2,Bool>>) -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!([targets.dims()[0],targets.dims()[1]],target.tokens.dims(),"paired KL targets/decoder rows differ");
        let hidden = self.forward_hidden_with(source,target,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_token_kl_terms(hidden,targets,criterion,output_group,output_layout,visible)
    }

    /// Flat paired KL terms over actual target tokens, without teacher detachment or renormalization.
    pub fn forward_packed_token_kl_terms_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>,1>,target:TensorParallelTransformerInput<Autodiff<B,S>,1>,
        targets:Tensor<Autodiff<B,S>,2>,source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,criterion:&KLDivLoss,visible:Option<Tensor<Autodiff<B,S>,1,Bool>>)
        -> Result<LossTerms<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(targets.dims()[0],target.tokens.dims()[0],"packed paired KL targets/decoder rows differ");
        let hidden = self.forward_packed_hidden_with(source,target,source_packed,target_packed,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_kl_terms(hidden,targets,criterion,output_group,output_layout,visible)
    }

    /// Complete paired target-sequence classification with actual decoder/source gradients.
    /// Original real-token pooling retains empty-row visibility and exact token counts.
    pub fn forward_sequence_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>>,target:TensorParallelTransformerInput<Autodiff<B,S>>,
        visible:Tensor<Autodiff<B,S>,2,Bool>,pooling:SequencePooling,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<SequenceHeadOutput<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!(visible.dims(),target.tokens.dims(),"paired sequence pool/decoder rows differ");
        let hidden = self.forward_hidden_with(source,target,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_sequence(hidden,visible,pooling,output_group,output_layout,gather_output)
    }

    /// Actual packed paired document classification with original independent-document pooling.
    pub fn forward_packed_sequences_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>,1>,target:TensorParallelTransformerInput<Autodiff<B,S>,1>,
        source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,pooling:SequencePooling,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<SequenceHeadOutput<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        let hidden = self.forward_packed_hidden_with(source,target,source_packed,target_packed,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_packed_sequences(hidden,target_packed,visible,pooling,output_group,output_layout,gather_output)
    }
}
