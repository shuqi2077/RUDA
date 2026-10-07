use super::{Backend,Tensor,Param,check_storage,ColumnParallelLinear};
use crate::Embedding;
use super::super::{VocabParallelLossLayout,VocabParallelEmbedding,VocabParallelProjection,VocabParallelTransformerHead,
    VocabParallelLoRAProjection,VocabParallelAdaptedTransformerHead};

fn rows<B:Backend,const D:usize>(parameter:Param<Tensor<B,D>>,layout:&VocabParallelLossLayout,rank:usize) -> Param<Tensor<B,D>> {
    assert!(D > 0,"vocabulary storage requires a row axis");
    let shape = parameter.val().dims();let interval = layout.interval(rank);
    assert_eq!(shape[0],layout.storage_size(),"full vocabulary rows differ from declared actual storage");
    if interval == (0..shape[0]) {return parameter;}
    check_storage(&parameter.val());
    parameter.map(|value| {
        let trainable = value.is_require_grad();value.slice_dim(0,interval).detach().set_require_grad(trainable)
    })
}

fn same_embedding<B:Backend>(embedding:&Embedding<B>,projection:&VocabParallelProjection<B>) {
    let source = embedding.weight.val();let target = projection.weight.val();
    assert!(embedding.weight.id == projection.weight.id && source.dims() == target.dims() && source.dtype() == target.dtype()
        && source.device() == target.device() && source.is_require_grad() == target.is_require_grad(),
        "full embedding/head must identify the same actual native parameter and contract");
}

impl<B:Backend> VocabParallelEmbedding<B> {
    /// Partition actual loaded complete floating embedding rows without dtype/ID/flag changes.
    /// The full table includes exactly the supplied storage rows. Packed partial tables are loaded
    /// explicitly as native local shards; no full dequantization, padding creation or optimizer conversion occurs.
    pub fn from_full_with_layout(mut embedding:Embedding<B>,layout:&VocabParallelLossLayout,rank:usize,padding_index:Option<usize>) -> Self {
        assert!(embedding.weight.val().dims()[1] > 0,"full embedding hidden width must be positive");
        embedding.weight = rows(embedding.weight,layout,rank);
        Self::from_shard_with_layout(embedding,layout,rank,padding_index)
    }
}

impl<B:Backend> VocabParallelProjection<B> {
    /// Partition an actual complete row-major floating head and matching output bias.
    /// Use the joint tied constructor when this weight is shared with an embedding, so both
    /// modules receive the same local leaf rather than independent leaves with the same ID.
    pub fn from_full_with_layout(self,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        assert!(self.weight.val().dims()[1] > 0,"full vocabulary head hidden width must be positive");
        if let Some(bias) = &self.bias {
            assert_eq!(bias.val().dims(),[layout.storage_size()],"full vocabulary head bias rows differ");
            assert_eq!(bias.val().device(),self.weight.val().device(),"full vocabulary head bias/weight devices differ");
        }
        Self {weight:rows(self.weight,layout,rank),bias:self.bias.map(|bias|rows(bias,layout,rank))}
    }
}

impl<B:Backend> VocabParallelTransformerHead<B> {
    /// Partition an actual loaded independent native head, retaining original norm/dropout.
    pub fn from_full(head:Self,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        Self::from_projection(VocabParallelProjection::from_full_with_layout(head.projection,layout,rank),head.normalization,head.dropout,layout,rank)
    }

    /// Partition one actual shared full embedding/head weight into one shared local leaf.
    /// Both native modules keep original IDs/mappers/flags and accumulate into that same leaf.
    pub fn from_full_tied(embedding:Embedding<B>,head:Self,layout:&VocabParallelLossLayout,rank:usize,padding_index:Option<usize>)
        -> (VocabParallelEmbedding<B>,Self) {
        same_embedding(&embedding,&head.projection);
        let embedding = VocabParallelEmbedding::from_full_with_layout(embedding,layout,rank,padding_index);
        let bias = head.projection.bias.map(|bias|rows(bias,layout,rank));
        let head = Self::from_embedding(&embedding,bias,head.normalization,head.dropout,layout,rank);
        (embedding,head)
    }
}

impl<B:Backend> VocabParallelLoRAProjection<B> {
    /// Partition actual full frozen vocabulary rows and B columns, retaining original replicated A.
    /// Rank, multiplier, storage, dropout and flags are preserved; no adapter initialization/merge occurs.
    pub fn from_full(mut layer:Self,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        let [classes,hidden] = layer.base.weight.val().dims();let adapter_rank = layer.adapter_a.weight.val().dims()[1];
        assert_eq!(classes,layout.storage_size(),"full vocabulary adapter base rows differ");
        assert_eq!(layer.adapter_a.weight.val().dims(),[hidden,adapter_rank],"full vocabulary adapter A geometry differs");
        assert_eq!(layer.adapter_b.weight.val().dims(),[adapter_rank,classes],"full vocabulary adapter B geometry differs");
        let interval = layout.interval(rank);
        if interval != (0..classes) {
            assert!(layer.base.weight.id != layer.adapter_a.weight.id && layer.base.weight.id != layer.adapter_b.weight.id
                && layer.adapter_a.weight.id != layer.adapter_b.weight.id,"tied full adapter roles require explicit compatible local loading");
        }
        layer.base = VocabParallelProjection::from_full_with_layout(layer.base,layout,rank);
        layer.adapter_b = ColumnParallelLinear::from_full(layer.adapter_b,interval).local;
        Self::from_adapters(layer.base,layer.adapter_a,layer.adapter_b,layer.dropout,layer.scale)
    }
}

impl<B:Backend> VocabParallelAdaptedTransformerHead<B> {
    /// Partition real full head adapter storage without changing normalization/dropout or LoRA options.
    pub fn from_full(head:Self,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        Self::from_projection(VocabParallelLoRAProjection::from_full(head.projection,layout,rank),head.normalization,head.dropout,layout,rank)
    }

    /// Partition a frozen tied base once, preserving the shared embedding/head leaf and original A/B.
    pub fn from_full_tied(embedding:Embedding<B>,head:Self,layout:&VocabParallelLossLayout,rank:usize,padding_index:Option<usize>)
        -> (VocabParallelEmbedding<B>,Self) {
        same_embedding(&embedding,&head.projection.base);
        let full = head.projection;
        let classes = full.base.weight.val().dims()[0];let hidden = full.base.weight.val().dims()[1];let adapter_rank = full.adapter_a.weight.val().dims()[1];
        assert_eq!(full.adapter_a.weight.val().dims(),[hidden,adapter_rank],"full tied head adapter A geometry differs");
        assert_eq!(full.adapter_b.weight.val().dims(),[adapter_rank,classes],"full tied head adapter B geometry differs");
        let interval = layout.interval(rank);
        if interval != (0..classes) {
            assert!(full.base.weight.id != full.adapter_a.weight.id && full.base.weight.id != full.adapter_b.weight.id
                && full.adapter_a.weight.id != full.adapter_b.weight.id,"tied full adapter roles require compatible explicit local loading");
        }
        let embedding = VocabParallelEmbedding::from_full_with_layout(embedding,layout,rank,padding_index);
        let base = VocabParallelProjection::from_embedding(&embedding,full.base.bias.map(|bias|rows(bias,layout,rank)));
        let projection = VocabParallelLoRAProjection::from_adapters(base,full.adapter_a,
            ColumnParallelLinear::from_full(full.adapter_b,interval).local,full.dropout,full.scale);
        let head = Self::from_projection(projection,head.normalization,head.dropout,layout,rank);
        (embedding,head)
    }
}
