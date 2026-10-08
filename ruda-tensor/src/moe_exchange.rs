//! Native routing/dispatch, received expert rows and ordered combine for expert-parallel graphs.
use alloc::vec::Vec;
use core::fmt::Debug;
use crate::{Backend,moe::{MoeOps,MoeOptions,MoeExpertStrategy,MoeCombineGradientStrategy},tensor::{FloatTensor,IntTensor}};

/// Actual original discrete dispatch and continuous weights, before any cross-rank exchange.
#[derive(Debug)]
pub struct MoeDispatched<B:MoeDispatchOps> {
    /// Original expert-sorted source row copies, one row per selected assignment.
    pub values:FloatTensor<B>,
    /// Original FP32 continuous `[tokens,top_k]` weights, with its actual logits derivative.
    pub weights:FloatTensor<B>,
    /// Original selected U32 `[tokens,top_k]` expert IDs.
    pub selected_experts:IntTensor<B>,
    /// Original sorted U32 `[assignments]` global expert IDs, aligned with values.
    pub row_experts:IntTensor<B>,
    /// Actual private original native mappings and router VJP state.
    pub state:B::MoeDispatchState,
}
/// Requested original combine VJP outputs.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct MoeCombineSelection {
    /// Original expert-row output derivative.
    pub experts:bool,
    /// Original FP32 continuous routing-weight derivative.
    pub weights:bool,
}
/// Actual optional original combine derivatives, not placeholder zero tensors.
#[derive(Debug)]
pub struct MoeCombineBackward<B:Backend> {
    /// Requested derivative of source expert-sorted row values.
    pub experts:Option<FloatTensor<B>>,
    /// Requested FP32 derivative of `[tokens,top_k]` continuous weights.
    pub weights:Option<FloatTensor<B>>,
}
/// Independent native source-side dispatch and combine, retaining discrete selection semantics.
pub trait MoeDispatchOps:MoeOps {
    /// Original valid private native row mappings, with no unchecked imported offsets.
    type MoeDispatchState:Clone+Send+Debug+'static;
    /// Original selection and continuous weights followed by device COPY dispatch.
    fn moe_dispatch(input:FloatTensor<Self>,logits:FloatTensor<Self>,bias:Option<FloatTensor<Self>>,options:MoeOptions)
        -> Result<MoeDispatched<Self>,Self::MoeError>;
    /// Actual row counts per caller-declared contiguous global expert range.
    /// Reads only expert-prefix coordination metadata, not activation/weight values.
    fn moe_dispatch_counts(state:&Self::MoeDispatchState,expert_prefix:&[usize]) -> Result<Vec<usize>,Self::MoeError>;
    /// COPY VJP sums selected expert-row seeds without multiplying routing weights again.
    fn moe_dispatch_backward(state:Self::MoeDispatchState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::MoeError>;
    /// Original fixed-selection source-logit VJP of FP32 continuous weights.
    fn moe_dispatch_weights_backward(state:Self::MoeDispatchState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::MoeError>;
    /// Original ascending expert-ID combine after actual expert rows have returned to their source.
    fn moe_combine(state:Self::MoeDispatchState,expert_values:FloatTensor<Self>,weights:FloatTensor<Self>,backward_strategy:MoeCombineGradientStrategy)
        -> Result<FloatTensor<Self>,Self::MoeError>;
    /// Original combine kernels with only requested actual expert/weight derivatives.
    fn moe_combine_backward(state:Self::MoeDispatchState,expert_values:FloatTensor<Self>,weights:FloatTensor<Self>,gradient:FloatTensor<Self>,
        strategy:MoeCombineGradientStrategy,selection:MoeCombineSelection) -> Result<MoeCombineBackward<Self>,Self::MoeError>;
}
/// Explicit original local expert range and independent forward/backward strategies.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct MoeReceivedOptions {
    /// First global expert ID owned by these actual local cubes; count comes from their original shape.
    pub expert_start:usize,
    /// Original actual segmented forward policy.
    pub forward:MoeExpertStrategy,
    /// Original actual independent segmented backward policy.
    pub backward:MoeExpertStrategy,
}
/// Actual original received-expert derivatives requested by native callers or tracked parents.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct MoeReceivedSelection {
    /// Real upstream activation derivative even when every expert matrix is frozen.
    pub input:bool,
    /// Original FP32 local gate-cube derivative.
    pub gate:bool,
    /// Original FP32 local up-cube derivative.
    pub up:bool,
    /// Original FP32 local down-cube derivative.
    pub down:bool,
}
/// Actual original optional received-expert VJP outputs in the original receive order.
#[derive(Debug)]
pub struct MoeReceivedBackward<B:Backend> {
    /// Original unsorted receive-axis input derivative.
    pub input:Option<FloatTensor<B>>,
    /// Original FP32 local gate cube derivative.
    pub gate:Option<FloatTensor<B>>,
    /// Original FP32 local up cube derivative.
    pub up:Option<FloatTensor<B>>,
    /// Original FP32 local down cube derivative.
    pub down:Option<FloatTensor<B>>,
}
/// Native local expert computation on actual transported assignments, without routing-weight application.
pub trait MoeReceivedOps:MoeOps {
    /// Private actual validated grouping, inverse permutation and original expert VJP cache.
    type MoeReceivedState:Clone+Send+Debug+'static;
    /// Sort received U32 assignments locally, execute original experts and restore receive order.
    /// Empty expert owners participate with real empty local cubes and zero received rows.
    fn moe_received_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,
        options:MoeReceivedOptions,selection:MoeReceivedSelection) -> Result<(FloatTensor<Self>,Self::MoeReceivedState),Self::MoeError>;
    /// Same original values without retained caches unless actual AD parents require derivatives.
    fn moe_received_inference(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeReceivedOptions)
        -> Result<FloatTensor<Self>,Self::MoeError> {
        Self::moe_received_forward(input,global_ids,gate,up,down,options,MoeReceivedSelection {input:false,gate:false,up:false,down:false}).map(|(output,_)|output)
    }
    /// Original expert VJP and inverse COPY mapping; local cube gradients retain FP32.
    fn moe_received_backward(state:Self::MoeReceivedState,gradient:FloatTensor<Self>,selection:MoeReceivedSelection)
        -> Result<MoeReceivedBackward<Self>,Self::MoeError>;
}
