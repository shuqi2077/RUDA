use super::{ExpertLinear,PackedExpertLoRA,LoRALinearConfig,ExpertAdapterTarget,NativeSwiGluExperts,NativeMoeLayer,
    FrozenExpertGeometry,FrozenSelectedExperts,Nf4MoeLayer,Nf4MoeRouting};
use super::transformer::TransformerProjectionShape;
use ruda_model::{module::{Module,ParamId},tensor::{Tensor,Int,DType,TensorPrimitive,
    ExpertProjectionOps,NativeSwiGluOps,MoeExpertStrategy,MoeOptions,backend::Backend}};
use alloc::vec::Vec;
use core::fmt;

/// Real native per-expert A/B residual on an original frozen floating cube.
pub type FloatingExpertLoRA<B> = PackedExpertLoRA<B,ExpertLinear<B>>;

/// Actual original floating base or native A/B projection failure.
#[derive(Debug)]
pub enum FloatingExpertLoRAError<G:fmt::Debug> {
    /// Original selected floating base projection failure.
    Base(G),
    /// Actual trainable A/B projection failure.
    Adapter(G),
}
impl<G:fmt::Debug> fmt::Display for FloatingExpertLoRAError<G> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {
        Self::Base(error)=>write!(f,"floating expert base: {error:?}"),
        Self::Adapter(error)=>write!(f,"native expert adapter: {error:?}")}}
}
impl<G:fmt::Debug> core::error::Error for FloatingExpertLoRAError<G> {}
impl<B:ExpertProjectionOps> FloatingExpertLoRA<B> {
    /// Original floating base plus actual grouped A/B, preserving the base output storage.
    /// Real input/A/B derivatives use the native selected-expert graph, without packed-backend requirements.
    pub fn forward(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize)
        -> Result<Tensor<B,2>,FloatingExpertLoRAError<B::ExpertProjectionError>> {
        self.validate();let base=self.base.forward(input.clone(),ids.clone(),expert_start).map_err(FloatingExpertLoRAError::Base)?;
        self.add_adapter_residual(base,input,ids,expert_start).map_err(FloatingExpertLoRAError::Adapter)
    }
}
impl<B:Backend> FloatingExpertLoRA<B> {
    /// Consume the adapter into original floating weight layout for dropout-free inference.
    /// Preserves base IDs/dtype and execution policies; cannot resume A/B optimizer state from the merged cube.
    pub fn merge(self) -> ExpertLinear<B> {
        self.validate();let mut base=self.base;
        if base.dimensions()[0]==0 {return base.no_grad();}
        let a=self.adapter_a.weight.val();let b=self.adapter_b.weight.val();
        let update=b.cast(DType::F32).matmul(a.cast(DType::F32)).mul_scalar(self.scale).detach();
        base.weight=base.weight.map(|weight| {
            let dtype=weight.dtype();(weight.cast(DType::F32)+update).cast(dtype).detach().set_require_grad(false)
        });base
    }
}

/// Actual original floating projection or explicitly attached per-expert A/B.
#[derive(Module,Debug)]
pub enum FloatingExpertProjection<B:Backend> {
    /// Original supplied cube and its unchanged trainability/parameter identity.
    Dense(ExpertLinear<B>),
    /// Explicitly selected frozen base plus actual native trainable A/B.
    LoRA(FloatingExpertLoRA<B>),
}
impl<B:Backend> FloatingExpertProjection<B> {
    /// Actual original `[experts,input,output]` widths, not the adapter rank.
    pub fn dimensions(&self) -> [usize;3] {self.base().dimensions()}
    /// Original cube, without materializing an adapter update or copying source weights.
    pub fn base(&self) -> &ExpertLinear<B> {match self {Self::Dense(value)=>value,Self::LoRA(value)=>&value.base}}
    /// Actual original storage dtype of the selected base.
    pub fn base_dtype(&self) -> DType {self.base().weight.val().dtype()}
    /// Actual original resident device.
    pub fn device(&self) -> B::Device {self.base().weight.val().device()}
    /// Validate original native geometry and independently stored actual adapter leaves.
    pub fn validate(&self) {match self {Self::Dense(value)=>value.validate(),Self::LoRA(value)=>value.validate()}}
    /// Merge only explicitly present adapters, retaining untouched original cubes and their flags.
    pub fn merge(self) -> ExpertLinear<B> {match self {Self::Dense(value)=>value,Self::LoRA(value)=>value.merge()}}
}
impl<B:ExpertProjectionOps> FloatingExpertProjection<B> {
    /// Execute the actual selected floating projection with original native first-order derivatives.
    pub fn forward(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize)
        -> Result<Tensor<B,2>,FloatingExpertLoRAError<B::ExpertProjectionError>> {
        match self {Self::Dense(value)=>value.forward(input,ids,expert_start).map_err(FloatingExpertLoRAError::Base),
            Self::LoRA(value)=>value.forward(input,ids,expert_start)}
    }
}

/// Actual floating expert projection or source storage-rounded SwiGLU failure.
#[derive(Debug)]
pub enum FloatingExpertError<G:fmt::Debug,S:fmt::Debug> {
    /// Actual original floating base or selected A/B failure.
    Projection(FloatingExpertLoRAError<G>),
    /// Original native SwiGLU activation failure.
    Activation(S),
}
impl<G:fmt::Debug,S:fmt::Debug> fmt::Display for FloatingExpertError<G,S> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {
        Self::Projection(error)=>write!(f,"{error}"),Self::Activation(error)=>write!(f,"native expert SwiGLU: {error:?}")}}
}
impl<G:fmt::Debug,S:fmt::Debug> core::error::Error for FloatingExpertError<G,S> {}

/// Original floating gate/up/down cubes with adapters only on explicitly selected roles.
#[derive(Module,Debug)]
pub struct AdaptedFloatingSwiGluExperts<B:Backend> {
    /// Actual original `[E,I,H]` gate, retaining unselected flags and values.
    pub gate:FloatingExpertProjection<B>,
    /// Actual independent original `[E,I,H]` up cube.
    pub up:FloatingExpertProjection<B>,
    /// Actual original `[E,H,I]` down cube.
    pub down:FloatingExpertProjection<B>,
}
impl<B:Backend> AdaptedFloatingSwiGluExperts<B> {
    /// Connect actual loaded original/adapted projections without replacement initialization.
    pub fn from_parts(gate:FloatingExpertProjection<B>,up:FloatingExpertProjection<B>,down:FloatingExpertProjection<B>) -> Self {
        let experts=Self {gate,up,down};experts.validate();experts
    }
    /// Retain original cube values, IDs, dtypes and trainability before explicit role selection.
    /// Base execution uses the source forward/backward policies supplied by the original routed layer.
    pub fn from_native(source:NativeSwiGluExperts<B>,forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> Self {
        source.validate();Self::from_parts(
            FloatingExpertProjection::Dense(ExpertLinear::from_parameters(source.gate,forward,backward)),
            FloatingExpertProjection::Dense(ExpertLinear::from_parameters(source.up,forward,backward)),
            FloatingExpertProjection::Dense(ExpertLinear::from_parameters(source.down,forward,backward)))
    }
    /// Attach actual new A/B only to explicit roles; unselected original matrices remain trainable when originally so.
    /// Adapter storage and execution are independent of the original base storage and source execution policies.
    pub fn with_adapters(mut self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType,use_rslora:bool,
        forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> Self {
        self.validate_adapter_targets(config,targets,dtype);
        let adapt=|value|match value {
            FloatingExpertProjection::Dense(base)=>FloatingExpertProjection::LoRA(config.init_grouped_expert_adapters(base,dtype,use_rslora,forward,backward)),
            FloatingExpertProjection::LoRA(_)=>unreachable!("validated original floating expert role is not adapted")};
        if targets.contains(&ExpertAdapterTarget::Gate) {self.gate=adapt(self.gate);}
        if targets.contains(&ExpertAdapterTarget::Up) {self.up=adapt(self.up);}
        if targets.contains(&ExpertAdapterTarget::Down) {self.down=adapt(self.down);}self.validate();self
    }
    pub(super) fn validate_adapter_targets(&self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType) {
        self.validate();assert!(!targets.is_empty(),"floating expert adapter selection requires an actual role");
        for (index,target) in targets.iter().enumerate() {
            assert!(!targets[..index].contains(target),"duplicate floating expert adapter role");
            let projection=match target {ExpertAdapterTarget::Gate=>&self.gate,ExpertAdapterTarget::Up=>&self.up,ExpertAdapterTarget::Down=>&self.down};
            let FloatingExpertProjection::Dense(base)=projection else {panic!("selected original floating expert projection is already adapted")};
            config.validate_expert_initialization::<B,_>(base,dtype);
        }
    }
    /// Canonical actual A/B IDs, excluding unselected original trainable cubes and frozen selected bases.
    pub fn adapter_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids=Vec::new();for projection in [&self.gate,&self.up,&self.down] {
            if let FloatingExpertProjection::LoRA(layer)=projection {for id in [layer.adapter_a.weight.id.clone(),layer.adapter_b.weight.id.clone()] {
                if !ids.contains(&id) {ids.push(id);}}}}ids
    }
    /// Original single native cube execution policy; reject independently altered policies instead of guessing.
    pub fn base_strategies(&self) -> (MoeExpertStrategy,MoeExpertStrategy) {
        let gate=self.gate.base();let policies=(gate.forward_strategy,gate.backward_strategy);
        for projection in [&self.up,&self.down] {let base=projection.base();
            assert_eq!((base.forward_strategy,base.backward_strategy),policies,"original native expert base execution policies differ");}policies
    }
    /// Restore the actual original cube-only structure after merging only selected A/B for inference.
    pub fn merge(self) -> NativeSwiGluExperts<B> {
        NativeSwiGluExperts::from_parameters(self.gate.merge().weight,self.up.merge().weight,self.down.merge().weight)
    }
}
impl<B:Backend> FrozenExpertGeometry<B> for AdaptedFloatingSwiGluExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.gate.dimensions()}
    fn validate(&self) {
        self.gate.validate();self.up.validate();self.down.validate();let [e,h,i]=self.dimensions();
        assert_eq!(self.up.dimensions(),[e,h,i],"floating expert gate/up geometry differs");
        assert_eq!(self.down.dimensions(),[e,i,h],"floating expert down geometry differs");
        for projection in [&self.up,&self.down] {
            assert_eq!(projection.device(),self.gate.device(),"floating expert base devices differ");
            assert_eq!(projection.base_dtype(),self.gate.base_dtype(),"floating expert original base storage differs");}
    }
    fn device(&self) -> B::Device {self.gate.device()}
}
impl<B:ExpertProjectionOps+NativeSwiGluOps> FrozenSelectedExperts<B> for AdaptedFloatingSwiGluExperts<B> {
    type Error=FloatingExpertError<B::ExpertProjectionError,B::SwiGluError>;
    fn forward(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,Self::Error> {
        self.validate();let gate=self.gate.forward(input.clone(),ids.clone(),expert_start).map_err(FloatingExpertError::Projection)?;
        let up=self.up.forward(input,ids.clone(),expert_start).map_err(FloatingExpertError::Projection)?;
        let hidden=B::native_swiglu(gate.into_primitive().tensor(),up.into_primitive().tensor()).map_err(FloatingExpertError::Activation)?;
        self.down.forward(Tensor::from_primitive(TensorPrimitive::Float(hidden)),ids,expert_start).map_err(FloatingExpertError::Projection)
    }
}

/// Complete native routing/combine graph with actual floating expert adapter training.
pub type AdaptedFloatingMoeLayer<B,P> = Nf4MoeLayer<B,P,AdaptedFloatingSwiGluExperts<B>>;
/// Complete native token/logit/causal-loss/cache graph with selected floating expert adapters.
pub type AdaptedFloatingMoeTransformerModel<B,P> = super::transformer::Nf4MoeTransformerModel<B,P,AdaptedFloatingSwiGluExperts<B>>;
/// Actual original attention/shared/residual block with explicitly adapted floating expert roles.
pub type AdaptedFloatingMoeTransformerBlock<B,P> = super::transformer::Nf4MoeTransformerBlock<B,P,AdaptedFloatingSwiGluExperts<B>>;
/// Actual original layer order, retaining native whole-expert execution on every unselected routed layer.
pub type AdaptedFloatingMoeTransformerLayer<B,P> = super::transformer::Nf4MoeTransformerLayer<B,P,AdaptedFloatingSwiGluExperts<B>>;

impl<B:Backend,P:TransformerProjectionShape<B>> NativeMoeLayer<B,P> {
    /// Attach only explicit floating expert roles, retaining the actual source router/correction bias and routing policies.
    pub fn with_expert_adapters(self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType,use_rslora:bool,
        forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> AdaptedFloatingMoeLayer<B,P> {
        self.validate();let experts=AdaptedFloatingSwiGluExperts::from_native(self.experts,self.options.forward,self.options.backward)
            .with_adapters(config,targets,dtype,use_rslora,forward,backward);
        Nf4MoeLayer::from_parts(self.router,experts,self.correction_bias,Nf4MoeRouting {
            selection:self.options.selection,weights:self.options.weights,combine_backward:self.options.combine_backward},self.router_input_dtype)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>> AdaptedFloatingMoeLayer<B,P> {
    /// Merge selected expert A/B for inference and restore the original native whole-expert routed execution.
    /// The router/shared-model parameters are not merged or frozen by this operation.
    pub fn merge_expert_adapters(self) -> NativeMoeLayer<B,P> {
        self.validate();let (forward,backward)=self.experts.base_strategies();let options=MoeOptions {
            selection:self.routing.selection,weights:self.routing.weights,combine_backward:self.routing.combine_backward,forward,backward};
        NativeMoeLayer::from_parts(self.router,self.experts.merge(),self.correction_bias,options,self.router_input_dtype)
    }
}
