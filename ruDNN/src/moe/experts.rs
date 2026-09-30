use super::{
    DispatchedTokens, MoeError, RoutingOptions, elements, float_tensor, kernels, route, same_device,
};
use rublas::tensor_grouped::{grouped_matmul_nt_segmented, grouped_matmul_nt_backward_segmented_with_strategy, GroupedStrategy};
use ruda_core::tensor::Shape;
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim},
    tensor::{RudaTensor, contiguous::into_contiguous},
};

#[derive(Debug)]
pub struct ExpertTrainingCache<R: Runtime> {
    input: RudaTensor<R>, gate_raw: RudaTensor<R>, up_raw: RudaTensor<R>,
    activated: RudaTensor<R>, row_experts:RudaTensor<R>, offsets:RudaTensor<R>,
    gate_weights:RudaTensor<R>, up_weights:RudaTensor<R>, down_weights:RudaTensor<R>,
}

#[derive(Debug)]
pub struct ExpertTrainingOutput<R: Runtime> { pub output:RudaTensor<R>, pub cache:ExpertTrainingCache<R> }
#[derive(Debug)]
pub struct ExpertBackward<R: Runtime> {
    pub dinput:RudaTensor<R>, pub dgate:RudaTensor<R>, pub dup:RudaTensor<R>, pub ddown:RudaTensor<R>,
}

#[derive(Debug, Clone)]
pub struct SwiGluExperts<R: Runtime> {
    gate: RudaTensor<R>,
    up: RudaTensor<R>,
    down: RudaTensor<R>,
}

impl<R: Runtime> SwiGluExperts<R> {
    /// Bias-free gate/up `[experts, intermediate, hidden]` and down `[experts, hidden, intermediate]`.
    pub fn new(
        gate: RudaTensor<R>,
        up: RudaTensor<R>,
        down: RudaTensor<R>,
    ) -> Result<Self, MoeError> {
        for tensor in [&gate, &up, &down] {
            float_tensor(tensor)?;
            same_device(&gate, tensor)?;
            if tensor.dtype != gate.dtype || tensor.meta.num_dims() != 3 {
                return Err(MoeError(
                    "MoE expert weights must be rank-three tensors with matching dtype",
                ));
            }
            elements(tensor.meta.shape())?;
        }
        let e = gate.meta.shape()[0];
        let intermediate = gate.meta.shape()[1];
        let hidden = gate.meta.shape()[2];
        if e == 0
            || intermediate == 0
            || hidden == 0
            || up.meta.shape() != gate.meta.shape()
            || down.meta.shape() != &Shape::from([e, hidden, intermediate])
        {
            return Err(MoeError("MoE expert weight dimensions do not match"));
        }
        Ok(Self {
            gate: into_contiguous(gate),
            up: into_contiguous(up),
            down: into_contiguous(down),
        })
    }

    pub fn forward_dispatched(
        &self,
        dispatched: &DispatchedTokens<R>,
    ) -> Result<RudaTensor<R>, MoeError> {
        self.forward_dispatched_with_strategy(dispatched, GroupedStrategy::Scalar)
    }

    /// Opt-in cooperative matrix path shared by all models using these experts.
    pub fn forward_dispatched_with_strategy(
        &self, dispatched: &DispatchedTokens<R>, strategy: GroupedStrategy,
    ) -> Result<RudaTensor<R>, MoeError> {
        same_device(&self.gate, &dispatched.values)?;
        if dispatched.routing.experts != self.gate.meta.shape()[0]
            || dispatched.values.meta.shape()[1] != self.gate.meta.shape()[2]
            || dispatched.values.dtype != self.gate.dtype
        {
            return Err(MoeError("MoE dispatch and expert weights do not match"));
        }
        // SAFETY: DispatchedTokens fields are private and RoutingPlan::dispatch
        // constructs complete, monotone expert segments without token dropping.
        let product = |input, weights| unsafe {
            grouped_matmul_nt_segmented(input, weights, dispatched.row_experts.clone(),
                dispatched.offsets.clone(), strategy)
        };
        let gate = product(
            dispatched.values.clone(),
            self.gate.clone(),
        )?;
        let up = product(
            dispatched.values.clone(),
            self.up.clone(),
        )?;
        let size = gate.meta.num_elements();
        if size != 0 {
            let dim = RudaDim::new(gate.client.properties(), size);
            kernels::experts::swiglu::launch::<R>(
                &gate.client,
                calculate_ruda_count_elemwise(&gate.client, size, dim),
                dim,
                gate.clone().into_array_arg(),
                up.into_array_arg(),
                gate.dtype.into(),
            );
        }
        Ok(product(
            gate,
            self.down.clone(),
        )?)
    }

    /// Training forward after routing/dispatch. Keeps only tensors required for
    /// first-order expert backward; routing/top-k selection remains caller-owned.
    pub fn forward_dispatched_training(&self, dispatched:&DispatchedTokens<R>, strategy:GroupedStrategy)
        ->Result<ExpertTrainingOutput<R>,MoeError> {
        same_device(&self.gate,&dispatched.values)?;
        if dispatched.routing.experts!=self.gate.meta.shape()[0]
            || dispatched.values.meta.shape()[1]!=self.gate.meta.shape()[2]
            || dispatched.values.dtype!=self.gate.dtype
        { return Err(MoeError("MoE training dispatch and expert weights do not match")); }
        let product=|input,weights| unsafe { grouped_matmul_nt_segmented(input,weights,
            dispatched.row_experts.clone(),dispatched.offsets.clone(),strategy) };
        let gate_raw=product(dispatched.values.clone(),self.gate.clone())?;
        let up_raw=product(dispatched.values.clone(),self.up.clone())?;
        let activated=super::empty(&gate_raw,gate_raw.meta.shape().clone(),gate_raw.dtype);
        let size=activated.meta.num_elements();
        if size!=0 { let dim=RudaDim::new(activated.client.properties(),size);
            kernels::experts::swiglu_out::launch::<R>(&activated.client,
                calculate_ruda_count_elemwise(&activated.client,size,dim),dim,
                gate_raw.clone().into_array_arg(),up_raw.clone().into_array_arg(),
                activated.clone().into_array_arg(),activated.dtype.into()); }
        let output=product(activated.clone(),self.down.clone())?;
        Ok(ExpertTrainingOutput{output,cache:ExpertTrainingCache{
            input:dispatched.values.clone(),gate_raw,up_raw,activated,
            row_experts:dispatched.row_experts.clone(),offsets:dispatched.offsets.clone(),
            gate_weights:self.gate.clone(),up_weights:self.up.clone(),down_weights:self.down.clone()}})
    }

    /// Shared sigmoid/group-limited route -> dispatch -> selected expert kernels -> combine.
    /// Shared experts and model residual/scaling outside this routed branch remain caller-owned.
    pub fn forward_sigmoid_grouped(&self, input:RudaTensor<R>,logits:RudaTensor<R>,
        bias:Option<RudaTensor<R>>, options:super::GroupRoutingOptions,strategy:GroupedStrategy)
        ->Result<RudaTensor<R>,MoeError> {
        let dispatched=super::route_sigmoid_grouped(logits,bias,options)?.dispatch(input)?;
        let output=self.forward_dispatched_with_strategy(&dispatched,strategy)?;
        dispatched.combine(output)
    }

    /// Complete local softmax/top-k -> dispatch -> SwiGLU experts -> combine path.
    /// Router logits are supplied by the caller; this does not load a model or run expert parallelism.
    pub fn forward(
        &self,
        input: RudaTensor<R>,
        logits: RudaTensor<R>,
        options: RoutingOptions,
    ) -> Result<RudaTensor<R>, MoeError> {
        let dispatched = route(logits, options)?.dispatch(input)?;
        let output = self.forward_dispatched(&dispatched)?;
        dispatched.combine(output)
    }
}

impl<R: Runtime> ExpertTrainingCache<R> {
    /// Backpropagate through down projection, SwiGLU, gate and up projections.
    /// Returned weight gradients are FP32 even for FP16/BF16 expert weights.
    /// `dinput` corresponds to dispatched rows, before routing combine/gather.
    pub fn backward(self, grad_output:RudaTensor<R>) -> Result<ExpertBackward<R>,MoeError> {
        self.backward_with_strategy(grad_output, GroupedStrategy::Scalar)
    }

    /// Explicit backward strategy, independent of the forward strategy. TensorCore
    /// requires supported FP16/BF16 hardware and errors otherwise; Auto is only a
    /// capability fallback. Defaults are not changed by forward selection.
    pub fn backward_with_strategy(self, grad_output:RudaTensor<R>, strategy:GroupedStrategy)
        -> Result<ExpertBackward<R>,MoeError> {
        float_tensor(&grad_output)?; same_device(&self.input,&grad_output)?;
        if grad_output.meta.num_dims()!=2 || grad_output.meta.shape()[0]!=self.input.meta.shape()[0]
            || grad_output.meta.shape()[1]!=self.down_weights.meta.shape()[1]
            || grad_output.dtype!=self.input.dtype
        { return Err(MoeError("MoE expert backward gradient shape/dtype mismatch")); }
        let down=unsafe { grouped_matmul_nt_backward_segmented_with_strategy(self.activated.clone(),self.down_weights,
            grad_output,self.row_experts.clone(),self.offsets.clone(),strategy) }?;
        let dgate_out=super::empty(&self.gate_raw,self.gate_raw.meta.shape().clone(),self.gate_raw.dtype);
        let dup_out=super::empty(&self.up_raw,self.up_raw.meta.shape().clone(),self.up_raw.dtype);
        let size=self.gate_raw.meta.num_elements();
        if size!=0 { let dim=RudaDim::new(self.gate_raw.client.properties(),size);
            kernels::experts::swiglu_backward::launch::<R>(&self.gate_raw.client,
                calculate_ruda_count_elemwise(&self.gate_raw.client,size,dim),dim,
                self.gate_raw.clone().into_array_arg(),self.up_raw.clone().into_array_arg(),down.dinput.into_array_arg(),
                dgate_out.clone().into_array_arg(),dup_out.clone().into_array_arg(),self.gate_raw.dtype.into()); }
        let gate=unsafe { grouped_matmul_nt_backward_segmented_with_strategy(self.input.clone(),self.gate_weights,dgate_out,
            self.row_experts.clone(),self.offsets.clone(),strategy) }?;
        let up=unsafe { grouped_matmul_nt_backward_segmented_with_strategy(self.input.clone(),self.up_weights,dup_out,
            self.row_experts,self.offsets,strategy) }?;
        let dinput=super::empty(&gate.dinput,gate.dinput.meta.shape().clone(),gate.dinput.dtype);
        let size=dinput.meta.num_elements();
        if size!=0 { let dim=RudaDim::new(dinput.client.properties(),size);
            kernels::experts::add::launch::<R>(&dinput.client,
                calculate_ruda_count_elemwise(&dinput.client,size,dim),dim,
                gate.dinput.into_array_arg(),up.dinput.into_array_arg(),dinput.clone().into_array_arg(),dinput.dtype.into()); }
        Ok(ExpertBackward{dinput,dgate:gate.dweights,dup:up.dweights,ddown:down.dweights})
    }
}
