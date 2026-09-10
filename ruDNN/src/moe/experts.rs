use super::{
    DispatchedTokens, MoeError, RoutingOptions, elements, float_tensor, kernels, route, same_device,
};
use rublas::tensor_grouped::grouped_matmul_nt;
use ruda_core::tensor::Shape;
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim},
    tensor::{RudaTensor, contiguous::into_contiguous},
};

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
        same_device(&self.gate, &dispatched.values)?;
        if dispatched.routing.experts != self.gate.meta.shape()[0]
            || dispatched.values.meta.shape()[1] != self.gate.meta.shape()[2]
            || dispatched.values.dtype != self.gate.dtype
        {
            return Err(MoeError("MoE dispatch and expert weights do not match"));
        }
        let gate = grouped_matmul_nt(
            dispatched.values.clone(),
            self.gate.clone(),
            dispatched.row_experts.clone(),
        )?;
        let up = grouped_matmul_nt(
            dispatched.values.clone(),
            self.up.clone(),
            dispatched.row_experts.clone(),
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
        Ok(grouped_matmul_nt(
            gate,
            self.down.clone(),
            dispatched.row_experts.clone(),
        )?)
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
