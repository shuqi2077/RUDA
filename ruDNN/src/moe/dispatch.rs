use super::{MoeError, RoutingPlan, elements, empty, float_tensor, kernels, same_device};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim},
    tensor::{RudaTensor, contiguous::into_contiguous},
};

#[derive(Debug, Clone)]
pub struct DispatchedTokens<R: Runtime> {
    pub(super) routing: RoutingPlan<R>,
    pub(super) values: RudaTensor<R>,
    pub(super) row_experts: RudaTensor<R>,
    pub(super) offsets: RudaTensor<R>,
    pub(super) slot_rows: RudaTensor<R>,
}

impl<R: Runtime> RoutingPlan<R> {
    /// Compact device-side dispatch with no token dropping or host readback.
    /// Expert segments are contiguous; row order within each expert is unspecified.
    pub fn dispatch(self, input: RudaTensor<R>) -> Result<DispatchedTokens<R>, MoeError> {
        float_tensor(&input)?;
        same_device(&input, &self.indices)?;
        if input.meta.num_dims() != 2
            || input.meta.shape()[0] != self.tokens
            || input.meta.shape()[1] == 0
        {
            return Err(MoeError(
                "MoE dispatch requires [tokens, nonzero hidden] input",
            ));
        }
        let hidden = input.meta.shape()[1];
        let rows = elements(&[self.tokens, self.top_k])?;
        let size = elements(&[rows, hidden])?;
        let offset_count = self
            .experts
            .checked_add(1)
            .ok_or(MoeError("MoE offset size overflow"))?;
        elements(&[offset_count])?;
        let counts = empty(&input, [self.experts], DType::U32);
        let offsets = empty(&input, [offset_count], DType::U32);
        let ranks = empty(&input, [rows], DType::U32);
        let sorted_slots = empty(&input, [rows], DType::U32);
        let slot_rows = empty(&input, [rows], DType::U32);
        let row_experts = empty(&input, [rows], DType::U32);
        let values = empty(&input, [rows, hidden], input.dtype);
        let dim = RudaDim::new(input.client.properties(), self.experts);
        kernels::dispatch::clear_counts::launch::<R>(
            &input.client,
            calculate_ruda_count_elemwise(&input.client, self.experts, dim),
            dim,
            counts.clone().into_array_arg(),
        );
        if rows != 0 {
            let dim = RudaDim::new(input.client.properties(), rows);
            kernels::dispatch::count_routes::launch::<R>(
                &input.client,
                calculate_ruda_count_elemwise(&input.client, rows, dim),
                dim,
                self.indices.clone().into_array_arg(),
                counts.clone().into_array_arg(),
                ranks.clone().into_array_arg(),
            );
        }
        let dim = RudaDim::new(input.client.properties(), 1);
        kernels::dispatch::prefix::launch::<R>(
            &input.client,
            calculate_ruda_count_elemwise(&input.client, 1, dim),
            dim,
            counts.into_array_arg(),
            offsets.clone().into_array_arg(),
        );
        if rows != 0 {
            let dim = RudaDim::new(input.client.properties(), rows);
            kernels::dispatch::scatter_routes::launch::<R>(
                &input.client,
                calculate_ruda_count_elemwise(&input.client, rows, dim),
                dim,
                self.indices.clone().into_array_arg(),
                ranks.into_array_arg(),
                offsets.clone().into_array_arg(),
                sorted_slots.clone().into_array_arg(),
                slot_rows.clone().into_array_arg(),
                row_experts.clone().into_array_arg(),
            );
            let input = into_contiguous(input);
            let dim = RudaDim::new(input.client.properties(), size);
            kernels::dispatch::gather::launch::<R>(
                &input.client,
                calculate_ruda_count_elemwise(&input.client, size, dim),
                dim,
                input.clone().into_array_arg(),
                sorted_slots.into_array_arg(),
                values.clone().into_array_arg(),
                hidden as u32,
                self.top_k as u32,
                input.dtype.into(),
            );
        }
        Ok(DispatchedTokens {
            routing: self,
            values,
            row_experts,
            offsets,
            slot_rows,
        })
    }
}

impl<R: Runtime> DispatchedTokens<R> {
    pub fn values(&self) -> &RudaTensor<R> {
        &self.values
    }
    pub fn row_experts(&self) -> &RudaTensor<R> {
        &self.row_experts
    }
    /// Exclusive prefix offsets `[experts + 1]`, including empty expert segments.
    pub fn expert_offsets(&self) -> &RudaTensor<R> {
        &self.offsets
    }
    pub fn routing(&self) -> &RoutingPlan<R> {
        &self.routing
    }

    /// Undo dispatch and add weighted expert outputs in ascending expert-ID order.
    /// Each weighted contribution and each addition is rounded to the output dtype.
    pub fn combine(&self, expert_output: RudaTensor<R>) -> Result<RudaTensor<R>, MoeError> {
        float_tensor(&expert_output)?;
        same_device(&self.values, &expert_output)?;
        let rows = self.routing.tokens * self.routing.top_k;
        if expert_output.meta.num_dims() != 2
            || expert_output.meta.shape()[0] != rows
            || expert_output.meta.shape()[1] == 0
            || expert_output.dtype != self.values.dtype
        {
            return Err(MoeError(
                "MoE combine requires one output row per dispatched row with matching dtype",
            ));
        }
        let width = expert_output.meta.shape()[1];
        elements(&[rows, width])?;
        let size = elements(&[self.routing.tokens, width])?;
        let output = empty(
            &expert_output,
            Shape::from([self.routing.tokens, width]),
            expert_output.dtype,
        );
        if size != 0 {
            let expert_output = into_contiguous(expert_output);
            let dim = RudaDim::new(output.client.properties(), size);
            kernels::dispatch::combine::launch::<R>(
                &output.client,
                calculate_ruda_count_elemwise(&output.client, size, dim),
                dim,
                expert_output.into_array_arg(),
                self.routing.weights.clone().into_array_arg(),
                self.routing.indices.clone().into_array_arg(),
                self.slot_rows.clone().into_array_arg(),
                output.clone().into_array_arg(),
                width as u32,
                self.routing.top_k as u32,
                output.dtype.into(),
                self.routing.weights.dtype.into(),
            );
        }
        Ok(output)
    }
}
