use super::*;
use ruda_tensor::graph::{
    EmbeddingBackwardOpIr, EmbeddingOpIr, LinearBiasBackwardOpIr, LinearOpIr,
    LinearWeightBackwardOpIr, LinearXBackwardOpIr,
};

impl<B: BackendIr> Runner<B> {
    pub(super) fn apply_embedding(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &EmbeddingOpIr,
    ) {
        let weights = handles.get_float_tensor::<B>(&desc.weights);
        let indices = handles.get_int_tensor::<B>(&desc.indices);

        let output = B::embedding(weights, indices);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_embedding_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &EmbeddingBackwardOpIr,
    ) {
        let weights = handles.get_float_tensor::<B>(&desc.weights);
        let indices = handles.get_int_tensor::<B>(&desc.indices);
        let output_grad = handles.get_float_tensor::<B>(&desc.out_grad);

        let output = B::embedding_backward(weights, output_grad, indices);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_linear(&self, handles: &mut HandleContainer<B::Handle>, desc: &LinearOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let bias = desc
            .bias
            .as_ref()
            .map(|bias| handles.get_float_tensor::<B>(bias));

        let output = B::linear(x, weight, bias);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_linear_xbackward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &LinearXBackwardOpIr,
    ) {
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::linear_x_backward(weight, output_grad);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_linear_weight_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &LinearWeightBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::linear_weight_backward(x, output_grad);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_linear_bias_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &LinearBiasBackwardOpIr,
    ) {
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::linear_bias_backward(output_grad);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }
}
