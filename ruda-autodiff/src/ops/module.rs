use crate::Autodiff;
use crate::checkpoint::base::Checkpointer;
use crate::checkpoint::strategy::CheckpointStrategy;
use crate::grads::Gradients;
use crate::graph::NodeId;
use crate::ops::{Backward, Ops, broadcast_shape, unary};
use crate::tensor::AutodiffTensor;

use ruda_tensor::ops::attention::attention_fallback;
use ruda_tensor::ops::*;
use ruda_tensor::tensor::{FloatTensor, IntTensor};
use ruda_tensor::{Backend, TensorMetadata, get_device_settings};
use ruda_core::tensor::{FloatDType, Shape};

use super::OpsKind;

fn causal_attention_mask<B: Backend>(scores: &B::FloatTensorPrimitive) -> B::BoolTensorPrimitive {
    let device = B::float_device(scores);
    let [batch, heads, query_sequence, key_sequence] = scores.shape().dims::<4>();
    let settings = get_device_settings::<B>(&device);
    let offset = key_sequence as i64 - query_sequence as i64;
    let rows = B::int_reshape(
        B::int_arange(0..query_sequence as i64, &device, settings.int_dtype),
        Shape::new([query_sequence, 1]),
    );
    let columns = B::int_reshape(
        B::int_arange(0..key_sequence as i64, &device, settings.int_dtype),
        Shape::new([1, key_sequence]),
    );
    let rows = B::int_add_scalar(rows, offset.into());
    let mask = B::int_lower(rows, columns, settings.bool_dtype);
    let mask = B::bool_reshape(mask, Shape::new([1, 1, query_sequence, key_sequence]));
    B::bool_expand(
        mask,
        Shape::new([batch, heads, query_sequence, key_sequence]),
    )
}

/// Recompute the causal softmax matrix used by the optimized attention
/// backward. Low-precision products and normalization use FP32, while FP64
/// inputs retain FP64 accumulation.
fn causal_attention_probabilities<B: Backend>(
    query: B::FloatTensorPrimitive,
    key: B::FloatTensorPrimitive,
    accumulator_dtype: FloatDType,
) -> B::FloatTensorPrimitive {
    let head_dimension = query.shape().dims::<4>()[3];
    let scale = 1.0 / (head_dimension as f64).sqrt();
    let query = B::float_cast(query, accumulator_dtype);
    let key = B::float_cast(key, accumulator_dtype);
    let scores = B::float_matmul(query, B::float_transpose(key));
    let scores = B::float_mul_scalar(scores, scale.into());
    let mask = causal_attention_mask::<B>(&scores);
    let scores = B::float_mask_fill(scores, mask, f32::NEG_INFINITY.into());

    let finfo = scores.dtype().finfo().expect("float tensor");
    let maximum = B::float_max_dim(scores.clone(), 3);
    let maximum = B::float_clamp_min(maximum, finfo.min.into());
    let numerator = B::float_exp(B::float_sub(scores, maximum));
    let denominator = B::float_sum_dim(numerator.clone(), 3);
    let denominator = B::float_clamp_min(denominator, finfo.min_positive.into());
    B::float_div(numerator, denominator)
}

impl<B: Backend, C: CheckpointStrategy> ModuleOps<Autodiff<B, C>> for Autodiff<B, C> {
    fn embedding(weights: AutodiffTensor<B>, indices: IntTensor<B>) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Embedding;

        impl<B: Backend> Backward<B, 1> for Embedding {
            type State = (B::FloatTensorPrimitive, IntTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (weights, indices) = ops.state;

                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::embedding_backward(weights, grad, indices)
                });
            }
        }

        match Embedding
            .prepare::<C>([weights.node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (weights.primitive.clone(), indices.clone()),
                B::embedding(weights.primitive, indices),
            ),
            OpsKind::UnTracked(prep) => prep.finish(B::embedding(weights.primitive, indices)),
        }
    }

    fn linear(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct LinearWithBias;
        #[derive(Debug)]
        struct LinearNoBias;

        impl<B: Backend> Backward<B, 3> for LinearWithBias {
            type State = (Option<NodeId>, Option<NodeId>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state) = ops.state;
                let x = x_state
                    .map(|id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id));
                let weight = weight_state
                    .map(|id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id));

                if let Some(node) = node_x {
                    let grad = B::linear_x_backward(weight.unwrap(), grad.clone());
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::linear_weight_backward(x.unwrap(), grad.clone());
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::linear_bias_backward(grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for LinearNoBias {
            type State = (Option<NodeId>, Option<NodeId>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state) = ops.state;
                let x = x_state
                    .map(|id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id));
                let weight = weight_state
                    .map(|id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id));

                if let Some(node) = node_x {
                    let grad = B::linear_x_backward(weight.unwrap(), grad.clone());
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::linear_weight_backward(x.unwrap(), grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        let x_tracked = x.is_tracked();
        let weight_tracked = weight.is_tracked();

        match bias {
            Some(bias) => match LinearWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    // x is only needed to compute the weight gradient, and vice versa.
                    let x_state = weight_tracked.then(|| prep.checkpoint(&x));
                    let weight_state = x_tracked.then(|| prep.checkpoint(&weight));
                    prep.finish(
                        (x_state, weight_state),
                        B::linear(x.primitive, weight.primitive, Some(bias.primitive)),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::linear(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                )),
            },
            None => match LinearNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = weight_tracked.then(|| prep.checkpoint(&x));
                    let weight_state = x_tracked.then(|| prep.checkpoint(&weight));
                    prep.finish(
                        (x_state, weight_state),
                        B::linear(x.primitive, weight.primitive, None),
                    )
                }
                OpsKind::UnTracked(prep) => {
                    prep.finish(B::linear(x.primitive, weight.primitive, None))
                }
            },
        }
    }

    fn conv1d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvOptions<1>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Conv1DWithBias;
        #[derive(Debug)]
        struct Conv1DNoBias;

        impl<B: Backend> Backward<B, 3> for Conv1DWithBias {
            type State = (NodeId, NodeId, NodeId, ConvOptions<1>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv1d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv1d_weight_backward(x.clone(), weight, grad.clone(), options);
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv1d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for Conv1DNoBias {
            type State = (NodeId, NodeId, ConvOptions<1>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv1d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv1d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }
        match bias {
            Some(bias) => match Conv1DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv1d(x.primitive, weight.primitive, Some(bias.primitive), options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv1d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match Conv1DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv1d(x.primitive, weight.primitive, None, options),
                    )
                }
                OpsKind::UnTracked(prep) => {
                    prep.finish(B::conv1d(x.primitive, weight.primitive, None, options))
                }
            },
        }
    }

    fn conv_transpose1d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvTransposeOptions<1>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct ConvTranspose1DWithBias;
        #[derive(Debug)]
        struct ConvTranspose1DNoBias;

        impl<B: Backend> Backward<B, 3> for ConvTranspose1DWithBias {
            type State = (NodeId, NodeId, NodeId, ConvTransposeOptions<1>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose1d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose1d_weight_backward(
                        x.clone(),
                        weight,
                        grad.clone(),
                        options,
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv_transpose1d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for ConvTranspose1DNoBias {
            type State = (NodeId, NodeId, ConvTransposeOptions<1>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose1d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose1d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match ConvTranspose1DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv_transpose1d(
                            x.primitive,
                            weight.primitive,
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose1d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match ConvTranspose1DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv_transpose1d(x.primitive, weight.primitive, None, options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose1d(
                    x.primitive,
                    weight.primitive,
                    None,
                    options,
                )),
            },
        }
    }

    fn conv2d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvOptions<2>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Conv2DWithBias;
        #[derive(Debug)]
        struct Conv2DNoBias;

        impl<B: Backend> Backward<B, 3> for Conv2DWithBias {
            type State = (NodeId, NodeId, NodeId, ConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv2d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad =
                        B::conv2d_weight_backward(x.clone(), weight.clone(), grad.clone(), options);
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv2d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for Conv2DNoBias {
            type State = (NodeId, NodeId, ConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv2d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv2d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match Conv2DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv2d(x.primitive, weight.primitive, Some(bias.primitive), options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv2d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match Conv2DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv2d(x.primitive, weight.primitive, None, options),
                    )
                }

                OpsKind::UnTracked(prep) => {
                    prep.finish(B::conv2d(x.primitive, weight.primitive, None, options))
                }
            },
        }
    }

    fn deform_conv2d(
        x: AutodiffTensor<B>,
        offset: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        mask: Option<AutodiffTensor<B>>,
        bias: Option<AutodiffTensor<B>>,
        options: DeformConvOptions<2>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct DeformConv2DWithMaskWithBias;
        #[derive(Debug)]
        struct DeformConv2DWithMaskNoBias;
        #[derive(Debug)]
        struct DeformConv2DNoMaskWithBias;
        #[derive(Debug)]
        struct DeformConv2DNoMaskNoBias;

        impl<B: Backend> Backward<B, 5> for DeformConv2DWithMaskWithBias {
            type State = (NodeId, NodeId, NodeId, NodeId, NodeId, DeformConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 5>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_offset, node_weight, node_mask, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, offset_state, weight_state, mask_state, bias_state, options) =
                    ops.state;
                let x = checkpointer.retrieve_node_output(x_state);
                let offset = checkpointer.retrieve_node_output(offset_state);
                let weight = checkpointer.retrieve_node_output(weight_state);
                let mask = Some(checkpointer.retrieve_node_output(mask_state));
                let bias = Some(checkpointer.retrieve_node_output(bias_state));

                let backward =
                    B::deform_conv2d_backward(x, offset, weight, mask, bias, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, backward.x_grad)
                }
                if let Some(node) = node_offset {
                    grads.register::<B>(node.id, backward.offset_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, backward.weight_grad)
                }
                if let Some(node) = node_mask {
                    grads.register::<B>(node.id, backward.mask_grad.unwrap())
                }
                if let Some(node) = node_bias {
                    grads.register::<B>(node.id, backward.bias_grad.unwrap())
                }
            }
        }

        impl<B: Backend> Backward<B, 4> for DeformConv2DWithMaskNoBias {
            type State = (NodeId, NodeId, NodeId, NodeId, DeformConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 4>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_offset, node_weight, node_mask] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, offset_state, weight_state, mask_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output(x_state);
                let offset = checkpointer.retrieve_node_output(offset_state);
                let weight = checkpointer.retrieve_node_output(weight_state);
                let mask = Some(checkpointer.retrieve_node_output(mask_state));

                let backward =
                    B::deform_conv2d_backward(x, offset, weight, mask, None, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, backward.x_grad)
                }
                if let Some(node) = node_offset {
                    grads.register::<B>(node.id, backward.offset_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, backward.weight_grad)
                }
                if let Some(node) = node_mask {
                    grads.register::<B>(node.id, backward.mask_grad.unwrap())
                }
            }
        }

        impl<B: Backend> Backward<B, 4> for DeformConv2DNoMaskWithBias {
            type State = (NodeId, NodeId, NodeId, NodeId, DeformConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 4>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_offset, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, offset_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output(x_state);
                let offset = checkpointer.retrieve_node_output(offset_state);
                let weight = checkpointer.retrieve_node_output(weight_state);
                let bias = Some(checkpointer.retrieve_node_output(bias_state));

                let backward =
                    B::deform_conv2d_backward(x, offset, weight, None, bias, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, backward.x_grad)
                }
                if let Some(node) = node_offset {
                    grads.register::<B>(node.id, backward.offset_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, backward.weight_grad)
                }
                if let Some(node) = node_bias {
                    grads.register::<B>(node.id, backward.bias_grad.unwrap())
                }
            }
        }

        impl<B: Backend> Backward<B, 3> for DeformConv2DNoMaskNoBias {
            type State = (NodeId, NodeId, NodeId, DeformConvOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_offset, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, offset_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output(x_state);
                let offset = checkpointer.retrieve_node_output(offset_state);
                let weight = checkpointer.retrieve_node_output(weight_state);

                let backward =
                    B::deform_conv2d_backward(x, offset, weight, None, None, grad, options);

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, backward.x_grad)
                }
                if let Some(node) = node_offset {
                    grads.register::<B>(node.id, backward.offset_grad)
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, backward.weight_grad)
                }
            }
        }

        match (mask, bias) {
            (Some(mask), Some(bias)) => match DeformConv2DWithMaskWithBias
                .prepare::<C>([
                    x.node.clone(),
                    offset.node.clone(),
                    weight.node.clone(),
                    mask.node.clone(),
                    bias.node.clone(),
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let offset_state = prep.checkpoint(&offset);
                    let weight_state = prep.checkpoint(&weight);
                    let mask_state = prep.checkpoint(&mask);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (
                            x_state,
                            offset_state,
                            weight_state,
                            mask_state,
                            bias_state,
                            options.clone(),
                        ),
                        B::deform_conv2d(
                            x.primitive,
                            offset.primitive,
                            weight.primitive,
                            Some(mask.primitive),
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::deform_conv2d(
                    x.primitive,
                    offset.primitive,
                    weight.primitive,
                    Some(mask.primitive),
                    Some(bias.primitive),
                    options,
                )),
            },
            (Some(mask), None) => match DeformConv2DWithMaskNoBias
                .prepare::<C>([
                    x.node.clone(),
                    offset.node.clone(),
                    weight.node.clone(),
                    mask.node.clone(),
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let offset_state = prep.checkpoint(&offset);
                    let weight_state = prep.checkpoint(&weight);
                    let mask_state = prep.checkpoint(&mask);
                    prep.finish(
                        (
                            x_state,
                            offset_state,
                            weight_state,
                            mask_state,
                            options.clone(),
                        ),
                        B::deform_conv2d(
                            x.primitive,
                            offset.primitive,
                            weight.primitive,
                            Some(mask.primitive),
                            None,
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::deform_conv2d(
                    x.primitive,
                    offset.primitive,
                    weight.primitive,
                    Some(mask.primitive),
                    None,
                    options,
                )),
            },
            (None, Some(bias)) => match DeformConv2DNoMaskWithBias
                .prepare::<C>([
                    x.node.clone(),
                    offset.node.clone(),
                    weight.node.clone(),
                    bias.node.clone(),
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let offset_state = prep.checkpoint(&offset);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (
                            x_state,
                            offset_state,
                            weight_state,
                            bias_state,
                            options.clone(),
                        ),
                        B::deform_conv2d(
                            x.primitive,
                            offset.primitive,
                            weight.primitive,
                            None,
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::deform_conv2d(
                    x.primitive,
                    offset.primitive,
                    weight.primitive,
                    None,
                    Some(bias.primitive),
                    options,
                )),
            },
            (None, None) => match DeformConv2DNoMaskNoBias
                .prepare::<C>([x.node.clone(), offset.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let offset_state = prep.checkpoint(&offset);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, offset_state, weight_state, options.clone()),
                        B::deform_conv2d(
                            x.primitive,
                            offset.primitive,
                            weight.primitive,
                            None,
                            None,
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::deform_conv2d(
                    x.primitive,
                    offset.primitive,
                    weight.primitive,
                    None,
                    None,
                    options,
                )),
            },
        }
    }

    fn deform_conv2d_backward(
        x: AutodiffTensor<B>,
        offset: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        mask: Option<AutodiffTensor<B>>,
        bias: Option<AutodiffTensor<B>>,
        output_grad: AutodiffTensor<B>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        super::deform_backward::backward::<Self>(
            x, offset, weight, mask, bias, output_grad, options,
        )
    }

    fn conv_transpose2d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvTransposeOptions<2>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct ConvTranspose2DWithBias;
        #[derive(Debug)]
        struct ConvTranspose2DNoBias;

        impl<B: Backend> Backward<B, 3> for ConvTranspose2DWithBias {
            type State = (NodeId, NodeId, NodeId, ConvTransposeOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose2d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose2d_weight_backward(
                        x.clone(),
                        weight,
                        grad.clone(),
                        options,
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv_transpose2d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for ConvTranspose2DNoBias {
            type State = (NodeId, NodeId, ConvTransposeOptions<2>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose2d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose2d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match ConvTranspose2DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);

                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv_transpose2d(
                            x.primitive,
                            weight.primitive,
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose2d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match ConvTranspose2DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);

                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv_transpose2d(x.primitive, weight.primitive, None, options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose2d(
                    x.primitive,
                    weight.primitive,
                    None,
                    options,
                )),
            },
        }
    }

    fn conv3d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvOptions<3>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Conv3DWithBias;
        #[derive(Debug)]
        struct Conv3DNoBias;

        impl<B: Backend> Backward<B, 3> for Conv3DWithBias {
            type State = (NodeId, NodeId, NodeId, ConvOptions<3>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv3d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad =
                        B::conv3d_weight_backward(x.clone(), weight.clone(), grad.clone(), options);
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv3d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for Conv3DNoBias {
            type State = (NodeId, NodeId, ConvOptions<3>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv3d_x_backward(
                        x.clone(),
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv3d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match Conv3DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);
                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv3d(x.primitive, weight.primitive, Some(bias.primitive), options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv3d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match Conv3DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv3d(x.primitive, weight.primitive, None, options),
                    )
                }

                OpsKind::UnTracked(prep) => {
                    prep.finish(B::conv3d(x.primitive, weight.primitive, None, options))
                }
            },
        }
    }

    fn conv_transpose3d(
        x: AutodiffTensor<B>,
        weight: AutodiffTensor<B>,
        bias: Option<AutodiffTensor<B>>,
        options: ConvTransposeOptions<3>,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct ConvTranspose3DWithBias;
        #[derive(Debug)]
        struct ConvTranspose3DNoBias;

        impl<B: Backend> Backward<B, 3> for ConvTranspose3DWithBias {
            type State = (NodeId, NodeId, NodeId, ConvTransposeOptions<3>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight, node_bias] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, bias_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                let bias = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(bias_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose3d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose3d_weight_backward(
                        x.clone(),
                        weight,
                        grad.clone(),
                        options,
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_bias {
                    let grad = B::conv_transpose3d_bias_backward(x, bias, grad);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        impl<B: Backend> Backward<B, 2> for ConvTranspose3DNoBias {
            type State = (NodeId, NodeId, ConvTransposeOptions<3>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, weight_state, options) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);

                if let Some(node) = node_x {
                    let grad = B::conv_transpose3d_x_backward(
                        weight.clone(),
                        grad.clone(),
                        options.clone(),
                    );
                    grads.register::<B>(node.id, grad)
                }
                if let Some(node) = node_weight {
                    let grad = B::conv_transpose3d_weight_backward(x, weight, grad, options);
                    grads.register::<B>(node.id, grad)
                }
            }
        }

        match bias {
            Some(bias) => match ConvTranspose3DWithBias
                .prepare::<C>([x.node.clone(), weight.node.clone(), bias.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);
                    let bias_state = prep.checkpoint(&bias);

                    prep.finish(
                        (x_state, weight_state, bias_state, options.clone()),
                        B::conv_transpose3d(
                            x.primitive,
                            weight.primitive,
                            Some(bias.primitive),
                            options,
                        ),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose3d(
                    x.primitive,
                    weight.primitive,
                    Some(bias.primitive),
                    options,
                )),
            },
            None => match ConvTranspose3DNoBias
                .prepare::<C>([x.node.clone(), weight.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let x_state = prep.checkpoint(&x);
                    let weight_state = prep.checkpoint(&weight);

                    prep.finish(
                        (x_state, weight_state, options.clone()),
                        B::conv_transpose3d(x.primitive, weight.primitive, None, options),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(B::conv_transpose3d(
                    x.primitive,
                    weight.primitive,
                    None,
                    options,
                )),
            },
        }
    }

    // TODO: Support a custom unfold4d operation by overriding the default implementation.
    //
    // We don't override it now because the fold operation isn't available for the backward pass.
    // This implies that when autodiff is enabled, custom unfold operations defined by backends
    // won't be used. Instead, the conv2d operation with custom weights matrix will be used.
    // Therefore, the conv2d backward pass will be used for the unfold4d backward pass.
    //
    // fn unfold4d(
    //     x:AutodiffTensor<B>,
    //     kernel_size: [usize; 2],
    //     options: UnfoldOptions,
    // ) -> AutodiffTensor<B> {
    //     todo!()
    // }

    fn avg_pool1d(
        x: AutodiffTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct AvgPool1D;

        impl<B: Backend> Backward<B, 1> for AvgPool1D {
            type State = (NodeId, usize, usize, usize, bool, bool);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (x_state, kernel_size, stride, padding, count_include_pad, ceil_mode) =
                    ops.state;
                let x = checkpointer.retrieve_node_output(x_state);

                if let Some(node) = node_parent {
                    let grad = B::avg_pool1d_backward(
                        x,
                        grad,
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                        ceil_mode,
                    );
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match AvgPool1D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                prep.finish(
                    (
                        x_state,
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                        ceil_mode,
                    ),
                    B::avg_pool1d(
                        x.primitive.clone(),
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                        ceil_mode,
                    ),
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::avg_pool1d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            )),
        }
    }

    fn avg_pool2d(
        x: AutodiffTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct AvgPool2D;

        impl<B: Backend> Backward<B, 1> for AvgPool2D {
            type State = (NodeId, [usize; 2], [usize; 2], [usize; 2], bool, bool);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (x_state, kernel_size, stride, padding, count_include_pad, ceil_mode) =
                    ops.state;
                let x = checkpointer.retrieve_node_output(x_state);

                if let Some(node) = node_parent {
                    let grad = B::avg_pool2d_backward(
                        x,
                        grad,
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                        ceil_mode,
                    );
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match AvgPool2D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                prep.finish(
                    (
                        x_state,
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                        ceil_mode,
                    ),
                    B::avg_pool2d(
                        x.primitive.clone(),
                        kernel_size,
                        stride,
                        padding,
                        count_include_pad,
                        ceil_mode,
                    ),
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::avg_pool2d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            )),
        }
    }

    fn avg_pool2d_backward(
        x: AutodiffTensor<B>,
        grad: AutodiffTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> AutodiffTensor<B> {
        super::pool_backward::average::<B, C>(x, grad, kernel_size, stride, padding, count_include_pad, ceil_mode)
    }

    fn max_pool1d(
        x: AutodiffTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> AutodiffTensor<B> {
        match MaxPool1D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let output = B::max_pool1d_with_indices(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                );
                prep.finish(
                    (
                        x_state,
                        output.indices,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        ceil_mode,
                    ),
                    output.output,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::max_pool1d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            )),
        }
    }

    fn max_pool1d_with_indices(
        x: AutodiffTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> MaxPool1dWithIndices<Self> {
        match MaxPool1D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let output = B::max_pool1d_with_indices(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                );

                let output_tensor = prep.finish(
                    (
                        x_state,
                        output.indices.clone(),
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        ceil_mode,
                    ),
                    output.output,
                );

                MaxPool1dWithIndices::new(output_tensor, output.indices)
            }
            OpsKind::UnTracked(prep) => {
                let output = B::max_pool1d_with_indices(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                );
                let output_tensor = prep.finish(output.output);

                MaxPool1dWithIndices::new(output_tensor, output.indices)
            }
        }
    }

    fn max_pool2d(
        x: AutodiffTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> AutodiffTensor<B> {
        match MaxPool2D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let output = B::max_pool2d_with_indices(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                );
                prep.finish(
                    (
                        x_state,
                        output.indices,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        ceil_mode,
                    ),
                    output.output,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::max_pool2d(
                x.primitive,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            )),
        }
    }

    fn max_pool2d_with_indices(
        x: AutodiffTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        match MaxPool2D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);

                let output = B::max_pool2d_with_indices(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                );

                let output_tensor = prep.finish(
                    (
                        x_state,
                        output.indices.clone(),
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        ceil_mode,
                    ),
                    output.output,
                );

                MaxPool2dWithIndices::new(output_tensor, output.indices)
            }
            OpsKind::UnTracked(prep) => {
                let output = B::max_pool2d_with_indices(
                    x.primitive,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                );
                let output_tensor = prep.finish(output.output);

                MaxPool2dWithIndices::new(output_tensor, output.indices)
            }
        }
    }

    fn max_pool2d_with_indices_backward(
        x: AutodiffTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: AutodiffTensor<B>,
        indices: IntTensor<B>,
    ) -> MaxPool2dBackward<Self> {
        MaxPool2dBackward::new(super::pool_backward::maximum::<B, C>(
            x, output_grad, indices, kernel_size, stride, padding, dilation, ceil_mode,
        ))
    }
    fn adaptive_avg_pool1d(x: AutodiffTensor<B>, output_size: usize) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct AdaptiveAvgPool1D;

        impl<B: Backend> Backward<B, 1> for AdaptiveAvgPool1D {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let state = checkpointer.retrieve_node_output(ops.state);

                if let Some(node) = node_parent {
                    let grad = B::adaptive_avg_pool1d_backward(state, grad);
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match AdaptiveAvgPool1D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                prep.finish(x_state, B::adaptive_avg_pool1d(x.primitive, output_size))
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::adaptive_avg_pool1d(x.primitive, output_size))
            }
        }
    }

    fn adaptive_avg_pool2d(x: AutodiffTensor<B>, output_size: [usize; 2]) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct AdaptiveAvgPool2D;

        impl<B: Backend> Backward<B, 1> for AdaptiveAvgPool2D {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let state = checkpointer.retrieve_node_output(ops.state);

                if let Some(node) = node_parent {
                    let grad = B::adaptive_avg_pool2d_backward(state, grad);
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match AdaptiveAvgPool2D
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                prep.finish(x_state, B::adaptive_avg_pool2d(x.primitive, output_size))
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::adaptive_avg_pool2d(x.primitive, output_size))
            }
        }
    }

    fn adaptive_avg_pool2d_backward(
        x: AutodiffTensor<B>,
        grad: AutodiffTensor<B>,
    ) -> AutodiffTensor<B> {
        super::pool_backward::adaptive_average::<B, C>(x, grad)
    }

    fn interpolate(
        x: AutodiffTensor<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> AutodiffTensor<B> {
        #[derive(Debug)]
        struct Interpolate;
        impl<B: Backend> Backward<B, 1> for Interpolate {
            type State = (NodeId, [usize; 2], InterpolateOptions);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, output_size, options) = ops.state;
                let state = checkpointer.retrieve_node_output(x_state);

                if let Some(node) = node_parent {
                    let grad = B::interpolate_backward(state, grad, output_size, options);
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match Interpolate
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let output = B::interpolate(x.primitive.clone(), output_size, options.clone());
                prep.finish((x_state, output_size, options), output)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::interpolate(x.primitive, output_size, options))
            }
        }
    }

    fn interpolate_backward(
        x: FloatTensor<Autodiff<B, C>>,
        grad: FloatTensor<Autodiff<B, C>>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> AutodiffTensor<B> {
        super::interpolation::backward::<B, C>(x, grad, output_size, options)
    }

    fn attention(
        query: FloatTensor<Autodiff<B, C>>,
        key: FloatTensor<Autodiff<B, C>>,
        value: FloatTensor<Autodiff<B, C>>,
        mask: Option<ruda_tensor::tensor::BoolTensor<Autodiff<B, C>>>,
        attn_bias: Option<FloatTensor<Autodiff<B, C>>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Autodiff<B, C>> {
        // Keep the compositional fallback for features whose gradients need
        // the full generic attention graph. Plain causal self-attention can
        // use the backend's FlashAttention forward and recompute its softmax
        // matrix during backward instead of materializing it in the graph.
        if mask.is_some()
            || attn_bias.is_some()
            || options.softcap.is_some()
            || options.scale.is_some()
            || !options.is_causal
        {
            return attention_fallback::<Self>(query, key, value, mask, attn_bias, options);
        }

        #[derive(Debug)]
        struct CausalAttention;

        impl<B: Backend> Backward<B, 3> for CausalAttention {
            type State = (NodeId, NodeId, NodeId);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [query_parent, key_parent, value_parent] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (query_state, key_state, value_state) = ops.state;
                let query =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(query_state);
                let key = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(key_state);
                let value =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(value_state);
                let query_dtype: FloatDType = query.dtype().into();
                let key_dtype: FloatDType = key.dtype().into();
                let value_dtype: FloatDType = value.dtype().into();
                let query_shape = query.shape();
                let key_shape = key.shape();
                let value_shape = value.shape();
                let accumulator_dtype = if query_dtype == FloatDType::F64 {
                    FloatDType::F64
                } else {
                    FloatDType::F32
                };
                let probabilities = causal_attention_probabilities::<B>(
                    query.clone(),
                    key.clone(),
                    accumulator_dtype,
                );
                let grad = B::float_cast(grad, accumulator_dtype);

                if let Some(node) = value_parent {
                    let value_grad =
                        B::float_matmul(B::float_transpose(probabilities.clone()), grad.clone());
                    let value_grad = broadcast_shape::<B>(value_grad, &value_shape);
                    let value_grad = B::float_cast(value_grad, value_dtype);
                    grads.register::<B>(node.id, value_grad);
                }

                if query_parent.is_none() && key_parent.is_none() {
                    return;
                }

                let value = B::float_cast(value, accumulator_dtype);
                let probability_grad = B::float_matmul(grad, B::float_transpose(value));
                let row_dot = B::float_sum_dim(
                    B::float_mul(probability_grad.clone(), probabilities.clone()),
                    3,
                );
                let scores_grad =
                    B::float_mul(probabilities, B::float_sub(probability_grad, row_dot));
                let head_dimension = query.shape().dims::<4>()[3];
                let scale = 1.0 / (head_dimension as f64).sqrt();
                let query = B::float_cast(query, accumulator_dtype);
                let key = B::float_cast(key, accumulator_dtype);

                if let Some(node) = query_parent {
                    let query_grad = B::float_matmul(scores_grad.clone(), key);
                    let query_grad = B::float_mul_scalar(query_grad, scale.into());
                    let query_grad = broadcast_shape::<B>(query_grad, &query_shape);
                    let query_grad = B::float_cast(query_grad, query_dtype);
                    grads.register::<B>(node.id, query_grad);
                }
                if let Some(node) = key_parent {
                    let key_grad = B::float_matmul(B::float_transpose(scores_grad), query);
                    let key_grad = B::float_mul_scalar(key_grad, scale.into());
                    let key_grad = broadcast_shape::<B>(key_grad, &key_shape);
                    let key_grad = B::float_cast(key_grad, key_dtype);
                    grads.register::<B>(node.id, key_grad);
                }
            }
        }

        match CausalAttention
            .prepare::<C>([query.node.clone(), key.node.clone(), value.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let query_state = prep.checkpoint(&query);
                let key_state = prep.checkpoint(&key);
                let value_state = prep.checkpoint(&value);
                let output = B::attention(
                    query.primitive,
                    key.primitive,
                    value.primitive,
                    None,
                    None,
                    options,
                );
                prep.finish((query_state, key_state, value_state), output)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::attention(
                query.primitive,
                key.primitive,
                value.primitive,
                None,
                None,
                options,
            )),
        }
    }

    fn ctc_loss(
        log_probs: FloatTensor<Autodiff<B, C>>,
        targets: IntTensor<Autodiff<B, C>>,
        input_lengths: IntTensor<Autodiff<B, C>>,
        target_lengths: IntTensor<Autodiff<B, C>>,
        blank: usize,
    ) -> FloatTensor<Autodiff<B, C>> {
        // Backends without a native ctc_loss_backward fall back to the default
        // implementation, which is built from differentiable tensor ops so the
        // autodiff layer derives the gradient automatically.
        if !B::has_ctc_loss_backward() {
            return ruda_tensor::ops::ctc::ctc_loss_default::<Self>(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank,
            );
        }

        #[derive(Debug)]
        struct CtcLoss;

        impl<B: Backend> Backward<B, 1> for CtcLoss {
            type State = (NodeId, IntTensor<B>, IntTensor<B>, IntTensor<B>, usize);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_parent] = ops.parents;
                let grad_loss = grads.consume::<B>(&ops.node);

                let (log_probs_state, targets, input_lengths, target_lengths, blank) = ops.state;
                let log_probs: B::FloatTensorPrimitive =
                    checkpointer.retrieve_node_output(log_probs_state);

                if let Some(node) = node_parent {
                    let grad = B::ctc_loss_backward(
                        log_probs,
                        targets,
                        input_lengths,
                        target_lengths,
                        grad_loss,
                        blank,
                    );
                    grads.register::<B>(node.id, grad);
                }
            }
        }

        match CtcLoss
            .prepare::<C>([log_probs.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let log_probs_state = prep.checkpoint(&log_probs);
                let output = B::ctc_loss(
                    log_probs.primitive.clone(),
                    targets.clone(),
                    input_lengths.clone(),
                    target_lengths.clone(),
                    blank,
                );
                prep.finish(
                    (
                        log_probs_state,
                        targets,
                        input_lengths,
                        target_lengths,
                        blank,
                    ),
                    output,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::ctc_loss(
                log_probs.primitive,
                targets,
                input_lengths,
                target_lengths,
                blank,
            )),
        }
    }

    fn rfft(
        signal: FloatTensor<Autodiff<B, C>>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Autodiff<B, C>>, FloatTensor<Autodiff<B, C>>) {
        super::fft::rfft::<B, C>(signal, dim, n)
    }

    fn irfft(
        spectrum_re: FloatTensor<Autodiff<B, C>>,
        spectrum_im: FloatTensor<Autodiff<B, C>>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Autodiff<B, C>> {
        super::fft::irfft::<B, C>(spectrum_re, spectrum_im, dim, n)
    }
}

#[derive(Debug)]
struct MaxPool1D;

impl<B: Backend> Backward<B, 1> for MaxPool1D {
    type State = (NodeId, IntTensor<B>, usize, usize, usize, usize, bool);

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let [node_parent] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let (x_state, indices, kernel_size, stride, padding, dilation, ceil_mode) = ops.state;
        let x = checkpointer.retrieve_node_output(x_state);

        if let Some(node) = node_parent {
            let grad = B::max_pool1d_with_indices_backward(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                grad,
                indices,
            );

            grads.register::<B>(node.id, grad.x_grad);
        }
    }
}

#[derive(Debug)]
struct MaxPool2D;

impl<B: Backend> Backward<B, 1> for MaxPool2D {
    type State = (
        NodeId,
        IntTensor<B>,
        [usize; 2],
        [usize; 2],
        [usize; 2],
        [usize; 2],
        bool,
    );

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let [node_parent] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let (x_state, indices, kernel_size, stride, padding, dilation, ceil_mode) = ops.state;
        let x = checkpointer.retrieve_node_output(x_state);

        if let Some(node) = node_parent {
            let grad = B::max_pool2d_with_indices_backward(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                grad,
                indices,
            );

            grads.register::<B>(node.id, grad.x_grad);
        }
    }
}
