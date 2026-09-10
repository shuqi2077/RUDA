use crate::{Autodiff, checkpoint::{base::Checkpointer, strategy::CheckpointStrategy}, grads::Gradients, tensor::AutodiffTensor};
use crate::ops::{Backward, Ops, OpsKind};
use ruda_tensor::{ElementConversion, TensorData, TensorMetadata, ops::{CsrAddition, SparseOps}, tensor::FloatTensor};
use alloc::vec::Vec;
use crate::graph::NodeId;

mod recompute;
use recompute::{CsrReplay, CsrUnaryReplay};

#[derive(Debug)]
struct CsrMatmul;

impl<B: SparseOps> Backward<B, 2> for CsrMatmul {
    type State = (B::CsrHandle, Option<NodeId>, Option<NodeId>, bool);

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
        let (matrix, values, rhs, transpose) = ops.state;
        let grad = grads.consume::<B>(&ops.node);
        let [values_parent, rhs_parent] = ops.parents;
        if let Some(parent) = values_parent {
            let rhs = checkpointer.retrieve_node_output::<FloatTensor<B>>(rhs.unwrap());
            let values_grad = if transpose {
                B::csr_sampled_matmul(&matrix, rhs.clone(), B::float_transpose(grad.clone()))
            } else {
                B::csr_sampled_matmul(&matrix, grad.clone(), B::float_transpose(rhs.clone()))
            }.unwrap_or_else(|error| panic!("CSR value gradient: {error}"));
            grads.register::<B>(parent.id, values_grad);
        }
        if let Some(parent) = rhs_parent {
            let values = checkpointer.retrieve_node_output::<FloatTensor<B>>(values.unwrap());
            let rhs_grad = B::csr_matmul(&matrix, values, grad, !transpose)
                .unwrap_or_else(|error| panic!("CSR dense gradient: {error}"));
            grads.register::<B>(parent.id, rhs_grad);
        }
    }
}

#[derive(Debug)]
struct CsrSampledMatmul;

impl<B: SparseOps> Backward<B, 2> for CsrSampledMatmul {
    type State = (B::CsrHandle, Option<NodeId>, Option<NodeId>);

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
        let (matrix, lhs, rhs) = ops.state;
        let grad = grads.consume::<B>(&ops.node);
        let [lhs_parent, rhs_parent] = ops.parents;
        if let Some(parent) = lhs_parent {
            let rhs = checkpointer.retrieve_node_output::<FloatTensor<B>>(rhs.unwrap());
            let lhs_grad = B::csr_matmul(&matrix, grad.clone(), B::float_transpose(rhs), false)
                .unwrap_or_else(|error| panic!("Sampled matmul lhs gradient: {error}"));
            grads.register::<B>(parent.id, lhs_grad);
        }
        if let Some(parent) = rhs_parent {
            let lhs = checkpointer.retrieve_node_output::<FloatTensor<B>>(lhs.unwrap());
            let rhs_grad = B::csr_matmul(&matrix, grad, lhs, true)
                .unwrap_or_else(|error| panic!("Sampled matmul rhs gradient: {error}"));
            grads.register::<B>(parent.id, B::float_transpose(rhs_grad));
        }
    }
}

impl<B: SparseOps, C: CheckpointStrategy> SparseOps for Autodiff<B, C> {
    type CsrHandle = B::CsrHandle;
    type CsrData = B::CsrData;
    type SparseError = B::SparseError;

    fn csr_from_data(data: &Self::CsrData, device: &Self::Device) -> Result<Self::CsrHandle, Self::SparseError> {
        B::csr_from_data(data, device)
    }

    fn csr_to_device(matrix: &Self::CsrHandle, device: &Self::Device) -> Self::CsrHandle {
        B::csr_to_device(matrix, device)
    }

    fn csr_transpose_with_permutation(matrix: &Self::CsrHandle) -> Result<(Self::CsrHandle, Vec<u32>), Self::SparseError> {
        B::csr_transpose_with_permutation(matrix)
    }

    async fn csr_to_data(matrix: &Self::CsrHandle, values: FloatTensor<Self>) -> Result<Self::CsrData, Self::SparseError> {
        B::csr_to_data(matrix, values.primitive).await
    }

    fn csr_shape(matrix: &Self::CsrHandle) -> [usize; 2] {
        B::csr_shape(matrix)
    }

    fn csr_nnz(matrix: &Self::CsrHandle) -> usize {
        B::csr_nnz(matrix)
    }

    fn csr_product_pattern(left: &Self::CsrHandle, right: &Self::CsrHandle) -> Result<Self::CsrHandle, Self::SparseError> {
        B::csr_product_pattern(left, right)
    }

    fn csr_sampled_sparse_matmul(
        pattern: &Self::CsrHandle, left: &Self::CsrHandle, left_values: FloatTensor<Self>,
        right: &Self::CsrHandle, right_values: FloatTensor<Self>, transpose_left: bool, transpose_right: bool,
    ) -> Result<FloatTensor<Self>, Self::SparseError> {
        let output = B::csr_sampled_sparse_matmul(pattern, left, left_values.primitive.clone(), right, right_values.primitive.clone(), transpose_left, transpose_right)?;
        Ok(match CsrSampledSparseMatmul.prepare::<C>([left_values.node.clone(), right_values.node.clone()]).compute_bound().stateful() {
            OpsKind::Tracked(mut prep) => {
                let left_state = right_values.is_tracked().then(|| prep.checkpoint(&left_values));
                let right_state = left_values.is_tracked().then(|| prep.checkpoint(&right_values));
                prep.finish((
                    pattern.clone(), left.clone(), right.clone(), left_state, right_state,
                    transpose_left, transpose_right,
                ), output)
            }
            OpsKind::UnTracked(prep) => prep.finish(output),
        })
    }

    fn csr_add_prepare(left: &Self::CsrHandle, right: &Self::CsrHandle) -> Result<CsrAddition<Self::CsrHandle>, Self::SparseError> {
        B::csr_add_prepare(left, right)
    }

    fn csr_add(plan: &CsrAddition<Self::CsrHandle>, left: FloatTensor<Self>, right: FloatTensor<Self>, alpha: f32, beta: f32) -> Result<FloatTensor<Self>, Self::SparseError> {
        let device = B::float_device(&left.primitive);
        let output = B::csr_add(plan, left.primitive, right.primitive, alpha, beta)?;
        Ok(match CsrAdd.prepare::<C>([left.node, right.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((
                [plan.left_entries.clone(), plan.right_entries.clone()], [alpha, beta], device,
            ), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        })
    }

    fn csr_validate_operand<T: TensorMetadata>(
        matrix: &Self::CsrHandle,
        operand: &T,
        device: &Self::Device,
        shape: &[usize],
    ) -> Result<(), Self::SparseError> {
        B::csr_validate_operand(matrix, operand, device, shape)
    }

    fn csr_values(matrix: &Self::CsrHandle) -> FloatTensor<Self> {
        AutodiffTensor::new(B::csr_values(matrix))
    }

    fn csr_validate_values(matrix: &Self::CsrHandle, values: &FloatTensor<Self>) -> Result<(), Self::SparseError> {
        B::csr_validate_values(matrix, &values.primitive)
    }

    fn csr_gather(matrix: &Self::CsrHandle, dense: FloatTensor<Self>) -> Result<FloatTensor<Self>, Self::SparseError> {
        csr_indexing::<B, C>(matrix, dense, true)
    }

    fn csr_scatter_add(matrix: &Self::CsrHandle, values: FloatTensor<Self>) -> Result<FloatTensor<Self>, Self::SparseError> {
        csr_indexing::<B, C>(matrix, values, false)
    }

    fn csr_to_dense(matrix: &Self::CsrHandle, values: FloatTensor<Self>) -> Result<FloatTensor<Self>, Self::SparseError> {
        csr_dense_conversion::<B, C>(matrix, values, false)
    }

    fn csr_to_dense_backward(matrix: &Self::CsrHandle, grad: FloatTensor<Self>) -> Result<FloatTensor<Self>, Self::SparseError> {
        csr_dense_conversion::<B, C>(matrix, grad, true)
    }

    fn csr_matmul(
        matrix: &Self::CsrHandle,
        values: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        transpose: bool,
    ) -> Result<FloatTensor<Self>, Self::SparseError> {
        let output = B::csr_matmul(matrix, values.primitive.clone(), rhs.primitive.clone(), transpose)?;
        Ok(match CsrMatmul.prepare::<C>([values.node.clone(), rhs.node.clone()]).compute_bound().stateful() {
            OpsKind::Tracked(mut prep) => {
                let values_state = rhs.is_tracked().then(|| prep.checkpoint(&values));
                let rhs_state = values.is_tracked().then(|| prep.checkpoint(&rhs));
                prep.finish((matrix.clone(), values_state, rhs_state, transpose), output)
            }
            OpsKind::UnTracked(prep) => prep.finish(output),
        })
    }

    fn csr_sampled_matmul(
        matrix: &Self::CsrHandle,
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
    ) -> Result<FloatTensor<Self>, Self::SparseError> {
        let output = B::csr_sampled_matmul(matrix, lhs.primitive.clone(), rhs.primitive.clone())?;
        Ok(match CsrSampledMatmul.prepare::<C>([lhs.node.clone(), rhs.node.clone()]).compute_bound().stateful() {
            OpsKind::Tracked(mut prep) => {
                let lhs_state = rhs.is_tracked().then(|| prep.checkpoint(&lhs));
                let rhs_state = lhs.is_tracked().then(|| prep.checkpoint(&rhs));
                prep.finish((matrix.clone(), lhs_state, rhs_state), output)
            }
            OpsKind::UnTracked(prep) => prep.finish(output),
        })
    }
}

#[derive(Debug)]
struct CsrDenseConversion;

#[derive(Debug)]
struct CsrIndexing;

impl<B: SparseOps> Backward<B, 1> for CsrIndexing {
    type State = (B::CsrHandle, bool);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let (matrix, gather) = ops.state;
        let grad = grads.consume::<B>(&ops.node);
        if let Some(parent) = ops.parents[0].as_ref() {
            let output = if gather {
                B::csr_scatter_add(&matrix, grad)
            } else {
                B::csr_gather(&matrix, grad)
            }.unwrap_or_else(|error| panic!("CSR indexing gradient: {error}"));
            grads.register::<B>(parent.id, output);
        }
    }
}

fn csr_indexing<B: SparseOps, C: CheckpointStrategy>(
    matrix: &B::CsrHandle, input: AutodiffTensor<B>, gather: bool,
) -> Result<AutodiffTensor<B>, B::SparseError> {
    let replay = if gather { CsrUnaryReplay::Gather } else { CsrUnaryReplay::ScatterAdd };
    let prep = CsrIndexing.prepare::<C>([input.node.clone()])
        .memory_bound()
        .retro_forward(CsrReplay::<B>::new(matrix.clone(), input.node.id, replay))
        .parents([&input])
        .stateful();
    let output = if gather {
        B::csr_gather(matrix, input.primitive)
    } else {
        B::csr_scatter_add(matrix, input.primitive)
    }?;
    Ok(match prep {
        OpsKind::Tracked(prep) => prep.finish((matrix.clone(), gather), output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    })
}

#[derive(Debug)]
struct CsrAdd;

#[derive(Debug)]
struct CsrSampledSparseMatmul;

impl<B: SparseOps> Backward<B, 2> for CsrSampledSparseMatmul {
    type State = (B::CsrHandle, B::CsrHandle, B::CsrHandle, Option<NodeId>, Option<NodeId>, bool, bool);

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
        let (pattern, left, right, left_values, right_values, transpose_left, transpose_right) = ops.state;
        let grad = grads.consume::<B>(&ops.node);
        let [left_parent, right_parent] = ops.parents;
        if let Some(parent) = left_parent {
            let right_values = checkpointer.retrieve_node_output::<FloatTensor<B>>(right_values.unwrap());
            let output = if transpose_left {
                B::csr_sampled_sparse_matmul(&left, &right, right_values, &pattern, grad.clone(), transpose_right, true)
            } else {
                B::csr_sampled_sparse_matmul(&left, &pattern, grad.clone(), &right, right_values, false, !transpose_right)
            }.unwrap_or_else(|error| panic!("Sampled sparse product left gradient: {error}"));
            grads.register::<B>(parent.id, output);
        }
        if let Some(parent) = right_parent {
            let left_values = checkpointer.retrieve_node_output::<FloatTensor<B>>(left_values.unwrap());
            let output = if transpose_right {
                B::csr_sampled_sparse_matmul(&right, &pattern, grad, &left, left_values, true, transpose_left)
            } else {
                B::csr_sampled_sparse_matmul(&right, &left, left_values, &pattern, grad, !transpose_left, false)
            }.unwrap_or_else(|error| panic!("Sampled sparse product right gradient: {error}"));
            grads.register::<B>(parent.id, output);
        }
    }
}

impl<B: SparseOps> Backward<B, 2> for CsrAdd {
    type State = ([Vec<u32>; 2], [f32; 2], B::Device);

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let (mappings, scales, device) = ops.state;
        let grad = grads.consume::<B>(&ops.node);
        for ((parent, mapping), scale) in ops.parents.into_iter().zip(mappings).zip(scales) {
            if let Some(parent) = parent {
                let count = mapping.len();
                let indices = B::int_from_data(TensorData::new(
                    mapping.into_iter().map(i64::from).collect::<Vec<_>>(), [count],
                ), &device);
                let values = B::float_select(grad.clone(), 0, indices);
                grads.register::<B>(parent.id, B::float_mul_scalar(values, scale.into()));
            }
        }
    }
}

impl<B: SparseOps> Backward<B, 1> for CsrDenseConversion {
    type State = (B::CsrHandle, bool);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let (matrix, backward) = ops.state;
        let grad = grads.consume::<B>(&ops.node);
        if let Some(parent) = ops.parents[0].as_ref() {
            let output = if backward {
                B::csr_to_dense(&matrix, grad)
            } else {
                B::csr_to_dense_backward(&matrix, grad)
            }.unwrap_or_else(|error| panic!("CSR dense conversion gradient: {error}"));
            grads.register::<B>(parent.id, output);
        }
    }
}

fn csr_dense_conversion<B: SparseOps, C: CheckpointStrategy>(
    matrix: &B::CsrHandle,
    input: AutodiffTensor<B>,
    backward: bool,
) -> Result<AutodiffTensor<B>, B::SparseError> {
    let replay = if backward { CsrUnaryReplay::ToDenseBackward } else { CsrUnaryReplay::ToDense };
    let prep = CsrDenseConversion.prepare::<C>([input.node.clone()])
        .memory_bound()
        .retro_forward(CsrReplay::<B>::new(matrix.clone(), input.node.id, replay))
        .parents([&input])
        .stateful();
    let output = if backward {
        B::csr_to_dense_backward(matrix, input.primitive)
    } else {
        B::csr_to_dense(matrix, input.primitive)
    }?;
    Ok(match prep {
        OpsKind::Tracked(prep) => prep.finish((matrix.clone(), backward), output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    })
}
