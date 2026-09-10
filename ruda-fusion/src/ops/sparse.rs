use crate::{
    Fusion, FusionBackend, get_client,
    ops::NoOp,
    stream::{OperationStreams, execution::Operation},
};
use ruda_tensor::{
    Shape, TensorMetadata,
    graph::{CustomOpIr, HandleContainer, InitOperationIr, OperationIr, OperationOutput, TensorIr},
    ops::{CsrAddition, SparseOps},
    tensor::FloatTensor,
};

#[derive(Debug)]
struct CsrMatmulOp<B: FusionBackend + SparseOps> {
    desc: CustomOpIr,
    matrix: B::CsrHandle,
    transpose: bool,
}

impl<B> Operation<B::FusionRuntime> for CsrMatmulOp<B>
where
    B: FusionBackend + SparseOps,
    B::CsrHandle: Sync,
{
    fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
        let values = handles.get_float_tensor::<B>(&self.desc.inputs[0]);
        let rhs = handles.get_float_tensor::<B>(&self.desc.inputs[1]);
        let output = B::csr_matmul(&self.matrix, values, rhs, self.transpose)
            .unwrap_or_else(|error| panic!("Fused CSR matmul execution: {error}"));
        handles.register_float_tensor::<B>(&self.desc.outputs[0].id, output);
    }
}

#[derive(Debug)]
struct CsrSampledMatmulOp<B: FusionBackend + SparseOps> {
    desc: CustomOpIr,
    matrix: B::CsrHandle,
}

impl<B> Operation<B::FusionRuntime> for CsrSampledMatmulOp<B>
where
    B: FusionBackend + SparseOps,
    B::CsrHandle: Sync,
{
    fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
        let lhs = handles.get_float_tensor::<B>(&self.desc.inputs[0]);
        let rhs = handles.get_float_tensor::<B>(&self.desc.inputs[1]);
        let output = B::csr_sampled_matmul(&self.matrix, lhs, rhs)
            .unwrap_or_else(|error| panic!("Fused sampled matmul execution: {error}"));
        handles.register_float_tensor::<B>(&self.desc.outputs[0].id, output);
    }
}

impl<B> SparseOps for Fusion<B>
where
    B: FusionBackend + SparseOps,
    B::CsrHandle: Sync,
{
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
        Self::csr_validate_values(matrix, &values)?;
        let values = values.client.clone().resolve_tensor_float::<B>(values);
        B::csr_to_data(matrix, values).await
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
        Self::csr_validate_values(left, &left_values)?;
        Self::csr_validate_values(right, &right_values)?;
        B::csr_validate_operand(left, &right_values, right_values.client.device(), &[B::csr_nnz(right)])?;
        let [left_rows, left_columns] = B::csr_shape(left);
        let [right_rows, right_columns] = B::csr_shape(right);
        let (rows, left_inner) = if transpose_left { (left_columns, left_rows) } else { (left_rows, left_columns) };
        let (right_inner, columns) = if transpose_right { (right_columns, right_rows) } else { (right_rows, right_columns) };
        let device = left_values.client.device();
        let shape = SparseProductMetadata { shape: Shape::new([rows, columns, left_inner]), dtype: left_values.dtype };
        let [pattern_rows, pattern_columns] = B::csr_shape(pattern);
        B::csr_validate_operand(pattern, &shape, device, &[pattern_rows, pattern_columns, right_inner])?;
        let streams = OperationStreams::with_inputs([&left_values, &right_values]);
        let client = left_values.client.clone();
        let out = TensorIr::uninit(client.create_empty_handle(), Shape::new([B::csr_nnz(pattern)]), left_values.dtype);
        let id = match (transpose_left, transpose_right) {
            (false, false) => "ruda.csr_sampled_sparse_matmul.nn.v1",
            (false, true) => "ruda.csr_sampled_sparse_matmul.nt.v1",
            (true, false) => "ruda.csr_sampled_sparse_matmul.tn.v1",
            (true, true) => "ruda.csr_sampled_sparse_matmul.tt.v1",
        };
        let desc = CustomOpIr::new(id, &[left_values.into_ir(), right_values.into_ir()], &[out]);
        let operation = CsrSampledSparseMatmulOp::<B> {
            desc: desc.clone(), pattern: pattern.clone(), left: left.clone(), right: right.clone(), transpose_left, transpose_right,
        };
        Ok(client.register(streams, OperationIr::Custom(desc), operation).output())
    }

    fn csr_add_prepare(left: &Self::CsrHandle, right: &Self::CsrHandle) -> Result<CsrAddition<Self::CsrHandle>, Self::SparseError> {
        B::csr_add_prepare(left, right)
    }

    fn csr_add(plan: &CsrAddition<Self::CsrHandle>, left: FloatTensor<Self>, right: FloatTensor<Self>, alpha: f32, beta: f32) -> Result<FloatTensor<Self>, Self::SparseError> {
        Self::csr_validate_values(&plan.left, &left)?;
        Self::csr_validate_values(&plan.right, &right)?;
        B::csr_validate_operand(&plan.left, &right, right.client.device(), &[B::csr_nnz(&plan.right)])?;
        let streams = OperationStreams::with_inputs([&left, &right]);
        let client = left.client.clone();
        let out = TensorIr::uninit(client.create_empty_handle(), Shape::new([B::csr_nnz(&plan.output)]), left.dtype);
        let desc = CustomOpIr::new("ruda.csr_add.v1", &[left.into_ir(), right.into_ir()], &[out]);
        let operation = CsrAddOp::<B> { desc: desc.clone(), plan: plan.clone(), alpha, beta };
        Ok(client.register(streams, OperationIr::Custom(desc), operation).output())
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
        let values = B::csr_values(matrix);
        let client = get_client::<B>(&B::float_device(&values));
        let shape = values.shape();
        let dtype = values.dtype();
        let handle = B::float_tensor_handle(values);
        let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));
        client.register(OperationStreams::default(), OperationIr::Init(desc), NoOp::<B>::new()).output()
    }

    fn csr_validate_values(matrix: &Self::CsrHandle, values: &FloatTensor<Self>) -> Result<(), Self::SparseError> {
        B::csr_validate_operand(matrix, values, values.client.device(), &[B::csr_nnz(matrix)])
    }

    fn csr_gather(matrix: &Self::CsrHandle, dense: FloatTensor<Self>) -> Result<FloatTensor<Self>, Self::SparseError> {
        csr_indexing::<B>(matrix, dense, true)
    }

    fn csr_scatter_add(matrix: &Self::CsrHandle, values: FloatTensor<Self>) -> Result<FloatTensor<Self>, Self::SparseError> {
        csr_indexing::<B>(matrix, values, false)
    }

    fn csr_to_dense(matrix: &Self::CsrHandle, values: FloatTensor<Self>) -> Result<FloatTensor<Self>, Self::SparseError> {
        csr_dense_conversion::<B>(matrix, values, false)
    }

    fn csr_to_dense_backward(matrix: &Self::CsrHandle, grad: FloatTensor<Self>) -> Result<FloatTensor<Self>, Self::SparseError> {
        csr_dense_conversion::<B>(matrix, grad, true)
    }

    fn csr_matmul(
        matrix: &Self::CsrHandle,
        values: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        transpose: bool,
    ) -> Result<FloatTensor<Self>, Self::SparseError> {
        Self::csr_validate_values(matrix, &values)?;
        let [rows, columns] = B::csr_shape(matrix);
        let (rows, inner) = if transpose { (columns, rows) } else { (rows, columns) };
        let columns = rhs.shape.get(1).copied().unwrap_or(0);
        B::csr_validate_operand(matrix, &rhs, rhs.client.device(), &[inner, columns])?;
        let streams = OperationStreams::with_inputs([&values, &rhs]);
        let client = values.client.clone();
        let out = TensorIr::uninit(client.create_empty_handle(), Shape::new([rows, columns]), values.dtype);
        let id = if transpose { "ruda.csr_matmul.transpose.v1" } else { "ruda.csr_matmul.v1" };
        let desc = CustomOpIr::new(id, &[values.into_ir(), rhs.into_ir()], &[out]);
        let operation = CsrMatmulOp::<B> { desc: desc.clone(), matrix: matrix.clone(), transpose };
        Ok(client.register(streams, OperationIr::Custom(desc), operation).output())
    }

    fn csr_sampled_matmul(
        matrix: &Self::CsrHandle,
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
    ) -> Result<FloatTensor<Self>, Self::SparseError> {
        let [rows, columns] = B::csr_shape(matrix);
        let inner = lhs.shape.get(1).copied().unwrap_or(0);
        B::csr_validate_operand(matrix, &lhs, lhs.client.device(), &[rows, inner])?;
        B::csr_validate_operand(matrix, &rhs, rhs.client.device(), &[inner, columns])?;
        let streams = OperationStreams::with_inputs([&lhs, &rhs]);
        let client = lhs.client.clone();
        let out = TensorIr::uninit(client.create_empty_handle(), Shape::new([B::csr_nnz(matrix)]), lhs.dtype);
        let desc = CustomOpIr::new("ruda.csr_sampled_matmul.v1", &[lhs.into_ir(), rhs.into_ir()], &[out]);
        let operation = CsrSampledMatmulOp::<B> { desc: desc.clone(), matrix: matrix.clone() };
        Ok(client.register(streams, OperationIr::Custom(desc), operation).output())
    }
}

#[derive(Debug)]
struct CsrDenseConversionOp<B: FusionBackend + SparseOps> {
    desc: CustomOpIr,
    matrix: B::CsrHandle,
    backward: bool,
}

#[derive(Debug)]
struct CsrIndexingOp<B: FusionBackend + SparseOps> {
    desc: CustomOpIr,
    matrix: B::CsrHandle,
    gather: bool,
}

impl<B> Operation<B::FusionRuntime> for CsrIndexingOp<B>
where
    B: FusionBackend + SparseOps,
    B::CsrHandle: Sync,
{
    fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
        let input = handles.get_float_tensor::<B>(&self.desc.inputs[0]);
        let output = if self.gather {
            B::csr_gather(&self.matrix, input)
        } else {
            B::csr_scatter_add(&self.matrix, input)
        }.unwrap_or_else(|error| panic!("Fused CSR indexing: {error}"));
        handles.register_float_tensor::<B>(&self.desc.outputs[0].id, output);
    }
}

fn csr_indexing<B>(
    matrix: &B::CsrHandle, input: FloatTensor<Fusion<B>>, gather: bool,
) -> Result<FloatTensor<Fusion<B>>, B::SparseError>
where
    B: FusionBackend + SparseOps,
    B::CsrHandle: Sync,
{
    let dense_shape = Shape::new(B::csr_shape(matrix));
    let values_shape = Shape::new([B::csr_nnz(matrix)]);
    let (input_shape, output_shape, id) = if gather {
        (dense_shape, values_shape, "ruda.csr_gather.v1")
    } else {
        (values_shape, dense_shape, "ruda.csr_scatter_add.v1")
    };
    B::csr_validate_operand(matrix, &input, input.client.device(), &input_shape[..])?;
    let streams = OperationStreams::with_inputs([&input]);
    let client = input.client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), output_shape, input.dtype);
    let desc = CustomOpIr::new(id, &[input.into_ir()], &[out]);
    let operation = CsrIndexingOp::<B> { desc: desc.clone(), matrix: matrix.clone(), gather };
    Ok(client.register(streams, OperationIr::Custom(desc), operation).output())
}

#[derive(Debug)]
struct CsrAddOp<B: FusionBackend + SparseOps> {
    desc: CustomOpIr,
    plan: CsrAddition<B::CsrHandle>,
    alpha: f32,
    beta: f32,
}

#[derive(Debug)]
struct CsrSampledSparseMatmulOp<B: FusionBackend + SparseOps> {
    desc: CustomOpIr,
    pattern: B::CsrHandle,
    left: B::CsrHandle,
    right: B::CsrHandle,
    transpose_left: bool,
    transpose_right: bool,
}

#[derive(Clone, Debug)]
struct SparseProductMetadata {
    shape: Shape,
    dtype: ruda_tensor::DType,
}

impl TensorMetadata for SparseProductMetadata {
    fn shape(&self) -> Shape { self.shape.clone() }
    fn dtype(&self) -> ruda_tensor::DType { self.dtype }
}

impl<B> Operation<B::FusionRuntime> for CsrSampledSparseMatmulOp<B>
where
    B: FusionBackend + SparseOps,
    B::CsrHandle: Sync,
{
    fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
        let left_values = handles.get_float_tensor::<B>(&self.desc.inputs[0]);
        let right_values = handles.get_float_tensor::<B>(&self.desc.inputs[1]);
        let output = B::csr_sampled_sparse_matmul(
            &self.pattern, &self.left, left_values, &self.right, right_values, self.transpose_left, self.transpose_right,
        ).unwrap_or_else(|error| panic!("Fused sampled sparse product: {error}"));
        handles.register_float_tensor::<B>(&self.desc.outputs[0].id, output);
    }
}

impl<B> Operation<B::FusionRuntime> for CsrAddOp<B>
where
    B: FusionBackend + SparseOps,
    B::CsrHandle: Sync,
{
    fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
        let left = handles.get_float_tensor::<B>(&self.desc.inputs[0]);
        let right = handles.get_float_tensor::<B>(&self.desc.inputs[1]);
        let output = B::csr_add(&self.plan, left, right, self.alpha, self.beta)
            .unwrap_or_else(|error| panic!("Fused CSR add execution: {error}"));
        handles.register_float_tensor::<B>(&self.desc.outputs[0].id, output);
    }
}

impl<B> Operation<B::FusionRuntime> for CsrDenseConversionOp<B>
where
    B: FusionBackend + SparseOps,
    B::CsrHandle: Sync,
{
    fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
        let input = handles.get_float_tensor::<B>(&self.desc.inputs[0]);
        let output = if self.backward {
            B::csr_to_dense_backward(&self.matrix, input)
        } else {
            B::csr_to_dense(&self.matrix, input)
        }.unwrap_or_else(|error| panic!("Fused CSR dense conversion: {error}"));
        handles.register_float_tensor::<B>(&self.desc.outputs[0].id, output);
    }
}

fn csr_dense_conversion<B>(
    matrix: &B::CsrHandle,
    input: FloatTensor<Fusion<B>>,
    backward: bool,
) -> Result<FloatTensor<Fusion<B>>, B::SparseError>
where
    B: FusionBackend + SparseOps,
    B::CsrHandle: Sync,
{
    let dense_shape = Shape::new(B::csr_shape(matrix));
    let values_shape = Shape::new([B::csr_nnz(matrix)]);
    let (input_shape, output_shape, id) = if backward {
        (dense_shape, values_shape, "ruda.csr_to_dense.backward.v1")
    } else {
        (values_shape, dense_shape, "ruda.csr_to_dense.v1")
    };
    B::csr_validate_operand(matrix, &input, input.client.device(), &input_shape[..])?;
    let streams = OperationStreams::with_inputs([&input]);
    let client = input.client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), output_shape, input.dtype);
    let desc = CustomOpIr::new(id, &[input.into_ir()], &[out]);
    let operation = CsrDenseConversionOp::<B> {
        desc: desc.clone(), matrix: matrix.clone(), backward,
    };
    Ok(client.register(streams, OperationIr::Custom(desc), operation).output())
}
