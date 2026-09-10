use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_base_quantized(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &BaseOperationIr,
    ) -> bool {
        match op {
            BaseOperationIr::Reshape(desc) if matches!(desc.input.dtype, DType::QFloat(_)) => {
                let input = handles.get_quantized_tensor::<B>(&desc.input);
                let output = B::q_reshape(input, desc.out.shape.clone());
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Expand(desc) if matches!(desc.input.dtype, DType::QFloat(_)) => {
                let input = handles.get_quantized_tensor::<B>(&desc.input);
                let output = B::q_expand(input, desc.out.shape.clone());
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SwapDims(desc) if matches!(desc.input.dtype, DType::QFloat(_)) => {
                let input = handles.get_quantized_tensor::<B>(&desc.input);
                let output = B::q_swap_dims(input, desc.dim1, desc.dim2);
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Permute(desc) if matches!(desc.input.dtype, DType::QFloat(_)) => {
                let input = handles.get_quantized_tensor::<B>(&desc.input);
                let output = B::q_permute(input, &desc.axes);
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Flip(desc) if matches!(desc.input.dtype, DType::QFloat(_)) => {
                let input = handles.get_quantized_tensor::<B>(&desc.input);
                let output = B::q_flip(input, &desc.axes);
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Slice(desc) if matches!(desc.tensor.dtype, DType::QFloat(_)) => {
                let input = handles.get_quantized_tensor::<B>(&desc.tensor);
                let output = B::q_slice(input, &desc.ranges);
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Gather(desc) if matches!(desc.tensor.dtype, DType::QFloat(_)) => {
                let input = handles.get_quantized_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let output = B::q_gather(desc.dim, input, indices);
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Select(desc) if matches!(desc.tensor.dtype, DType::QFloat(_)) => {
                let input = handles.get_quantized_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let output = B::q_select(input, desc.dim, indices);
                handles.register_quantized_tensor::<B>(&desc.out.id, output);
            }
            _ => return false,
        }
        true
    }
}
