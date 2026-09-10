use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_base_float(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &BaseOperationIr,
    ) {
        if self.run_base_quantized(handles, op) { return; }
        match op {
            BaseOperationIr::Reshape(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);

                let output = B::float_reshape(tensor, desc.out.shape.clone());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SwapDims(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);

                let output = B::float_swap_dims(tensor, desc.dim1, desc.dim2);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Permute(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);

                let output = B::float_permute(tensor, &desc.axes);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Flip(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);

                let output = B::float_flip(tensor, &desc.axes);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Expand(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);

                let output = B::float_expand(tensor, desc.out.shape.clone());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Unfold(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);

                let output = B::float_unfold(tensor, desc.dim, desc.size, desc.step);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Slice(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                let output = B::float_slice(tensor, &desc.ranges);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SliceAssign(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                let value = handles.get_float_tensor::<B>(&desc.value);

                let output = B::float_slice_assign(tensor, &desc.ranges, value);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Gather(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);

                let output = B::float_gather(desc.dim, tensor, indices);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Scatter(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let value = handles.get_float_tensor::<B>(&desc.value);

                let output = match desc.update {
                    IndexingUpdateOp::Add => B::float_scatter_add(desc.dim, tensor, indices, value),
                    _ => unimplemented!(),
                };
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::ScatterNd(desc) => {
                let data = handles.get_float_tensor::<B>(&desc.data);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let values = handles.get_float_tensor::<B>(&desc.values);

                let output = B::float_scatter_nd(data, indices, values, desc.reduction);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::GatherNd(desc) => {
                let data = handles.get_float_tensor::<B>(&desc.data);
                let indices = handles.get_int_tensor::<B>(&desc.indices);

                let output = B::float_gather_nd(data, indices);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Select(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);

                let output = B::float_select(tensor, desc.dim, indices);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SelectAssign(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let value = handles.get_float_tensor::<B>(&desc.value);

                let output = match desc.update {
                    IndexingUpdateOp::Add => B::float_select_add(tensor, desc.dim, indices, value),
                    _ => unimplemented!(),
                };
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::MaskWhere(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&desc.mask);
                let value = handles.get_float_tensor::<B>(&desc.value);

                let output = B::float_mask_where(tensor, mask, value);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::MaskFill(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&desc.mask);

                let output = B::float_mask_fill(tensor, mask, desc.value.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Equal(desc) => {
                binary_float_cmp_ops!(handles, desc, B::float_equal)
            }
            BaseOperationIr::EqualElem(desc) => {
                scalar_float_cmp_ops!(handles, desc, B::float_equal_elem)
            }
            BaseOperationIr::RepeatDim(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                let output = B::float_repeat_dim(tensor, desc.dim, desc.times);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Cat(desc) => {
                let tensors = desc
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_float_tensor::<B>(tensor))
                    .collect();

                let output = B::float_cat(tensors, desc.dim);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Cast(desc) => {
                let tensor = handles.get_float_tensor::<B>(&desc.input);
                let output = B::float_cast(tensor, desc.out.dtype.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Empty(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::float_empty(shape, &self.device, desc.out.dtype.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Ones(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::float_ones(shape, &self.device, desc.out.dtype.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Zeros(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::float_zeros(shape, &self.device, desc.out.dtype.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
        }
    }
}
