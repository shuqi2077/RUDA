use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_base_int(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &BaseOperationIr,
    ) {
        match op {
            BaseOperationIr::Reshape(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);

                let output = B::int_reshape(tensor, desc.out.shape.clone());
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SwapDims(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);

                let output = B::int_swap_dims(tensor, desc.dim1, desc.dim2);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Permute(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);

                let output = B::int_permute(tensor, &desc.axes);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Flip(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);

                let output = B::int_flip(tensor, &desc.axes);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Expand(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);

                let output = B::int_expand(tensor, desc.out.shape.clone());
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Unfold(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);

                let output = B::int_unfold(tensor, desc.dim, desc.size, desc.step);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Slice(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                let output = B::int_slice(tensor, &desc.ranges);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SliceAssign(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                let value = handles.get_int_tensor::<B>(&desc.value);

                let output = B::int_slice_assign(tensor, &desc.ranges, value);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Gather(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);

                let output = B::int_gather(desc.dim, tensor, indices);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Scatter(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let value = handles.get_int_tensor::<B>(&desc.value);

                let output = match desc.update {
                    IndexingUpdateOp::Add => B::int_scatter_add(desc.dim, tensor, indices, value),
                    _ => unimplemented!(),
                };
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::ScatterNd(desc) => {
                let data = handles.get_int_tensor::<B>(&desc.data);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let values = handles.get_int_tensor::<B>(&desc.values);

                let output = B::int_scatter_nd(data, indices, values, desc.reduction);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::GatherNd(desc) => {
                let data = handles.get_int_tensor::<B>(&desc.data);
                let indices = handles.get_int_tensor::<B>(&desc.indices);

                let output = B::int_gather_nd(data, indices);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Select(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);

                let output = B::int_select(tensor, desc.dim, indices);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SelectAssign(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let value = handles.get_int_tensor::<B>(&desc.value);

                let output = match desc.update {
                    IndexingUpdateOp::Add => B::int_select_add(tensor, desc.dim, indices, value),
                    _ => unimplemented!(),
                };
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::MaskWhere(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&desc.mask);
                let value = handles.get_int_tensor::<B>(&desc.value);

                let output = B::int_mask_where(tensor, mask, value);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::MaskFill(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&desc.mask);

                let output = B::int_mask_fill(tensor, mask, desc.value.into());
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Equal(desc) => {
                binary_int_cmp_ops!(handles, desc, B::int_equal)
            }
            BaseOperationIr::EqualElem(desc) => {
                scalar_int_cmp_ops!(handles, desc, B::int_equal_elem)
            }
            BaseOperationIr::RepeatDim(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                let output = B::int_repeat_dim(tensor, desc.dim, desc.times);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Cat(desc) => {
                let tensors = desc
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_int_tensor::<B>(tensor))
                    .collect();

                let output = B::int_cat(tensors, desc.dim);
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Cast(_) => unreachable!(),
            BaseOperationIr::Empty(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::int_empty(shape, &self.device, desc.out.dtype.into());
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Ones(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::int_ones(shape, &self.device, desc.out.dtype.into());
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Zeros(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::int_zeros(shape, &self.device, desc.out.dtype.into());
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
        }
    }
}
