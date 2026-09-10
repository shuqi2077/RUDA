use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_base_bool(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &BaseOperationIr,
    ) {
        match op {
            BaseOperationIr::Reshape(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.input);

                let output = B::bool_reshape(tensor, desc.out.shape.clone());
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SwapDims(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.input);

                let output = B::bool_swap_dims(tensor, desc.dim1, desc.dim2);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Permute(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.input);

                let output = B::bool_permute(tensor, &desc.axes);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Flip(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.input);

                let output = B::bool_flip(tensor, &desc.axes);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Expand(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.input);

                let output = B::bool_expand(tensor, desc.out.shape.clone());
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Unfold(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.input);

                let output = B::bool_unfold(tensor, desc.dim, desc.size, desc.step);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Slice(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.tensor);

                let output = B::bool_slice(tensor, &desc.ranges);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SliceAssign(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                let value = handles.get_bool_tensor::<B>(&desc.value);

                let output = B::bool_slice_assign(tensor, &desc.ranges, value);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Gather(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);

                let output = B::bool_gather(desc.dim, tensor, indices);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Scatter(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let value = handles.get_bool_tensor::<B>(&desc.value);

                let output = match desc.update {
                    IndexingUpdateOp::Add => B::bool_scatter_or(desc.dim, tensor, indices, value),
                    _ => unimplemented!(),
                };
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::ScatterNd(_) => {
                unreachable!("scatter_nd not supported for bool tensors")
            }
            BaseOperationIr::GatherNd(_) => {
                unreachable!("gather_nd not supported for bool tensors")
            }
            BaseOperationIr::Select(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);

                let output = B::bool_select(tensor, desc.dim, indices);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::SelectAssign(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                let indices = handles.get_int_tensor::<B>(&desc.indices);
                let value = handles.get_bool_tensor::<B>(&desc.value);

                let output = match desc.update {
                    IndexingUpdateOp::Add => B::bool_select_or(tensor, desc.dim, indices, value),
                    _ => unimplemented!(),
                };
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::MaskWhere(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&desc.mask);
                let value = handles.get_bool_tensor::<B>(&desc.value);

                let output = B::bool_mask_where(tensor, mask, value);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::MaskFill(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&desc.mask);

                let output = B::bool_mask_fill(tensor, mask, desc.value.into());
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Equal(desc) => {
                let lhs = handles.get_bool_tensor::<B>(&desc.lhs);
                let rhs = handles.get_bool_tensor::<B>(&desc.rhs);

                let output = B::bool_equal(lhs, rhs);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::EqualElem(desc) => {
                let lhs = handles.get_bool_tensor::<B>(&desc.lhs);

                let output = B::bool_equal_elem(lhs, desc.rhs.into());
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::RepeatDim(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.tensor);

                let output = B::bool_repeat_dim(tensor, desc.dim, desc.times);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Cat(desc) => {
                let tensors = desc
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_bool_tensor::<B>(tensor))
                    .collect();

                let output = B::bool_cat(tensors, desc.dim);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Cast(_) => unreachable!(),
            BaseOperationIr::Empty(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::bool_empty(shape, &self.device, desc.out.dtype.into());
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Zeros(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::bool_zeros(shape, &self.device, desc.out.dtype.into());
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BaseOperationIr::Ones(desc) => {
                let shape = desc.out.shape.clone();
                let output = B::bool_ones(shape, &self.device, desc.out.dtype.into());
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
        }
    }
}
