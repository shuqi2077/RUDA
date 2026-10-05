use super::{ModuleMapper, Param, ParamId};
use ruda_tensor::{FloatDType, api::Tensor, backend::Backend, container::TensorContainer};

pub(super) struct DtypeMapper {
    dtype: FloatDType,
    converted: TensorContainer<ParamId>,
}

impl DtypeMapper {
    pub(super) fn new(dtype: FloatDType) -> Self {
        Self {
            dtype,
            converted: TensorContainer::new(),
        }
    }
}

impl<B: Backend> ModuleMapper<B> for DtypeMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        if let Some(converted) = self.converted.get::<B>(&id) {
            return Param::from_mapped_value(id, Tensor::<B, D>::from_primitive(converted), mapper);
        }
        let requires_grad = tensor.is_require_grad();
        let target_dtype: ruda_tensor::DType = self.dtype.into();
        let tensor = if tensor.dtype() == target_dtype {
            tensor
        } else {
            tensor
                .cast(self.dtype)
                .detach()
                .set_require_grad(requires_grad)
        };
        self.converted
            .register::<B>(id, tensor.clone().into_primitive());
        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestAutodiffBackend, module::Module};

    #[derive(Module, Debug)]
    struct SharedModule<B: Backend> {
        first: Param<Tensor<B, 1>>,
        alias: Param<Tensor<B, 1>>,
        frozen: Param<Tensor<B, 1>>,
    }

    #[test]
    fn dtype_conversion_preserves_shared_leaf_gradients_and_frozen_params() {
        let device = Default::default();
        type B = TestAutodiffBackend;
        for dtype in [FloatDType::F16, FloatDType::BF16] {
            let original = Tensor::<B, 1>::ones([2], &device).require_grad();
            let first = Param::from_tensor(original.clone());
            let id = first.id;
            let frozen = Param::from_tensor(Tensor::<B, 1>::ones([2], &device)).no_grad();
            let frozen_id = frozen.id;
            let model = SharedModule {
                first: first.clone(),
                alias: first,
                frozen,
            }
            .to_dtype(dtype);
            assert_eq!(model.first.id, id);
            assert_eq!(model.alias.id, id);
            assert_eq!(model.frozen.id, frozen_id);
            assert_eq!(model.first.val().dtype(), dtype.into());
            assert!(model.first.val().is_require_grad());
            assert!(!model.frozen.val().is_require_grad());
            let input = Tensor::<B, 1>::ones([2], &device).cast(dtype);
            let output =
                input.clone() * model.first.val() + input * model.alias.val() + model.frozen.val();
            let gradients = output.sum().backward();
            for param in [&model.first, &model.alias] {
                let gradient = param
                    .val()
                    .grad(&gradients)
                    .expect("missing converted parameter gradient");
                assert_eq!(gradient.dtype(), dtype.into());
                assert_eq!(
                    gradient
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    alloc::vec![2.; 2]
                );
            }
            assert!(model.frozen.val().grad(&gradients).is_none());
            assert!(original.grad(&gradients).is_none());
        }
    }
}
