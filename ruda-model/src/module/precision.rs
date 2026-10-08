use super::{ModuleMapper, Param, ParamId};
use alloc::collections::BTreeSet;
use ruda_tensor::{FloatDType, api::Tensor, backend::Backend, tensor::TensorContainer};

pub(super) struct DtypeMapper {
    dtype: FloatDType,
    selected: Option<BTreeSet<ParamId>>,
    converted: TensorContainer<(ParamId, bool)>,
}

impl DtypeMapper {
    pub(super) fn new(dtype: FloatDType) -> Self {
        Self {
            dtype,
            selected: None,
            converted: TensorContainer::new(),
        }
    }
    pub(super) fn new_selected(dtype: FloatDType, parameters: &[ParamId]) -> Self {
        Self { dtype, selected: Some(parameters.iter().copied().collect()), converted: TensorContainer::new() }
    }
}

impl<B: Backend> ModuleMapper<B> for DtypeMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if self.selected.as_ref().is_some_and(|selected| !selected.contains(&param.id)) {
            return param;
        }
        let (id, tensor, mapper) = param.consume();
        let requires_grad = tensor.is_require_grad();
        let key = (id, requires_grad);
        if let Some(converted) = self.converted.get::<B>(&key) {
            return Param::from_mapped_value(id, Tensor::<B, D>::from_primitive(converted), mapper);
        }
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
            .register::<B>(key, tensor.clone().into_primitive());
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
        frozen_alias: Param<Tensor<B, 1>>,
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
                frozen_alias: first.clone().no_grad(),
                alias: first,
                frozen,
            }
            .to_dtype(dtype);
            assert_eq!(model.first.id, id);
            assert_eq!(model.alias.id, id);
            assert_eq!(model.frozen_alias.id, id);
            assert_eq!(model.frozen.id, frozen_id);
            assert_eq!(model.first.val().dtype(), dtype.into());
            assert!(model.first.val().is_require_grad());
            assert!(!model.frozen.val().is_require_grad());
            assert!(!model.frozen_alias.val().is_require_grad());
            let input = Tensor::<B, 1>::ones([2], &device).cast(dtype);
            let output = input.clone() * model.first.val()
                + input * model.alias.val()
                + model.frozen.val()
                + model.frozen_alias.val();
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
            assert!(model.frozen_alias.val().grad(&gradients).is_none());
            assert!(original.grad(&gradients).is_none());
        }
    }

    #[test]
    fn dtype_conversion_keeps_diverged_frozen_alias_values_and_record_restore() {
        use crate::{
            module::ModuleDTypeRecord,
            record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        };
        let device = Default::default();
        type B = TestAutodiffBackend;
        for dtype in [FloatDType::F16, FloatDType::BF16] {
            let first = Param::from_tensor(Tensor::<B, 1>::from_floats([1., 3.], &device));
            let frozen_alias = first.clone().no_grad();
            let first = first.map(|tensor| (tensor + 1.).detach().require_grad());
            let module = SharedModule {
                first: first.clone(),
                alias: first,
                frozen_alias,
                frozen: Param::from_tensor(Tensor::<B, 1>::ones([2], &device)).no_grad(),
            }
            .to_dtype(dtype);
            assert_eq!(module.first.id, module.frozen_alias.id);
            let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
            let profile = ModuleDTypeRecord::capture(&module).unwrap();
            let bytes = <BinBytesRecorder<FullPrecisionSettings> as Recorder<B>>::record(
                &recorder,
                (module.clone().into_record(), profile),
                (),
            )
            .unwrap();
            let (record, profile): (<SharedModule<B> as Module<B>>::Record, ModuleDTypeRecord) =
                <BinBytesRecorder<FullPrecisionSettings> as Recorder<B>>::load(
                    &recorder, bytes, &device,
                )
                .unwrap();
            let restored = profile
                .apply(
                    module
                        .clone()
                        .to_dtype(FloatDType::F32)
                        .load_record(record)
                        .fork(&device),
                )
                .unwrap();
            for model in [module, restored] {
                assert_eq!(model.first.val().dtype(), dtype.into());
                assert_eq!(model.frozen_alias.val().dtype(), dtype.into());
                assert_eq!(
                    model
                        .first
                        .val()
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    alloc::vec![2., 4.]
                );
                assert_eq!(
                    model
                        .frozen_alias
                        .val()
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    alloc::vec![1., 3.]
                );
                assert!(!model.frozen_alias.val().is_require_grad());
                let gradients = (model.first.val() + model.alias.val() + model.frozen_alias.val())
                    .sum()
                    .backward();
                assert_eq!(
                    model
                        .first
                        .val()
                        .grad(&gradients)
                        .unwrap()
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    alloc::vec![2.; 2]
                );
                assert!(model.frozen_alias.val().grad(&gradients).is_none());
            }
        }
    }
}
