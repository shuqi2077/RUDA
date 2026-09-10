use ruda_model::module::Module;
use ruda_nn::conv::Conv2d;
use ruda_tensor::api::backend::Backend;

#[derive(Module, Debug)]
#[allow(unused)]
pub struct Net<B: Backend> {
    do_not_exist_in_pt: Conv2d<B>,
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use ruda_nn::conv::Conv2dConfig;
    use ruda_store::{ModuleSnapshot, PytorchStore};

    use super::*;

    impl<B: Backend> Net<B> {
        pub fn init(device: &B::Device) -> Self {
            Self {
                do_not_exist_in_pt: Conv2dConfig::new([2, 2], [2, 2]).init(device),
            }
        }
    }

    #[test]
    #[should_panic(expected = "do_not_exist_in_pt")]
    fn should_fail_if_struct_field_is_missing() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store =
            PytorchStore::from_file("tests/missing_module_field/missing_module_field.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");
    }
}
