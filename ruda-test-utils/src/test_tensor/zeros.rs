use ruda_kernel::dsl as kernel_dsl;
use ruda_test_runtime::TestRuntime;
use ruda_kernel::library::tensor::TensorHandle;
use ruda_kernel::dsl::zspace::metadata::Metadata;

use crate::BaseInputSpec;

pub(crate) fn build_zeros(spec: BaseInputSpec) -> TensorHandle<TestRuntime> {
    let mut tensor = TensorHandle::zeros(&spec.client, spec.shape.clone(), spec.dtype);

    // This manipulation is only valid since all the data is the same
    *tensor.metadata = Metadata::new(tensor.shape(), spec.strides());

    tensor
}
