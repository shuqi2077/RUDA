use ruda::runtime::{
    client::ComputeClient,
    server::{RudaCount, Handle, KernelArguments},
};
use derive_new::new;

use crate::dummy::{DummyRuntime, KernelTask};

#[derive(new, Clone)]
/// Extended kernel that accounts for additional parameters, i.e. needed
/// information that does not count as an input/output.
pub struct OneKernelAutotuneOperation {
    kernel: KernelTask,
    client: ComputeClient<DummyRuntime>,
}

impl OneKernelAutotuneOperation {
    pub fn run(&self, inputs: Vec<Handle>) -> Result<(), String> {
        self.client.launch(
            Box::new(self.kernel.clone()),
            RudaCount::Static(1, 1, 1),
            KernelArguments::new().with_buffers(inputs.into_iter().map(|h| h.binding()).collect()),
        );
        Ok(())
    }
}
