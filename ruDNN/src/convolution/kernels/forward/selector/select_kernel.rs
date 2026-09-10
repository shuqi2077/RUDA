use ruda_kernel::dsl as kernel_dsl;
use crate::convolution::{
    components::global::args::RuntimeArgs,
    forward::args::{ConcreteArgs, ConcreteInputsFactory, ConcreteOutputFactory},
};
use ruda_kernel::dsl::prelude::TensorBinding;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::client::ComputeClient;
use rublas::kernel_ir::{
    definition::{MatmulElems, MatmulVectorSizes},
    routines::Routine,
};
use rublas::kernel_ir::{
    launch::{InputArg, OutputArg},
    routines::BlueprintStrategy,
};
use ruda_kernel::tiling::InputBinding;

use crate::convolution::components::{ConvSetupError, ConvolutionProblem};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<R: Runtime, Args: ConcreteArgs<A>, A: Routine<RuntimeArgs>>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    out: TensorBinding<R>,
    problem: ConvolutionProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<Args::Config, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    let mut view_vector_sizes = vector_sizes;

    if let InputBinding::Quantized { scheme, .. } = input {
        view_vector_sizes.lhs *= scheme.num_quants();
    }
    if let InputBinding::Quantized { scheme, .. } = weight {
        view_vector_sizes.rhs *= scheme.num_quants();
    }

    let device_settings = A::device_settings(client, view_vector_sizes);
    let expand_info = A::expand_blueprint(
        &problem.as_matmul_problem(),
        &device_settings,
        blueprint_strategy,
    )?;

    let problem = Args::adjust_problem(client, problem, &expand_info.blueprint, dtypes);
    let launch_info = A::prepare(&problem.as_matmul_problem(), &device_settings, expand_info)?;

    let (input, runtime_args) = <InputArg<Args> as ConcreteInputsFactory<A>>::create(
        input,
        weight,
        bias,
        &launch_info.blueprint,
        &problem,
        &launch_info.dtypes,
    );
    let output = <OutputArg<Args> as ConcreteOutputFactory<A>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &launch_info.dtypes,
    );

    rublas::kernel_ir::launch::launch_kernel::<Args, R, A>(
        client,
        input,
        output,
        runtime_args,
        launch_info,
    )
    .map_err(ConvSetupError::Matmul)
}
