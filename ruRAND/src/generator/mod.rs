use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords1d;
use ruda_kernel::library::tensor::layout::linear::linear_view;

pub(crate) const N_VALUES_PER_THREAD: usize = 128;

use crate::state::get_seeds;

/// Pseudo-random generator
pub(crate) fn random<F: RandomFamily, R: Runtime>(
    client: &ComputeClient<R>,
    prng: F::Runtime,
    output: TensorBinding<R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let seeds = get_seeds();
    let args = prng.args();

    let ruda_dim = RudaDim::new(client.properties(), output.size().div_ceil(N_VALUES_PER_THREAD));
    let ruda_count = prng_ruda_count(output.size(), ruda_dim, N_VALUES_PER_THREAD);

    let output_vector_size = 1;
    // TODO: Higher vectorization can add some correlation locally.
    //
    // let output_line_size = tensor_vector_size_parallel(
    //     R::line_size_elem(&E::as_elem_native_unchecked()),
    //     output.shape,
    //     output.strides,
    //     output.strides.len() - 1,
    // );

    let address_type = output.required_address_type(dtype.size());
    let output = linear_view(output);

    prng_kernel::launch::<F, R>(
        client,
        ruda_count,
        ruda_dim,
        address_type,
        output_vector_size,
        output,
        seeds[0],
        seeds[1],
        seeds[2],
        seeds[3],
        args,
        N_VALUES_PER_THREAD,
        dtype,
    );

    Ok(())
}

fn prng_ruda_count(num_elems: usize, ruda_dim: RudaDim, n_values_per_thread: usize) -> RudaCount {
    let num_threads = f32::ceil(num_elems as f32 / n_values_per_thread as f32);
    let num_invocations = f32::ceil(num_threads / ruda_dim.num_elems() as f32);
    let rudas_x = f32::ceil(f32::sqrt(num_invocations));
    let rudas_y = f32::ceil(num_invocations / rudas_x);

    RudaCount::Static(rudas_x as u32, rudas_y as u32, 1)
}

pub(crate) trait PrngArgs: Send + Sync + 'static {
    type Args: LaunchArg;

    fn args<R: Runtime>(self) -> <Self::Args as LaunchArg>::RuntimeArg<R>;
}

pub(crate) trait RandomFamily: Send + Sync + 'static + std::fmt::Debug {
    type Runtime: PrngRuntime;
}

#[ruda]
pub(crate) trait PrngRuntime: Send + Sync + 'static + PrngArgs {
    #[allow(clippy::too_many_arguments)]
    fn inner_loop<E: Numeric, N: Size>(
        args: Self::Args,
        write_index_base: usize,
        n_invocations: u32,
        #[comptime] n_values_per_thread: usize,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut View<Vector<E, N>, Coords1d, ReadWrite>,
    );
}

type Args<F> = <<F as RandomFamily>::Runtime as PrngArgs>::Args;

mod kernel;
use kernel::prng_kernel;
