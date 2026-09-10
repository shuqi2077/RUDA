use ruda_kernel::dsl as kernel_dsl;
use super::*;
use ruda_kernel::library::tensor::layout::linear::LinearView;

#[ruda(launch, address_type = "dynamic")]
pub(super) fn prng_kernel<F: RandomFamily, E: Numeric, N: Size>(
    output: &mut LinearView<Vector<E, N>, ReadWrite>,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    args: Args<F>,
    #[comptime] n_values_per_thread: usize,
    #[define(E)] _dtype: StorageType,
) {
    let ruda_offset = RUDA_POS * RUDA_DIM as usize;

    let write_index_base = ruda_offset * n_values_per_thread / N::value() + UNIT_POS as usize;

    // Truncating position should be fine here, it's no issue if the seed repeats
    #[allow(arithmetic_overflow)]
    let thread_seed = 1000000007u32 * ABSOLUTE_POS as u32;

    let mut state_0 = thread_seed + seed_0;
    let mut state_1 = thread_seed + seed_1;
    let mut state_2 = thread_seed + seed_2;
    let mut state_3 = thread_seed + seed_3;

    // Creation of n_values_per_thread values, specific to the distribution
    F::Runtime::inner_loop(
        args,
        write_index_base,
        RUDA_DIM,
        n_values_per_thread,
        &mut state_0,
        &mut state_1,
        &mut state_2,
        &mut state_3,
        output,
    );
}
