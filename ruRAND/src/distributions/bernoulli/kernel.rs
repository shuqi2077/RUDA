use super::*;

#[ruda]
impl PrngRuntime for Bernoulli {
    fn inner_loop<E: Numeric, N: Size>(
        args: Bernoulli,
        write_index_base: usize,
        n_invocations: u32,
        #[comptime] n_values_per_thread: usize,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut View<Vector<E, N>, usize, ReadWrite>,
    ) {
        let prob = args.probability;

        let mut output_vector = Vector::empty();

        let num_iterations = n_values_per_thread / N::value();
        #[unroll(num_iterations <=8)]
        for vector_index in 0..num_iterations {
            // vectorization
            #[unroll]
            for i in 0..N::value() {
                *state_0 = taus_step_0(*state_0);
                *state_1 = taus_step_1(*state_1);
                *state_2 = taus_step_2(*state_2);
                *state_3 = lcg_step(*state_3);

                let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
                let float_random = to_unit_interval_closed_open(int_random);
                output_vector[i] = E::cast_from(float_random < prob);
            }
            let write_index = vector_index * n_invocations as usize + write_index_base;

            output.write_checked(write_index, output_vector);
        }
    }
}
