use ruda_kernel::dsl as kernel_dsl;
use crate::reduce::{
    ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        global::idle_check,
        instructions::{
            Accumulator, ReduceStep, SharedAccumulator, fuse_accumulator_inplace, reduce_inplace,
        },
        readers::{Reader, ruda::RudaReader},
        writers::Writer,
    },
    routines::RudaBlueprint,
};
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::r#virtual::VirtualTensor;

#[derive(RudaType)]
pub struct GlobalFullRudaReduce;

#[ruda]
impl GlobalFullRudaReduce {
    pub fn execute<P: ReducePrecision, Out: NumericVector, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        inst: &I,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: RudaBlueprint,
    ) {
        let acc_format = I::accumulator_format(inst);
        let write_index = RUDA_POS;

        let accumulator_size = blueprint.num_shared_accumulators;
        let worker_pos = Self::worker_pos(blueprint);

        let mut writer = Writer::<Out>::new::<P>(
            input,
            output,
            reduce_axis,
            out_vec_axis,
            write_index,
            vectorization_mode,
            acc_format,
        );

        let write_count = writer.write_count();

        let reduce_index_start = write_index * write_count;

        let idle = idle_check::<P, Out>(
            input,
            output,
            reduce_axis,
            reduce_index_start,
            vectorization_mode,
            blueprint.ruda_idle,
        );

        for b in 0..write_count {
            let reduce_index = reduce_index_start + b;

            let mut accumulator_shared = Self::reduce_shared::<P, Out, I>(
                input,
                output,
                reduce_axis,
                reduce_index,
                inst,
                idle,
                vectorization_mode,
                blueprint,
            );

            let mut accumulator_final = I::null_accumulator(inst);

            match blueprint.use_planes {
                true => {
                    if UNIT_POS == 0 {
                        reduce_scan::<P, I>(
                            inst,
                            &mut accumulator_shared,
                            &mut accumulator_final,
                            PLANE_COUNT as usize,
                        );
                        writer.write::<P, I>(b, accumulator_final, inst);
                    }

                    // Wait for plane 0 to finish reading SM before next iter overwrites it.
                    sync_ruda();
                }
                false => {
                    reduce_tree::<P, I>(
                        inst,
                        &mut accumulator_shared,
                        &mut accumulator_final,
                        worker_pos,
                        accumulator_size,
                    );
                    if worker_pos == 0 {
                        writer.write::<P, I>(b, accumulator_final, inst);
                    }
                }
            };
        }

        let commit_required = writer.commit_required();

        #[allow(clippy::collapsible_if)]
        if commit_required {
            if UNIT_POS == 0 {
                writer.commit();
            }
        }
    }

    fn worker_pos(#[comptime] blueprint: RudaBlueprint) -> usize {
        match blueprint.use_planes {
            true => PLANE_POS as usize,
            false => UNIT_POS as usize,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_shared<P: ReducePrecision, Out: NumericVector, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        reduce_index: usize,
        inst: &I,
        idle: ComptimeOption<bool>,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: RudaBlueprint,
    ) -> I::SharedAccumulator {
        let reader = Reader::<P>::new::<I, Out>(
            input,
            output,
            inst,
            reduce_axis,
            reduce_index,
            idle,
            blueprint.bound_checks,
            vectorization_mode,
            false,
        );
        let reader = RudaReader::<P>::new(reader);
        let mut accumulator = I::null_accumulator(inst);

        for i in 0..reader.length() {
            let item = reader.read(i);
            reduce_inplace::<P, I>(inst, &mut accumulator, item, ReduceStep::Identity);
        }

        let worker_pos = Self::worker_pos(blueprint);

        let accumulator_plane = match blueprint.use_planes {
            true => {
                I::plane_reduce_inplace(inst, &mut accumulator);
                accumulator
            }
            false => accumulator,
        };

        // Sync at the ruda level.
        let accumulator_size = blueprint.num_shared_accumulators;
        let requirements = I::requirements(inst);
        let mut accumulator_shared =
            I::SharedAccumulator::allocate(accumulator_size, requirements.coordinates, inst);

        if blueprint.use_planes {
            if UNIT_POS_PLANE == 0 {
                I::SharedAccumulator::write(&mut accumulator_shared, worker_pos, accumulator_plane);
            }
        } else {
            I::SharedAccumulator::write(&mut accumulator_shared, worker_pos, accumulator_plane);
        }

        sync_ruda();

        accumulator_shared
    }
}

#[ruda]
fn reduce_scan<P: ReducePrecision, I: ReduceInstruction<P>>(
    inst: &I,
    shared_accumulator: &mut I::SharedAccumulator,
    accumulator: &mut Accumulator<P>,
    size: usize,
) {
    for i in 0..size {
        let acc = I::SharedAccumulator::read(shared_accumulator, i);
        I::fuse_accumulators(inst, accumulator, &acc);
    }
}

/// Use all units within a ruda to fuse the first `size` elements of `accumulator` inplace like this with some padding if `size` is not a power of 2.
///
///
/// ```ignored
///
///     0   1   2   3   4   5   6   7
///     |   |   |   |   |   |   |   |
///     +---+   +---+   +---+   +---+
///     |       |       |       |
///     +-------+       +-------+
///     |               |
///     +---------------+
///     |
///     *
///
/// ```
///
/// The outcome is stored in the first element of the accumulator and also returned by this function for convenience.
///
/// Since each individual ruda performs a reduction, this function is meant to be called
/// with a different `accumulator` for each ruda based on `RUDA_POS`.
///
/// There is no out-of-bound check, so it is the responsibility of the caller to ensure that `size` is at most the length
/// of the shared memory and that there are at least `size` units within each ruda.
#[ruda]
fn reduce_tree<P: ReducePrecision, I: ReduceInstruction<P>>(
    inst: &I,
    shared_accumulator: &mut I::SharedAccumulator,
    accumulator: &mut Accumulator<P>,
    worker_index: usize,
    #[comptime] size: usize,
) {
    if size.is_power_of_two() {
        let mut num_active_units = size.runtime();
        let mut jump = 1;
        while num_active_units > 1 {
            num_active_units /= 2;
            let destination = jump * 2 * worker_index;
            let origin = jump * (2 * worker_index + 1);
            if worker_index < num_active_units {
                fuse_accumulator_inplace::<P, I>(inst, shared_accumulator, destination, origin);
            }
            jump *= 2;
            sync_ruda();
        }
    } else {
        let mut num_remaining_items = size.runtime();
        let mut jump = 1;
        while num_remaining_items > 1 {
            let destination = jump * 2 * worker_index;
            let origin = jump * (2 * worker_index + 1);
            if worker_index < num_remaining_items / 2 {
                fuse_accumulator_inplace::<P, I>(inst, shared_accumulator, destination, origin);
            }
            num_remaining_items = num_remaining_items.div_ceil(2);
            jump *= 2;
            sync_ruda();
        }
    }
    sync_ruda();

    let acc = I::SharedAccumulator::read(shared_accumulator, 0);
    I::fuse_accumulators(inst, accumulator, &acc);
}
