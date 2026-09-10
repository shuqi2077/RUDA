use ruda_kernel::dsl as kernel_dsl;
use crate::reduce::{
    ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        global::idle_check,
        instructions::{Accumulator, reduce_inplace},
        readers::{Reader, plane::PlaneReader},
        writers::Writer,
    },
    routines::{PlaneMergeStrategy, PlaneReduceBlueprint},
};

use crate::reduce::components::instructions::ReduceStep;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::r#virtual::VirtualTensor;

#[derive(RudaType)]
pub struct GlobalFullPlaneReduce;

#[ruda]
impl GlobalFullPlaneReduce {
    pub fn execute<P: ReducePrecision, Out: NumericVector, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        inst: &I,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: PlaneReduceBlueprint,
    ) {
        let acc_format = I::accumulator_format(inst);
        let planes_per_ruda = RUDA_DIM_Y as usize * RUDA_DIM_Z as usize;
        let plane_index = PLANE_POS as usize;
        let assigned = plane_index < planes_per_ruda;
        let leader = plane_exclusive_sum(1u32) == 0;
        let reduction_index = RUDA_POS * planes_per_ruda + plane_index;
        let write_index = reduction_index;

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
            blueprint.plane_idle,
        );
        #[comptime]
        let idle = match idle {
            ComptimeOption::Some(idle) => idle || !assigned,
            ComptimeOption::None => !assigned,
        };
        let idle = ComptimeOption::new_Some(idle);

        for b in 0..write_count {
            let reduce_index = reduce_index_start + b;
            let result = Self::reduce_single::<P, Out, I>(
                input,
                output,
                reduce_axis,
                reduce_index,
                inst,
                idle,
                vectorization_mode,
                blueprint,
            );

            if leader && assigned {
                writer.write::<P, I>(b, result, inst);
            }
        }

        let commit_required = writer.commit_required();

        #[allow(clippy::collapsible_if)]
        if commit_required {
            if leader && assigned {
                writer.commit();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_single<P: ReducePrecision, Out: NumericVector, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        reduce_index: usize,
        inst: &I,
        idle: ComptimeOption<bool>,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: PlaneReduceBlueprint,
    ) -> Accumulator<P> {
        let reader = Reader::<P>::new::<I, Out>(
            input,
            output,
            inst,
            reduce_axis,
            reduce_index,
            idle,
            blueprint.bound_checks,
            vectorization_mode,
            blueprint.plane_dim_ceil,
        );
        let reader = PlaneReader::<P>::new(reader);

        let mut accumulator = I::null_accumulator(inst);

        let iteration_plane_reduce_mode = match blueprint.plane_merge_strategy {
            PlaneMergeStrategy::Eager => ReduceStep::Plane,
            PlaneMergeStrategy::Lazy => ReduceStep::Identity,
        };
        for i in 0..reader.length() {
            let item = reader.read(i);
            reduce_inplace::<P, I>(inst, &mut accumulator, item, iteration_plane_reduce_mode);
        }

        match blueprint.plane_merge_strategy {
            PlaneMergeStrategy::Lazy => {
                I::plane_reduce_inplace(inst, &mut accumulator);
                accumulator
            }
            PlaneMergeStrategy::Eager => accumulator,
        }
    }
}
