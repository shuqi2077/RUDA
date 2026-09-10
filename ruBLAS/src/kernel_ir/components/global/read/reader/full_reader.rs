use ruda_kernel::dsl as kernel_dsl;
use std::marker::PhantomData;

use crate::kernel_ir::{
    components::global::multi_stage::JobExecutor,
    components::global::multi_stage::LoadMaxRoundPlaneCount,
    components::global::read::LoadingJob,
    components::global::read::LoadingValidation,
    components::global::read::StageBuffer,
    components::global::read::SyncStrategy,
    components::global::read::TaskCounter,
    components::global::{multi_stage::JobIterator, read::FullLoaderStage},
    components::stage::TilingLayout,
    components::{global::memory::GlobalIterator, stage::LoadStageFamily},
    {components::global::GlobalReaderConfig, launch::RuntimeConfig},
};
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords2d;
use ruda_kernel::tiling::tile::TileKind;

pub type SyncBarrier<S> = <S as SyncStrategy>::Barrier;

#[ruda]
/// A strategy for synchronously loading a full stage memory.
pub trait FullLoadingStrategy<RC: RuntimeConfig>:
    'static + Send + Sync + Clone + LoadingValidation + LoadMaxRoundPlaneCount
{
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;
    /// The synchronization strategy that should be used with this loading strategy
    type SyncStrategy: SyncStrategy;
    type Stage: LoadStageFamily<ReadOnly>;
    type TileKind: TileKind;

    /// The [LoadingJob] for this strategy.
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>: LoadingJob<EG, NG, ES, NS, Self::TilingLayout, Self::SyncStrategy, Stage = Self::Stage>;

    const SHOULD_CLEAR: bool = false;

    /// Returns the job with preliminary calculations done.
    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        config: RC,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, NG, ES, NS>;
}

#[derive(Clone, RudaType)]
/// Loads the entire stage memory.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct FullStageGlobalReader<
    EG: Numeric,
    NG: Size,
    ES: Numeric,
    NS: Size,
    RC: RuntimeConfig,
    L: FullLoadingStrategy<RC>,
> {
    global_iter: GlobalIterator<Vector<EG, NG>>,
    runtime_config: RC,
    stage: FullLoaderStage<RC, L, ES, NS>,
    loading_job: ComptimeOption<L::Job<EG, NG, ES, NS>>,
    #[ruda(comptime)]
    _phantom: PhantomData<L>,
}

#[ruda]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, RC: RuntimeConfig, L: FullLoadingStrategy<RC>>
    FullStageGlobalReader<EG, NG, ES, NS, RC, L>
{
    /// Create a new SyncFullStageGlobalReader
    pub fn new(
        view: View<Vector<EG, NG>, Coords2d>,
        runtime_config: RC,
        k_step: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self {
        Self::new_with_checked_view(view, runtime_config, k_step, config, false)
    }

    pub fn new_with_checked_view(
        view: View<Vector<EG, NG>, Coords2d>,
        runtime_config: RC,
        k_step: u32,
        #[comptime] config: GlobalReaderConfig,
        #[comptime] checked: bool,
    ) -> Self {
        // Maybe make align a property on the strategy, but it's fine to over-align so this works
        // for now. Swizzling will require more though.
        let stage = L::Stage::create(128usize, config.smem_config);

        let global_iter =
            GlobalIterator::new(view, k_step, config.gmem_config.view_direction, checked);

        let loading_job = match config.precompute_job {
            true => ComptimeOption::new_Some(L::new_job::<EG, NG, ES, NS>(
                runtime_config.clone(),
                config,
            )),
            false => ComptimeOption::new_None(),
        };

        FullStageGlobalReader::<EG, NG, ES, NS, RC, L> {
            global_iter,
            runtime_config,
            stage,
            loading_job,
            _phantom: PhantomData::<L>,
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn stage(&self) -> FullLoaderStage<RC, L, ES, NS> {
        L::Stage::with_buffer_index(&self.stage, 0)
    }

    /// Frees the stage memory for reuse
    pub fn free_stage(self) {
        L::Stage::free(&self.stage);
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(&mut self) {
        self.global_iter.advance();
    }

    pub fn set_first_k_step(&mut self, first: u32) {
        self.global_iter.set_first_step(first);
    }

    /// Accomplish the entire job of loading data into the stage memory
    pub fn load_stage(
        &mut self,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut loading_job = self
            .loading_job
            .clone()
            .unwrap_or_else(|| L::new_job::<EG, NG, ES, NS>(self.runtime_config.clone(), config));

        let len = L::Job::task_count(&loading_job);

        #[unroll]
        for task_id in 0..len {
            L::Job::<EG, NG, ES, NS>::execute_task(
                &mut loading_job,
                task_id,
                &self.global_iter,
                &mut self.stage,
                barrier,
                config,
            );
        }
    }
}

#[ruda]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, RC: RuntimeConfig, L: FullLoadingStrategy<RC>>
    JobExecutor<L::SyncStrategy> for FullStageGlobalReader<EG, NG, ES, NS, RC, L>
{
    type JobIterator = FullStageJobIterator<EG, NG, ES, NS, RC, L>;

    fn create_job_iterator(
        this: &Self,
        #[comptime] _stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::JobIterator {
        let job = this
            .loading_job
            .clone()
            .unwrap_or_else(|| L::new_job::<EG, NG, ES, NS>(this.runtime_config.clone(), config));

        let num_tasks = L::Job::task_count(&job);

        FullStageJobIterator::<EG, NG, ES, NS, RC, L> {
            job,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
        }
    }

    fn execute_task(
        this: &mut Self,
        job_iterator: &mut FullStageJobIterator<EG, NG, ES, NS, RC, L>,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let task_id = job_iterator.current.read().counter.comptime();

        L::Job::<EG, NG, ES, NS>::execute_task(
            &mut job_iterator.job,
            task_id,
            &this.global_iter,
            &mut this.stage,
            barrier,
            config,
        );

        job_iterator.current.store(TaskCounter {
            counter: task_id + 1,
        });
    }

    fn execute_all_remaining_tasks(
        this: &mut Self,
        job_iterator: &mut Self::JobIterator,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let task_counter = job_iterator.current.read().counter;

        #[unroll]
        for task_id in task_counter..job_iterator.num_tasks {
            L::Job::<EG, NG, ES, NS>::execute_task(
                &mut job_iterator.job,
                task_id,
                &this.global_iter,
                &mut this.stage,
                barrier,
                config,
            );
        }

        job_iterator.current.store(TaskCounter {
            counter: job_iterator.num_tasks,
        });
    }

    fn execute_whole_job(
        this: &mut Self,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) {
        Self::execute_all_remaining_tasks(
            this,
            &mut Self::create_job_iterator(this, stage_buffer, config),
            barrier,
            config,
        );
    }
}

#[derive(RudaType)]
/// A comptime iterator over a job for sync full stage reader
pub struct FullStageJobIterator<
    EG: Numeric,
    NG: Size,
    ES: Numeric,
    NS: Size,
    RC: RuntimeConfig,
    L: FullLoadingStrategy<RC>,
> {
    job: L::Job<EG, NG, ES, NS>,
    #[ruda(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}

#[ruda]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, RC: RuntimeConfig, L: FullLoadingStrategy<RC>>
    JobIterator for FullStageJobIterator<EG, NG, ES, NS, RC, L>
{
    fn current(this: &Self) -> comptime_type!(u32) {
        this.current.read().counter
    }

    fn num_tasks(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
