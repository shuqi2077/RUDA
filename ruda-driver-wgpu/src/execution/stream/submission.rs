#[cfg(not(target_family = "wasm"))]
mod __submission_load {
    #[derive(Default, Debug)]
    pub enum SubmissionLoad {
        Init {
            last_index: wgpu::SubmissionIndex,
            tasks_count_submitted: usize,
        },
        #[default]
        Empty,
    }

    impl SubmissionLoad {
        pub fn regulate(
            &mut self,
            device: &wgpu::Device,
            tasks_count: usize,
            mut index: wgpu::SubmissionIndex,
        ) {
            match self {
                SubmissionLoad::Init {
                    last_index,
                    tasks_count_submitted,
                } => {
                    *tasks_count_submitted += tasks_count;

                    // Enough to keep the GPU busy.
                    //
                    // - Too much can hang the GPU and create slowdown.
                    // - Too little and GPU utilization is really bad.
                    //
                    // TODO: Could be smarter and dynamic based on stats.
                    const MAX_TOTAL_TASKS: usize = 512;

                    if *tasks_count_submitted >= MAX_TOTAL_TASKS {
                        core::mem::swap(last_index, &mut index);
                        if let Err(e) = device.poll(wgpu::PollType::Wait {
                            submission_index: Some(index),
                            timeout: None,
                        }) {
                            log::warn!(
                                "wgpu: requested wait timed out before the submission was completed during sync. ({e})"
                            )
                        }
                        *tasks_count_submitted = 0;
                    }
                }
                SubmissionLoad::Empty => {
                    *self = Self::Init {
                        last_index: index,
                        tasks_count_submitted: 0,
                    }
                }
            }
        }
    }
}

#[cfg(target_family = "wasm")]
mod __submission_load_wasm {
    #[derive(Default, Debug)]
    pub struct SubmissionLoad;

    impl SubmissionLoad {
        pub fn regulate(
            &mut self,
            _device: &wgpu::Device,
            _tasks_count: usize,
            _index: wgpu::SubmissionIndex,
        ) {
            // Nothing to do.
        }
    }
}

#[cfg(not(target_family = "wasm"))]
pub(super) use __submission_load::*;
#[cfg(target_family = "wasm")]
pub(super) use __submission_load_wasm::*;
