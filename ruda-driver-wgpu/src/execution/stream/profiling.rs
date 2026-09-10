use super::*;

impl WgpuStream {
    // Bit silly but needed to make the borrow checker happy.
    fn system_profiler(&mut self) -> &mut TimestampProfiler {
        let Timings::System(timing) = &mut self.timings else {
            panic!("Unexpected timings type");
        };
        timing
    }

    pub fn start_profile(&mut self) -> Result<ProfilingToken, ServerError> {
        if matches!(self.timings, Timings::System(_)) {
            ruda_core::future::block_on(self.sync())?;
        } else {
            self.flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })?;
        }

        match &mut self.timings {
            Timings::System(_) => {
                let profiler = self.system_profiler();
                Ok(profiler.start())
            }
            Timings::Device(query) => {
                self.compute_pass = None;
                let token = query.start_profile();
                Ok(token)
            }
        }
    }

    pub fn profile_error(&mut self, error: ProfileError) {
        match &mut self.timings {
            Timings::Device(profiler) => {
                profiler.error(error);
            }
            Timings::System(profiler) => {
                profiler.error(error);
            }
        }
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        match &mut self.timings {
            Timings::System(..) => {
                // Nb: WASM _has_ to use device timing and will panic here if query timestamps are not supported.
                let result = future::block_on(self.sync());
                let profiler = self.system_profiler();

                if let Err(err) = result {
                    profiler.error(ProfileError::Server(Box::new(err)));
                }
                profiler.stop(token)
            }
            Timings::Device(..) => {
                let poll = self.poll.start_polling();
                self.compute_pass = None;

                // Submit commands needed for profiling.
                let buffer = {
                    let Timings::Device(timing) = &mut self.timings else {
                        return Err(ProfileError::Unknown {
                            reason: "Unexpected timings type".to_string(),
                            backtrace: BackTrace::capture(),
                        });
                    };
                    timing.stop_profile_setup(token, &self.device, &mut self.encoder)?
                };

                // This flushes the queue to execute the encoder write command to write the
                // timings.
                self.tasks_count += 1;
                let result = self.flush(StreamErrorMode {
                    ignore: false,
                    flush: true,
                });

                let Timings::Device(timing) = &mut self.timings else {
                    return Err(ProfileError::Unknown {
                        reason: "Unexpected timings type".to_string(),
                        backtrace: BackTrace::capture(),
                    });
                };

                match result {
                    Ok(_) => timing.stop_profile(buffer, poll),
                    Err(err) => {
                        // Just to clean the timing buffer.
                        let _ = timing.stop_profile(buffer, poll).ok();
                        Err(ProfileError::Server(Box::new(err)))
                    }
                }
            }
        }
    }
}
