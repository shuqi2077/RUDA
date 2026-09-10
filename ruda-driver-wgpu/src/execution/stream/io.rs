use super::*;

impl WgpuStream {
    /// Read multiple buffers lazily to [Bytes], potentially using pinned memory.
    ///
    /// # Arguments
    ///
    /// * `self` - The current stream.
    /// * `descriptors` - A vector of copy descriptors specifying the source data.
    ///
    /// # Returns
    ///
    /// A [Result] containing a vector of [Bytes] with the copied data, or an [`IoError`] if any copy fails.
    pub fn read_resources(
        &mut self,
        descriptors: Vec<(WgpuResource, Shape, usize)>,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        self.compute_pass = None;
        let mut staging_info = Vec::with_capacity(descriptors.len());
        let mut callbacks = Vec::with_capacity(descriptors.len());

        for (resource, shape, elem_size) in descriptors {
            let size = shape.iter().product::<usize>() * elem_size;

            // Zero-sized resources don't need a GPU copy.
            if resource.size == 0 {
                staging_info.push(None);
                continue;
            }

            // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
            // memory is 32 bytes aligned (see WgpuStorage).
            let align = wgpu::COPY_BUFFER_ALIGNMENT;
            let aligned_len = resource.size.div_ceil(align) * align;
            let (staging, binding) = self.mem_manage.reserve_staging(aligned_len).unwrap();

            self.tasks_count += 1;
            self.encoder.copy_buffer_to_buffer(
                &resource.buffer,
                resource.offset,
                &staging.buffer,
                0,
                aligned_len,
            );
            staging_info.push(Some((staging, binding, size)));
        }

        // Flush all commands to the queue, so GPU gets started on copying to the staging buffer.
        let _ = self
            .flush(StreamErrorMode {
                ignore: true,
                flush: false,
            })
            .ok();

        for entry in staging_info.iter() {
            if let Some((staging, _binding, _size)) = entry {
                let (sender, receiver) = async_channel::bounded(1);
                staging
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |v| {
                        // This might fail if the channel is closed (eg. the future is dropped).
                        // This is fine, just means results aren't needed anymore.
                        let _ = sender.try_send(v);
                    });

                callbacks.push(Some(receiver));
            } else {
                callbacks.push(None);
            }
        }

        let poll = self.poll.start_polling();

        Box::pin(async move {
            for receiver in callbacks.iter().flatten() {
                receiver
                    .recv()
                    .await
                    .expect("Unable to receive buffer slice result.")
                    .expect("Failed to map buffer");
            }

            // Can stop polling now.
            core::mem::drop(poll);

            let result = {
                staging_info
                    .into_iter()
                    .map(|entry| {
                        if let Some((staging, binding, size)) = entry {
                            let controller =
                                Box::new(WgpuAllocController::init(binding, staging.buffer));
                            // SAFETY: The binding has initialized memory for at least `size` bytes.
                            unsafe { Bytes::from_controller(controller, size) }
                        } else {
                            Bytes::from_bytes_vec(vec![])
                        }
                    })
                    .collect()
            };

            Ok(result)
        })
    }

    pub(super) fn write_to_buffer(&mut self, resource: &WgpuResource, data: &[u8]) {
        // Nothing to write for zero-sized resources.
        if resource.size == 0 {
            return;
        }

        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is also aligned (see WgpuStorage). Per the WebGPU spec, this
        // just has to be a multiple of 4: https://www.w3.org/TR/webgpu/#dom-gpuqueue-writebuffer
        let copy_align = wgpu::COPY_BUFFER_ALIGNMENT;
        let size = resource.size.next_multiple_of(copy_align);

        if size == data.len() as u64 {
            // write_buffer is the recommended way to write this data, as:
            // - On WebGPU, from WASM, this can save a copy to the JS memory.
            // - On devices with unified memory, this could skip the staging buffer entirely.
            self.queue
                .write_buffer(&resource.buffer, resource.offset, data);
        } else {
            // For sizes not aligned we need to only write a part of the staging buffer, do this
            // with `write_buffer_with`.
            let mut buffer = self
                .queue
                .write_buffer_with(
                    &resource.buffer,
                    resource.offset,
                    NonZero::new(size).unwrap(),
                )
                .expect("Internal error: Failed to call `write_buffer_with`, this likely means no staging buffer could be allocated.");
            buffer.slice(0..data.len()).copy_from_slice(data);
        }

        self.pending_write_count += 1;

        // Prevent wgpu staging buffer pool exhaustion during bulk writes (e.g. model
        // loading with hundreds of tensors). queue.write_buffer() is async — wgpu
        // copies data into an internal staging buffer, then transfers to GPU on the
        // next queue.submit(). Without periodic submits, hundreds of writes accumulate
        // and staging buffers get recycled before the GPU copy completes, silently
        // corrupting early tensors.
        // See: https://github.com/shuqi2077/RUDA/blob/main/THIRD_PARTY_NOTICES.md
        const MAX_PENDING_WRITES: usize = 64;

        if self.pending_write_count >= MAX_PENDING_WRITES {
            // Submit a fresh, empty command buffer to flush all pending write_buffer work.
            // wgpu flushes its internal staging-buffer copies on any queue.submit(),
            // so we don't need to touch the main compute encoder here.
            let write_flush_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Ruda Write Flush Encoder"),
                    });
            let index = self.queue.submit([write_flush_encoder.finish()]);

            // Wait for the GPU to finish processing these writes before continuing.
            #[cfg(not(target_family = "wasm"))]
            if let Err(e) = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            }) {
                log::warn!("wgpu: write flush poll failed ({e})");
            }

            self.pending_write_count = 0;
        }
    }
}
