use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn all_gather_ring(
        &self,
        input: &D::Buffer,
        ring_channels: usize,
    ) -> Result<(D::Buffer, CollectiveStats), D::Error> {
        let world_size = self.world_size() as usize;
        let output_length = self
            .execution
            .buffer_len(input)
            .checked_mul(world_size)
            .ok_or(RankError::Overflow("ring all-gather output length"))?;
        let output = RankDevice::<T>::alloc(self.execution, output_length)?;
        let local = RankDevice::<T>::copy_bytes_from_device_at(
            self.execution,
            input,
            0,
            self.execution.buffer_len(input),
        )?;
        RankDevice::<T>::copy_bytes_to_device_at(
            self.execution,
            &output,
            self.rank() as usize * self.execution.buffer_len(input),
            &local,
        )?;
        let operation = self.internal_sequence.fetch_add(1, Ordering::Relaxed);
        let channels =
            self.effective_ring_channels(ring_channels, self.execution.buffer_len(input));
        self.ring_agreement(operation, self.execution.buffer_len(input), 0x200, channels)?;
        let channel_ranges = balanced_ranges(self.execution.buffer_len(input), channels);
        let transferred = std::thread::scope(|scope| {
            let handles = channel_ranges
                .into_iter()
                .enumerate()
                .map(|(channel, channel_range)| {
                    let communicator = self.clone();
                    let output = output.clone();
                    scope.spawn(move || {
                        let result = communicator.all_gather_ring_channel(
                            &output,
                            self.execution.buffer_len(input),
                            channel_range,
                            operation,
                            channel,
                        );
                        communicator.propagate_channel_failure("ring all-gather", result)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| WorkError::WorkerPanic)?)
                .collect::<Result<Vec<_>, D::Error>>()
        })?;
        let transferred_bytes = transferred.into_iter().try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(RankError::Overflow("ring all-gather transferred bytes"))
        })?;
        let stats = self.stats(
            CollectiveAlgorithm::Ring,
            (world_size - 1) as u32,
            transferred_bytes,
            0,
        )?;
        Ok((output, stats))
    }

    fn all_gather_ring_channel(
        &self,
        output: &D::Buffer,
        input_length: usize,
        channel_range: Range<usize>,
        operation: u64,
        channel: usize,
    ) -> Result<usize, D::Error> {
        let world_size = self.world_size() as usize;
        let ring = self.tuning.ring_order_for_channel(channel);
        let position = self.ring_position(ring)?;
        let previous_rank = ring[(position + world_size - 1) % world_size];
        let next_rank = ring[(position + 1) % world_size];
        let mut transferred_bytes = 0_usize;
        for step in 0..world_size - 1 {
            let send_chunk = ring[(position + world_size - step) % world_size] as usize;
            let receive_chunk = ring[(position + world_size - step - 1) % world_size] as usize;
            let send_start = send_chunk
                .checked_mul(input_length)
                .and_then(|start| start.checked_add(channel_range.start))
                .ok_or(RankError::Overflow("ring all-gather channel send offset"))?;
            let received = self.exchange_ring_chunk_bytes(
                output,
                send_start..send_start + channel_range.len(),
                channel_range.len(),
                next_rank,
                previous_rank,
                channel,
                ring_data_tag(operation, 0, channel, step),
            )?;
            let receive_start = receive_chunk
                .checked_mul(input_length)
                .and_then(|start| start.checked_add(channel_range.start))
                .ok_or(RankError::Overflow(
                    "ring all-gather channel receive offset",
                ))?;
            RankDevice::<T>::copy_bytes_to_device_at(
                self.execution,
                output,
                receive_start,
                &received,
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(received.len().saturating_mul(2))
                .ok_or(RankError::Overflow("ring all-gather transferred bytes"))?;
        }
        Ok(transferred_bytes)
    }
}
