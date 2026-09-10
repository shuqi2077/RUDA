use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn reduce_scatter_ring(
        &self,
        input: &D::Buffer,
        output_length: usize,
        reduction_operation: ReductionOperation,
        function: &D::Kernel,
        ring_channels: usize,
    ) -> Result<(D::Buffer, CollectiveStats), D::Error> {
        let world_size = self.world_size() as usize;
        let working = RankDevice::<T>::alloc(self.execution, self.execution.buffer_len(input))?;
        RankDevice::<T>::copy_to_device(
            self.execution,
            &working,
            &RankDevice::<T>::copy_from_device(self.execution, input)?,
        )?;
        let output = RankDevice::<T>::alloc(self.execution, output_length)?;
        let operation = self.internal_sequence.fetch_add(1, Ordering::Relaxed);
        let channels = self.effective_ring_channels(ring_channels, output_length);
        self.ring_agreement(
            operation,
            self.execution.buffer_len(input),
            0x300 + reduction_operation as u32,
            channels,
        )?;
        let channel_ranges = balanced_ranges(output_length, channels);
        let channel_results = std::thread::scope(|scope| {
            let handles = channel_ranges
                .into_iter()
                .enumerate()
                .map(|(channel, channel_range)| {
                    let communicator = self.clone();
                    let working = working.clone();
                    let output = output.clone();
                    let function = (*function).clone();
                    scope.spawn(move || {
                        let result = communicator.reduce_scatter_ring_channel(
                            &working,
                            &output,
                            output_length,
                            channel_range,
                            operation,
                            channel,
                            &function,
                        );
                        communicator.propagate_channel_failure("ring reduce-scatter", result)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| WorkError::WorkerPanic)?)
                .collect::<Result<Vec<_>, D::Error>>()
        })?;
        let (transferred_bytes, reduction_kernel_launches) = channel_results.into_iter().try_fold(
            (0_usize, 0_u32),
            |(total_bytes, total_launches), (bytes, launches)| {
                Ok::<(usize, u32), D::Error>((
                    total_bytes
                        .checked_add(bytes)
                        .ok_or(RankError::Overflow("ring reduce-scatter transferred bytes"))?,
                    total_launches
                        .checked_add(launches)
                        .ok_or(RankError::Overflow("ring reduce-scatter kernel launches"))?,
                ))
            },
        )?;
        let stats = self.stats(
            CollectiveAlgorithm::Ring,
            (world_size - 1) as u32,
            transferred_bytes,
            reduction_kernel_launches,
        )?;
        Ok((output, stats))
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_scatter_ring_channel(
        &self,
        working: &D::Buffer,
        output: &D::Buffer,
        output_length: usize,
        channel_range: Range<usize>,
        operation: u64,
        channel: usize,
        function: &D::Kernel,
    ) -> Result<(usize, u32), D::Error> {
        let world_size = self.world_size() as usize;
        let ring = self.tuning.ring_order_for_channel(channel);
        let position = self.ring_position(ring)?;
        let previous_rank = ring[(position + world_size - 1) % world_size];
        let next_rank = ring[(position + 1) % world_size];
        let ranges = (0..world_size)
            .map(|rank| {
                let rank_start = rank * output_length;
                rank_start + channel_range.start..rank_start + channel_range.end
            })
            .collect::<Vec<_>>();
        let mut transferred_bytes = 0_usize;
        let mut reduction_kernel_launches = 0_u32;
        for step in 0..world_size - 1 {
            let send_chunk = ring[(position + world_size - 1 - step) % world_size] as usize;
            let receive_chunk = ring[(position + 2 * world_size - 2 - step) % world_size] as usize;
            let send_range = ranges[send_chunk].clone();
            let receive_range = ranges[receive_chunk].clone();
            let received = self.exchange_ring_chunk_bytes(
                working,
                send_range.clone(),
                receive_range.len(),
                next_rank,
                previous_rank,
                channel,
                ring_data_tag(operation, 0, channel, step),
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(
                    send_range
                        .len()
                        .saturating_mul(D::ELEMENT_SIZE)
                        .saturating_add(received.len()),
                )
                .ok_or(RankError::Overflow("ring reduce-scatter transferred bytes"))?;
            reduction_kernel_launches = reduction_kernel_launches
                .checked_add(self.reduce_bytes_into(
                    working,
                    receive_range.start,
                    receive_range.len(),
                    &received,
                    function,
                )?)
                .ok_or(RankError::Overflow("ring reduce-scatter kernel launches"))?;
        }
        let output_values = RankDevice::<T>::copy_bytes_from_device_at(
            self.execution,
            working,
            self.rank() as usize * output_length + channel_range.start,
            channel_range.len(),
        )?;
        RankDevice::<T>::copy_bytes_to_device_at(
            self.execution,
            output,
            channel_range.start,
            &output_values,
        )?;
        Ok((transferred_bytes, reduction_kernel_launches))
    }
}
