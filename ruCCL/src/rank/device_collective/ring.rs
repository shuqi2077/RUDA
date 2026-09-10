use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn all_reduce_ring(
        &self,
        buffer: &D::Buffer,
        reduction_operation: ReductionOperation,
        function: &D::Kernel,
        ring_channels: usize,
    ) -> Result<CollectiveStats, D::Error> {
        let world_size = self.world_size() as usize;
        let operation = self.internal_sequence.fetch_add(1, Ordering::Relaxed);
        let channels =
            self.effective_ring_channels(ring_channels, self.execution.buffer_len(buffer));
        self.ring_agreement(
            operation,
            self.execution.buffer_len(buffer),
            0x100 + reduction_operation as u32,
            channels,
        )?;
        let channel_ranges = balanced_ranges(self.execution.buffer_len(buffer), channels);
        let channel_results = std::thread::scope(|scope| {
            let handles = channel_ranges
                .into_iter()
                .enumerate()
                .map(|(channel, channel_range)| {
                    let communicator = self.clone();
                    let buffer = buffer.clone();
                    let function = (*function).clone();
                    scope.spawn(move || {
                        let result = communicator.all_reduce_ring_channel(
                            &buffer,
                            channel_range,
                            operation,
                            channel,
                            &function,
                        );
                        communicator.propagate_channel_failure("ring all-reduce", result)
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
                        .ok_or(RankError::Overflow("ring all-reduce transferred bytes"))?,
                    total_launches
                        .checked_add(launches)
                        .ok_or(RankError::Overflow("ring all-reduce kernel launches"))?,
                ))
            },
        )?;
        self.stats(
            CollectiveAlgorithm::Ring,
            (2 * (world_size - 1)) as u32,
            transferred_bytes,
            reduction_kernel_launches,
        )
    }

    fn all_reduce_ring_channel(
        &self,
        buffer: &D::Buffer,
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
        let ranges = balanced_ranges(channel_range.len(), world_size)
            .into_iter()
            .map(|range| range.start + channel_range.start..range.end + channel_range.start)
            .collect::<Vec<_>>();
        let mut transferred_bytes = 0_usize;
        let mut reduction_kernel_launches = 0_u32;
        for step in 0..world_size - 1 {
            let send_chunk = ring[(position + world_size - step) % world_size] as usize;
            let receive_chunk = ring[(position + world_size - step - 1) % world_size] as usize;
            let send_range = ranges[send_chunk].clone();
            let receive_range = ranges[receive_chunk].clone();
            let received = self.exchange_ring_chunk_bytes(
                buffer,
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
                .ok_or(RankError::Overflow("ring all-reduce transferred bytes"))?;
            reduction_kernel_launches = reduction_kernel_launches
                .checked_add(self.reduce_bytes_into(
                    buffer,
                    receive_range.start,
                    receive_range.len(),
                    &received,
                    function,
                )?)
                .ok_or(RankError::Overflow("ring all-reduce kernel launches"))?;
        }

        for step in 0..world_size - 1 {
            let send_chunk = ring[(position + 1 + world_size - step) % world_size] as usize;
            let receive_chunk = ring[(position + world_size - step) % world_size] as usize;
            let send_range = ranges[send_chunk].clone();
            let receive_range = ranges[receive_chunk].clone();
            let received = self.exchange_ring_chunk_bytes(
                buffer,
                send_range.clone(),
                receive_range.len(),
                next_rank,
                previous_rank,
                channel,
                ring_data_tag(operation, 1, channel, step),
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(
                    send_range
                        .len()
                        .saturating_mul(D::ELEMENT_SIZE)
                        .saturating_add(received.len()),
                )
                .ok_or(RankError::Overflow("ring all-reduce transferred bytes"))?;
            if !received.is_empty() {
                RankDevice::<T>::copy_bytes_to_device_at(
                    self.execution,
                    buffer,
                    receive_range.start,
                    &received,
                )?;
            }
        }

        Ok((transferred_bytes, reduction_kernel_launches))
    }

    pub fn exchange_ring_chunk_bytes(
        &self,
        buffer: &D::Buffer,
        send_range: Range<usize>,
        receive_length: usize,
        destination: u32,
        source: u32,
        rail_hint: usize,
        tag: u64,
    ) -> Result<Vec<u8>, D::Error> {
        let payload = RankDevice::<T>::copy_bytes_from_device_at(
            self.execution,
            buffer,
            send_range.start,
            send_range.len(),
        )?;
        let rail = self.collective_rail_for_channel(rail_hint);
        self.session.send_on_rail(
            rail,
            destination,
            tag,
            self.element_type,
            send_range.len() as u64,
            payload,
        )?;
        let response = self.session.receive_on_rail(
            rail,
            Some(source),
            tag,
            self.element_type,
            receive_length as u64,
        )?;
        let receive_bytes = receive_length
            .checked_mul(D::ELEMENT_SIZE)
            .ok_or(RankError::Overflow("ring receive byte length"))?;
        if response.payload.len() != receive_bytes {
            return Err(NetworkError::InvalidConfiguration(format!(
                "ring response has {} bytes, expected {receive_bytes}",
                response.payload.len()
            ))
            .into());
        }
        Ok(response.payload)
    }
}
