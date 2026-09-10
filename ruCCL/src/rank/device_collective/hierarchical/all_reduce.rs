use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn all_reduce_hierarchical(
        &self,
        buffer: &D::Buffer,
        reduction_operation: ReductionOperation,
        function: &D::Kernel,
        ring_channels: usize,
    ) -> Result<CollectiveStats, D::Error> {
        let (groups, group_index, _) = self.hierarchy_membership()?;
        let group = &groups[group_index];
        let leader = group[0];
        let leaders = groups.iter().map(|group| group[0]).collect::<Vec<_>>();
        let operation = self.internal_sequence.fetch_add(1, Ordering::Relaxed);
        self.hierarchy_agreement(
            operation,
            self.execution.buffer_len(buffer),
            0x600 + reduction_operation as u32,
            ring_channels,
        )?;
        let steps = hierarchy_steps(groups, 2 * leaders.len().saturating_sub(1));

        if self.rank() != leader {
            let values = RankDevice::<T>::copy_from_device(self.execution, buffer)?;
            let payload = D::encode(&values);
            let mut transferred_bytes = payload.len();
            self.session.send(
                leader,
                hierarchy_data_tag(operation, 0, self.rank()),
                self.element_type,
                self.execution.buffer_len(buffer) as u64,
                payload,
            )?;
            let reduced = self.receive_values(
                leader,
                hierarchy_data_tag(operation, 3, self.rank()),
                self.execution.buffer_len(buffer),
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(reduced.len().saturating_mul(D::ELEMENT_SIZE))
                .ok_or(RankError::Overflow(
                    "hierarchical all-reduce transferred bytes",
                ))?;
            RankDevice::<T>::copy_to_device(self.execution, buffer, &reduced)?;
            return self.stats(
                CollectiveAlgorithm::Hierarchical,
                steps,
                transferred_bytes,
                0,
            );
        }

        let mut transferred_bytes = 0_usize;
        let mut reduction_kernel_launches = 0_u32;
        for member in &group[1..] {
            let values = self.receive_values(
                *member,
                hierarchy_data_tag(operation, 0, *member),
                self.execution.buffer_len(buffer),
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(values.len().saturating_mul(D::ELEMENT_SIZE))
                .ok_or(RankError::Overflow(
                    "hierarchical all-reduce transferred bytes",
                ))?;
            reduction_kernel_launches = reduction_kernel_launches
                .checked_add(self.reduce_values_into(buffer, 0, &values, function)?)
                .ok_or(RankError::Overflow(
                    "hierarchical all-reduce kernel launches",
                ))?;
        }

        let (leader_bytes, leader_launches) = self.leader_all_reduce_ring(
            buffer,
            &leaders,
            group_index,
            operation,
            function,
            ring_channels,
        )?;
        transferred_bytes =
            transferred_bytes
                .checked_add(leader_bytes)
                .ok_or(RankError::Overflow(
                    "hierarchical all-reduce transferred bytes",
                ))?;
        reduction_kernel_launches = reduction_kernel_launches
            .checked_add(leader_launches)
            .ok_or(RankError::Overflow(
                "hierarchical all-reduce kernel launches",
            ))?;

        let reduced = RankDevice::<T>::copy_from_device(self.execution, buffer)?;
        let reduced_payload = D::encode(&reduced);
        for member in &group[1..] {
            self.session.send(
                *member,
                hierarchy_data_tag(operation, 3, *member),
                self.element_type,
                self.execution.buffer_len(buffer) as u64,
                reduced_payload.clone(),
            )?;
            transferred_bytes =
                transferred_bytes
                    .checked_add(reduced_payload.len())
                    .ok_or(RankError::Overflow(
                        "hierarchical all-reduce transferred bytes",
                    ))?;
        }
        self.stats(
            CollectiveAlgorithm::Hierarchical,
            steps,
            transferred_bytes,
            reduction_kernel_launches,
        )
    }

    fn leader_all_reduce_ring(
        &self,
        buffer: &D::Buffer,
        leaders: &[u32],
        position: usize,
        operation: u64,
        function: &D::Kernel,
        ring_channels: usize,
    ) -> Result<(usize, u32), D::Error> {
        let channels =
            self.effective_ring_channels(ring_channels, self.execution.buffer_len(buffer));
        let channel_ranges = balanced_ranges(self.execution.buffer_len(buffer), channels);
        let channel_results = std::thread::scope(|scope| {
            let handles = channel_ranges
                .into_iter()
                .enumerate()
                .map(|(channel, channel_range)| {
                    let communicator = self.clone();
                    let buffer = buffer.clone();
                    let leaders = leaders.to_vec();
                    let function = (*function).clone();
                    scope.spawn(move || {
                        let result = communicator.leader_all_reduce_ring_channel(
                            &buffer,
                            &leaders,
                            position,
                            channel_range,
                            operation,
                            channel,
                            &function,
                        );
                        communicator.propagate_channel_failure("hierarchical all-reduce", result)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| WorkError::WorkerPanic)?)
                .collect::<Result<Vec<_>, D::Error>>()
        })?;
        channel_results.into_iter().try_fold(
            (0_usize, 0_u32),
            |(total_bytes, total_launches), (bytes, launches)| {
                Ok::<(usize, u32), D::Error>((
                    total_bytes.checked_add(bytes).ok_or(RankError::Overflow(
                        "hierarchical leader all-reduce transferred bytes",
                    ))?,
                    total_launches
                        .checked_add(launches)
                        .ok_or(RankError::Overflow(
                            "hierarchical leader all-reduce kernel launches",
                        ))?,
                ))
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn leader_all_reduce_ring_channel(
        &self,
        buffer: &D::Buffer,
        leaders: &[u32],
        position: usize,
        channel_range: Range<usize>,
        operation: u64,
        channel: usize,
        function: &D::Kernel,
    ) -> Result<(usize, u32), D::Error> {
        let leader_count = leaders.len();
        let previous_rank = leaders[(position + leader_count - 1) % leader_count];
        let next_rank = leaders[(position + 1) % leader_count];
        let ranges = balanced_ranges(channel_range.len(), leader_count)
            .into_iter()
            .map(|range| range.start + channel_range.start..range.end + channel_range.start)
            .collect::<Vec<_>>();
        let mut transferred_bytes = 0_usize;
        let mut reduction_kernel_launches = 0_u32;

        for step in 0..leader_count - 1 {
            let send_chunk = (position + leader_count - step) % leader_count;
            let receive_chunk = (position + leader_count - step - 1) % leader_count;
            let send_range = ranges[send_chunk].clone();
            let receive_range = ranges[receive_chunk].clone();
            let received = self.exchange_ring_chunk_bytes(
                buffer,
                send_range.clone(),
                receive_range.len(),
                next_rank,
                previous_rank,
                channel,
                hierarchy_ring_data_tag(operation, 1, channel, step),
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(
                    send_range
                        .len()
                        .saturating_mul(D::ELEMENT_SIZE)
                        .saturating_add(received.len()),
                )
                .ok_or(RankError::Overflow(
                    "hierarchical leader all-reduce transferred bytes",
                ))?;
            reduction_kernel_launches = reduction_kernel_launches
                .checked_add(self.reduce_bytes_into(
                    buffer,
                    receive_range.start,
                    receive_range.len(),
                    &received,
                    function,
                )?)
                .ok_or(RankError::Overflow(
                    "hierarchical leader all-reduce kernel launches",
                ))?;
        }

        for step in 0..leader_count - 1 {
            let send_chunk = (position + 1 + leader_count - step) % leader_count;
            let receive_chunk = (position + leader_count - step) % leader_count;
            let send_range = ranges[send_chunk].clone();
            let receive_range = ranges[receive_chunk].clone();
            let received = self.exchange_ring_chunk_bytes(
                buffer,
                send_range.clone(),
                receive_range.len(),
                next_rank,
                previous_rank,
                channel,
                hierarchy_ring_data_tag(operation, 2, channel, step),
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(
                    send_range
                        .len()
                        .saturating_mul(D::ELEMENT_SIZE)
                        .saturating_add(received.len()),
                )
                .ok_or(RankError::Overflow(
                    "hierarchical leader all-reduce transferred bytes",
                ))?;
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
}
