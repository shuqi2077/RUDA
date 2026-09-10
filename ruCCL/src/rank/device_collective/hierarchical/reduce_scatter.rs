use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn reduce_scatter_hierarchical(
        &self,
        input: &D::Buffer,
        output_length: usize,
        reduction_operation: ReductionOperation,
        function: &D::Kernel,
        ring_channels: usize,
    ) -> Result<(D::Buffer, CollectiveStats), D::Error> {
        let (groups, group_index, member_index) = self.hierarchy_membership()?;
        let group = &groups[group_index];
        let leader = group[0];
        let leaders = groups.iter().map(|group| group[0]).collect::<Vec<_>>();
        let operation = self.internal_sequence.fetch_add(1, Ordering::Relaxed);
        self.hierarchy_agreement(
            operation,
            self.execution.buffer_len(input),
            0x800 + reduction_operation as u32,
            ring_channels,
        )?;
        let steps = hierarchy_steps(groups, leaders.len().saturating_sub(1));

        if self.rank() != leader {
            let input_values = RankDevice::<T>::copy_from_device(self.execution, input)?;
            let payload = D::encode(&input_values);
            let mut transferred_bytes = payload.len();
            self.session.send(
                leader,
                hierarchy_data_tag(operation, 7, self.rank()),
                self.element_type,
                self.execution.buffer_len(input) as u64,
                payload,
            )?;
            let values = self.receive_values(
                leader,
                hierarchy_data_tag(operation, 9, self.rank()),
                output_length,
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(values.len().saturating_mul(D::ELEMENT_SIZE))
                .ok_or(RankError::Overflow(
                    "hierarchical reduce-scatter transferred bytes",
                ))?;
            let output = RankDevice::<T>::alloc(self.execution, output_length)?;
            RankDevice::<T>::copy_to_device(self.execution, &output, &values)?;
            return Ok((
                output,
                self.stats(
                    CollectiveAlgorithm::Hierarchical,
                    steps,
                    transferred_bytes,
                    0,
                )?,
            ));
        }

        let working = RankDevice::<T>::alloc(self.execution, self.execution.buffer_len(input))?;
        RankDevice::<T>::copy_to_device(
            self.execution,
            &working,
            &RankDevice::<T>::copy_from_device(self.execution, input)?,
        )?;
        let mut transferred_bytes = 0_usize;
        let mut reduction_kernel_launches = 0_u32;
        for member in &group[1..] {
            let values = self.receive_values(
                *member,
                hierarchy_data_tag(operation, 7, *member),
                self.execution.buffer_len(input),
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(values.len().saturating_mul(D::ELEMENT_SIZE))
                .ok_or(RankError::Overflow(
                    "hierarchical reduce-scatter transferred bytes",
                ))?;
            reduction_kernel_launches = reduction_kernel_launches
                .checked_add(self.reduce_values_into(&working, 0, &values, function)?)
                .ok_or(RankError::Overflow(
                    "hierarchical reduce-scatter kernel launches",
                ))?;
        }

        let reduced_values = RankDevice::<T>::copy_from_device(self.execution, &working)?;
        let mut packed_values = Vec::with_capacity(self.execution.buffer_len(input));
        let mut group_ranges = Vec::with_capacity(groups.len());
        for hierarchy_group in groups {
            let start = packed_values.len();
            for rank in hierarchy_group {
                let rank_start =
                    (*rank as usize)
                        .checked_mul(output_length)
                        .ok_or(RankError::Overflow(
                            "hierarchical reduce-scatter rank offset",
                        ))?;
                let rank_end = rank_start
                    .checked_add(output_length)
                    .ok_or(RankError::Overflow(
                        "hierarchical reduce-scatter rank range",
                    ))?;
                packed_values.extend_from_slice(&reduced_values[rank_start..rank_end]);
            }
            group_ranges.push(start..packed_values.len());
        }
        if packed_values.len() != self.execution.buffer_len(input) {
            return Err(NetworkError::InvalidConfiguration(
                "hierarchical reduce-scatter packing changed the input length".into(),
            )
            .into());
        }
        let packed = RankDevice::<T>::alloc(self.execution, packed_values.len())?;
        RankDevice::<T>::copy_to_device(self.execution, &packed, &packed_values)?;
        let (leader_bytes, leader_launches) = self.leader_reduce_scatter_ring(
            &packed,
            &leaders,
            group_index,
            &group_ranges,
            operation,
            function,
            ring_channels,
        )?;
        transferred_bytes =
            transferred_bytes
                .checked_add(leader_bytes)
                .ok_or(RankError::Overflow(
                    "hierarchical reduce-scatter transferred bytes",
                ))?;
        reduction_kernel_launches = reduction_kernel_launches
            .checked_add(leader_launches)
            .ok_or(RankError::Overflow(
                "hierarchical reduce-scatter kernel launches",
            ))?;

        let local_group_values = RankDevice::<T>::copy_from_device_at(
            self.execution,
            &packed,
            group_ranges[group_index].start,
            group_ranges[group_index].len(),
        )?;
        for (position, member) in group.iter().enumerate().skip(1) {
            let start = position
                .checked_mul(output_length)
                .ok_or(RankError::Overflow(
                    "hierarchical reduce-scatter member offset",
                ))?;
            let end = start.checked_add(output_length).ok_or(RankError::Overflow(
                "hierarchical reduce-scatter member range",
            ))?;
            let values = &local_group_values[start..end];
            let payload = D::encode(values);
            self.session.send(
                *member,
                hierarchy_data_tag(operation, 9, *member),
                self.element_type,
                output_length as u64,
                payload,
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(values.len().saturating_mul(D::ELEMENT_SIZE))
                .ok_or(RankError::Overflow(
                    "hierarchical reduce-scatter transferred bytes",
                ))?;
        }
        let own_start = member_index
            .checked_mul(output_length)
            .ok_or(RankError::Overflow(
                "hierarchical reduce-scatter own offset",
            ))?;
        let own_end = own_start
            .checked_add(output_length)
            .ok_or(RankError::Overflow("hierarchical reduce-scatter own range"))?;
        let output = RankDevice::<T>::alloc(self.execution, output_length)?;
        RankDevice::<T>::copy_to_device(
            self.execution,
            &output,
            &local_group_values[own_start..own_end],
        )?;
        Ok((
            output,
            self.stats(
                CollectiveAlgorithm::Hierarchical,
                steps,
                transferred_bytes,
                reduction_kernel_launches,
            )?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn leader_reduce_scatter_ring(
        &self,
        buffer: &D::Buffer,
        leaders: &[u32],
        position: usize,
        ranges: &[Range<usize>],
        operation: u64,
        function: &D::Kernel,
        ring_channels: usize,
    ) -> Result<(usize, u32), D::Error> {
        let minimum_range = ranges.iter().map(Range::len).min().unwrap_or(0);
        let channels = self.effective_ring_channels(ring_channels, minimum_range);
        let channel_results = std::thread::scope(|scope| {
            let handles = (0..channels)
                .map(|channel| {
                    let channel_ranges = ranges
                        .iter()
                        .map(|range| {
                            let slice = balanced_ranges(range.len(), channels)[channel].clone();
                            range.start + slice.start..range.start + slice.end
                        })
                        .collect::<Vec<_>>();
                    let communicator = self.clone();
                    let buffer = buffer.clone();
                    let leaders = leaders.to_vec();
                    let function = (*function).clone();
                    scope.spawn(move || {
                        let result = communicator.leader_reduce_scatter_ring_channel(
                            &buffer,
                            &leaders,
                            position,
                            &channel_ranges,
                            operation,
                            channel,
                            &function,
                        );
                        communicator
                            .propagate_channel_failure("hierarchical reduce-scatter", result)
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
                        "hierarchical leader reduce-scatter transferred bytes",
                    ))?,
                    total_launches
                        .checked_add(launches)
                        .ok_or(RankError::Overflow(
                            "hierarchical leader reduce-scatter kernel launches",
                        ))?,
                ))
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn leader_reduce_scatter_ring_channel(
        &self,
        buffer: &D::Buffer,
        leaders: &[u32],
        position: usize,
        ranges: &[Range<usize>],
        operation: u64,
        channel: usize,
        function: &D::Kernel,
    ) -> Result<(usize, u32), D::Error> {
        let leader_count = leaders.len();
        let previous_rank = leaders[(position + leader_count - 1) % leader_count];
        let next_rank = leaders[(position + 1) % leader_count];
        let mut transferred_bytes = 0_usize;
        let mut reduction_kernel_launches = 0_u32;
        for step in 0..leader_count - 1 {
            let send_chunk = (position + leader_count - 1 - step) % leader_count;
            let receive_chunk = (position + 2 * leader_count - 2 - step) % leader_count;
            let send_range = ranges[send_chunk].clone();
            let receive_range = ranges[receive_chunk].clone();
            let received = self.exchange_ring_chunk_bytes(
                buffer,
                send_range.clone(),
                receive_range.len(),
                next_rank,
                previous_rank,
                channel,
                hierarchy_ring_data_tag(operation, 8, channel, step),
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(
                    send_range
                        .len()
                        .saturating_mul(D::ELEMENT_SIZE)
                        .saturating_add(received.len()),
                )
                .ok_or(RankError::Overflow(
                    "hierarchical leader reduce-scatter transferred bytes",
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
                    "hierarchical leader reduce-scatter kernel launches",
                ))?;
        }
        Ok((transferred_bytes, reduction_kernel_launches))
    }
}
