use super::*;

impl<T, D, E> InProcessCollective<'_, T, D, E>
where
    T: Copy + Send + Sync + 'static,
    D: InProcessDevice<T>,
    E: From<D::Error> + From<InProcessError>,
{
    pub fn broadcast(
        &self,
        buffer: &DistributedBuffer<T, D>,
        root: usize,
    ) -> Result<CollectiveStats, E> {
        self.validate_buffer(buffer)?;
        let root_values = self.download_rank(buffer, root)?;
        for rank in 0..self.world_size() {
            if rank != root {
                <D as InProcessDevice<T>>::copy_to_device(
                    &self.contexts[rank],
                    &buffer.buffers[rank],
                    &root_values,
                )?;
            }
        }
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Direct);
        stats.steps = u32::from(self.world_size() > 1);
        stats.transferred_bytes = transferred_bytes(
            self.world_size().saturating_sub(1),
            buffer.length_per_rank,
            D::ELEMENT_SIZE,
        )?;
        Ok(stats)
    }

    pub fn all_gather(
        &self,
        input: &DistributedBuffer<T, D>,
    ) -> Result<(DistributedBuffer<T, D>, CollectiveStats), E> {
        self.validate_buffer(input)?;
        let output_length = input
            .length_per_rank
            .checked_mul(self.world_size())
            .ok_or(InProcessError::Overflow("all-gather output length"))?;
        let output = self.allocate(output_length)?;
        for rank in 0..self.world_size() {
            let local = self.download_rank(input, rank)?;
            <D as InProcessDevice<T>>::copy_to_device_at(
                &self.contexts[rank],
                &output.buffers[rank],
                rank * input.length_per_rank,
                &local,
            )?;
        }
        for step in 0..self.world_size().saturating_sub(1) {
            let staged = (0..self.world_size())
                .map(|rank| {
                    let send_rank = (rank + self.world_size() - step) % self.world_size();
                    <D as InProcessDevice<T>>::copy_from_device_at(
                        &self.contexts[rank],
                        &output.buffers[rank],
                        send_rank * input.length_per_rank,
                        input.length_per_rank,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            for rank in 0..self.world_size() {
                let source_rank = (rank + self.world_size() - 1) % self.world_size();
                let receive_rank = (rank + self.world_size() - step - 1) % self.world_size();
                <D as InProcessDevice<T>>::copy_to_device_at(
                    &self.contexts[rank],
                    &output.buffers[rank],
                    receive_rank * input.length_per_rank,
                    &staged[source_rank],
                )?;
            }
        }
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Ring);
        stats.steps = self.world_size().saturating_sub(1) as u32;
        stats.transferred_bytes = transferred_bytes(
            self.world_size()
                .saturating_mul(self.world_size().saturating_sub(1)),
            input.length_per_rank,
            D::ELEMENT_SIZE,
        )?;
        Ok((output, stats))
    }

    pub fn gather(
        &self,
        input: &DistributedBuffer<T, D>,
        root: usize,
    ) -> Result<(RootedBuffer<T, D>, CollectiveStats), E> {
        self.validate_buffer(input)?;
        let output_length = input
            .length_per_rank
            .checked_mul(self.world_size())
            .ok_or(InProcessError::Overflow("gather output length"))?;
        let output = self.allocate_rooted(root, output_length)?;
        let root_context = self.context(root)?;
        for rank in 0..self.world_size() {
            let values = self.download_rank(input, rank)?;
            <D as InProcessDevice<T>>::copy_to_device_at(
                root_context,
                &output.buffer,
                rank * input.length_per_rank,
                &values,
            )?;
        }
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Direct);
        stats.steps = u32::from(self.world_size() > 1);
        stats.transferred_bytes = transferred_bytes(
            self.world_size().saturating_sub(1),
            input.length_per_rank,
            D::ELEMENT_SIZE,
        )?;
        Ok((output, stats))
    }

    pub fn scatter(
        &self,
        input: &RootedBuffer<T, D>,
    ) -> Result<(DistributedBuffer<T, D>, CollectiveStats), E> {
        self.validate_rooted_buffer(input)?;
        if !input.len().is_multiple_of(self.world_size()) {
            return Err(InProcessError::InvalidLength(
                "scatter input length must be divisible by world size",
            )
            .into());
        }
        let output_length = input.len() / self.world_size();
        let output = self.allocate(output_length)?;
        let root_context = self.context(input.root)?;
        for rank in 0..self.world_size() {
            let values = <D as InProcessDevice<T>>::copy_from_device_at(
                root_context,
                &input.buffer,
                rank * output_length,
                output_length,
            )?;
            <D as InProcessDevice<T>>::copy_to_device(
                &self.contexts[rank],
                &output.buffers[rank],
                &values,
            )?;
        }
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Direct);
        stats.steps = u32::from(self.world_size() > 1);
        stats.transferred_bytes = transferred_bytes(
            self.world_size().saturating_sub(1),
            output_length,
            D::ELEMENT_SIZE,
        )?;
        Ok((output, stats))
    }

    /// Tagged point-to-point transfer between two rank-local buffers. This is
    /// synchronous on the host-staged transport; `tag` is retained in the API
    /// because network transports match send and receive by that value.
    pub fn send_recv(
        &self,
        source: &DistributedBuffer<T, D>,
        destination: &DistributedBuffer<T, D>,
        transfer: PointToPointTransfer,
    ) -> Result<CollectiveStats, E> {
        self.validate_buffer(source)?;
        self.validate_buffer(destination)?;
        let source_context = self.context(transfer.source_rank)?;
        let destination_context = self.context(transfer.destination_rank)?;
        if transfer.source_range.start > transfer.source_range.end
            || transfer.source_range.end > source.length_per_rank
        {
            return Err(InProcessError::InvalidLength(
                "send source range is outside the rank buffer",
            )
            .into());
        }
        let length = transfer.source_range.len();
        let destination_end = transfer
            .destination_offset
            .checked_add(length)
            .ok_or(InProcessError::Overflow("send destination range"))?;
        if destination_end > destination.length_per_rank {
            return Err(InProcessError::InvalidLength(
                "receive destination range is outside the rank buffer",
            )
            .into());
        }
        let values = <D as InProcessDevice<T>>::copy_from_device_at(
            source_context,
            &source.buffers[transfer.source_rank],
            transfer.source_range.start,
            length,
        )?;
        <D as InProcessDevice<T>>::copy_to_device_at(
            destination_context,
            &destination.buffers[transfer.destination_rank],
            transfer.destination_offset,
            &values,
        )?;
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Direct);
        stats.steps = u32::from(length != 0);
        stats.transferred_bytes = transferred_bytes(1, length, D::ELEMENT_SIZE)?;
        Ok(stats)
    }
}
