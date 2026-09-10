use super::*;

impl<T, D, E> InProcessCollective<'_, T, D, E>
where
    T: Copy + Send + Sync + 'static,
    D: InProcessDevice<T>,
    E: From<D::Error> + From<InProcessError>,
{
    pub fn reduce_sum(
        &self,
        input: &DistributedBuffer<T, D>,
        root: usize,
        function: &D::Kernel,
    ) -> Result<(RootedBuffer<T, D>, CollectiveStats), E> {
        self.validate_buffer(input)?;
        let root_context = self.context(root)?;
        let output = self.allocate_rooted(root, input.length_per_rank)?;
        let root_values = self.download_rank(input, root)?;
        <D as InProcessDevice<T>>::copy_to_device(root_context, &output.buffer, &root_values)?;
        if input.length_per_rank == 0 {
            return Ok((output, CollectiveStats::new(CollectiveAlgorithm::Direct)));
        }
        let scratch = <D as InProcessDevice<T>>::alloc(root_context, input.length_per_rank)?;
        let length = u32::try_from(input.length_per_rank)
            .map_err(|_| InProcessError::InvalidLength("GX kernels use u32 element indices"))?;
        let launch = <D as InProcessDevice<T>>::prepare_reduction(length, 0);
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Direct);
        for rank in 0..self.world_size() {
            if rank == root {
                continue;
            }
            let values = self.download_rank(input, rank)?;
            <D as InProcessDevice<T>>::copy_to_device(root_context, &scratch, &values)?;
            <D as InProcessDevice<T>>::launch_reduction(
                root_context,
                function,
                &launch,
                &scratch,
                &output.buffer,
            )?;
            stats.steps += 1;
            stats.reduction_kernel_launches += 1;
        }
        stats.transferred_bytes = transferred_bytes(
            self.world_size().saturating_sub(1),
            input.length_per_rank,
            D::ELEMENT_SIZE,
        )?;
        Ok((output, stats))
    }

    pub fn all_reduce_sum(
        &self,
        buffer: &DistributedBuffer<T, D>,
        function: &D::Kernel,
    ) -> Result<CollectiveStats, E> {
        self.validate_buffer(buffer)?;
        let world_size = self.world_size();
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Ring);
        if world_size == 1 {
            return Ok(stats);
        }
        let length_u32 = u32::try_from(buffer.length_per_rank)
            .map_err(|_| InProcessError::InvalidLength("GX kernels use u32 element indices"))?;
        let chunks = ring_chunks(length_u32, world_size)?;
        let max_chunk = chunks.iter().map(|range| range.len()).max().unwrap_or(0);
        let scratch = self
            .contexts
            .iter()
            .map(|context| <D as InProcessDevice<T>>::alloc(context, max_chunk.max(1)))
            .collect::<Result<Vec<_>, _>>()?;

        for step in 0..world_size - 1 {
            let staged = (0..world_size)
                .map(|rank| {
                    let send_chunk = (rank + world_size - step - 1) % world_size;
                    let range = &chunks[send_chunk];
                    <D as InProcessDevice<T>>::copy_from_device_at(
                        &self.contexts[rank],
                        &buffer.buffers[rank],
                        range.start,
                        range.len(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (rank, scratch_buffer) in scratch.iter().enumerate() {
                let source_rank = (rank + world_size - 1) % world_size;
                let receive_chunk = (rank + world_size - step - 2) % world_size;
                let range = &chunks[receive_chunk];
                if range.is_empty() {
                    continue;
                }
                <D as InProcessDevice<T>>::copy_to_device_at(
                    &self.contexts[rank],
                    scratch_buffer,
                    0,
                    &staged[source_rank],
                )?;
                let launch = <D as InProcessDevice<T>>::prepare_reduction(
                    range.len() as u32,
                    range.start as u32,
                );
                <D as InProcessDevice<T>>::launch_reduction(
                    &self.contexts[rank],
                    function,
                    &launch,
                    scratch_buffer,
                    &buffer.buffers[rank],
                )?;
                stats.reduction_kernel_launches += 1;
            }
            stats.steps += 1;
        }

        for step in 0..world_size - 1 {
            let staged = (0..world_size)
                .map(|rank| {
                    let send_chunk = (rank + world_size - step) % world_size;
                    let range = &chunks[send_chunk];
                    <D as InProcessDevice<T>>::copy_from_device_at(
                        &self.contexts[rank],
                        &buffer.buffers[rank],
                        range.start,
                        range.len(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            for rank in 0..world_size {
                let source_rank = (rank + world_size - 1) % world_size;
                let receive_chunk = (rank + world_size - step - 1) % world_size;
                let range = &chunks[receive_chunk];
                if !range.is_empty() {
                    <D as InProcessDevice<T>>::copy_to_device_at(
                        &self.contexts[rank],
                        &buffer.buffers[rank],
                        range.start,
                        &staged[source_rank],
                    )?;
                }
            }
            stats.steps += 1;
        }

        stats.transferred_bytes = transferred_bytes(
            2 * (world_size - 1),
            buffer.length_per_rank,
            D::ELEMENT_SIZE,
        )?;
        Ok(stats)
    }

    pub fn reduce_scatter_sum(
        &self,
        input: &DistributedBuffer<T, D>,
        function: &D::Kernel,
    ) -> Result<(DistributedBuffer<T, D>, CollectiveStats), E> {
        self.validate_buffer(input)?;
        let world_size = self.world_size();
        if !input.length_per_rank.is_multiple_of(world_size) {
            return Err(InProcessError::InvalidLength(
                "reduce-scatter input length must be divisible by world size",
            )
            .into());
        }
        let output_length = input.length_per_rank / world_size;
        let length_u32 = u32::try_from(input.length_per_rank)
            .map_err(|_| InProcessError::InvalidLength("GX kernels use u32 element indices"))?;
        let chunks = ring_chunks(length_u32, world_size)?;
        let working = self.allocate(input.length_per_rank)?;
        for rank in 0..world_size {
            let values = self.download_rank(input, rank)?;
            <D as InProcessDevice<T>>::copy_to_device(
                &self.contexts[rank],
                &working.buffers[rank],
                &values,
            )?;
        }
        let scratch = self
            .contexts
            .iter()
            .map(|context| <D as InProcessDevice<T>>::alloc(context, output_length))
            .collect::<Result<Vec<_>, _>>()?;
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Ring);
        for step in 0..world_size.saturating_sub(1) {
            let staged = (0..world_size)
                .map(|rank| {
                    let send_chunk = (rank + world_size - step - 1) % world_size;
                    let range = &chunks[send_chunk];
                    <D as InProcessDevice<T>>::copy_from_device_at(
                        &self.contexts[rank],
                        &working.buffers[rank],
                        range.start,
                        range.len(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (rank, scratch_buffer) in scratch.iter().enumerate() {
                let source_rank = (rank + world_size - 1) % world_size;
                let receive_chunk = (rank + world_size - step - 2) % world_size;
                let range = &chunks[receive_chunk];
                if range.is_empty() {
                    continue;
                }
                <D as InProcessDevice<T>>::copy_to_device(
                    &self.contexts[rank],
                    scratch_buffer,
                    &staged[source_rank],
                )?;
                let launch = <D as InProcessDevice<T>>::prepare_reduction(
                    range.len() as u32,
                    range.start as u32,
                );
                <D as InProcessDevice<T>>::launch_reduction(
                    &self.contexts[rank],
                    function,
                    &launch,
                    scratch_buffer,
                    &working.buffers[rank],
                )?;
                stats.reduction_kernel_launches += 1;
            }
            stats.steps += 1;
        }

        let output = self.allocate(output_length)?;
        for (rank, range) in chunks.iter().enumerate() {
            let reduced = <D as InProcessDevice<T>>::copy_from_device_at(
                &self.contexts[rank],
                &working.buffers[rank],
                range.start,
                range.len(),
            )?;
            <D as InProcessDevice<T>>::copy_to_device(
                &self.contexts[rank],
                &output.buffers[rank],
                &reduced,
            )?;
        }
        stats.transferred_bytes = transferred_bytes(
            world_size.saturating_sub(1),
            input.length_per_rank,
            D::ELEMENT_SIZE,
        )?;
        Ok((output, stats))
    }
}
