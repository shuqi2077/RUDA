use super::*;

impl<B, T, E> InProcessCollective<'_, T, crate::tensor_device::TensorDevice<B>, E>
where
    B: ruda_tensor::Backend,
    T: crate::tensor_device::TensorElement,
    E: From<crate::tensor_device::TensorDeviceError> + From<InProcessError>,
{
    /// Attach existing rank tensors without downloading and re-uploading values.
    pub fn import_buffers(
        &self,
        buffers: Vec<crate::tensor_device::TensorBuffer<B, T>>,
    ) -> Result<DistributedBuffer<T, crate::tensor_device::TensorDevice<B>>, E> {
        if buffers.len() != self.world_size() {
            return Err(InProcessError::InvalidLength("one tensor buffer is required per rank").into());
        }
        let length_per_rank = buffers.first().ok_or(InProcessError::EmptyWorld)?.len();
        for (buffer, context) in buffers.iter().zip(self.contexts) {
            if buffer.len() != length_per_rank {
                return Err(InProcessError::InvalidLength("rank tensor lengths must match").into());
            }
            if buffer.device() != context.device() {
                return Err(crate::tensor_device::TensorDeviceError::DeviceMismatch.into());
            }
        }
        Ok(DistributedBuffer {
            communicator_id: self.id,
            marker: PhantomData,
            buffers,
            length_per_rank,
        })
    }
}

impl<T, D, E> InProcessCollective<'_, T, D, E>
where
    T: Copy + Send + Sync + 'static,
    D: InProcessDevice<T>,
    E: From<D::Error> + From<InProcessError>,
{
    pub fn allocate(&self, length_per_rank: usize) -> Result<DistributedBuffer<T, D>, E> {
        let buffers = self
            .contexts
            .iter()
            .map(|context| <D as InProcessDevice<T>>::alloc(context, length_per_rank))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DistributedBuffer {
            communicator_id: self.id,
            marker: PhantomData,
            buffers,
            length_per_rank,
        })
    }

    pub fn allocate_rooted(&self, root: usize, length: usize) -> Result<RootedBuffer<T, D>, E> {
        let context = self.context(root)?;
        Ok(RootedBuffer {
            communicator_id: self.id,
            marker: PhantomData,
            root,
            buffer: <D as InProcessDevice<T>>::alloc(context, length)?,
        })
    }

    pub fn allocate_variable(
        &self,
        lengths: &[usize],
    ) -> Result<VariableDistributedBuffer<T, D>, E> {
        if lengths.len() != self.world_size() {
            return Err(InProcessError::InvalidLength(
                "variable buffer lengths must contain one entry per rank",
            )
            .into());
        }
        let buffers = self
            .contexts
            .iter()
            .zip(lengths)
            .map(|(context, length)| {
                if *length == 0 {
                    Ok(None)
                } else {
                    <D as InProcessDevice<T>>::alloc(context, *length).map(Some)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(VariableDistributedBuffer {
            communicator_id: self.id,
            marker: PhantomData,
            buffers,
            lengths: lengths.to_vec(),
        })
    }

    pub fn upload_variable_rank(
        &self,
        buffer: &VariableDistributedBuffer<T, D>,
        rank: usize,
        values: &[T],
    ) -> Result<(), E> {
        self.validate_variable_buffer(buffer)?;
        let context = self.context(rank)?;
        let expected = buffer
            .rank_length(rank)
            .ok_or(InProcessError::RankOutOfRange {
                rank,
                world_size: self.world_size(),
            })?;
        if values.len() != expected {
            return Err(InProcessError::InvalidLength(
                "variable rank upload length does not match allocation",
            )
            .into());
        }
        if let Some(rank_buffer) = buffer.rank_buffer(rank) {
            <D as InProcessDevice<T>>::copy_to_device(context, rank_buffer, values)?;
        }
        Ok(())
    }

    pub fn download_variable_rank(
        &self,
        buffer: &VariableDistributedBuffer<T, D>,
        rank: usize,
    ) -> Result<Vec<T>, E> {
        self.validate_variable_buffer(buffer)?;
        let context = self.context(rank)?;
        match buffer.rank_buffer(rank) {
            Some(rank_buffer) => Ok(<D as InProcessDevice<T>>::copy_from_device(
                context,
                rank_buffer,
            )?),
            None if buffer.rank_length(rank) == Some(0) => Ok(Vec::new()),
            None => Err(InProcessError::RankOutOfRange {
                rank,
                world_size: self.world_size(),
            }
            .into()),
        }
    }

    pub fn upload_root(&self, buffer: &RootedBuffer<T, D>, values: &[T]) -> Result<(), E> {
        self.validate_rooted_buffer(buffer)?;
        <D as InProcessDevice<T>>::copy_to_device(
            self.context(buffer.root)?,
            &buffer.buffer,
            values,
        )?;
        Ok(())
    }

    pub fn download_root(&self, buffer: &RootedBuffer<T, D>) -> Result<Vec<T>, E> {
        self.validate_rooted_buffer(buffer)?;
        Ok(<D as InProcessDevice<T>>::copy_from_device(
            self.context(buffer.root)?,
            &buffer.buffer,
        )?)
    }

    pub fn upload_rank(
        &self,
        buffer: &DistributedBuffer<T, D>,
        rank: usize,
        values: &[T],
    ) -> Result<(), E> {
        self.validate_buffer(buffer)?;
        let context = self.context(rank)?;
        let rank_buffer = buffer
            .buffers
            .get(rank)
            .ok_or(InProcessError::RankOutOfRange {
                rank,
                world_size: self.world_size(),
            })?;
        <D as InProcessDevice<T>>::copy_to_device(context, rank_buffer, values)?;
        Ok(())
    }

    pub fn download_rank(
        &self,
        buffer: &DistributedBuffer<T, D>,
        rank: usize,
    ) -> Result<Vec<T>, E> {
        self.validate_buffer(buffer)?;
        let context = self.context(rank)?;
        let rank_buffer = buffer
            .buffers
            .get(rank)
            .ok_or(InProcessError::RankOutOfRange {
                rank,
                world_size: self.world_size(),
            })?;
        Ok(<D as InProcessDevice<T>>::copy_from_device(
            context,
            rank_buffer,
        )?)
    }
}
