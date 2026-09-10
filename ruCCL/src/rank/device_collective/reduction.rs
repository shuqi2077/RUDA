use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn reduce_concatenated(
        &self,
        payload: &[u8],
        output: &D::Buffer,
        length: usize,
        function: &D::Kernel,
    ) -> Result<u32, D::Error> {
        let rank_bytes = length
            .checked_mul(D::ELEMENT_SIZE)
            .ok_or(RankError::Overflow("TCP reduction rank bytes"))?;
        let expected = rank_bytes
            .checked_mul(self.world_size() as usize)
            .ok_or(RankError::Overflow("TCP reduction response bytes"))?;
        if payload.len() != expected {
            return Err(NetworkError::InvalidConfiguration(format!(
                "TCP reduction response has {} bytes, expected {expected}",
                payload.len()
            ))
            .into());
        }
        let first = D::decode(&payload[..rank_bytes])?;
        RankDevice::<T>::copy_to_device(self.execution, output, &first)?;
        if length == 0 {
            return Ok(0);
        }
        if self.world_size() == 1 {
            return Ok(0);
        }
        let scratch = RankDevice::<T>::alloc(self.execution, length)?;
        let launch = RankDevice::<T>::prepare_reduction(self.execution, length, 0)?;
        for source in 1..self.world_size() as usize {
            let start = source * rank_bytes;
            let values = D::decode(&payload[start..start + rank_bytes])?;
            RankDevice::<T>::copy_to_device(self.execution, &scratch, &values)?;
            RankDevice::<T>::launch_reduction(self.execution, function, &launch, &scratch, output)?;
        }
        Ok(self.world_size() - 1)
    }

    pub fn reduce_values_into(
        &self,
        destination: &D::Buffer,
        destination_offset: usize,
        values: &[T],
        function: &D::Kernel,
    ) -> Result<u32, D::Error> {
        let bytes = D::encode(values);
        self.reduce_bytes_into(
            destination,
            destination_offset,
            values.len(),
            &bytes,
            function,
        )
    }

    pub fn reduce_bytes_into(
        &self,
        destination: &D::Buffer,
        destination_offset: usize,
        element_count: usize,
        values: &[u8],
        function: &D::Kernel,
    ) -> Result<u32, D::Error> {
        let expected_bytes = element_count
            .checked_mul(D::ELEMENT_SIZE)
            .ok_or(RankError::Overflow("reduction input byte length"))?;
        if values.len() != expected_bytes {
            return Err(RankError::InvalidLength(
                "reduction input does not match its element count",
            )
            .into());
        }
        if element_count == 0 {
            return Ok(0);
        }
        let end = destination_offset
            .checked_add(element_count)
            .ok_or(RankError::Overflow(
                "hierarchical reduction destination range",
            ))?;
        if end > self.execution.buffer_len(destination) {
            return Err(RankError::InvalidLength(
                "hierarchical reduction destination range is out of bounds",
            )
            .into());
        }
        let scratch = RankDevice::<T>::alloc(self.execution, element_count)?;
        RankDevice::<T>::copy_bytes_to_device_at(self.execution, &scratch, 0, values)?;
        let launch =
            RankDevice::<T>::prepare_reduction(self.execution, element_count, destination_offset)?;
        RankDevice::<T>::launch_reduction(
            self.execution,
            function,
            &launch,
            &scratch,
            destination,
        )?;
        Ok(1)
    }
}
