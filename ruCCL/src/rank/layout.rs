use super::error::RankError;
use std::ops::Range;

#[cfg(test)]
mod tests;

pub fn balanced_ranges(length: usize, partitions: usize) -> Vec<Range<usize>> {
    let base = length / partitions;
    let remainder = length % partitions;
    let mut start = 0_usize;
    (0..partitions)
        .map(|partition| {
            let part_length = base + usize::from(partition < remainder);
            let range = start..start + part_length;
            start += part_length;
            range
        })
        .collect()
}

pub fn prefix_offsets(counts: &[usize]) -> Result<Vec<usize>, RankError> {
    let mut total = 0_usize;
    counts
        .iter()
        .map(|count| {
            let offset = total;
            total = total
                .checked_add(*count)
                .ok_or(RankError::Overflow("pairwise count prefix"))?;
            Ok(offset)
        })
        .collect()
}

pub fn hierarchy_steps(groups: &[Vec<u32>], inter_steps: usize) -> u32 {
    let local_steps = groups
        .iter()
        .map(|group| group.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    u32::try_from(local_steps.saturating_mul(2).saturating_add(inter_steps)).unwrap_or(u32::MAX)
}
