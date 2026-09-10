use super::ExchangeOptions;

#[cfg(test)]
mod tests;

const INTERNAL_RING_TAG_PREFIX: u64 = 0x4758_0000_0000_0000;
const INTERNAL_HIERARCHY_TAG_PREFIX: u64 = 0x4759_0000_0000_0000;

pub fn reduction_exchange_options(
    root_rank: u32,
    element_count: usize,
    operation: u32,
) -> ExchangeOptions {
    ExchangeOptions {
        root_rank,
        element_count: element_count as u64,
        flags: 0,
        tag: u64::from(operation) + 1,
    }
}

pub fn ring_agreement_tag(
    operation: u64,
    length: usize,
    rings: &[&[u32]],
    discriminator: u32,
    channels: usize,
    channel_rails: &[usize],
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in operation
        .to_le_bytes()
        .into_iter()
        .chain((length as u64).to_le_bytes())
        .chain(discriminator.to_le_bytes())
        .chain((channels as u64).to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for ring in rings {
        for byte in (ring.len() as u64)
            .to_le_bytes()
            .into_iter()
            .chain(ring.iter().flat_map(|rank| rank.to_le_bytes()))
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for rail in channel_rails {
        for byte in (*rail as u64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

pub fn ring_data_tag(operation: u64, phase: u64, channel: usize, step: usize) -> u64 {
    INTERNAL_RING_TAG_PREFIX
        | ((operation & 0x00ff_ffff) << 24)
        | ((phase & 0x0f) << 20)
        | ((channel as u64 & 0xff) << 12)
        | (step as u64 & 0x0fff)
}

pub fn hierarchy_agreement_tag(
    operation: u64,
    length: usize,
    groups: &[Vec<u32>],
    discriminator: u32,
    ring_channels: usize,
    channel_rails: &[usize],
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in operation
        .to_le_bytes()
        .into_iter()
        .chain((length as u64).to_le_bytes())
        .chain(discriminator.to_le_bytes())
        .chain((ring_channels as u64).to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for group in groups {
        for byte in (group.len() as u64)
            .to_le_bytes()
            .into_iter()
            .chain(group.iter().flat_map(|rank| rank.to_le_bytes()))
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for rail in channel_rails {
        for byte in (*rail as u64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

pub fn hierarchy_data_tag(operation: u64, phase: u32, index: u32) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in operation
        .to_le_bytes()
        .into_iter()
        .chain(phase.to_le_bytes())
        .chain(index.to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    INTERNAL_HIERARCHY_TAG_PREFIX | (hash & 0x0000_ffff_ffff_ffff)
}

pub fn hierarchy_ring_data_tag(operation: u64, phase: u32, channel: usize, step: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in operation
        .to_le_bytes()
        .into_iter()
        .chain(phase.to_le_bytes())
        .chain((channel as u64).to_le_bytes())
        .chain((step as u64).to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    INTERNAL_HIERARCHY_TAG_PREFIX | (hash & 0x0000_ffff_ffff_ffff)
}
