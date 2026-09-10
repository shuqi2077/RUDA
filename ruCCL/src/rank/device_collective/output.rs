#[derive(Debug, Clone)]
pub struct VariableCollectiveOutput<B> {
    pub(super) buffer: Option<B>,
    pub(super) counts: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ReceivedMessage<B> {
    pub(super) buffer: B,
    pub(super) source_rank: u32,
    pub(super) tag: u64,
}

impl<B> ReceivedMessage<B> {
    pub const fn buffer(&self) -> &B {
        &self.buffer
    }

    pub const fn source_rank(&self) -> u32 {
        self.source_rank
    }

    pub const fn tag(&self) -> u64 {
        self.tag
    }
}

impl<B> VariableCollectiveOutput<B> {
    pub fn buffer(&self) -> Option<&B> {
        self.buffer.as_ref()
    }

    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    pub fn len(&self) -> usize {
        self.counts.iter().sum()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_none()
    }
}
