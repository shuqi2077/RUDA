use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum InProcessError {
    EmptyWorld,
    RankOutOfRange { rank: usize, world_size: usize },
    WrongCommunicator,
    InvalidLength(&'static str),
    Overflow(&'static str),
}

impl Display for InProcessError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyWorld => {
                write!(formatter, "collective world must contain at least one rank")
            }
            Self::RankOutOfRange { rank, world_size } => write!(
                formatter,
                "rank {rank} is outside collective world size {world_size}"
            ),
            Self::WrongCommunicator => write!(
                formatter,
                "distributed buffer belongs to another communicator"
            ),
            Self::InvalidLength(message) => {
                write!(formatter, "invalid collective length: {message}")
            }
            Self::Overflow(what) => {
                write!(formatter, "collective size overflow while computing {what}")
            }
        }
    }
}

impl Error for InProcessError {}
