use super::NetworkError;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum RankError {
    RankOutOfRange { rank: usize, world_size: usize },
    InvalidLength(&'static str),
    Overflow(&'static str),
    Network(NetworkError),
}

impl Display for RankError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RankOutOfRange { rank, world_size } => {
                write!(formatter, "rank {rank} is outside collective world size {world_size}")
            }
            Self::InvalidLength(message) => {
                write!(formatter, "invalid collective length: {message}")
            }
            Self::Overflow(what) => {
                write!(formatter, "collective size overflow while computing {what}")
            }
            Self::Network(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for RankError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Network(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NetworkError> for RankError {
    fn from(error: NetworkError) -> Self {
        Self::Network(error)
    }
}
