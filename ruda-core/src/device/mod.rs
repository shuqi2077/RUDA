mod base;

pub use base::*;

pub(crate) mod handle;

#[cfg(feature = "communication")]
mod communication;
#[cfg(feature = "communication")]
pub use communication::CommunicationId;
