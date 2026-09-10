#![cfg_attr(not(feature = "std"), no_std)]

mod state;
mod tensor;

pub use state::{HostRng, seed};
pub use tensor::{float_random, int_random};
