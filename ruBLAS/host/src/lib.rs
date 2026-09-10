#![cfg_attr(not(feature = "std"), no_std)]

//! CPU matrix multiplication for rublas.

extern crate alloc;

mod matmul;

pub use matmul::{int_matmul, matmul};

mod cross;
pub use cross::cross;
