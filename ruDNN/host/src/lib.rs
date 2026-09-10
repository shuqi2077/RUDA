#![cfg_attr(not(feature = "std"), no_std)]

//! CPU neural-network operators for ruDNN.

extern crate alloc;

pub mod activation;
pub mod attention;
pub mod convolution;
pub mod grid_sample;
pub mod interpolate;
pub mod pool;

pub mod embedding;
