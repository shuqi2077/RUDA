#![cfg_attr(not(feature = "std"), no_std)]

//! CPU tensor primitives for ruPRIM.

extern crate alloc;

pub mod binary;
pub mod comparison;
pub mod expand;
pub mod flip;
pub mod simd;
pub mod unary;

pub mod cumulative;
pub mod gather_scatter;
pub mod mask;
pub mod reduce;
pub mod slice;
pub mod sort;

pub mod boolean;
pub mod cast;
pub mod cat;
pub mod repeat_dim;
pub mod unfold;

pub mod fill;

pub mod quantization;
