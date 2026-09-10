#![cfg_attr(not(feature = "std"), no_std)]

//! CPU real Fourier transforms for ruFFT.

extern crate alloc;

mod transform;

pub use transform::{
    irfft_bf16, irfft_f16, irfft_f32, irfft_f64,
    rfft_bf16, rfft_f16, rfft_f32, rfft_f64,
};

pub mod dispatch;
