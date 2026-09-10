#![no_std]
//! Ruda portable host runtime.

#[cfg(feature = "runtime-std")]
extern crate std;

#[cfg(feature = "runtime")]
extern crate alloc;

#[cfg(feature = "runtime")]
#[macro_use]
extern crate derive_new;

extern crate self as ruda;




#[cfg(feature = "runtime")]
#[allow(unsafe_code)]
pub mod runtime;




