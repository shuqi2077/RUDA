//! Compile-time bridge from Rust kernel functions to Ruda IR.

use proc_macro::TokenStream;
use quote::quote;
use syn::Item;



#[cfg(feature = "kernel-ir")]
#[allow(clippy::large_enum_variant)]
mod ir;

#[cfg(feature = "kernel-ir")]
include!("ir/entrypoints.rs");
