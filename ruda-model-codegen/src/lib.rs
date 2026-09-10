#![warn(missing_docs)]

//! The derive crate of Ruda.

#[macro_use]
extern crate derive_new;

pub(crate) mod config;
pub(crate) mod module;
pub(crate) mod record;
pub(crate) mod shared;

#[cfg(test)]
mod path_tests;

pub(crate) struct DerivePaths {
    model: proc_macro2::TokenStream,
    serde: &'static str,
}

impl DerivePaths {
    fn native() -> Self {
        Self { model: quote::quote!(::ruda_model), serde: "::ruda_model::serde" }
    }
}

/// Generate config code using the native model owner.
pub fn derive_config(input: &syn::DeriveInput) -> proc_macro2::TokenStream {
    config::derive_impl(input, &DerivePaths::native())
}

/// Generate config code using the native model owner.
pub fn derive_config_native(input: &syn::DeriveInput) -> proc_macro2::TokenStream {
    config::derive_impl(input, &DerivePaths::native())
}

/// Generate module code using the native model owner.
pub fn derive_module(input: &syn::DeriveInput) -> proc_macro2::TokenStream {
    module::derive_impl(input, &DerivePaths::native())
}

/// Generate module code using the native model owner.
pub fn derive_module_native(input: &syn::DeriveInput) -> proc_macro2::TokenStream {
    module::derive_impl(input, &DerivePaths::native())
}

/// Generate record code using the native model owner.
pub fn derive_record(input: &syn::DeriveInput) -> proc_macro2::TokenStream {
    record::derive_impl(input, &DerivePaths::native())
}

/// Generate record code using the native model owner.
pub fn derive_record_native(input: &syn::DeriveInput) -> proc_macro2::TokenStream {
    record::derive_impl(input, &DerivePaths::native())
}
