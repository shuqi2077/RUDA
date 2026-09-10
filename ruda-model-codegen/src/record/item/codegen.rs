use proc_macro2::{Ident, TokenStream};
use syn::Generics;

/// Basic trait to be implemented for record generation.
pub(crate) trait RecordItemCodegen {
    /// Initialize the record item.
    fn from_ast(ast: &syn::DeriveInput, _paths: &crate::DerivePaths) -> syn::Result<Self>
    where
        Self: Sized;
    /// Generate the record item type.
    fn gen_item_type(
        &self,
        item_name: &Ident,
        generics: &Generics,
        has_backend: bool,
        _paths: &crate::DerivePaths,
    ) -> TokenStream;
    /// Generate the into_item function.
    fn gen_into_item(&self, item_name: &Ident, _paths: &crate::DerivePaths) -> TokenStream;
    /// Generate the from item function.
    fn gen_from_item(&self, _paths: &crate::DerivePaths) -> TokenStream;
}
