use super::*;
use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use syn::DeriveInput;

#[derive(Debug)]
pub(crate) enum RudaType {
    Enum(RudaTypeEnum),
    Struct(RudaTypeStruct),
}

impl FromDeriveInput for RudaType {
    fn from_derive_input(input: &syn::DeriveInput) -> darling::Result<Self> {
        match &input.data {
            syn::Data::Struct(_) => Ok(Self::Struct(RudaTypeStruct::from_derive_input(input)?)),
            syn::Data::Enum(_) => Ok(Self::Enum(RudaTypeEnum::from_derive_input(input)?)),
            syn::Data::Union(_) => Err(darling::Error::custom("Union not supported")),
        }
    }
}

pub fn generate_ruda_type(input: &DeriveInput, with_launch: bool) -> syn::Result<TokenStream> {
    let ruda_type = RudaType::from_derive_input(input)?;
    Ok(ruda_type.generate(with_launch))
}
