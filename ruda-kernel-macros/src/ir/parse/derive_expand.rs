use darling::{FromMeta, util::Flag};
use proc_macro2::TokenStream;
use syn::{DeriveInput, Meta, parse_quote, parse2};

use crate::ir::{
    generate::{assign::generate_ruda_type_mut, into_runtime::generate_into_runtime},
    parse::ruda_type::generate_ruda_type,
};

#[derive(FromMeta)]
#[darling(rename_all = "PascalCase")]
pub struct DeriveExpand {
    ruda_type: Flag,
    ruda_type_mut: Flag,
    ruda_launch: Flag,
    into_runtime: Flag,
}

pub fn generate_derive_expand(input: TokenStream, meta: TokenStream) -> syn::Result<TokenStream> {
    let input: DeriveInput = parse2(input)?;
    let meta: Meta = parse_quote!(derive_expand(#meta));
    let derives = DeriveExpand::from_meta(&meta)?;

    let mut out = TokenStream::new();

    if derives.ruda_type.is_present() {
        out.extend(generate_ruda_type(&input, false)?);
    }
    if derives.ruda_type_mut.is_present() {
        out.extend(generate_ruda_type_mut(&input)?);
    }
    if derives.ruda_launch.is_present() {
        out.extend(generate_ruda_type(&input, true)?);
    }
    if derives.into_runtime.is_present() {
        out.extend(generate_into_runtime(&input)?)
    }

    Ok(out)
}
