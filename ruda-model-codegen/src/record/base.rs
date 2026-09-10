use super::{
    codegen::generate_record,
    item::{codegen_enum::EnumRecordItemCodegen, codegen_struct::StructRecordItemCodegen},
};

pub fn derive_impl(ast: &syn::DeriveInput, paths: &crate::DerivePaths) -> proc_macro2::TokenStream {
    match &ast.data {
        syn::Data::Struct(_) => generate_record::<StructRecordItemCodegen>(ast, paths),
        syn::Data::Enum(_) => generate_record::<EnumRecordItemCodegen>(ast, paths),
        syn::Data::Union(_) => panic!("Union modules aren't supported yet."),
    }
}
