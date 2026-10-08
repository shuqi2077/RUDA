use crate::DerivePaths;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Fields, Generics, Visibility, parse_quote};

use super::{codegen::RecordItemCodegen, codegen_enum_full::contains_self};

pub(crate) struct FullStructRecordItemCodegen {
    name: Ident,
    fields: Fields,
    vis: Visibility,
}

pub(crate) fn derive_full_struct(ast: &syn::DeriveInput, paths: &DerivePaths) -> TokenStream {
    let mut ast = ast.clone();
    let model = &paths.model;
    let syn::Data::Struct(data) = &ast.data else { unreachable!() };
    let bounds: Vec<syn::WherePredicate> = data.fields.iter()
        .filter(|field| !contains_self(&field.ty, &ast.ident))
        .map(|field| { let ty = &field.ty; parse_quote!(#ty: #model::record::Record<B>) }).collect();
    if !bounds.is_empty() { ast.generics.make_where_clause().predicates.extend(bounds); }
    crate::record::codegen::generate_record::<FullStructRecordItemCodegen>(&ast, paths)
}

impl FullStructRecordItemCodegen {
    fn construct(&self, constructor: TokenStream, fields: Vec<TokenStream>) -> TokenStream {
        match &self.fields {
            Fields::Unit => constructor,
            Fields::Unnamed(_) => quote!(#constructor(#(#fields),*)),
            Fields::Named(named) => {
                let names = named.named.iter().map(|field| field.ident.as_ref().unwrap());
                quote!(#constructor { #(#names: #fields),* })
            }
        }
    }

    fn members(&self) -> Vec<syn::Member> {
        self.fields.iter().enumerate().map(|(index, field)| match &field.ident {
            Some(name) => syn::Member::Named(name.clone()),
            None => syn::Member::Unnamed(syn::Index::from(index)),
        }).collect()
    }
}

impl RecordItemCodegen for FullStructRecordItemCodegen {
    fn item_has_payload(&self) -> bool { !self.fields.is_empty() }

    fn from_ast(ast: &syn::DeriveInput, _paths: &DerivePaths) -> syn::Result<Self> {
        let syn::Data::Struct(data) = &ast.data else {
            return Err(syn::Error::new_spanned(ast, "Expected a record struct"));
        };
        Ok(Self { name: ast.ident.clone(), fields: data.fields.clone(), vis: ast.vis.clone() })
    }

    fn gen_item_type(&self, item_name: &Ident, generics: &Generics, has_backend: bool,
        paths: &DerivePaths) -> TokenStream {
        let model = &paths.model;
        let serde_path = paths.serde;
        let vis = &self.vis;
        let mut generics = generics.clone();
        if !has_backend && self.item_has_payload() {
            generics.params.push(parse_quote!(B: #model::tensor::backend::Backend));
        }
        let mut serde_bounds = TokenStream::new();
        let fields: Vec<_> = self.fields.iter().map(|field| {
            let ty = &field.ty;
            let item = quote!(<#ty as #model::record::Record<B>>::Item<S>);
            if !contains_self(ty, &self.name) {
                serde_bounds.extend(quote!(#item: #model::serde::Serialize + #model::serde::de::DeserializeOwned,));
            }
            item
        }).collect();
        let serde_bound = serde_bounds.to_string();
        let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
        let declaration = match &self.fields {
            Fields::Unit => quote!(#vis struct #item_name #impl_generics #where_clause;),
            Fields::Unnamed(_) => quote!(#vis struct #item_name #impl_generics(#(pub #fields),*) #where_clause;),
            Fields::Named(named) => {
                let names = named.named.iter().map(|field| field.ident.as_ref().unwrap());
                quote!(#vis struct #item_name #impl_generics #where_clause { #(pub #names: #fields),* })
            }
        };
        let clone = self.construct(quote!(Self), self.members().iter()
            .map(|member| quote!(Clone::clone(&self.#member))).collect());
        quote! {
            #[allow(missing_docs)]
            #[derive(#model::serde::Serialize, #model::serde::Deserialize)]
            #[serde(crate = #serde_path, bound = #serde_bound)]
            #declaration

            impl #impl_generics Clone for #item_name #type_generics #where_clause {
                fn clone(&self) -> Self { #clone }
            }
        }
    }

    fn gen_into_item(&self, item_name: &Ident, paths: &DerivePaths) -> TokenStream {
        let model = &paths.model;
        let body = self.construct(quote!(#item_name), self.members().iter()
            .map(|member| quote!(#model::record::Record::<B>::into_item::<S>(self.#member))).collect());
        quote! {
            fn into_item<S: #model::record::PrecisionSettings>(self) -> Self::Item<S> { #body }
        }
    }

    fn gen_from_item(&self, paths: &DerivePaths) -> TokenStream {
        let model = &paths.model;
        let body = self.construct(quote!(Self), self.members().iter()
            .map(|member| quote!(#model::record::Record::<B>::from_item::<S>(item.#member, device))).collect());
        quote! {
            fn from_item<S: #model::record::PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self { #body }
        }
    }
}
