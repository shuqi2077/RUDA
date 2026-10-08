use crate::{DerivePaths, shared::enum_variant::map_enum_variant};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Fields, Generics, Type, Visibility, parse_quote};

use super::codegen::RecordItemCodegen;

pub(crate) struct FullEnumRecordItemCodegen {
    name: Ident,
    variants: Vec<syn::Variant>,
    vis: Visibility,
}

pub(super) fn contains_self(ty: &Type, name: &Ident) -> bool {
    match ty {
        Type::Array(ty) => contains_self(&ty.elem, name),
        Type::Slice(ty) => contains_self(&ty.elem, name),
        Type::Ptr(ty) => contains_self(&ty.elem, name),
        Type::Reference(ty) => contains_self(&ty.elem, name),
        Type::Paren(ty) => contains_self(&ty.elem, name),
        Type::Group(ty) => contains_self(&ty.elem, name),
        Type::Tuple(ty) => ty.elems.iter().any(|ty| contains_self(ty, name)),
        Type::Path(ty) => ty.qself.as_ref().is_some_and(|q| contains_self(&q.ty, name)) ||
            (ty.path.leading_colon.is_none() && ty.path.segments.len() == 1 &&
                (ty.path.segments[0].ident == *name || ty.path.segments[0].ident == "Self")) ||
            ty.path.segments.iter().any(|segment| {
                match &segment.arguments {
                    syn::PathArguments::AngleBracketed(args) => args.args.iter().any(|arg| match arg {
                        syn::GenericArgument::Type(ty) => contains_self(ty, name),
                        syn::GenericArgument::AssocType(ty) => contains_self(&ty.ty, name),
                        _ => false,
                    }),
                    syn::PathArguments::Parenthesized(args) => args.inputs.iter().any(|ty| contains_self(ty, name)) ||
                        matches!(&args.output, syn::ReturnType::Type(_, ty) if contains_self(ty, name)),
                    _ => false,
                }
            }),
        _ => false,
    }
}

pub(crate) fn derive_full_enum(ast: &syn::DeriveInput, paths: &DerivePaths) -> TokenStream {
    let mut ast = ast.clone();
    let model = &paths.model;
    let syn::Data::Enum(data) = &ast.data else { unreachable!() };
    let bounds: Vec<syn::WherePredicate> = data.variants.iter().flat_map(|variant| variant.fields.iter())
        .filter(|field| !contains_self(&field.ty, &ast.ident))
        .map(|field| { let ty = &field.ty; parse_quote!(#ty: #model::record::Record<B>) }).collect();
    if !bounds.is_empty() { ast.generics.make_where_clause().predicates.extend(bounds); }
    crate::record::codegen::generate_record::<FullEnumRecordItemCodegen>(&ast, paths)
}

impl RecordItemCodegen for FullEnumRecordItemCodegen {
    fn item_has_payload(&self) -> bool {
        self.variants.iter().any(|variant| !variant.fields.is_empty())
    }

    fn from_ast(ast: &syn::DeriveInput, _paths: &DerivePaths) -> syn::Result<Self> {
        let syn::Data::Enum(data) = &ast.data else {
            return Err(syn::Error::new_spanned(ast, "Expected a record enum"));
        };
        for variant in &data.variants {
            if variant.attrs.iter().any(|attr| attr.path().is_ident("module")) {
                return Err(syn::Error::new_spanned(variant, "Module attributes are not supported for enum variants."));
            }
        }
        Ok(Self { name: ast.ident.clone(), variants: data.variants.iter().cloned().collect(), vis: ast.vis.clone() })
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
        let mut variants = TokenStream::new();
        let mut clone_arms = TokenStream::new();
        let mut serde_bounds = TokenStream::new();
        for variant in &self.variants {
            let name = &variant.ident;
            let types: Vec<_> = variant.fields.iter().map(|field| {
                let ty = &field.ty;
                let item = quote!(<#ty as #model::record::Record<B>>::Item<S>);
                if !contains_self(ty, &self.name) {
                    serde_bounds.extend(quote!(#item: #model::serde::Serialize + #model::serde::de::DeserializeOwned,));
                }
                item
            }).collect();
            let fields = match &variant.fields {
                Fields::Unit => quote!(),
                Fields::Unnamed(_) => quote!((#(#types),*)),
                Fields::Named(fields) => {
                    let names = fields.named.iter().map(|field| field.ident.as_ref().unwrap());
                    quote!({ #(#names: #types),* })
                }
            };
            variants.extend(quote!(#name #fields,));
            let (pattern, cloned) = map_enum_variant(variant, |binding| quote!(Clone::clone(#binding)));
            clone_arms.extend(quote!(Self::#name #pattern => Self::#name #cloned,));
        }
        let serde_bound = serde_bounds.to_string();
        let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
        let clone_body = if self.variants.is_empty() { quote!(match *self {}) }
            else { quote!(match self { #clone_arms }) };
        quote! {
            #[allow(missing_docs)]
            #[derive(#model::serde::Serialize, #model::serde::Deserialize)]
            #[serde(crate = #serde_path, bound = #serde_bound)]
            #vis enum #item_name #impl_generics #where_clause { #variants }

            impl #impl_generics Clone for #item_name #type_generics #where_clause {
                fn clone(&self) -> Self { #clone_body }
            }
        }
    }

    fn gen_into_item(&self, item_name: &Ident, paths: &DerivePaths) -> TokenStream {
        let model = &paths.model;
        let mut arms = TokenStream::new();
        for variant in &self.variants {
            let name = &variant.ident;
            let (pattern, output) = map_enum_variant(variant,
                |binding| quote!(#model::record::Record::<B>::into_item::<S>(#binding)));
            arms.extend(quote!(Self::#name #pattern => #item_name::#name #output,));
        }
        quote! {
            fn into_item<S: #model::record::PrecisionSettings>(self) -> Self::Item<S> {
                match self { #arms }
            }
        }
    }

    fn gen_from_item(&self, paths: &DerivePaths) -> TokenStream {
        let model = &paths.model;
        let item_name = Ident::new(&format!("{}Item", self.name), self.name.span());
        let mut arms = TokenStream::new();
        for variant in &self.variants {
            let name = &variant.ident;
            let (pattern, output) = map_enum_variant(variant,
                |binding| quote!(#model::record::Record::<B>::from_item::<S>(#binding, device)));
            arms.extend(quote!(#item_name::#name #pattern => Self::#name #output,));
        }
        quote! {
            fn from_item<S: #model::record::PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
                match item { #arms }
            }
        }
    }
}
