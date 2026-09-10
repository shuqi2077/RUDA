use crate::{DerivePaths, shared::enum_variant::EnumVariant};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Generics, parse_quote};

pub(super) fn generate(
    item_name: &Ident,
    generics: &Generics,
    variants: &[EnumVariant],
    paths: &DerivePaths,
) -> TokenStream {
    let model = &paths.model;
    let serde_path = paths.serde;
    let serialized_name = item_name.to_string();
    let mut fields = TokenStream::new();
    let mut tagged_arms = TokenStream::new();
    let mut probes = TokenStream::new();
    let mut serde_bounds = TokenStream::new();
    let mut deserialize_generics = generics.clone();
    for variant in variants {
        let name = &variant.ident;
        let ty = &variant.ty;
        let item = quote!(<#ty as #model::record::Record<B>>::Item<S>);
        fields.extend(quote!(#name(#item),));
        tagged_arms.extend(quote! {
            __RudaTaggedRecord::#name(value) => #item_name::#name(value),
        });
        probes.extend(quote! {
            if let Ok(Some(value)) = sequence.next_element::<#item>() {
                return Ok(#item_name::#name(value));
            }
        });
        serde_bounds.extend(quote!(#item: #model::serde::de::DeserializeOwned,));
        deserialize_generics.make_where_clause().predicates.push(parse_quote! {
            #item: #model::serde::de::DeserializeOwned
        });
    }
    let serde_bound = serde_bounds.to_string();
    deserialize_generics.params.insert(0, parse_quote!('__ruda_de));
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    let (deserialize_impl_generics, _, deserialize_where_clause) = deserialize_generics.split_for_impl();

    quote! {
        const _: () = {
            #[derive(#model::serde::Deserialize)]
            #[serde(crate = #serde_path, rename = #serialized_name, bound = #serde_bound)]
            enum __RudaTaggedRecord #impl_generics #where_clause {
                #fields
            }

            struct __RudaRecordEnumVisitor #impl_generics #where_clause {
                marker: ::core::marker::PhantomData<#item_name #type_generics>,
            }

            impl #deserialize_impl_generics #model::serde::de::Visitor<'__ruda_de>
                for __RudaRecordEnumVisitor #type_generics #deserialize_where_clause
            {
                type Value = #item_name #type_generics;

                fn expecting(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    formatter.write_str(#serialized_name)
                }

                fn visit_newtype_struct<__RudaDeserializer>(
                    self,
                    deserializer: __RudaDeserializer,
                ) -> Result<Self::Value, __RudaDeserializer::Error>
                where
                    __RudaDeserializer: #model::serde::Deserializer<'__ruda_de>,
                {
                    let tagged = <__RudaTaggedRecord #type_generics as
                        #model::serde::Deserialize<'__ruda_de>>::deserialize(deserializer)?;
                    Ok(match tagged { #tagged_arms })
                }

                fn visit_seq<__RudaSequence>(
                    self,
                    mut sequence: __RudaSequence,
                ) -> Result<Self::Value, __RudaSequence::Error>
                where
                    __RudaSequence: #model::serde::de::SeqAccess<'__ruda_de>,
                {
                    #probes
                    Err(<__RudaSequence::Error as #model::serde::de::Error>::custom("No variant match"))
                }
            }

            impl #deserialize_impl_generics #model::serde::Deserialize<'__ruda_de>
                for #item_name #type_generics #deserialize_where_clause
            {
                fn deserialize<__RudaDeserializer>(
                    deserializer: __RudaDeserializer,
                ) -> Result<Self, __RudaDeserializer::Error>
                where
                    __RudaDeserializer: #model::serde::Deserializer<'__ruda_de>,
                {
                    deserializer.deserialize_newtype_struct(
                        "__ruda_record_enum_v1",
                        __RudaRecordEnumVisitor {
                            marker: ::core::marker::PhantomData::<#item_name #type_generics>,
                        },
                    )
                }
            }
        };
    }
}
