use super::{display, record::ModuleRecordCodegen};
use crate::{
    module::generics::{GenericKind, ModuleGenerics},
    shared::generics::GenericsHelper,
};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Attribute, Generics, parse_quote};

/// Basic trait to be implemented for Module generation.
pub(crate) trait ModuleCodegen {
    type RecordCodegen: ModuleRecordCodegen;

    fn gen_num_params(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_visit(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_collect_devices(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_to_device(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_fork(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_map(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_valid(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_from_inner(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_into_record(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_load_record(&self, _paths: &crate::DerivePaths) -> TokenStream;
    fn gen_clone(&self) -> TokenStream;

    fn record_codegen(self) -> Self::RecordCodegen;

    fn gen_display(&self, _paths: &crate::DerivePaths) -> TokenStream;

    fn module_generics(&self) -> &ModuleGenerics;
}

pub(crate) fn generate_module_standard<Codegen: ModuleCodegen>(
    ast: &syn::DeriveInput,
    codegen: Codegen,
    paths: &crate::DerivePaths,
) -> TokenStream {
    let model = &paths.model;
    let name = &ast.ident;

    let generics = GenericsParser::from_ast(&ast.generics, codegen.module_generics(), paths);

    let display_fn = display::display_fn(ast, paths);
    let attributes_fn = codegen.gen_display(paths);
    let num_params_fn = codegen.gen_num_params(paths);
    let visit = codegen.gen_visit(paths);
    let map_mut = codegen.gen_map(paths);
    let collect_devices = codegen.gen_collect_devices(paths);
    let to_device = codegen.gen_to_device(paths);
    let fork = codegen.gen_fork(paths);
    let valid_fn = codegen.gen_valid(paths);
    let from_inner_fn = codegen.gen_from_inner(paths);
    let into_record_fn = codegen.gen_into_record(paths);
    let load_record_fn = codegen.gen_load_record(paths);
    let clone_fn = codegen.gen_clone();

    let record = codegen.record_codegen();
    let record_name = Ident::new(format!("{name}Record").as_str(), name.span());
    let (record_type, record_generics) = record.gen_record_type(&record_name, &generics.module, paths);

    let (generics_module, generics_ty_module, generics_where_module) =
        generics.module.split_for_impl();
    let (generics_module_autodiff, generics_ty_module_autodiff, generics_where_module_autodiff) =
        generics.module_autodiff.split_for_impl();
    let (generics_module_has_autodiff, _generics_ty, generics_where_module_has_autodiff) =
        generics.module_has_autodiff.split_for_impl();
    let (_, generics_ty_record, _) = record_generics.split_for_impl();

    let generics_ty_inner_module = generics.inner_module_ty;
    let generics_ty_train_module = generics.train_module_ty;
    let generics_ty_train_inner_module = generics.train_inner_ty;

    let mut codegen = quote! {
        impl #generics_module #model::module::Module<B> for #name #generics_ty_module #generics_where_module {
            type Record = #record_name #generics_ty_record;

            #load_record_fn
            #into_record_fn

            #num_params_fn

            #visit
            #map_mut

            #collect_devices
            #to_device
            #fork

        }

        impl #generics_module_autodiff #model::module::AutodiffModule<B> for #name #generics_ty_module_autodiff #generics_where_module_autodiff
        {
            type InnerModule=#name<B::InnerBackend, #generics_ty_inner_module>;

            #valid_fn

            #from_inner_fn
        }

        impl #generics_module_has_autodiff #model::module::HasAutodiffModule<B> for #name<B::InnerBackend, #generics_ty_train_module> #generics_where_module_has_autodiff
        {
            type TrainModule=#name<B, #generics_ty_train_inner_module>;
        }

        impl #generics_module core::fmt::Display for #name #generics_ty_module #generics_where_module {
            #display_fn
        }


        impl #generics_module #model::module::ModuleDisplayDefault for #name #generics_ty_module #generics_where_module {
            #attributes_fn

            fn num_params(&self) -> usize {
                #model::module::Module::num_params(self)
            }
        }

        impl #generics_module Clone for #name #generics_ty_module #generics_where_module {
            #clone_fn
        }

        #record_type
    };

    if !has_custom_display(&ast.attrs) {
        codegen.extend(quote! {
            impl #generics_module #model::module::ModuleDisplay for #name #generics_ty_module #generics_where_module {

            }
        });
    }

    codegen
}

// TODO: wait that means nothing is persistent... (empty!)

// When there is no backend in the generic parameter, the type is considered as a constant.
pub(crate) fn generate_module_const(ast: &syn::DeriveInput, paths: &crate::DerivePaths) -> TokenStream {
    let model = &paths.model;
    let name = &ast.ident;
    let (generics, generics_ty, generics_where) = ast.generics.split_for_impl();

    let backend: syn::Generics = parse_quote! { <B: #model::tensor::backend::Backend >};
    let backend_ad: syn::Generics = parse_quote! { <B: #model::tensor::backend::AutodiffBackend >};

    let mut generics_module = ast.generics.clone();
    let mut generics_module_autodiff = ast.generics.clone();

    for param in backend.params.into_iter() {
        generics_module.params.push(param);
    }
    for param in backend_ad.params.into_iter() {
        generics_module_autodiff.params.push(param);
    }
    let (generics_module, _, _) = generics_module.split_for_impl();
    let (generics_module_ad, _, _) = generics_module_autodiff.split_for_impl();

    let display_fn = display::display_fn(ast, paths);
    let attributes_fn = display::attributes_fn(ast, paths);

    let mut codegen = quote! {
        impl #generics_module #model::module::Module<B> for #name #generics_ty #generics_where {
            #model::empty!(module);
        }

        impl #generics_module_ad #model::module::AutodiffModule<B>
            for #name #generics_ty #generics_where {
            #model::empty!(ad_module, #name #generics_ty);
        }

        impl #generics core::fmt::Display for #name #generics_ty #generics_where {
            #display_fn
        }


        impl #generics #model::module::ModuleDisplayDefault for #name #generics_ty #generics_where {
            #attributes_fn
        }

    };

    if !has_custom_display(&ast.attrs) {
        codegen.extend(quote! {
            impl  #generics #model::module::ModuleDisplay for #name #generics_ty #generics_where {

            }
        });
    }

    codegen
}

struct GenericsParser {
    module: Generics,
    module_autodiff: Generics,
    module_has_autodiff: Generics,
    inner_module_ty: TokenStream,
    train_module_ty: TokenStream,
    train_inner_ty: TokenStream,
}

impl GenericsParser {
    fn from_ast(generics: &Generics, module_generics: &ModuleGenerics, paths: &crate::DerivePaths) -> Self {
        let model = &paths.model;
        let mut module = GenericsHelper::new(generics.clone());
        let mut module_autodiff = GenericsHelper::new(generics.clone());
        let mut module_has_autodiff = GenericsHelper::new(generics.clone());

        let backend_trait = module.fetch_backend_trait();

        module_autodiff.add_predicate(parse_quote! {
                B: #model::tensor::backend::AutodiffBackend
        });

        module_autodiff.add_predicate(parse_quote! {
                <B as #model::tensor::backend::AutodiffBackend>::InnerBackend: #backend_trait
        });

        module_has_autodiff.add_predicate(parse_quote! {
                B: #model::tensor::backend::AutodiffBackend
        });

        module_has_autodiff.add_predicate(parse_quote! {
                <B as #model::tensor::backend::AutodiffBackend>::InnerBackend: #backend_trait
        });

        let mut generics_names_except_backend = quote! {};
        let mut train_generics_names_except_backend = quote! {};
        let mut train_inner_generics_names_except_backend = quote! {};

        module
        .types()
        .into_iter()
        .filter(|ident| ident != "B")
        .for_each(|ident| {
            // By default, require module bound
            let mut requires_module_bound = true;
            let mut generic_kind = None;
            if !module_generics.is_empty() {
                generic_kind = module_generics.get_generic_kind(&ident);
                let has_module_bound = matches!(generic_kind, Some(GenericKind::Module));
                let is_unbounded = matches!(generic_kind, Some(GenericKind::Plain));

                requires_module_bound = has_module_bound || is_unbounded;
            }

            if requires_module_bound {
                module.add_predicate(
                    parse_quote! {
                        #ident: #model::module::Module<B>
                    }
                );

                module.add_predicate(
                    parse_quote! {
                        #ident: #model::module::ModuleDisplay
                    }
                );

                module_autodiff.add_predicate(
                    parse_quote! {
                        #ident: #model::module::AutodiffModule<B>
                    }
                );

                module_autodiff.add_predicate(
                    parse_quote! {
                        <#ident as #model::module::AutodiffModule<B>>::InnerModule: #model::module::Module<B::InnerBackend>
                    }
                );

                module_autodiff.add_predicate(
                    parse_quote! {
                        <#ident as #model::module::AutodiffModule<B>>::InnerModule: #model::module::ModuleDisplay
                    }
                );

                generics_names_except_backend.extend(quote! { <#ident as #model::module::AutodiffModule<B>>::InnerModule, });

                module_autodiff.add_predicate(
                    parse_quote! {
                        #ident: #model::module::ModuleDisplay
                    }
                );

                module_has_autodiff.add_predicate(
                    parse_quote! {
                        #ident: #model::module::Module<B::InnerBackend>
                    }
                );

                module_has_autodiff.add_predicate(
                    parse_quote! {
                        #ident: #model::module::ModuleDisplay
                    }
                );

                module_has_autodiff.add_predicate(
                    parse_quote! {
                        #ident: #model::module::HasAutodiffModule<B>
                    }
                );

                module_has_autodiff.add_predicate(
                    parse_quote! {
                        #ident::TrainModule: #model::module::ModuleDisplay
                    }
                );
                train_generics_names_except_backend.extend(quote! { #ident, });
                train_inner_generics_names_except_backend.extend(quote! { #ident::TrainModule, });
            }
            else {
                // Add required bounds to impl
                if let Some(GenericKind::Skip) = generic_kind {
                    module.add_predicate(
                        parse_quote! {
                            #ident: Clone + core::fmt::Debug + Send
                        }
                    );
                    module_autodiff.add_predicate(
                        parse_quote! {
                            #ident: Clone + core::fmt::Debug + Send
                        }
                    );
                    module_has_autodiff.add_predicate(
                        parse_quote! {
                            #ident: Clone + core::fmt::Debug + Send
                        }
                    );
                }

                // Pass through
                generics_names_except_backend.extend(quote! { #ident, });
                train_generics_names_except_backend.extend(quote! { #ident, });
                train_inner_generics_names_except_backend.extend(quote! { #ident, });
            }

        });

        module.consts().into_iter().for_each(|ident| {
            generics_names_except_backend.extend(quote! { #ident, });
            train_generics_names_except_backend.extend(quote! { #ident, });
            train_inner_generics_names_except_backend.extend(quote! { #ident, });
        });

        Self {
            module: module.generics,
            module_autodiff: module_autodiff.generics,
            module_has_autodiff: module_has_autodiff.generics,
            inner_module_ty: generics_names_except_backend,
            train_module_ty: train_generics_names_except_backend,
            train_inner_ty: train_inner_generics_names_except_backend,
        }
    }
}

fn has_custom_display(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("module")
            && attr
                .parse_nested_meta(|meta| {
                    if meta.path.is_ident("custom_display") {
                        Ok(())
                    } else {
                        Err(meta.error("unsupported attribute"))
                    }
                })
                .is_ok()
    })
}
