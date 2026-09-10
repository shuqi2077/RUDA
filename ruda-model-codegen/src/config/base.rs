use super::ConfigAnalyzerFactory;
use quote::quote;

pub fn derive_impl(item: &syn::DeriveInput, paths: &crate::DerivePaths) -> proc_macro2::TokenStream {
    let factory = ConfigAnalyzerFactory::new();
    let analyzer = factory.create_analyzer(item);

    let constructor = analyzer.gen_new_fn();
    let builders = analyzer.gen_builder_fns();
    let serde = analyzer.gen_serde_impl(paths);
    let clone = analyzer.gen_clone_impl();
    let display = analyzer.gen_display_impl(paths);
    let config_impl = analyzer.gen_config_impl(paths);

    quote! {
        #config_impl
        #constructor
        #builders
        #serde
        #clone
        #display
    }
}
