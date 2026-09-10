use quote::format_ident;
use proc_macro_crate::{FoundCrate, crate_name};
use std::cell::LazyCell;
use syn::Path;

#[allow(clippy::declare_interior_mutable_const)]
const CORE_PATH: LazyCell<Path> = LazyCell::new(|| {
    resolve_core_path(|name| crate_name(name).ok())
});

fn resolve_core_path(mut lookup: impl FnMut(&str) -> Option<FoundCrate>) -> Path {
    if let Some(found) = lookup("ruda-kernel") {
        let name = match found {
            FoundCrate::Itself => "ruda_kernel".into(),
            FoundCrate::Name(name) => name,
        };
        let ident = format_ident!("{name}");
        return syn::parse_quote!(::#ident::dsl);
    }
    syn::parse_quote!(kernel_dsl)
}
#[allow(clippy::declare_interior_mutable_const)]
const FRONTEND_PATH: LazyCell<Path> = LazyCell::new(|| {
    let mut path = core_path();
    path.segments.push(format_ident!("frontend").into());
    path
});
#[allow(clippy::declare_interior_mutable_const)]
const PRELUDE_PATH: LazyCell<Path> = LazyCell::new(|| {
    let mut path = core_path();
    path.segments.push(format_ident!("prelude").into());
    path
});
#[allow(clippy::declare_interior_mutable_const)]
const TUNE_PATH: LazyCell<Path> = LazyCell::new(|| {
    let mut path = core_path();
    path.segments.push(format_ident!("tune").into());
    path
});

pub fn frontend_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    FRONTEND_PATH.clone()
}

pub fn prelude_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    PRELUDE_PATH.clone()
}

pub fn tune_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    TUNE_PATH.clone()
}

pub fn core_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    CORE_PATH.clone()
}

pub fn core_type(ty: &str) -> Path {
    let mut path = core_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}

pub fn frontend_type(ty: &str) -> Path {
    let mut path = frontend_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}

pub fn prelude_type(ty: &str) -> Path {
    let mut path = prelude_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}

pub fn tune_type(ty: &str) -> Path {
    let mut path = tune_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_dependency_uses_kernel_owner() {
        let path = resolve_core_path(|name| (name == "ruda-kernel")
            .then(|| FoundCrate::Name("ruda_kernel".into())));
        assert_eq!(path, syn::parse_quote!(::ruda_kernel::dsl));
    }

    #[test]
    fn renamed_native_dependency_is_resolved() {
        let path = resolve_core_path(|name| (name == "ruda-kernel")
            .then(|| FoundCrate::Name("renamed_kernel".into())));
        assert_eq!(path, syn::parse_quote!(::renamed_kernel::dsl));
    }

    #[test]
    fn kernel_itself_uses_existing_self_alias() {
        let path = resolve_core_path(|name| (name == "ruda-kernel")
            .then_some(FoundCrate::Itself));
        assert_eq!(path, syn::parse_quote!(::ruda_kernel::dsl));
    }

    #[test]
    fn native_owner_uses_declared_dependency() {
        let path = resolve_core_path(|_| Some(FoundCrate::Name("ruda_kernel".into())));
        assert_eq!(path, syn::parse_quote!(::ruda_kernel::dsl));
    }

    #[test]
    fn existing_local_alias_remains_a_fallback() {
        assert_eq!(resolve_core_path(|_| None), syn::parse_quote!(kernel_dsl));
    }

}
