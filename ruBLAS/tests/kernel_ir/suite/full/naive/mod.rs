use ruda_kernel::dsl as kernel_dsl;
mod f16_ty {
    use ruda_kernel::dsl::frontend::RudaPrimitive;
    use rublas::kernel_ir::definition::{MatmulElems, MatmulGlobalElems};

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
    }

    include!("suite.rs");
}

mod f32_ty {
    use ruda_kernel::dsl::frontend::RudaPrimitive;
    use rublas::kernel_ir::definition::{MatmulElems, MatmulGlobalElems};

    fn elems() -> MatmulGlobalElems {
        MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
    }

    include!("suite.rs");
}
