mod s1x1x1 {
    use super::*;
    use rublas::kernel_ir::definition::TilingSchemeBuilder;
    use ruda_kernel::tiling::StageSize;

    fn stage(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_stage_size(StageSize { m: 1, n: 1, k: 1 })
    }

    include!("../../common/advanced/specialized.rs");
}

mod s1x2x1 {
    use super::*;
    use rublas::kernel_ir::definition::TilingSchemeBuilder;
    use ruda_kernel::tiling::StageSize;

    fn stage(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_stage_size(StageSize { m: 1, n: 2, k: 1 })
    }

    include!("../../common/advanced/specialized.rs");
}
