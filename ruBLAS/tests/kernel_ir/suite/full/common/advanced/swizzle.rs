mod no_swizzle {
    use super::*;
    use ruda_kernel::tiling::SwizzleModes;
    use ruda_kernel::tiling::stage::SwizzleMode;

    fn swizzle() -> SwizzleModes {
        SwizzleModes {
            lhs: SwizzleMode::None,
            rhs: SwizzleMode::None,
            ..Default::default()
        }
    }

    include!("hyperruda.rs");
}

mod b32 {
    use super::*;
    use ruda_kernel::tiling::SwizzleModes;
    use ruda_kernel::tiling::stage::SwizzleMode;

    fn swizzle() -> SwizzleModes {
        SwizzleModes {
            lhs: SwizzleMode::B32,
            rhs: SwizzleMode::B32,
            ..Default::default()
        }
    }

    include!("hyperruda.rs");
}

mod b64 {
    use super::*;
    use ruda_kernel::tiling::SwizzleModes;
    use ruda_kernel::tiling::stage::SwizzleMode;

    fn swizzle() -> SwizzleModes {
        SwizzleModes {
            lhs: SwizzleMode::B64,
            rhs: SwizzleMode::B64,
            ..Default::default()
        }
    }

    include!("hyperruda.rs");
}

mod b128 {
    use super::*;
    use ruda_kernel::tiling::SwizzleModes;
    use ruda_kernel::tiling::stage::SwizzleMode;

    fn swizzle() -> SwizzleModes {
        SwizzleModes {
            lhs: SwizzleMode::B128,
            rhs: SwizzleMode::B128,
            ..Default::default()
        }
    }

    include!("hyperruda.rs");
}
