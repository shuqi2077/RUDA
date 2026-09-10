mod mm {
    use super::*;
    use rublas::kernel_ir::components::global::{InputLoadFlow, LoadFlows};

    fn specialization() -> LoadFlows {
        LoadFlows {
            lhs: InputLoadFlow::MainOnly,
            rhs: InputLoadFlow::MainOnly,
        }
    }

    include!("swizzle.rs");
}

mod ml {
    use super::*;
    use rublas::kernel_ir::components::global::{InputLoadFlow, LoadFlows};

    fn specialization() -> LoadFlows {
        LoadFlows {
            lhs: InputLoadFlow::MainOnly,
            rhs: InputLoadFlow::LoadOnly,
        }
    }

    include!("swizzle.rs");
}

mod lm {
    use super::*;
    use rublas::kernel_ir::components::global::{InputLoadFlow, LoadFlows};

    fn specialization() -> LoadFlows {
        LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::MainOnly,
        }
    }

    include!("swizzle.rs");
}

mod ll {
    use super::*;
    use rublas::kernel_ir::components::global::{InputLoadFlow, LoadFlows};

    fn specialization() -> LoadFlows {
        LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
        }
    }

    include!("swizzle.rs");
}
