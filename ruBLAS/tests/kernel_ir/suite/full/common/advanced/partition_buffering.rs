pub mod pb1 {
    use super::*;
    use rublas::kernel_ir::components::stage::PartitionBuffering;

    fn partition_buffering() -> PartitionBuffering {
        PartitionBuffering::Single
    }

    include!("../problem/layouts.rs");
}

pub mod pb2 {
    use super::*;
    use rublas::kernel_ir::components::stage::PartitionBuffering;

    fn partition_buffering() -> PartitionBuffering {
        PartitionBuffering::Double
    }

    include!("../problem/layouts.rs");
}
