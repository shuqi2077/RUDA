#[macro_export]
macro_rules! testgen_convolution_partition_buffering {
    ($algorithm: expr, $dtypes: expr, $tiling_scheme: expr, $swizzle: expr) => {
        use rublas::kernel_ir::components::stage::PartitionBuffering;

        $crate::testgen_convolution_problem!(
            $algorithm,
            $dtypes,
            $tiling_scheme,
            $swizzle,
            PartitionBuffering::Single
        );
    };
}
