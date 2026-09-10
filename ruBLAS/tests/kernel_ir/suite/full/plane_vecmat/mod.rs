use ruda_kernel::dsl as kernel_dsl;
mod matmul_plane_vecmat {
    use ruda_test_runtime::TestRuntime;
    use ruda_kernel::dsl::client::ComputeClient;
    use rublas::kernel_ir::{
        definition::{MatmulProblem, TilingBlueprint},
        launch::Strategy,
        routines::BlueprintStrategy,
    };

    use crate::suite::test_matmul_strategy;

    fn launch_simple_cyclic(
        client: ComputeClient<TestRuntime>,
        problem: MatmulProblem,
        bp: TilingBlueprint,
    ) {
        test_matmul_strategy(
            client,
            problem,
            Strategy::SimpleVecMat(BlueprintStrategy::Forced(bp)),
        );
    }

    fn launch_double_buffering_cyclic(
        client: ComputeClient<TestRuntime>,
        problem: MatmulProblem,
        bp: TilingBlueprint,
    ) {
        test_matmul_strategy(
            client,
            problem,
            Strategy::DoubleVecMat(BlueprintStrategy::Forced(bp)),
        );
    }

    include!("algorithm.rs");
}
