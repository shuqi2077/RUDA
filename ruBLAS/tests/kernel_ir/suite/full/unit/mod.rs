use ruda_kernel::dsl as kernel_dsl;
mod matmul_unit {
    use ruda_test_runtime::TestRuntime;
    use ruda_kernel::dsl::client::ComputeClient;
    use rublas::kernel_ir::{
        definition::{MatmulProblem, TilingBlueprint},
        launch::{Strategy, test_only::TestStrategy},
        routines::BlueprintStrategy,
    };

    use crate::suite::{test_matmul_strategy, test_matmul_test_strategy};

    fn launch_simple(c: ComputeClient<TestRuntime>, p: MatmulProblem, bp: TilingBlueprint) {
        test_matmul_strategy(c, p, Strategy::SimpleUnit(BlueprintStrategy::Forced(bp)));
    }

    fn launch_double_buffering(
        c: ComputeClient<TestRuntime>,
        p: MatmulProblem,
        bp: TilingBlueprint,
    ) {
        test_matmul_strategy(c, p, Strategy::DoubleUnit(BlueprintStrategy::Forced(bp)));
    }

    fn launch_interleaved(c: ComputeClient<TestRuntime>, p: MatmulProblem, bp: TilingBlueprint) {
        test_matmul_test_strategy(
            c,
            p,
            TestStrategy::Interleaved(BlueprintStrategy::Forced(bp)),
        );
    }

    include!("algorithm.rs");
}
