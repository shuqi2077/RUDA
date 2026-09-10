use ruda_kernel::dsl as kernel_dsl;
pub mod test_case;

macro_rules! testgen_reduce {
    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
        strategy: $strategy:expr,
    ) => {
        use ruda_kernel::dsl::zspace::Shape;
        use ruda_kernel::dsl::zspace::Strides;
        use ruda_kernel::dsl::zspace::shape;
        use ruda_kernel::dsl::zspace::strides;

        type TestDType = $dtype;

        fn test_shape() -> Shape {
            $shape
        }
        fn test_strides() -> Strides {
            $strides
        }
        fn test_axis() -> Option<usize> {
            $axis
        }

        fn test_strategy() -> ReduceStrategy {
            $strategy
        }

        mod reduce_dim {
            use super::*;
            include!("reduce_dim.rs");
        }

    };

    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
    ) => {
        mod reduce_shared {
            use ruda_kernel::dsl::zspace::Shape;
            use ruda_kernel::dsl::zspace::Strides;
            use ruda_kernel::dsl::zspace::shape;
            use ruda_kernel::dsl::zspace::strides;

            type TestDType = $dtype;

            fn test_shape() -> Shape {
                $shape
            }
            fn test_strides() -> Strides {
                $strides
            }

            include!("reduce_shared.rs");
        }
    };

    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
    ) => {
        mod parallel_vectorization_enabled {
            use ruprim::reduce::launch::VectorizationStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                vectorization_strategy: VectorizationStrategy {
                    parallel_output_vectorization: true,
                },
            );
        }

        mod parallel_vectorization_disabled {
            use ruprim::reduce::launch::VectorizationStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                vectorization_strategy: VectorizationStrategy {
                    parallel_output_vectorization: false,
                },
            );
        }
    };

    (
        dtype: $dtype:ty,
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
        vectorization_strategy: $vectorization_strategy:expr,
    ) => {
        use ruprim::reduce::{ReduceStrategy, routines::BlueprintStrategy, launch::RoutineStrategy};
        use ruprim::reduce::routines::PlaneMergeStrategy;

        /// Ruda-routine tests are expensive on CPU and can stall CI, so they
        /// are excluded from light test suite
        #[cfg(feature = "heavy")]
        mod full_ruda {
            use super::*;
            use ruprim::reduce::routines::ruda::RudaStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy {
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Ruda(
                        BlueprintStrategy::Inferred(RudaStrategy{ use_planes: false })
                    ),
                },
            );
        }

        #[cfg(feature = "heavy")]
        mod full_ruda_plane {
            use super::*;
            use ruprim::reduce::routines::ruda::RudaStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy {
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Ruda(
                        BlueprintStrategy::Inferred(RudaStrategy{ use_planes: true })
                    ),
                },
            );
        }

        /// The goal of that test is to limit the size of a ruda to `8` to validate multiple rudas
        /// arithmetic.
        ///
        /// With this test, we can't have `use_planes` to true since the `ruda_dim.x !=
        /// plane_size`.
        #[cfg(feature = "heavy")]
        mod full_ruda_single_plane {
            use super::*;
            use ruprim::reduce::{routines::RudaBlueprint, {BoundChecks, IdleMode}};
            use ruda_kernel::dsl::prelude::RudaDim;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy {
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Ruda(
                        BlueprintStrategy::Forced(
                            RudaBlueprint {
                                ruda_idle: IdleMode::Terminate,
                                bound_checks: BoundChecks::Mask,
                                num_shared_accumulators: 8,
                                use_planes: false,
                            },
                            RudaDim::new_2d(8, 1),
                        )
                    ),
                },
            );
        }

        /// The goal of that test is to limit the size of a ruda to `plane_size` to validate multiple planes
        /// arithmetic.
        mod full_plane_single_plane {
            use super::*;
            use ruprim::reduce::{routines::PlaneReduceBlueprint, {BoundChecks, IdleMode}};
            use ruda_kernel::dsl::prelude::RudaDim;

            mod plane_size_32 {
                use super::*;

                testgen_reduce!(
                    dtype: $dtype,
                    shape: $shape,
                    strides: $strides,
                    axis: $axis,
                    strategy: ReduceStrategy {
                        vectorization: $vectorization_strategy,
                        routine: RoutineStrategy::Plane(
                            BlueprintStrategy::Forced(
                                PlaneReduceBlueprint {
                                    plane_idle: IdleMode::Terminate,
                                    bound_checks: BoundChecks::Mask,
                                    plane_merge_strategy: PlaneMergeStrategy::Lazy,
                                    plane_dim_ceil: true,
                                },
                                RudaDim::new_2d(32, 2),
                            )
                        ),
                    },
                );
            }

            mod plane_size_64 {
                use super::*;

                testgen_reduce!(
                    dtype: $dtype,
                    shape: $shape,
                    strides: $strides,
                    axis: $axis,
                    strategy: ReduceStrategy {
                        vectorization: $vectorization_strategy,
                        routine: RoutineStrategy::Plane(
                            BlueprintStrategy::Forced(
                                PlaneReduceBlueprint {
                                    plane_idle: IdleMode::Terminate,
                                    bound_checks: BoundChecks::Mask,
                                    plane_merge_strategy: PlaneMergeStrategy::Lazy,
                                    plane_dim_ceil: true,
                                },
                                RudaDim::new_2d(64, 2),
                            )
                        ),
                    },
                );
            }
        }

        mod full_plane {
            use super::*;
            use ruprim::reduce::routines::plane::PlaneStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy {
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Plane(
                        BlueprintStrategy::Inferred(PlaneStrategy{ independent: false })
                    ),
                },
            );
        }

        mod full_plane_unit {
            use super::*;
            use ruprim::reduce::routines::plane::PlaneStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy {
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Plane(
                        BlueprintStrategy::Inferred(PlaneStrategy{ independent: true })
                    ),
                },
            );
        }

        mod full_unit {
            use super::*;
            use ruprim::reduce::routines::unit::UnitStrategy;

            testgen_reduce!(
                dtype: $dtype,
                shape: $shape,
                strides: $strides,
                axis: $axis,
                strategy: ReduceStrategy {
                    vectorization: $vectorization_strategy,
                    routine: RoutineStrategy::Unit(
                        BlueprintStrategy::Inferred(UnitStrategy)
                    ),
                },
            );
        }
    };

    (
        shape: $shape:expr,
        strides: $strides:expr,
        axis: $axis:expr,
    ) => {
        mod f32 {
            testgen_reduce!(
                dtype: f32,
                shape: $shape,
                strides: $strides,
                axis: $axis,
            );
        }
        mod f16 {
            testgen_reduce!(
                dtype: half::f16,
                shape: $shape,
                strides: $strides,
                axis: $axis,
            );
        }
    };
    (
        shape: $shape:expr,
        strides: $strides:expr,
    ) => {
        mod f32 {
            testgen_reduce!(
                dtype: f32,
                shape: $shape,
                strides: $strides,
            );
        }
    };
}

mod reduce_dim {
    mod vector_small {
        testgen_reduce!(
            shape: shape![22],
            strides: strides![1],
            axis: Some(0),
        );
    }

    mod vector_large {
        testgen_reduce!(
            shape: shape![1024],
            strides: strides![1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_small {
        testgen_reduce!(
            shape: shape![4, 8],
            strides: strides![8, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_small {
        testgen_reduce!(
            shape: shape![4, 8],
            strides: strides![8, 1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_large {
        testgen_reduce!(
            shape: shape![8, 256],
            strides: strides![256, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_large {
        testgen_reduce!(
            shape: shape![8, 256],
            strides: strides![256, 1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_xlarge {
        testgen_reduce!(
            shape: shape![64, 1024],
            strides: strides![1024, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_xlarge {
        testgen_reduce!(
            shape: shape![64, 1024],
            strides: strides![1024, 1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_xxlarge {
        testgen_reduce!(
            shape: shape![64*4, 1024*4],
            strides: strides![1024*4, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_xxlarge {
        testgen_reduce!(
            shape: shape![64*4, 1024*4],
            strides: strides![1024*4, 1],
            axis: Some(0),
        );
    }

    mod parallel_matrix_large_odd_batch {
        testgen_reduce!(
            shape: shape![33, 1024],
            strides: strides![1024, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_large_odd_batch {
        testgen_reduce!(
            shape: shape![33, 1024],
            strides: strides![1024, 1],
            axis: Some(0),
        );
    }

    mod parallel_rank_three_tensor {
        testgen_reduce!(
            shape: shape![16, 16, 16],
            strides: strides![1, 256, 16],
            axis: Some(0),
        );
    }

    mod perpendicular_rank_three_tensor {
        testgen_reduce!(
            shape: shape![16, 16, 16],
            strides: strides![1, 256, 16],
            axis: Some(1),
        );
    }

    mod parallel_rank_three_tensor_unexact_shape {
        testgen_reduce!(
            shape: shape![11, 12, 13],
            strides: strides![156, 13, 1],
            axis: Some(2),
        );
    }

    mod parallel_rank_four_tensor {
        testgen_reduce!(
            shape: shape![4, 4, 4, 4],
            strides: strides![1, 16, 64, 4],
            axis: Some(0),
        );
    }

    mod perpendicular_rank_four_tensor {
        testgen_reduce!(
            shape: shape![4, 4, 4, 4],
            strides: strides![1, 16, 64, 4],
            axis: Some(1),
        );
    }

    mod decreasing_rank_four_tensor {
        testgen_reduce!(
            shape: shape![4, 4, 4, 4],
            strides: strides![64, 16, 4, 1],
            axis: Some(3),
        );
    }

    mod parallel_matrix_with_jumps {
        testgen_reduce!(
            shape: shape![256, 256],
            strides: strides![512, 1],
            axis: Some(1),
        );
    }

    mod perpendicular_matrix_with_jumps {
        testgen_reduce!(
            shape: shape![256, 256],
            strides: strides![512, 1],
            axis: Some(0),
        );
    }

    mod broadcast_slice_0 {
        testgen_reduce!(
            shape: shape![4, 32],
            strides: strides![0, 1],
            axis: Some(0),
        );
    }
}

mod reduce {
    mod vector_small {
        testgen_reduce!(
            shape: shape![22],
            strides: strides![1],
        );
    }

    mod vector_large {
        testgen_reduce!(
            shape: shape![1024],
            strides: strides![1],
        );
    }

    mod matrix_small {
        testgen_reduce!(
            shape: shape![4, 8],
            strides: strides![8, 1],
        );
    }

    mod matrix_large {
        testgen_reduce!(
            shape: shape![8, 256],
            strides: strides![256, 1],
        );
    }

    mod rank_three_tensor {
        testgen_reduce!(
            shape: shape![16, 16, 16],
            strides: strides![1, 256, 16],
        );
    }

    mod rank_three_tensor_unexact_shape {
        testgen_reduce!(
            shape: shape![11, 12, 13],
            strides: strides![156, 13, 1],
        );
    }

    mod rank_four_tensor {
        testgen_reduce!(
            shape: shape![4, 4, 4, 4],
            strides: strides![64, 16, 4, 1],
        );
    }

    // TODO not well supported in test-utils
    // mod matrix_with_jumps {
    //     testgen_reduce!(
    //         shape: shape![8, 8],
    //         strides: strides![64, 1],
    //     );
    // }

    mod broadcast_slice_0 {
        testgen_reduce!(
            shape: shape![4, 32],
            strides: strides![0, 1],
        );
    }
}
