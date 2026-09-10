use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_test_runtime::TestRuntime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::dsl::ir::AddressType;
use ruda_kernel::dsl::ir::BarrierLevel;
use ruda_kernel::dsl::ir::OpaqueType;
use ruda_kernel::dsl::ir::SemanticType;
use ruda_kernel::dsl::zspace::shape;
use rublas::kernel_ir::definition::{MatmulElems, MatmulGlobalElems, MatmulProblem};
use ruda_kernel::tiling::MatrixLayout;

pub(crate) fn client() -> ComputeClient<TestRuntime> {
    TestRuntime::client(&Default::default())
}

pub(crate) fn f16_elems() -> MatmulGlobalElems {
    use ruda_kernel::dsl::frontend::RudaPrimitive;
    MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
}

pub(crate) fn f32_elems() -> MatmulGlobalElems {
    use ruda_kernel::dsl::frontend::RudaPrimitive;
    MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
}

pub(crate) fn square(dim: usize, elems: MatmulGlobalElems) -> MatmulProblem {
    rect(dim, dim, dim, elems)
}

pub(crate) fn rect(m: usize, n: usize, k: usize, elems: MatmulGlobalElems) -> MatmulProblem {
    rect_with_layouts(
        m,
        n,
        k,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        elems,
    )
}

pub(crate) fn rect_with_layouts(
    m: usize,
    n: usize,
    k: usize,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    elems: MatmulGlobalElems,
) -> MatmulProblem {
    MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![1],
        shape![1],
        lhs_layout,
        rhs_layout,
        MatrixLayout::RowMajor,
        None,
        None,
        elems,
        AddressType::U32,
    )
}
