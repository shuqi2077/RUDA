// Re-export for testgen macros.
pub use test_log;

pub mod all_reduce;
pub mod assign;
pub mod atomic;
pub mod barrier;
pub mod binary;
pub mod branch;
pub mod cluster;
pub mod cmma;
pub mod comparison;
pub mod const_match;
pub mod constants;
pub mod debug;
pub mod different_rank;
pub mod enums;
pub mod file;
pub mod index;
pub mod launch;
pub mod metadata;
pub mod minifloat;
pub mod numeric;
pub mod plane;
pub mod properties;
pub mod saturating;
pub mod sequence;
pub mod slice;
pub mod stream;
pub mod synchronization;
pub mod tensor;
pub mod tensormap;
pub mod to_client;
pub mod topology;
pub mod traits;
pub mod unary;
pub mod unroll;
pub mod vector;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all {
    () => {
        use $crate::dsl::Runtime;

        type FloatType = f32;
        type IntType = i32;
        type UintType = u32;

        $crate::dsl::testgen_float!();
        $crate::dsl::testgen_int!();
        $crate::dsl::testgen_uint!();
        $crate::dsl::testgen_untyped!();
    };
    ($f_def:ident: [$($float:ident),*], $i_def:ident: [$($int:ident),*], $u_def:ident: [$($uint:ident),*]) => {
        use $crate::dsl::Runtime;

        ::paste::paste! {
            $(mod [<$float _ty>] {
                use super::*;

                type FloatType = $float;
                type IntType = $i_def;
                type UintType = $u_def;

                $crate::dsl::testgen_float!();
            })*
            $(mod [<$int _ty>] {
                use super::*;

                type FloatType = $f_def;
                type IntType = $int;
                type UintType = $u_def;

                $crate::dsl::testgen_int!();
            })*
            $(mod [<$uint _ty>] {
                use super::*;

                type FloatType = $f_def;
                type IntType = $i_def;
                type UintType = $uint;

                $crate::dsl::testgen_uint!();
            })*
        }
        $crate::dsl::testgen_untyped!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_float {
    () => {
        ruda_kernel::dsl::testgen_assign!();
        ruda_kernel::dsl::testgen_barrier!();
        ruda_kernel::dsl::testgen_binary!();
        ruda_kernel::dsl::testgen_branch!();
        ruda_kernel::dsl::testgen_different_rank!();
        ruda_kernel::dsl::testgen_index!();
        ruda_kernel::dsl::testgen_launch!();
        ruda_kernel::dsl::testgen_vector!();
        ruda_kernel::dsl::testgen_plane!();
        ruda_kernel::dsl::testgen_sequence!();
        ruda_kernel::dsl::testgen_slice!();
        ruda_kernel::dsl::testgen_stream!();
        ruda_kernel::dsl::testgen_unary!();
        ruda_kernel::dsl::testgen_atomic_float!();
        ruda_kernel::dsl::testgen_tensormap!();
        ruda_kernel::dsl::testgen_minifloat!();
        ruda_kernel::dsl::testgen_unroll!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_int {
    () => {
        ruda_kernel::dsl::testgen_unary_int!();
        ruda_kernel::dsl::testgen_atomic_int!();
        ruda_kernel::dsl::testgen_saturating_int!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_uint {
    () => {
        ruda_kernel::dsl::testgen_const_match!();
        ruda_kernel::dsl::testgen_saturating_uint!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_untyped {
    () => {
        ruda_kernel::dsl::testgen_launch_untyped!();

        ruda_kernel::dsl::testgen_cmma!();
        ruda_kernel::dsl::testgen_numeric!();
        ruda_kernel::dsl::testgen_file!();
        ruda_kernel::dsl::testgen_metadata!();
        ruda_kernel::dsl::testgen_topology!();
        ruda_kernel::dsl::testgen_properties!();

        ruda_kernel::dsl::testgen_constants!();
        ruda_kernel::dsl::testgen_sync_plane!();
        ruda_kernel::dsl::testgen_atomic_untyped!();
        ruda_kernel::dsl::testgen_tensor_indexing!();
        ruda_kernel::dsl::testgen_debug!();
        ruda_kernel::dsl::testgen_binary_untyped!();
        ruda_kernel::dsl::testgen_cluster!();

        ruda_kernel::dsl::testgen_enums!();
        ruda_kernel::dsl::testgen_comparison!();

        ruda_kernel::dsl::testgen_to_client!();
        ruda_kernel::dsl::testgen_all_reduce!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_bytes {
    ($ty:ident: $($elem:expr),*) => {
        $ty::as_bytes(&[$($ty::new($elem),)*])
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_type {
    ($ty:ident: $($elem:expr),*) => {
        &[$($ty::new($elem),)*]
    };
}
