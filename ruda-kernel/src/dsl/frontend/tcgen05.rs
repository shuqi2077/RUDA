//! SM100 tensor memory and asynchronous matrix multiply intrinsics.
use alloc::vec;
use ruda_core::ir::{CoopMma, Instruction, Tcgen05Op};
use ruda_kernel_macros::intrinsic;
use crate::dsl::prelude::*;

#[ruda]
pub fn allocate(destination: &mut SliceMut<u32>, #[comptime] columns: u32) {
    intrinsic!(|scope| {
        let (buffer, offset) = destination.__to_raw_parts();
        let out = scope.create_local(u32::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell {
            kind: Tcgen05Op::Allocate { columns }, args: vec![buffer, offset],
        }, *out));
    })
}

#[ruda]
pub fn deallocate(address: u32, #[comptime] columns: u32) {
    intrinsic!(|scope| {
        let out = scope.create_local(u32::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell {
            kind: Tcgen05Op::Deallocate { columns }, args: vec![*address.expand],
        }, *out));
    })
}

#[ruda]
pub fn relinquish() {
    intrinsic!(|scope| {
        let out = scope.create_local(u32::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell { kind: Tcgen05Op::Relinquish, args: vec![] }, *out));
    })
}

#[ruda]
pub fn descriptor<I: Numeric>(source: &Slice<I>) -> u64 {
    intrinsic!(|scope| {
        let (buffer, offset) = source.__to_raw_parts();
        let out = scope.create_local(u64::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell { kind: Tcgen05Op::Descriptor, args: vec![buffer, offset] }, *out));
        out.into()
    })
}

#[ruda]
pub fn mma(address: u32, a: u64, b: u64, accumulate: bool, #[comptime] n: u32, #[comptime] bf16: bool) {
    intrinsic!(|scope| {
        let out = scope.create_local(u32::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell { kind: Tcgen05Op::Mma { n, bf16 },
            args: vec![*address.expand, *a.expand, *b.expand, *accumulate.expand] }, *out));
    })
}

#[ruda]
pub fn mma_fp8(address: u32, a: u64, b: u64, accumulate: bool, #[comptime] n: u32,
    #[comptime] e5m2_a: bool, #[comptime] e5m2_b: bool,
) {
    intrinsic!(|scope| {
        let out = scope.create_local(u32::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell { kind: Tcgen05Op::MmaFp8 { n, e5m2_a, e5m2_b },
            args: vec![*address.expand, *a.expand, *b.expand, *accumulate.expand] }, *out));
    })
}

/// 128B-swizzled descriptor with explicit leading/stride byte offsets (MN-major operands).
#[ruda]
pub fn descriptor_strided<I: Numeric>(source: &Slice<I>, #[comptime] lbo: u32, #[comptime] sbo: u32) -> u64 {
    intrinsic!(|scope| {
        let (buffer, offset) = source.__to_raw_parts();
        let out = scope.create_local(u64::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell { kind: Tcgen05Op::DescriptorStrided { lbo, sbo }, args: vec![buffer, offset] }, *out));
        out.into()
    })
}

#[ruda]
pub fn descriptor_tf32_mn<I: Numeric>(source: &Slice<I>, #[comptime] lbo: u32) -> u64 {
    descriptor_strided(source, lbo, 512u32) ^ (3u64 << 61u32)
}

/// M=128 tcgen05 MMA of any supported kind (0 f16, 1 bf16, 2 tf32, 3 e4m3, 4 e5m2).
#[ruda]
pub fn mma_kind(address: u32, a: u64, b: u64, accumulate: bool, #[comptime] n: u32, #[comptime] kind: u32,
    #[comptime] a_mn: bool, #[comptime] b_mn: bool,
) {
    intrinsic!(|scope| {
        let out = scope.create_local(u32::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell { kind: Tcgen05Op::MmaKind { n, kind, a_mn, b_mn },
            args: vec![*address.expand, *a.expand, *b.expand, *accumulate.expand] }, *out));
    })
}

#[ruda]
pub fn commit(completion: &barrier::Barrier) {
    intrinsic!(|scope| {
        let out = scope.create_local(u32::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell { kind: Tcgen05Op::Commit, args: vec![*completion.expand] }, *out));
    })
}

#[ruda]
pub fn load(address: u32) -> Array<f32> {
    intrinsic!(|scope| {
        let out = Array::<f32>::__expand_new(scope, 32usize);
        scope.register(Instruction::new(CoopMma::Blackwell { kind: Tcgen05Op::Load { columns: 32 }, args: vec![*address.expand] }, *out.expand));
        out
    })
}

#[ruda]
pub fn fence(#[comptime] before: bool) {
    intrinsic!(|scope| {
        let out = scope.create_local(u32::as_type(scope));
        scope.register(Instruction::new(CoopMma::Blackwell { kind: Tcgen05Op::Fence { before }, args: vec![] }, *out));
    })
}
