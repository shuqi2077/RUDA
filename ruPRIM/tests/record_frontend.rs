#![cfg(feature = "kernel-ir")]

use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl::ir::{Bitwise, Operation, Operator};
use ruprim::collective::decompose::*;
use ruprim::collective::record::*;

ruprim::ruda_record! {
    struct Fields { low: u32, high: u64 }
}
ruprim::ruda_record! {
    struct Nested { fields: Fields, tag: u16 }
}
ruprim::ruda_decomposer!(FieldsBits, FieldsBitsLaunch for Fields { high: u64, low: u32 });

#[ruda]
fn scalar_bit(value: u64, index: usize) -> bool {
    let decomposer = RudaScalarDecomposer {};
    <RudaScalarDecomposer as RudaDecomposer<u64>>::bit(&decomposer, value, index)
}

#[test]
fn composite_iterators_remain_launch_arguments() {
    fn launch<T: LaunchArg>() {}
    launch::<RudaZip<RudaNativeRecords, RudaNativeRecords>>();
    launch::<RudaZip<RudaZip<RudaNativeRecords, RudaNativeRecords>, RudaNativeRecords>>();
    launch::<RudaReferences<Nested, RudaNativeRecords>>();
}

#[test]
fn reference_assignment_retains_compile_time_marker() {
    let mut scope = Scope::root(false);
    let first = RudaReference::<Nested>::__expand_new(&mut scope, 64u64.into());
    let second = RudaReference::<Nested>::__expand_new(&mut scope, 128u64.into());
    let mut mutable = first.init_mut(&mut scope);
    mutable.expand_assign(&mut scope, second);
    assert!(!scope.instructions.is_empty());
}

#[test]
fn record_allocations_expand_with_native_addresses() {
    let mut scope = Scope::root(false);
    scope.register_type::<usize>(u32::as_type(&scope).storage_type());
    let array = RudaRecordArray::<Nested>::__expand_new(&mut scope, 3);
    let shared = RudaRecordShared::<Nested>::__expand_new(&mut scope, 3);
    array.__expand_address_method(&mut scope, 1usize.into());
    shared.__expand_address_method(&mut scope, 2usize.into());
    assert_eq!(scope.instructions.iter().filter(|instruction|
        matches!(instruction.operation, Operation::Operator(Operator::NativeAddress(_)))
    ).count(), 2);
}

#[test]
fn scalar_decomposition_uses_matching_shift_types() {
    let mut scope = Scope::root(false);
    scope.register_type::<usize>(u32::as_type(&scope).storage_type());
    scalar_bit::expand(&mut scope, (1u64 << 63).into(), 63usize.into());
    let shifts: Vec<_> = scope.instructions.iter().filter_map(|instruction| {
        if let Operation::Bitwise(Bitwise::ShiftRight(op)) = &instruction.operation {
            Some(op)
        } else { None }
    }).collect();
    assert_eq!(shifts.len(), 1);
    assert_eq!(shifts[0].lhs.ty, shifts[0].rhs.ty);
}
