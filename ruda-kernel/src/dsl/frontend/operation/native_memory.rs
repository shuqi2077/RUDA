use crate::dsl::prelude::*;
use crate::dsl::ir::{BinaryOperator, Instruction, ManagedVariable, Operator, UnaryOperator};

/// Obtain the native byte address of a slice element. The address remains valid
/// only while the underlying allocation is alive and in its owning address space.
pub fn native_address<T: RudaPrimitive>(_input: &Slice<T>, _index: usize) -> u64 {
    crate::dsl::unexpanded!()
}

pub mod native_address {
    use super::*;
    pub fn expand<T: RudaPrimitive>(scope: &mut Scope, input: SliceExpand<T, ReadOnly>, index: NativeExpand<usize>) -> NativeExpand<u64> {
        let (array, offset) = input.__to_raw_parts();
        let index = add::expand(scope, index, NativeExpand::new(ManagedVariable::Plain(offset)));
        let ty = u64::as_type(scope);
        let output = scope.create_local(ty);
        scope.register(Instruction::new(Operator::NativeAddress(BinaryOperator {
            lhs: array, rhs: index.expand.consume(),
        }), *output));
        output.into()
    }
}

/// Read a value at a native byte address. The caller supplies a valid, aligned
/// address in the current device's address space, with enough readable storage.
pub fn native_load<T: RudaPrimitive>(_address: u64) -> T {
    crate::dsl::unexpanded!()
}

pub mod native_load {
    use super::*;
    pub fn expand<T: RudaPrimitive>(scope: &mut Scope, address: NativeExpand<u64>) -> NativeExpand<T> {
        let ty = T::as_type(scope);
        let output = scope.create_local(ty);
        scope.register(Instruction::new(Operator::NativeLoad(UnaryOperator { input: address.expand.consume() }), *output));
        output.into()
    }
}

/// Write through a native byte address. The caller supplies a valid, aligned,
/// writable allocation and obeys the device's synchronization requirements.
pub fn native_store<T: RudaPrimitive>(_address: u64, _value: T) {
    crate::dsl::unexpanded!()
}

pub mod native_store {
    use super::*;
    pub fn expand<T: RudaPrimitive>(scope: &mut Scope, address: NativeExpand<u64>, value: NativeExpand<T>) {
        scope.register(Instruction::no_out(Operator::NativeStore(BinaryOperator {
            lhs: address.expand.consume(), rhs: value.expand.consume(),
        })));
    }
}
