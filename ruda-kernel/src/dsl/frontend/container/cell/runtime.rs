use crate::dsl::prelude::RudaPrimitive;
use crate::dsl::frontend::assign::expand_no_check;
use crate::dsl::prelude::*;
use ruda_core::ir::Operation;
use ruda_kernel_macros::intrinsic;

#[derive(Clone, Copy)]
pub struct RuntimeCell<T: RudaType> {
    #[allow(unused)]
    value: T,
}

pub struct RuntimeCellExpand<T: RudaType> {
    value: <T as crate::dsl::prelude::RudaType>::ExpandType,
}
impl<T: RudaType> Clone for RuntimeCellExpand<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}
impl<T: RudaType> crate::dsl::prelude::RudaType for RuntimeCell<T> {
    type ExpandType = RuntimeCellExpand<T>;
}
impl<T: RudaType> crate::dsl::prelude::IntoMut for RuntimeCellExpand<T> {
    fn into_mut(self, _scope: &mut crate::dsl::prelude::Scope) -> Self {
        Self {
            // We keep the same as a cell would do.
            value: self.value.clone(),
        }
    }
}
impl<T: RudaType> crate::dsl::prelude::RudaDebug for RuntimeCellExpand<T> {}

#[ruda]
impl<T: RudaPrimitive> RuntimeCell<T> {
    /// Create a new runtime cell with the given initial value.
    #[allow(unused_variables)]
    pub fn new(init: T) -> Self {
        intrinsic!(|scope| {
            let value = init_expand(scope, init.expand, true, Operation::Copy);
            RuntimeCellExpand {
                value: value.into(),
            }
        })
    }

    /// Store a new value in the cell.
    #[allow(unused_variables)]
    pub fn store(&self, value: T) {
        intrinsic!(|scope| {
            expand_no_check(scope, value, self.value);
        })
    }

    /// Get the value from the call
    pub fn read(&self) -> T {
        intrinsic!(|scope| {
            let value = init_expand(scope, self.value.expand, false, Operation::Copy);
            value.into()
        })
    }

    /// Consume the cell.
    pub fn consume(self) -> T {
        intrinsic!(|scope| { self.value })
    }
}

#[ruda]
impl<T: RudaIndexMut> RuntimeCell<T> {
    /// Store a new value in the cell at the given index.
    #[allow(unused_variables)]
    pub fn store_at(&mut self, index: <T as RudaIndex>::Idx, value: <T as RudaIndex>::Output) {
        intrinsic!(|scope| { self.value.expand_index_mut(scope, index, value) })
    }
}

#[ruda]
impl<T: RudaIndex> RuntimeCell<T> {
    /// Read a value in the cell at the given index.
    #[allow(unused_variables)]
    pub fn read_at(&self, index: T::Idx) -> T::Output {
        intrinsic!(|scope| { self.value.expand_index(scope, index) })
    }
}
