use ruda_kernel::dsl::prelude::*;

use crate::library::tensor::{View, ViewExpand, layout::Coordinates};

impl<E: RudaPrimitive, C: Coordinates, IO: Clone> RudaIndex for View<E, C, IO> {
    type Output = E;
    type Idx = C;
}

impl<E: RudaPrimitive, C: Coordinates, IO: Clone> RudaIndexExpand for ViewExpand<E, C, IO> {
    type Output = <E as RudaType>::ExpandType;
    type Idx = <C as RudaType>::ExpandType;

    fn expand_index(self, scope: &mut Scope, index: C::ExpandType) -> Self::Output {
        self.__expand_read_method(scope, index)
    }

    fn expand_index_unchecked(self, scope: &mut Scope, index: C::ExpandType) -> Self::Output {
        self.__expand_read_unchecked_method(scope, index)
    }
}

impl<E: RudaPrimitive, C: Coordinates> RudaIndexMut for View<E, C, ReadWrite> {}
impl<E: RudaPrimitive, C: Coordinates> RudaIndexMutExpand for ViewExpand<E, C, ReadWrite> {
    fn expand_index_mut(self, scope: &mut Scope, index: C::ExpandType, value: Self::Output) {
        self.__expand_write_method(scope, index, value)
    }
}
