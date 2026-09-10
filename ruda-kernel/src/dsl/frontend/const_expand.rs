use ruda_core::ir::Scope;

use super::RudaType;

pub trait OptionExt<T: RudaType> {
    fn __expand_unwrap_or_else_method(
        self,
        _scope: &mut Scope,
        other: impl FnOnce(&mut Scope) -> T::ExpandType,
    ) -> T::ExpandType;

    fn __expand_unwrap_or_method(self, _scope: &mut Scope, other: T::ExpandType) -> T::ExpandType;
}

impl<T: RudaType + Into<T::ExpandType>> OptionExt<T> for Option<T> {
    fn __expand_unwrap_or_else_method(
        self,
        scope: &mut Scope,
        other: impl FnOnce(&mut Scope) -> <T as RudaType>::ExpandType,
    ) -> <T as RudaType>::ExpandType {
        self.map(Into::into).unwrap_or_else(|| other(scope))
    }

    fn __expand_unwrap_or_method(
        self,
        _scope: &mut Scope,
        other: <T as RudaType>::ExpandType,
    ) -> <T as RudaType>::ExpandType {
        self.map(Into::into).unwrap_or(other)
    }
}
