use core::ops::{Deref, DerefMut};

use super::{RudaType, NativeExpand};
use crate::dsl::{prelude::*, unexpanded};
use ruda_core::ir::{Scope, VectorSize};

/// Type from which we can read values in ruda functions.
/// For a mutable version, see [`ListMut`].
#[allow(clippy::len_without_is_empty)]
#[ruda(self_type = "ref", expand_base_traits = "SliceOperatorExpand<T>")]
pub trait List<T: RudaPrimitive>: SliceOperator<T> + Vectorized + Deref<Target = [T]> {
    #[allow(unused)]
    fn read(&self, index: usize) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_unchecked(&self, index: usize) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn len(&self) -> usize {
        unexpanded!();
    }
}

/// Type for which we can read and write values in ruda functions.
/// For an immutable version, see [List].
#[ruda(self_type = "ref", expand_base_traits = "SliceMutOperatorExpand<T>")]
pub trait ListMut<T: RudaPrimitive>:
    List<T> + SliceMutOperator<T> + DerefMut<Target = [T]>
{
    #[allow(unused)]
    fn write(&self, index: usize, value: T) {
        unexpanded!()
    }
}

// Automatic implementation for references to List.
impl<'a, T: RudaPrimitive, L: List<T>> List<T> for &'a L
where
    &'a L: RudaType<ExpandType = L::ExpandType>,
    &'a L: Deref<Target = [T]>,
{
    fn read(&self, index: usize) -> T {
        L::read(self, index)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: NativeExpand<usize>,
    ) -> <T as RudaType>::ExpandType {
        L::__expand_read(scope, this, index)
    }
}

// Automatic implementation for mutable references to List.
impl<'a, T: RudaPrimitive, L: List<T>> List<T> for &'a mut L
where
    &'a mut L: RudaType<ExpandType = L::ExpandType>,
    &'a mut L: Deref<Target = [T]>,
{
    fn read(&self, index: usize) -> T {
        L::read(self, index)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: NativeExpand<usize>,
    ) -> <T as RudaType>::ExpandType {
        L::__expand_read(scope, this, index)
    }
}

// Automatic implementation for references to ListMut.
impl<'a, T: RudaPrimitive, L: ListMut<T>> ListMut<T> for &'a L
where
    &'a L: RudaType<ExpandType = L::ExpandType>,
    &'a L: DerefMut<Target = [T]>,
{
    fn write(&self, index: usize, value: T) {
        L::write(self, index, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: NativeExpand<usize>,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, index, value);
    }
}

// Automatic implementation for mutable references to ListMut.
impl<'a, T: RudaPrimitive, L: ListMut<T>> ListMut<T> for &'a mut L
where
    &'a mut L: RudaType<ExpandType = L::ExpandType>,
    &'a mut L: DerefMut<Target = [T]>,
{
    fn write(&self, index: usize, value: T) {
        L::write(self, index, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: NativeExpand<usize>,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, index, value);
    }
}

pub trait Vectorized: RudaType<ExpandType: VectorizedExpand> {
    fn vector_size(&self) -> VectorSize {
        unexpanded!()
    }
    fn __expand_vector_size(_scope: &mut Scope, this: Self::ExpandType) -> VectorSize {
        this.vector_size()
    }
}

pub trait VectorizedExpand {
    fn vector_size(&self) -> VectorSize;
    fn __expand_vector_size_method(&self, _scope: &mut Scope) -> VectorSize {
        self.vector_size()
    }
}

impl<'a, L: Vectorized> Vectorized for &'a L where &'a L: RudaType<ExpandType: VectorizedExpand> {}
impl<'a, L: Vectorized> Vectorized for &'a mut L where
    &'a mut L: RudaType<ExpandType: VectorizedExpand>
{
}
