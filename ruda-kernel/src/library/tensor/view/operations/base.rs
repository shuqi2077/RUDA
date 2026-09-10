use crate::library::tensor::layout::Coordinates;
use crate::dsl::prelude::*;
use ruda_kernel::dsl::{prelude::barrier::Barrier, unexpanded};

/// Type from which we can read values in ruda functions.
/// For a mutable version, see [`ListMut`].
#[allow(clippy::len_without_is_empty)]
#[ruda(self_type = "ref", expand_base_traits = "VectorizedExpand")]
pub trait ViewOperations<T: RudaPrimitive, C: Coordinates>: Vectorized {
    #[allow(unused)]
    fn read(&self, pos: C) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_checked(&self, pos: C) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_masked(&self, pos: C, value: T) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_unchecked(&self, pos: C) -> T {
        unexpanded!()
    }

    /// Create a slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    #[allow(unused)]
    fn to_linear_slice(&self, pos: C, size: C) -> Slice<T, ReadOnly> {
        unexpanded!()
    }

    ///.Execute a TMA load into shared memory, if the underlying storage supports it.
    /// Panics if it's unsupported.
    #[allow(unused)]
    fn tensor_map_load(&self, barrier: &Barrier, shared_memory: &mut Slice<T, ReadWrite>, pos: C) {
        unexpanded!()
    }

    #[allow(unused)]
    fn shape(&self) -> C {
        unexpanded!();
    }

    #[allow(unused)]
    fn is_in_bounds(&self, pos: C) -> bool {
        unexpanded!();
    }
}

/// Type for which we can read and write values in ruda functions.
/// For an immutable version, see [List].
#[ruda(expand_base_traits = "ViewOperationsExpand<T, C>", self_type = "ref")]
pub trait ViewOperationsMut<T: RudaPrimitive, C: Coordinates>: ViewOperations<T, C> {
    #[allow(unused)]
    fn write(&self, pos: C, value: T) {
        unexpanded!()
    }

    #[allow(unused)]
    fn write_checked(&self, pos: C, value: T) {
        unexpanded!()
    }

    /// Create a mutable slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    #[allow(unused, clippy::wrong_self_convention)]
    fn to_linear_slice_mut(&self, pos: C, size: C) -> Slice<T, ReadWrite> {
        unexpanded!()
    }

    ///.Execute a TMA store into global memory, if the underlying storage supports it.
    /// Panics if it's unsupported.
    #[allow(unused)]
    fn tensor_map_store(&self, shared_memory: &Slice<T>, pos: C) {
        unexpanded!()
    }
}

// Automatic implementation for references to List.
impl<'a, T: RudaPrimitive, C: Coordinates, V: ViewOperations<T, C>> ViewOperations<T, C> for &'a V
where
    &'a V: RudaType<ExpandType = V::ExpandType>,
{
    fn read(&self, pos: C) -> T {
        V::read(self, pos)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        pos: C::ExpandType,
    ) -> <T as RudaType>::ExpandType {
        V::__expand_read(scope, this, pos)
    }
}

// Automatic implementation for mutable references to List.
impl<'a, T: RudaPrimitive, C: Coordinates, L: ViewOperations<T, C>> ViewOperations<T, C>
    for &'a mut L
where
    &'a mut L: RudaType<ExpandType = L::ExpandType>,
{
    fn read(&self, pos: C) -> T {
        L::read(self, pos)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        pos: C::ExpandType,
    ) -> <T as RudaType>::ExpandType {
        L::__expand_read(scope, this, pos)
    }
}

// Automatic implementation for references to ListMut.
impl<'a, T: RudaPrimitive, C: Coordinates, L: ViewOperationsMut<T, C>> ViewOperationsMut<T, C>
    for &'a L
where
    &'a L: RudaType<ExpandType = L::ExpandType>,
{
    fn write(&self, pos: C, value: T) {
        L::write(self, pos, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        pos: C::ExpandType,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, pos, value);
    }
}

// Automatic implementation for mutable references to ListMut.
impl<'a, T: RudaPrimitive, C: Coordinates, L: ViewOperationsMut<T, C>> ViewOperationsMut<T, C>
    for &'a mut L
where
    &'a mut L: RudaType<ExpandType = L::ExpandType>,
{
    fn write(&self, pos: C, value: T) {
        L::write(self, pos, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        pos: C::ExpandType,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, pos, value);
    }
}
