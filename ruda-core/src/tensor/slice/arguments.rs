use super::Slice;
use crate::tensor::Shape;
use alloc::vec::Vec;

/// Trait for slice arguments that can be converted into an array of slices.
/// This allows the `slice` method to accept both single slices (from `s![..]`)
/// and arrays of slices (from `s![.., ..]` or `[0..5, 1..3]`).
pub trait SliceArg {
    /// Convert to an vec of slices with clamping to shape dimensions.
    ///
    /// Returns a [Slice] for each dimension in `shape`.
    fn into_slices(self, shape: &Shape) -> Vec<Slice>;
}

impl<S: Into<Slice> + Clone> SliceArg for &[S] {
    fn into_slices(self, shape: &Shape) -> Vec<Slice> {
        assert!(
            self.len() <= shape.num_dims(),
            "Too many slices provided for shape, got {} but expected at most {}",
            self.len(),
            shape.num_dims()
        );

        shape
            .iter()
            .enumerate()
            .map(|(i, dim_size)| {
                let slice = if i >= self.len() {
                    Slice::full()
                } else {
                    self[i].clone().into()
                };
                // Apply shape clamping by converting to range and back
                let clamped_range = slice.to_range(*dim_size);
                Slice::new(
                    clamped_range.start as isize,
                    Some(clamped_range.end as isize),
                    slice.step(),
                )
            })
            .collect::<Vec<_>>()
    }
}

impl SliceArg for &Vec<Slice> {
    fn into_slices(self, shape: &Shape) -> Vec<Slice> {
        self.as_slice().into_slices(shape)
    }
}

impl<const R: usize, T> SliceArg for [T; R]
where
    T: Into<Slice> + Clone,
{
    fn into_slices(self, shape: &Shape) -> Vec<Slice> {
        self.as_slice().into_slices(shape)
    }
}

impl<T> SliceArg for T
where
    T: Into<Slice>,
{
    fn into_slices(self, shape: &Shape) -> Vec<Slice> {
        let slice: Slice = self.into();
        [slice].as_slice().into_slices(shape)
    }
}
