use super::Slice;

/// Defines an [`Iterator`] over a [`Slice`].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SliceIter {
    slice: Slice,
    current: isize,
}

impl Iterator for SliceIter {
    type Item = isize;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.current;
        self.current += self.slice.step;

        if let Some(end) = self.slice.end {
            if self.slice.is_reversed() {
                if next <= end {
                    return None;
                }
            } else if next >= end {
                return None;
            }
        }

        Some(next)
    }
}

/// Note: Unbounded [`Slice`]s produce infinite iterators.
impl IntoIterator for Slice {
    type Item = isize;
    type IntoIter = SliceIter;

    fn into_iter(self) -> Self::IntoIter {
        SliceIter {
            slice: self,
            current: self.start,
        }
    }
}
