use super::Slice;
use crate::tensor::indexing::AsIndex;
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

pub(super) fn convert_signed_index(index: isize, size: usize) -> usize {
    if index < 0 {
        (size as isize + index).max(0) as usize
    } else {
        (index as usize).min(size)
    }
}

pub(super) fn handle_signed_inclusive_end(end: isize) -> Option<isize> {
    match end {
        -1 => None,
        end => Some(end + 1),
    }
}

impl<I: AsIndex> From<Range<I>> for Slice {
    fn from(r: Range<I>) -> Self {
        Self {
            start: r.start.as_index(),
            end: Some(r.end.as_index()),
            step: 1,
        }
    }
}

impl<I: AsIndex + Copy> From<RangeInclusive<I>> for Slice {
    fn from(r: RangeInclusive<I>) -> Self {
        Self {
            start: r.start().as_index(),
            end: handle_signed_inclusive_end(r.end().as_index()),
            step: 1,
        }
    }
}

impl<I: AsIndex> From<RangeFrom<I>> for Slice {
    fn from(r: RangeFrom<I>) -> Self {
        Self {
            start: r.start.as_index(),
            end: None,
            step: 1,
        }
    }
}

impl<I: AsIndex> From<RangeTo<I>> for Slice {
    fn from(r: RangeTo<I>) -> Self {
        Self {
            start: 0,
            end: Some(r.end.as_index()),
            step: 1,
        }
    }
}

impl<I: AsIndex> From<RangeToInclusive<I>> for Slice {
    fn from(r: RangeToInclusive<I>) -> Self {
        Self {
            start: 0,
            end: handle_signed_inclusive_end(r.end.as_index()),
            step: 1,
        }
    }
}

impl From<RangeFull> for Slice {
    fn from(_: RangeFull) -> Self {
        Self {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

impl From<usize> for Slice {
    fn from(i: usize) -> Self {
        Slice::index(i as isize)
    }
}

impl From<isize> for Slice {
    fn from(i: isize) -> Self {
        Slice::index(i)
    }
}

impl From<i32> for Slice {
    fn from(i: i32) -> Self {
        Slice::index(i as isize)
    }
}

impl From<i64> for Slice {
    fn from(i: i64) -> Self {
        Slice::index(i as isize)
    }
}
