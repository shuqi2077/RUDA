use alloc::{boxed::Box, collections::{BTreeMap, BTreeSet, LinkedList, VecDeque}, string::String};
use core::ops::{Bound, Range, RangeFrom, RangeFull, RangeTo, RangeToInclusive};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::{PrecisionSettings, Record};
use crate::tensor::backend::Backend;

impl<B: Backend, T: Record<B>> Record<B> for Box<T> {
    type Item<S: PrecisionSettings> = Box<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        Box::new((*self).into_item::<S>())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Box::new(T::from_item::<S>(*item, device))
    }
}

impl<B: Backend, T: Record<B>> Record<B> for VecDeque<T> {
    type Item<S: PrecisionSettings> = VecDeque<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.into_iter().map(T::into_item::<S>).collect()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        item.into_iter().map(|value| T::from_item::<S>(value, device)).collect()
    }
}

impl<B: Backend, T: Record<B>> Record<B> for LinkedList<T> {
    type Item<S: PrecisionSettings> = LinkedList<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.into_iter().map(T::into_item::<S>).collect()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        item.into_iter().map(|value| T::from_item::<S>(value, device)).collect()
    }
}

impl<B: Backend, T: Record<B>, E: Record<B>> Record<B> for Result<T, E> {
    type Item<S: PrecisionSettings> = Result<T::Item<S>, E::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        match self {
            Ok(value) => Ok(value.into_item::<S>()),
            Err(value) => Err(value.into_item::<S>()),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        match item {
            Ok(value) => Ok(T::from_item::<S>(value, device)),
            Err(value) => Err(E::from_item::<S>(value, device)),
        }
    }
}

impl<B, K, T> Record<B> for BTreeMap<K, T>
where B: Backend, K: Ord + Clone + Send + Serialize + DeserializeOwned, T: Record<B> {
    type Item<S: PrecisionSettings> = BTreeMap<K, T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.into_iter().map(|(key, value)| (key, value.into_item::<S>())).collect()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        item.into_iter().map(|(key, value)| (key, T::from_item::<S>(value, device))).collect()
    }
}

impl<B, K> Record<B> for BTreeSet<K>
where B: Backend, K: Ord + Clone + Send + Serialize + DeserializeOwned {
    type Item<S: PrecisionSettings> = Self;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> { self }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self { item }
}

impl<B: Backend, T: Record<B>> Record<B> for hashbrown::HashMap<String, T> {
    type Item<S: PrecisionSettings> = hashbrown::HashMap<String, T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.into_iter().map(|(key, value)| (key, value.into_item::<S>())).collect()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        item.into_iter().map(|(key, value)| (key, T::from_item::<S>(value, device))).collect()
    }
}

#[cfg(feature = "std")]
impl<B: Backend, T: Record<B>> Record<B> for std::collections::HashMap<String, T> {
    type Item<S: PrecisionSettings> = std::collections::HashMap<String, T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.into_iter().map(|(key, value)| (key, value.into_item::<S>())).collect()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        item.into_iter().map(|(key, value)| (key, T::from_item::<S>(value, device))).collect()
    }
}

impl<B: Backend, T: Record<B>> Record<B> for Range<T> {
    type Item<S: PrecisionSettings> = (T::Item<S>, T::Item<S>);

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.start.into_item::<S>(), self.end.into_item::<S>())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        T::from_item::<S>(item.0, device)..T::from_item::<S>(item.1, device)
    }
}

impl<B: Backend, T: Record<B>> Record<B> for RangeFrom<T> {
    type Item<S: PrecisionSettings> = T::Item<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> { self.start.into_item::<S>() }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        T::from_item::<S>(item, device)..
    }
}

impl<B: Backend, T: Record<B>> Record<B> for RangeTo<T> {
    type Item<S: PrecisionSettings> = T::Item<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> { self.end.into_item::<S>() }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        ..T::from_item::<S>(item, device)
    }
}

impl<B: Backend, T: Record<B>> Record<B> for RangeToInclusive<T> {
    type Item<S: PrecisionSettings> = T::Item<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> { self.end.into_item::<S>() }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        ..=T::from_item::<S>(item, device)
    }
}

impl<B: Backend> Record<B> for RangeFull {
    type Item<S: PrecisionSettings> = ();

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {}

    fn from_item<S: PrecisionSettings>(_item: Self::Item<S>, _device: &B::Device) -> Self { .. }
}

/// Serialized original included/excluded/unbounded range endpoints.
#[derive(Clone, Serialize, Deserialize, Record)]
pub enum BoundItem<T> {
    /// An included endpoint.
    Included(T),
    /// An excluded endpoint.
    Excluded(T),
    /// No endpoint.
    Unbounded,
}

impl<B: Backend, T: Record<B>> Record<B> for Bound<T> {
    type Item<S: PrecisionSettings> = BoundItem<T::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        match self {
            Self::Included(value) => BoundItem::Included(value.into_item::<S>()),
            Self::Excluded(value) => BoundItem::Excluded(value.into_item::<S>()),
            Self::Unbounded => BoundItem::Unbounded,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        match item {
            BoundItem::Included(value) => Self::Included(T::from_item::<S>(value, device)),
            BoundItem::Excluded(value) => Self::Excluded(T::from_item::<S>(value, device)),
            BoundItem::Unbounded => Self::Unbounded,
        }
    }
}
