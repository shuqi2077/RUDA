use super::{RudaModuleAdapter, Deserializer, Error, NestedValue};
use serde::de::{DeserializeSeed, SeqAccess};

pub(super) struct RecordEnumProbe<A: RudaModuleAdapter> {
    value: NestedValue,
    default_for_missing_fields: bool,
    adapter: core::marker::PhantomData<A>,
}

impl<A: RudaModuleAdapter> RecordEnumProbe<A> {
    pub(super) fn new(value: NestedValue, default_for_missing_fields: bool) -> Self {
        Self { value, default_for_missing_fields, adapter: core::marker::PhantomData }
    }
}

impl<'de, A: RudaModuleAdapter> SeqAccess<'de> for RecordEnumProbe<A> {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        seed.deserialize(Deserializer::<A>::new(
            self.value.clone(), self.default_for_missing_fields,
        )).map(Some)
    }
}
