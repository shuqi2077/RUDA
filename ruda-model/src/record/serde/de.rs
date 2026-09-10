use core::ptr;
use std::collections::HashMap;

use super::data::NestedValue;
use super::{adapter::RudaModuleAdapter, error::Error};

use serde::de::{EnumAccess, VariantAccess};
use serde::{
    de::{self, DeserializeSeed, IntoDeserializer, MapAccess, SeqAccess, Visitor},
    forward_to_deserialize_any,
};

const RECORD_ITEM_SUFFIX: &str = "RecordItem";

mod enum_probe;

/// A deserializer for the nested value data structure.
pub struct Deserializer<A: RudaModuleAdapter> {
    // This string starts with the input data and characters are truncated off
    // the beginning as data is parsed.
    value: Option<NestedValue>,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

impl<A: RudaModuleAdapter> Deserializer<A> {
    /// Creates a new deserializer with the given nested value.
    ///
    /// # Arguments
    ///
    /// * `value` - A nested value.
    /// * `default_for_missing_fields` - A boolean indicating whether to add missing fields with default value.
    pub fn new(value: NestedValue, default_for_missing_fields: bool) -> Self {
        Self {
            value: Some(value),
            default_for_missing_fields,
            phantom: std::marker::PhantomData,
        }
    }

    fn convert<T>(
        self,
        conversion: impl FnOnce(NestedValue) -> Option<T>,
        expected: &'static str,
    ) -> Result<T, Error> {
        self.value.and_then(conversion)
            .ok_or_else(|| <Error as de::Error>::custom(expected))
    }
}

impl<'de, A: RudaModuleAdapter> serde::Deserializer<'de> for Deserializer<A> {
    type Error = Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            Some(NestedValue::Bool(value)) => visitor.visit_bool(value),
            Some(NestedValue::String(value)) => visitor.visit_string(value),
            Some(NestedValue::F32(value)) => visitor.visit_f32(value),
            Some(NestedValue::F64(value)) => visitor.visit_f64(value),
            Some(NestedValue::I16(value)) => visitor.visit_i16(value),
            Some(NestedValue::I32(value)) => visitor.visit_i32(value),
            Some(NestedValue::I64(value)) => visitor.visit_i64(value),
            Some(NestedValue::U8(value)) => visitor.visit_u8(value),
            Some(NestedValue::U16(value)) => visitor.visit_u16(value),
            Some(NestedValue::U64(value)) => visitor.visit_u64(value),
            Some(NestedValue::Map(map)) => visitor.visit_map(HashMapAccess::<A>::new(
                map, self.default_for_missing_fields,
            )),
            Some(value @ (NestedValue::Vec(_) | NestedValue::U8s(_)
                | NestedValue::U16s(_) | NestedValue::F32s(_))) => {
                Deserializer::<A>::new(value, self.default_for_missing_fields)
                    .deserialize_seq(visitor)
            }
            Some(value @ NestedValue::Bytes(_)) => {
                Deserializer::<A>::new(value, self.default_for_missing_fields)
                    .deserialize_byte_buf(visitor)
            }
            Some(NestedValue::Default(None)) => visitor.visit_none(),
            Some(NestedValue::Default(Some(field))) => Err(de::Error::custom(format!(
                "Cannot infer the default type for field {field}"
            ))),
            None => Err(de::Error::custom("Expected a value")),
        }
    }

    fn deserialize_struct<V>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let value = match self.value {
            Some(value) => {
                // Adapt modules
                if let Some(name) = name.strip_suffix(RECORD_ITEM_SUFFIX) {
                    A::adapt(name, value)
                } else {
                    value
                }
            }
            None => {
                return Err(de::Error::custom(format!(
                    "Expected some value but got {:?}",
                    self.value
                )));
            }
        };

        match value {
            NestedValue::Map(map) => {
                // Add missing fields into the map with default value if needed.
                let map = if self.default_for_missing_fields {
                    let mut map = map;
                    for field in fields.iter().map(|s| s.to_string()) {
                        map.entry(field.clone())
                            .or_insert(NestedValue::Default(Some(field)));
                    }
                    map
                } else {
                    map
                };

                visitor.visit_map(HashMapAccess::<A>::new(
                    map,
                    self.default_for_missing_fields,
                ))
            }

            _ => Err(de::Error::custom(format!(
                "Expected struct but got {value:?}"
            ))),
        }
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_string(self.convert(NestedValue::as_string, "Expected a string")?)
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_unit()
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            Some(NestedValue::Map(map)) => visitor.visit_map(HashMapAccess::<A>::new(
                map,
                self.default_for_missing_fields,
            )),

            _ => Err(de::Error::custom(format!(
                "Expected map value but got {:?}",
                self.value
            ))),
        }
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_bool(self.convert(NestedValue::as_bool, "Expected a boolean")?)
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let value = self.value.and_then(|value| match value {
            NestedValue::U8(value) => i8::try_from(value).ok(),
            value => value.as_i16().and_then(|value| i8::try_from(value).ok()),
        }).ok_or_else(|| <Error as de::Error>::custom("Expected an integer in the i8 range"))?;
        visitor.visit_i8(value)
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i16(self.convert(NestedValue::as_i16, "Expected an integer in the i16 range")?)
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i32(self.convert(NestedValue::as_i32, "Expected an integer in the i32 range")?)
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(self.convert(NestedValue::as_i64, "Expected an integer in the i64 range")?)
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u8(self.convert(NestedValue::as_u8, "Expected an integer in the u8 range")?)
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u16(self.convert(NestedValue::as_u16, "Expected an integer in the u16 range")?)
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let value = self.value.and_then(|value| match value {
            NestedValue::U8(value) => Some(u32::from(value)),
            value => value.as_u64().and_then(|value| u32::try_from(value).ok()),
        }).ok_or_else(|| <Error as de::Error>::custom("Expected an integer in the u32 range"))?;
        visitor.visit_u32(value)
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u64(self.convert(NestedValue::as_u64, "Expected an integer in the u64 range")?)
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f32(self.convert(NestedValue::as_f32, "Expected an f32-compatible floating value")?)
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f64(self.convert(NestedValue::as_f64, "Expected an f64-compatible floating value")?)
    }

    fn deserialize_char<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let value = self.value.and_then(NestedValue::as_string)
            .ok_or_else(|| <Error as de::Error>::custom("Expected a single-character string"))?;
        let mut chars = value.chars();
        match (chars.next(), chars.next()) {
            (Some(value), None) => visitor.visit_char(value),
            _ => Err(de::Error::custom("Expected a single-character string")),
        }
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let value = self.convert(NestedValue::as_string, "Expected a string")?;
        visitor.visit_str(&value)
    }

    fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_byte_buf(visitor)
    }

    fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let bytes = self.convert(NestedValue::as_bytes, "Expected a byte buffer")?;
        match bytes.try_into_vec::<u8>() {
            Ok(bytes) => visitor.visit_byte_buf(bytes),
            Err(bytes) => visitor.visit_bytes(&bytes),
        }
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            None | Some(NestedValue::Default(_)) => visitor.visit_none(),
            Some(value) => visitor.visit_some(Deserializer::<A>::new(
                value,
                self.default_for_missing_fields,
            )),
        }
    }

    fn deserialize_unit<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_unit is not implemented")
    }

    fn deserialize_unit_struct<V>(
        self,
        _name: &'static str,
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_unit_struct is not implemented")
    }

    fn deserialize_newtype_struct<V>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        if name == "__ruda_record_enum_v1" {
            let value = self.value
                .ok_or_else(|| <Error as de::Error>::custom("Expected a record enum value"))?;
            return visitor.visit_seq(enum_probe::RecordEnumProbe::<A>::new(
                value, self.default_for_missing_fields,
            ));
        }
        visitor.visit_newtype_struct(Deserializer::<A>::new(
            self.value.ok_or_else(|| <Error as de::Error>::custom("Expected a newtype value"))?,
            self.default_for_missing_fields,
        ))
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        if let Some(value) = self.value {
            match value {
                NestedValue::Vec(_) => visitor.visit_seq(VecSeqAccess::<A, NestedValue>::new(
                    value,
                    self.default_for_missing_fields,
                )),
                NestedValue::U8s(_) => visitor.visit_seq(VecSeqAccess::<A, u8>::new(
                    value,
                    self.default_for_missing_fields,
                )),
                NestedValue::U16s(_) => visitor.visit_seq(VecSeqAccess::<A, u16>::new(
                    value,
                    self.default_for_missing_fields,
                )),
                NestedValue::F32s(_) => visitor.visit_seq(VecSeqAccess::<A, f32>::new(
                    value,
                    self.default_for_missing_fields,
                )),
                _ => Err(de::Error::custom(format!("Expected Vec but got {value:?}"))),
            }
        } else {
            Err(de::Error::custom("Expected Vec but got None"))
        }
    }

    fn deserialize_tuple<V>(self, len: usize, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let actual = match &self.value {
            Some(NestedValue::Vec(values)) => values.len(),
            Some(NestedValue::U8s(values)) => values.len(),
            Some(NestedValue::U16s(values)) => values.len(),
            Some(NestedValue::F32s(values)) => values.len(),
            _ => return Err(de::Error::custom("Expected tuple sequence")),
        };
        if actual != len {
            return Err(de::Error::custom(format!(
                "Expected tuple of length {len}, got {actual}"
            )));
        }
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_tuple(len, visitor)
    }

    /// Deserializes an enum by attempting to match its variants against the provided data.
    ///
    /// This function attempts to deserialize an enum by iterating over its possible variants
    /// and trying to deserialize the data into each until one succeeds. We need to do this
    /// because we don't have a way to know which variant to deserialize from the data.
    ///
    /// This is similar to Serde's
    /// [untagged enum deserialization](https://serde.rs/enum-representations.html#untagged),
    /// but it's on the deserializer side. Using `#[serde(untagged)]` on the enum will force
    /// using `deserialize_any`, which is not what we want because we want to use methods, such
    /// as `visit_struct`.
    fn deserialize_enum<V>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let tag = match &self.value {
            Some(NestedValue::Map(map)) if name == "DType" || map.len() == 1 => match map.get(name) {
                Some(NestedValue::String(variant)) => Some(variant.clone()),
                _ => None,
            },
            _ => None,
        };
        if let Some(variant) = tag {
            if !variants.contains(&variant.as_str()) {
                return Err(<Error as de::Error>::unknown_variant(&variant, variants));
            }
            return visitor.visit_enum(ProbeEnumAccess::<A>::new(
                NestedValue::String(variant.clone()), variant, self.default_for_missing_fields,
            ));
        }
        fn clone_unsafely<T>(thing: &T) -> T {
            unsafe {
                // Allocate memory for the clone.
                let mut clone = std::mem::MaybeUninit::<T>::uninit();
                // Get a mutable pointer to the allocated memory.
                let clone_ptr = clone.as_mut_ptr();
                // Copy the memory
                ptr::copy_nonoverlapping(thing as *const T, clone_ptr, 1);
                // Assume the cloned data is initialized and convert it to an owned instance of T.
                clone.assume_init()
            }
        }

        // Try each variant in order
        for &variant in variants {
            // clone visitor to avoid moving it
            let cloned_visitor = clone_unsafely(&visitor);
            let result = cloned_visitor.visit_enum(ProbeEnumAccess::<A>::new(
                self.value.clone().unwrap(),
                variant.to_owned(),
                self.default_for_missing_fields,
            ));

            if result.is_ok() {
                return result;
            }
        }

        Err(de::Error::custom("No variant match"))
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_string(visitor)
    }
}

/// A sequence access for a vector in the nested value data structure.
struct VecSeqAccess<A: RudaModuleAdapter, I> {
    iter: Box<dyn Iterator<Item = I>>,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

// Concrete implementation for `Vec<NestedValue>`
impl<A: RudaModuleAdapter> VecSeqAccess<A, NestedValue> {
    fn new(vec: NestedValue, default_for_missing_fields: bool) -> Self {
        match vec {
            NestedValue::Vec(v) => VecSeqAccess {
                iter: Box::new(v.into_iter()),
                default_for_missing_fields,
                phantom: std::marker::PhantomData,
            },
            _ => panic!("Invalid vec sequence"),
        }
    }
}

// Concrete implementation for `Vec<u8>`
impl<A: RudaModuleAdapter> VecSeqAccess<A, u8> {
    fn new(vec: NestedValue, default_for_missing_fields: bool) -> Self {
        match vec {
            NestedValue::U8s(v) => VecSeqAccess {
                iter: Box::new(v.into_iter()),
                default_for_missing_fields,
                phantom: std::marker::PhantomData,
            },
            _ => panic!("Invalid vec sequence"),
        }
    }
}

// Concrete implementation for `Vec<u16>`
impl<A: RudaModuleAdapter> VecSeqAccess<A, u16> {
    fn new(vec: NestedValue, default_for_missing_fields: bool) -> Self {
        match vec {
            NestedValue::U16s(v) => VecSeqAccess {
                iter: Box::new(v.into_iter()),
                default_for_missing_fields,
                phantom: std::marker::PhantomData,
            },
            _ => panic!("Invalid vec sequence"),
        }
    }
}

// Concrete implementation for `Vec<f32>`
impl<A: RudaModuleAdapter> VecSeqAccess<A, f32> {
    fn new(vec: NestedValue, default_for_missing_fields: bool) -> Self {
        match vec {
            NestedValue::F32s(v) => VecSeqAccess {
                iter: Box::new(v.into_iter()),
                default_for_missing_fields,
                phantom: std::marker::PhantomData,
            },
            _ => panic!("Invalid vec sequence"),
        }
    }
}

// Concrete implementation for `Vec<NestedValue>`
impl<'de, A> SeqAccess<'de> for VecSeqAccess<A, NestedValue>
where
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: RudaModuleAdapter,
{
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let item = match self.iter.next() {
            Some(v) => v,
            None => return Ok(None),
        };

        seed.deserialize(
            NestedValueWrapper::<A>::new(item, self.default_for_missing_fields).into_deserializer(),
        )
        .map(Some)
    }
}

// Concrete implementation for `Vec<u8>`
impl<'de, A> SeqAccess<'de> for VecSeqAccess<A, u8>
where
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: RudaModuleAdapter,
{
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let item = match self.iter.next() {
            Some(v) => v,
            None => return Ok(None),
        };

        seed.deserialize(
            NestedValueWrapper::<A>::new(NestedValue::U8(item), self.default_for_missing_fields)
                .into_deserializer(),
        )
        .map(Some)
    }
}

// Concrete implementation for `Vec<u16>`
impl<'de, A> SeqAccess<'de> for VecSeqAccess<A, u16>
where
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: RudaModuleAdapter,
{
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let item = match self.iter.next() {
            Some(v) => v,
            None => return Ok(None),
        };

        seed.deserialize(
            NestedValueWrapper::<A>::new(NestedValue::U16(item), self.default_for_missing_fields)
                .into_deserializer(),
        )
        .map(Some)
    }
}

// Concrete implementation for `Vec<f32>`
impl<'de, A> SeqAccess<'de> for VecSeqAccess<A, f32>
where
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: RudaModuleAdapter,
{
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let item = match self.iter.next() {
            Some(v) => v,
            None => return Ok(None),
        };

        seed.deserialize(
            NestedValueWrapper::<A>::new(NestedValue::F32(item), self.default_for_missing_fields)
                .into_deserializer(),
        )
        .map(Some)
    }
}

/// A map access for a map in the nested value data structure.
struct HashMapAccess<A: RudaModuleAdapter> {
    iter: std::collections::hash_map::IntoIter<String, NestedValue>,
    next_value: Option<NestedValue>,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

impl<A: RudaModuleAdapter> HashMapAccess<A> {
    fn new(map: HashMap<String, NestedValue>, default_for_missing_fields: bool) -> Self {
        HashMapAccess {
            iter: map.into_iter(),
            next_value: None,
            default_for_missing_fields,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'de, A> MapAccess<'de> for HashMapAccess<A>
where
    String: IntoDeserializer<'de, Error>,
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: RudaModuleAdapter,
{
    type Error = Error;

    fn next_key_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.iter.next() {
            Some((k, v)) => {
                // Keep the value for the next call to next_value_seed.
                self.next_value = Some(v);
                // Deserialize the key.
                seed.deserialize(k.into_deserializer()).map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<T>(&mut self, seed: T) -> Result<T::Value, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.next_value.take() {
            Some(NestedValue::Default(originator)) => {
                seed.deserialize(DefaultDeserializer::new(originator))
            }
            Some(v) => seed.deserialize(
                NestedValueWrapper::new(v, self.default_for_missing_fields).into_deserializer(),
            ),
            None => seed.deserialize(DefaultDeserializer::new(None)),
        }
    }
}

struct ProbeEnumAccess<A: RudaModuleAdapter> {
    value: NestedValue,
    current_variant: String,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

impl<A: RudaModuleAdapter> ProbeEnumAccess<A> {
    fn new(value: NestedValue, current_variant: String, default_for_missing_fields: bool) -> Self {
        ProbeEnumAccess {
            value,
            current_variant,
            default_for_missing_fields,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'de, A> EnumAccess<'de> for ProbeEnumAccess<A>
where
    A: RudaModuleAdapter,
{
    type Error = Error;
    type Variant = Self;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant), Self::Error>
    where
        V: DeserializeSeed<'de>,
    {
        seed.deserialize(self.current_variant.clone().into_deserializer())
            .map(|v| (v, self))
    }
}

impl<'de, A> VariantAccess<'de> for ProbeEnumAccess<A>
where
    A: RudaModuleAdapter,
{
    type Error = Error;

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let value = seed.deserialize(
            NestedValueWrapper::<A>::new(self.value, self.default_for_missing_fields)
                .into_deserializer(),
        )?;
        Ok(value)
    }

    fn unit_variant(self) -> Result<(), Self::Error> {
        // Support tensor `DType` deserialization
        match self.value {
            NestedValue::String(variant) if variant == self.current_variant => Ok(()),
            NestedValue::Map(value) if value.contains_key("DType") => {
                match value.get("DType") {
                    Some(NestedValue::String(variant)) => {
                        if *variant == self.current_variant {
                            Ok(())
                        } else {
                            Err(Error::Other("Wrong variant".to_string())) // wrong match
                        }
                    }
                    _ => panic!("expected DType variant as string"),
                }
            }
            _ => unimplemented!(
                "unit variant is not implemented because it is not used in the ruda module"
            ),
        }
    }

    fn tuple_variant<V>(self, _len: usize, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("tuple variant is not implemented because it is not used in the ruda module")
    }

    fn struct_variant<V>(
        self,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!(
            "struct variant is not implemented because it is not used in the ruda module"
        )
    }
}

/// A wrapper for the nested value data structure with a ruda module adapter.
struct NestedValueWrapper<A: RudaModuleAdapter> {
    value: NestedValue,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

impl<A: RudaModuleAdapter> NestedValueWrapper<A> {
    fn new(value: NestedValue, default_for_missing_fields: bool) -> Self {
        Self {
            value,
            default_for_missing_fields,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<A: RudaModuleAdapter> IntoDeserializer<'_, Error> for NestedValueWrapper<A> {
    type Deserializer = Deserializer<A>;

    fn into_deserializer(self) -> Self::Deserializer {
        Deserializer::<A>::new(self.value, self.default_for_missing_fields)
    }
}

/// A default deserializer that always returns the default value.
struct DefaultDeserializer {
    /// The originator field name (the top-level missing field name)
    originator_field_name: Option<String>,
}

impl DefaultDeserializer {
    fn new(originator_field_name: Option<String>) -> Self {
        Self {
            originator_field_name,
        }
    }
}

impl<'de> serde::Deserializer<'de> for DefaultDeserializer {
    type Error = Error;

    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!()
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i32(Default::default())
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f32(Default::default())
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i16(Default::default())
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(Default::default())
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u16(Default::default())
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u64(Default::default())
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f64(Default::default())
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_bool(Default::default())
    }

    fn deserialize_char<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_char(Default::default())
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_str(Default::default())
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i8(Default::default())
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u8(Default::default())
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u32(Default::default())
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_none()
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(DefaultSeqAccess::new(None))
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_string(Default::default())
    }

    fn deserialize_struct<V>(
        self,
        name: &'static str,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        // Return an error if the originator field name is not set
        Err(Error::Other(format!(
            "Missing source values for the '{}' field of type '{}'. Please verify the source data and ensure the field name is correct",
            self.originator_field_name.unwrap_or("UNKNOWN".to_string()),
            name,
        )))
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(DefaultSeqAccess::new(Some(len)))
    }

    fn deserialize_tuple<V>(self, len: usize, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(DefaultSeqAccess::new(Some(len)))
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(DefaultMapAccess::new())
    }

    forward_to_deserialize_any! {
        u128 bytes byte_buf unit unit_struct newtype_struct
        enum identifier ignored_any
    }
}

/// A default sequence access that always returns None (empty sequence).
pub struct DefaultSeqAccess {
    size: Option<usize>,
}

impl Default for DefaultSeqAccess {
    fn default() -> Self {
        Self::new(None)
    }
}

impl DefaultSeqAccess {
    /// Creates a new default sequence access with the given size hint.
    pub fn new(size: Option<usize>) -> Self {
        DefaultSeqAccess { size }
    }
}

impl<'de> SeqAccess<'de> for DefaultSeqAccess {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.size {
            Some(0) => Ok(None),
            Some(ref mut size) => {
                *size -= 1;
                seed.deserialize(DefaultDeserializer::new(None)).map(Some)
            }
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        self.size
    }
}

/// A default map access that always returns None (empty map).
pub struct DefaultMapAccess;

impl Default for DefaultMapAccess {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultMapAccess {
    /// Creates a new default map access.
    pub fn new() -> Self {
        DefaultMapAccess
    }
}

impl<'de> MapAccess<'de> for DefaultMapAccess {
    type Error = Error;

    fn next_key_seed<T>(&mut self, _seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        // Since this is a default implementation, we'll just return None.
        Ok(None)
    }

    fn next_value_seed<T>(&mut self, _seed: T) -> Result<T::Value, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        unimplemented!("This should never be called since next_key_seed always returns None")
    }

    fn size_hint(&self) -> Option<usize> {
        // Since this is a default implementation, we'll just return None.
        None
    }
}
