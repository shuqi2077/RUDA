use super::{Error, NestedValue, Serializer};
use serde::{Serialize, ser::{SerializeMap, SerializeStruct, SerializeStructVariant,
    SerializeTuple, SerializeTupleStruct, SerializeTupleVariant}};
use std::collections::HashMap;

/// Serializes fixed-length heterogeneous values without sequence specialization.
pub struct TupleSerializer {
    values: Vec<NestedValue>,
    len: usize,
}

impl TupleSerializer {
    pub(super) fn new(len: usize) -> Self {
        Self { values: Vec::new(), len }
    }

    fn push<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        if self.values.len() == self.len {
            return Err(Error::Serialize("too many tuple elements".into()));
        }
        self.values.push(value.serialize(Serializer::new())?);
        Ok(())
    }

    fn finish(self) -> Result<NestedValue, Error> {
        if self.values.len() != self.len {
            return Err(Error::Serialize("tuple element count differs from declared length".into()));
        }
        Ok(NestedValue::Vec(self.values))
    }
}

impl SerializeTuple for TupleSerializer {
    type Ok = NestedValue;
    type Error = Error;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<NestedValue, Error> {
        self.finish()
    }
}

impl SerializeTupleStruct for TupleSerializer {
    type Ok = NestedValue;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<NestedValue, Error> {
        self.finish()
    }
}

pub(super) fn variant_value(name: &str, variant: &str, value: NestedValue) -> NestedValue {
    NestedValue::Map(HashMap::from([(name.to_owned(),
        NestedValue::Map(HashMap::from([(variant.to_owned(), value)])))]))
}

/// Serializes a named enum's fixed-length positional payload.
pub struct TupleVariantSerializer {
    name: &'static str,
    variant: &'static str,
    tuple: TupleSerializer,
}

impl TupleVariantSerializer {
    pub(super) fn new(name: &'static str, variant: &'static str, len: usize) -> Self {
        Self { name, variant, tuple: TupleSerializer::new(len) }
    }
}

impl SerializeTupleVariant for TupleVariantSerializer {
    type Ok = NestedValue;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.tuple.push(value)
    }

    fn end(self) -> Result<NestedValue, Error> {
        Ok(variant_value(self.name, self.variant, self.tuple.finish()?))
    }
}

/// Serializes a named enum's field payload using the existing struct serializer.
pub struct StructVariantSerializer {
    name: &'static str,
    variant: &'static str,
    fields: Serializer,
}

impl StructVariantSerializer {
    pub(super) fn new(name: &'static str, variant: &'static str) -> Self {
        Self { name, variant, fields: Serializer::new() }
    }
}

impl SerializeStructVariant for StructVariantSerializer {
    type Ok = NestedValue;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, value: &T) -> Result<(), Error> {
        SerializeStruct::serialize_field(&mut self.fields, key, value)
    }

    fn skip_field(&mut self, key: &'static str) -> Result<(), Error> {
        SerializeStruct::skip_field(&mut self.fields, key)
    }

    fn end(self) -> Result<NestedValue, Error> {
        Ok(variant_value(self.name, self.variant, SerializeStruct::end(self.fields)?))
    }
}

/// Serializes maps with the string keys required by NestedValue.
pub struct MapSerializer {
    values: HashMap<String, NestedValue>,
    pending_key: Option<String>,
}

impl MapSerializer {
    pub(super) fn new() -> Self {
        Self { values: HashMap::new(), pending_key: None }
    }
}

impl SerializeMap for MapSerializer {
    type Ok = NestedValue;
    type Error = Error;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Error> {
        if self.pending_key.is_some() {
            return Err(Error::Serialize("map key has no value".into()));
        }
        let NestedValue::String(key) = key.serialize(Serializer::new())? else {
            return Err(Error::Serialize("NestedValue map keys must serialize as strings".into()));
        };
        self.pending_key = Some(key);
        Ok(())
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        let key = self.pending_key.as_ref()
            .ok_or_else(|| Error::Serialize("map value has no key".into()))?;
        let value = value.serialize(Serializer::new())?;
        self.values.insert(key.clone(), value);
        self.pending_key = None;
        Ok(())
    }

    fn end(self) -> Result<NestedValue, Error> {
        if self.pending_key.is_some() {
            return Err(Error::Serialize("map key has no value".into()));
        }
        Ok(NestedValue::Map(self.values))
    }
}
