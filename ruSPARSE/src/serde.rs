use crate::{CsrMatrixOwned, IndexBase};
use ::serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};

impl Serialize for CsrMatrixOwned {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (
            self.rows,
            self.columns,
            &self.row_offsets,
            &self.column_indices,
            &self.values,
            self.index_base.value(),
        ).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CsrMatrixOwned {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (rows, columns, offsets, indices, values, base) =
            <(usize, usize, Vec<u32>, Vec<u32>, Vec<f32>, u32)>::deserialize(deserializer)?;
        let base = match base {
            0 => IndexBase::Zero,
            1 => IndexBase::One,
            _ => return Err(D::Error::custom("CSR index base must be zero or one")),
        };
        Self::new(rows, columns, offsets, indices, values, base).map_err(D::Error::custom)
    }
}
