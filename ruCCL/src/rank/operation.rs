use super::error::RankError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ReductionOperation {
    Sum = 0,
    Product = 1,
    Minimum = 2,
    Maximum = 3,
    BitAnd = 4,
    BitOr = 5,
    BitXor = 6,
}

impl TryFrom<u32> for ReductionOperation {
    type Error = RankError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Sum),
            1 => Ok(Self::Product),
            2 => Ok(Self::Minimum),
            3 => Ok(Self::Maximum),
            4 => Ok(Self::BitAnd),
            5 => Ok(Self::BitOr),
            6 => Ok(Self::BitXor),
            _ => Err(RankError::InvalidLength(
                "unknown collective reduction operation",
            )),
        }
    }
}
