use alloc::vec::Vec;
use crate::ir::StorageType;
use crate::kernel::ScalarKernelArg;
use super::Metadata;

pub const INFO_ALIGN: usize = size_of::<u64>();

/// Helper to calculate info struct fields
#[derive(Clone, Debug, Default)]
pub struct Info {
    pub scalars: Vec<SizedInfoField>,
    pub sized_meta: Option<SizedInfoField>,
    pub has_dynamic_meta: bool,
    pub dynamic_meta_offset: usize,
    pub metadata: Metadata,
}

#[derive(Clone, Copy, Debug)]
pub struct SizedInfoField {
    pub ty: StorageType,
    pub size: usize,
    pub offset: usize,
}

impl SizedInfoField {
    pub fn padded_size(&self) -> usize {
        let padding_factor = INFO_ALIGN / self.ty.size();
        self.size.next_multiple_of(padding_factor)
    }
}

impl Info {
    pub fn new(scalars: &[ScalarKernelArg], metadata: Metadata, address_type: StorageType) -> Self {
        let mut scalar_fields = Vec::with_capacity(scalars.len());
        let mut sized_meta = None;

        let mut offset = 0;

        for scalar in scalars {
            scalar_fields.push(SizedInfoField {
                ty: scalar.ty,
                size: scalar.count,
                offset,
            });
            offset += (scalar.ty.size() * scalar.count).next_multiple_of(INFO_ALIGN);
        }

        if metadata.static_len() > 0 {
            let size = metadata.static_len() as usize;
            sized_meta = Some(SizedInfoField {
                ty: address_type,
                size,
                offset,
            });
            offset += (address_type.size() * size).next_multiple_of(INFO_ALIGN);
        }

        Info {
            scalars: scalar_fields,
            sized_meta,
            has_dynamic_meta: metadata.num_extended_meta() > 0,
            dynamic_meta_offset: offset,
            metadata,
        }
    }

    pub fn has_info(&self) -> bool {
        !self.scalars.is_empty() || self.sized_meta.is_some()
    }
}

