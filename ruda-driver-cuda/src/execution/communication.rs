use std::{collections::HashMap, sync::OnceLock};

use ruda_kernel::dsl::{
    device::DeviceId,
    ir::ElemType,
    server::{CommunicationId, ReduceOperation},
    stub::Mutex,
};

/// Global state map from [`CommunicationId`] to boxed [`cudarc::nccl::sys::ncclUniqueId`].
static UNIQUE_IDS_MAP: OnceLock<Mutex<HashMap<CommunicationId, cudarc::nccl::sys::ncclUniqueId>>> =
    OnceLock::new();

pub(crate) fn get_nccl_comm_id(device_ids: Vec<DeviceId>) -> cudarc::nccl::sys::ncclUniqueId {
    let mut unique_ids_map = UNIQUE_IDS_MAP.get_or_init(Default::default).lock().unwrap();
    let comm_id = CommunicationId::from(device_ids);
    match unique_ids_map.get_mut(&comm_id) {
        Some(id) => *id,
        None => {
            let id = cudarc::nccl::result::get_uniqueid().unwrap();
            unique_ids_map.insert(comm_id, id);
            id
        }
    }
}

pub(crate) fn to_nccl_op(op: ReduceOperation) -> cudarc::nccl::sys::ncclRedOp_t {
    match op {
        ReduceOperation::Sum => cudarc::nccl::sys::ncclRedOp_t::ncclSum,
        ReduceOperation::Mean => cudarc::nccl::sys::ncclRedOp_t::ncclAvg,
    }
}

pub(crate) fn get_nccl_dtype_count(
    dtype: ElemType,
    size: u64,
) -> (cudarc::nccl::sys::ncclDataType_t, usize) {
    match dtype {
        ElemType::Float(
            ruda_kernel::dsl::ir::FloatKind::E2M1
            | ruda_kernel::dsl::ir::FloatKind::E2M3
            | ruda_kernel::dsl::ir::FloatKind::E3M2
            | ruda_kernel::dsl::ir::FloatKind::UE8M0,
        ) => panic!("Minifloat not supported in NCCL"),
        ElemType::Float(ruda_kernel::dsl::ir::FloatKind::E4M3) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat8e4m3,
            size as usize,
        ),
        ElemType::Float(ruda_kernel::dsl::ir::FloatKind::E5M2) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat8e5m2,
            size as usize,
        ),
        ElemType::Float(ruda_kernel::dsl::ir::FloatKind::F16) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat16,
            (size / 2) as usize,
        ),
        ElemType::Float(ruda_kernel::dsl::ir::FloatKind::BF16) => (
            cudarc::nccl::sys::ncclDataType_t::ncclBfloat16,
            (size / 2) as usize,
        ),
        ElemType::Float(ruda_kernel::dsl::ir::FloatKind::Flex32) => {
            panic!("NCCL doesn't support Flex32 format.")
        }

        ElemType::Float(ruda_kernel::dsl::ir::FloatKind::F32) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
            (size / 4) as usize,
        ),
        ElemType::Float(ruda_kernel::dsl::ir::FloatKind::TF32) => {
            panic!("NCCL doesn't support TF32 format.")
        }
        ElemType::Float(ruda_kernel::dsl::ir::FloatKind::F64) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat64,
            (size / 8) as usize,
        ),
        ElemType::Int(int_kind) => match int_kind {
            ruda_kernel::dsl::ir::IntKind::I8 => {
                (cudarc::nccl::sys::ncclDataType_t::ncclInt8, size as usize)
            }
            ruda_kernel::dsl::ir::IntKind::I16 => panic!("NCCL doesn't support Int16 format."),
            ruda_kernel::dsl::ir::IntKind::I32 => (
                cudarc::nccl::sys::ncclDataType_t::ncclInt32,
                (size / 4) as usize,
            ),
            ruda_kernel::dsl::ir::IntKind::I64 => (
                cudarc::nccl::sys::ncclDataType_t::ncclInt64,
                (size / 8) as usize,
            ),
        },
        ElemType::UInt(uint_kind) => match uint_kind {
            ruda_kernel::dsl::ir::UIntKind::U8 => {
                (cudarc::nccl::sys::ncclDataType_t::ncclUint8, size as usize)
            }
            ruda_kernel::dsl::ir::UIntKind::U16 => panic!("NCCL doesn't support UInt16 format."),
            ruda_kernel::dsl::ir::UIntKind::U32 => (
                cudarc::nccl::sys::ncclDataType_t::ncclUint32,
                (size / 4) as usize,
            ),
            ruda_kernel::dsl::ir::UIntKind::U64 => (
                cudarc::nccl::sys::ncclDataType_t::ncclUint64,
                (size / 8) as usize,
            ),
        },
        ElemType::Bool => panic!("NCCL doesn't support Bool format."),
    }
}
