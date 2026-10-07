use super::{AdamRecords,Manifest,MuonError,AdaptorRecord,AdaptorRecordV1};
use alloc::format;
use ruda_model::tensor::backend::AutodiffBackend;

pub(super) fn validate_adam_records<B: AutodiffBackend>(records: &AdamRecords<B>,manifest: &Manifest) -> Result<(),MuonError> {
    for (id,state) in records {
        let expected = manifest.iter().find(|entry|entry.0 == id.val()).ok_or(MuonError::IncompatibleRecord)?;
        macro_rules! check_adam {
            ($state:expr) => {{
                let m = &$state.momentum;
                if m.time == 0 || m.moment_1.shape().to_vec() != expected.1 || m.moment_2.shape().to_vec() != expected.1
                    || format!("{:?}",m.moment_1.dtype()) != expected.3 || format!("{:?}",m.moment_2.dtype()) != expected.3
                    || m.max_moment_2.as_ref().is_some_and(|value|value.shape().to_vec() != expected.1 || format!("{:?}",value.dtype()) != expected.3) {
                    return Err(MuonError::IncompatibleRecord);
                }
            }};
        }
        match state {AdaptorRecord::V1(state)=>match state {
            AdaptorRecordV1::Rank0(value)=>check_adam!(value),AdaptorRecordV1::Rank1(value)=>check_adam!(value),
            AdaptorRecordV1::Rank2(value)=>check_adam!(value),AdaptorRecordV1::Rank3(value)=>check_adam!(value),
            AdaptorRecordV1::Rank4(value)=>check_adam!(value),AdaptorRecordV1::Rank5(value)=>check_adam!(value),
            AdaptorRecordV1::Rank6(value)=>check_adam!(value),AdaptorRecordV1::Rank7(value)=>check_adam!(value),
            AdaptorRecordV1::Rank8(value)=>check_adam!(value),
        }}
    }
    Ok(())
}
