use super::{OptimizerChoice,OptimizerChoiceState,SimpleOptimizer};
use hashbrown::HashMap;
use ruda_model::{module::ParamId,tensor::backend::AutodiffBackend};
use crate::record::{AdaptorRecord,AdaptorRecordV1};

macro_rules! wrap_record {
    ($record:expr,$side:ident;$($rank:ident),+) => {
        match $record {AdaptorRecord::V1(record)=>AdaptorRecord::V1(match record {
            $(AdaptorRecordV1::$rank(state)=>AdaptorRecordV1::$rank(OptimizerChoiceState::$side(state))),+
        })}
    };
}
macro_rules! unwrap_record {
    ($record:expr,$side:ident;$($rank:ident),+) => {
        match $record {AdaptorRecord::V1(record)=>match record {
            $(AdaptorRecordV1::$rank(OptimizerChoiceState::$side(state))=>Ok(AdaptorRecord::V1(AdaptorRecordV1::$rank(state))),
              original@AdaptorRecordV1::$rank(_)=>Err(AdaptorRecord::V1(original))),+
        }}
    };
}
macro_rules! record_is {
    ($record:expr,$side:ident;$($rank:ident),+) => {
        match $record {AdaptorRecord::V1(record)=>match record {
            $(AdaptorRecordV1::$rank(OptimizerChoiceState::$side(_))=>true),+,
            _=>false,
        }}
    };
}

impl<L,R> OptimizerChoice<L,R> {
    /// Move an original left algorithm's native rank-tagged history into a choice
    /// without tensor reads/clones/casts, changed clocks or default moment buffers.
    /// The source model IDs/configuration and existing original history stay exact.
    pub fn wrap_left_record<B>(record:AdaptorRecord<L,B>) -> AdaptorRecord<Self,B>
    where B:AutodiffBackend,L:SimpleOptimizer<B::InnerBackend>,R:SimpleOptimizer<B::InnerBackend> {
        wrap_record!(record,Left;Rank0,Rank1,Rank2,Rank3,Rank4,Rank5,Rank6,Rank7,Rank8)
    }
    pub fn wrap_right_record<B>(record:AdaptorRecord<R,B>) -> AdaptorRecord<Self,B>
    where B:AutodiffBackend,L:SimpleOptimizer<B::InnerBackend>,R:SimpleOptimizer<B::InnerBackend> {
        wrap_record!(record,Right;Rank0,Rank1,Rank2,Rank3,Rank4,Rank5,Rank6,Rank7,Rank8)
    }
    /// Remove only the explicit left tag, returning the entire unchanged original
    /// choice record on a branch mismatch. No native payload is discarded.
    pub fn try_unwrap_left_record<B>(record:AdaptorRecord<Self,B>) -> Result<AdaptorRecord<L,B>,AdaptorRecord<Self,B>>
    where B:AutodiffBackend,L:SimpleOptimizer<B::InnerBackend>,R:SimpleOptimizer<B::InnerBackend> {
        unwrap_record!(record,Left;Rank0,Rank1,Rank2,Rank3,Rank4,Rank5,Rank6,Rank7,Rank8)
    }
    pub fn try_unwrap_right_record<B>(record:AdaptorRecord<Self,B>) -> Result<AdaptorRecord<R,B>,AdaptorRecord<Self,B>>
    where B:AutodiffBackend,L:SimpleOptimizer<B::InnerBackend>,R:SimpleOptimizer<B::InnerBackend> {
        unwrap_record!(record,Right;Rank0,Rank1,Rank2,Rank3,Rank4,Rank5,Rank6,Rank7,Rank8)
    }
    /// Wrap actual original whole-group histories without changing parameter IDs,
    /// native device/rank, history absence or any recorder precision setting.
    pub fn wrap_left_records<B>(records:HashMap<ParamId,AdaptorRecord<L,B>>) -> HashMap<ParamId,AdaptorRecord<Self,B>>
    where B:AutodiffBackend,L:SimpleOptimizer<B::InnerBackend>,R:SimpleOptimizer<B::InnerBackend> {
        records.into_iter().map(|(id,record)|(id,Self::wrap_left_record::<B>(record))).collect()
    }
    pub fn wrap_right_records<B>(records:HashMap<ParamId,AdaptorRecord<R,B>>) -> HashMap<ParamId,AdaptorRecord<Self,B>>
    where B:AutodiffBackend,L:SimpleOptimizer<B::InnerBackend>,R:SimpleOptimizer<B::InnerBackend> {
        records.into_iter().map(|(id,record)|(id,Self::wrap_right_record::<B>(record))).collect()
    }
    /// All-or-nothing tag removal for a complete original group. Metadata is
    /// checked before consuming entries; any other branch returns the entire
    /// input map unchanged, not a partially converted replacement record.
    pub fn try_unwrap_left_records<B>(records:HashMap<ParamId,AdaptorRecord<Self,B>>)
        -> Result<HashMap<ParamId,AdaptorRecord<L,B>>,HashMap<ParamId,AdaptorRecord<Self,B>>>
    where B:AutodiffBackend,L:SimpleOptimizer<B::InnerBackend>,R:SimpleOptimizer<B::InnerBackend> {
        if records.values().any(|record|!record_is!(record,Left;Rank0,Rank1,Rank2,Rank3,Rank4,Rank5,Rank6,Rank7,Rank8)) {return Err(records);}
        Ok(records.into_iter().map(|(id,record)| {
            let record=Self::try_unwrap_left_record::<B>(record).unwrap_or_else(|_|unreachable!("validated original left algorithm branch"));(id,record)
        }).collect())
    }
    pub fn try_unwrap_right_records<B>(records:HashMap<ParamId,AdaptorRecord<Self,B>>)
        -> Result<HashMap<ParamId,AdaptorRecord<R,B>>,HashMap<ParamId,AdaptorRecord<Self,B>>>
    where B:AutodiffBackend,L:SimpleOptimizer<B::InnerBackend>,R:SimpleOptimizer<B::InnerBackend> {
        if records.values().any(|record|!record_is!(record,Right;Rank0,Rank1,Rank2,Rank3,Rank4,Rank5,Rank6,Rank7,Rank8)) {return Err(records);}
        Ok(records.into_iter().map(|(id,record)| {
            let record=Self::try_unwrap_right_record::<B>(record).unwrap_or_else(|_|unreachable!("validated original right algorithm branch"));(id,record)
        }).collect())
    }
}
