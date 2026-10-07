use super::*;
use crate::OptimizerTensorShard;

/// One actual destination leaf and its explicit original complete-parameter coordinates.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct LBFGSMasterTensorShard {
    /// Original complete native parameter ID.
    pub source_parameter:u64,
    /// Actual loaded destination leaf ID; no new ID is generated during conversion.
    pub local_parameter:u64,
    /// Explicit original tensor axis/interval, or None to own the complete parameter.
    pub shard:Option<OptimizerTensorShard>,
}

fn axis_view(shape:&[usize],shard:&OptimizerTensorShard) -> Result<([usize;3],usize),LBFGSShardError> {
    let product = |axes:&[usize]|axes.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis))
        .ok_or(LBFGSShardError::Shape("master axis view overflows"));
    let before = product(&shape[..shard.axis])?;
    let after = product(&shape[shard.axis+1..])?;
    let length = before.checked_mul(shard.interval.len()).and_then(|length|length.checked_mul(after))
        .ok_or(LBFGSShardError::Shape("master local axis view overflows"))?;
    Ok(([before,shape[shard.axis],after],length))
}

impl<B:Backend> LBFGSFp32MasterState<B> {
    /// Convert a complete original model/master/history record into actual per-parameter shards.
    /// Supply every rank's native unique-leaf order and explicit tensor coordinates. Complete
    /// nonoverlapping coverage is checked before tensor work, so a flattened global checkpoint
    /// is not incorrectly sliced as rank-concatenated local models. Returned masters/history
    /// retain FP32 values and original counters; destination model weights must match these slices.
    pub fn partition_from_full(
        &self,
        placements:&[Vec<LBFGSMasterTensorShard>],
        rank:u32,
    ) -> Result<Self,LBFGSShardError> {
        self.validate()?;
        if self.placement.is_some() {return Err(LBFGSShardError::Record);}
        let full_master = self.master.as_ref().ok_or(LBFGSShardError::Record)?;
        let world = u32::try_from(placements.len()).map_err(|_|LBFGSShardError::Layout("master destination rank count overflows"))?;
        if world == 0 || rank >= world {return Err(LBFGSShardError::Layout("master destination rank is outside placements"));}

        let mut originals = HashMap::new();
        let mut offset = 0usize;
        for parameter in &self.parameters {
            let length = parameter_length(core::slice::from_ref(parameter))?;
            let end = offset.checked_add(length).ok_or(LBFGSShardError::Shape("complete parameter interval overflows"))?;
            originals.insert(parameter.id,(parameter,offset..end));offset = end;
        }
        let mut coverage:HashMap<u64,Vec<Option<(usize,core::ops::Range<usize>)>>> = HashMap::new();
        let mut rank_parameters = Vec::new();
        let mut lengths = Vec::new();
        for placement in placements {
            let mut parameters = Vec::new();
            for entry in placement {
                let (original,_) = originals.get(&entry.source_parameter).ok_or(LBFGSShardError::Record)?;
                let shape = if let Some(shard) = &entry.shard {
                    shard.validate().map_err(|_|LBFGSShardError::Layout("invalid master tensor coordinates"))?;
                    if shard.global_shape != original.shape {return Err(LBFGSShardError::Shape("master tensor source axes"));}
                    axis_view(&original.shape,shard)?;
                    coverage.entry(entry.source_parameter).or_default().push(Some((shard.axis,shard.interval.clone())));
                    shard.local_shape().map_err(|_|LBFGSShardError::Shape("master local parameter axes"))?
                } else {
                    coverage.entry(entry.source_parameter).or_default().push(None);
                    original.shape.clone()
                };
                parameters.push(LBFGSMasterParameter {id:entry.local_parameter,shape,storage:original.storage});
            }
            lengths.push(parameter_length(&parameters)?);
            rank_parameters.push(parameters);
        }
        let layout = LBFGSShardLayout::new(lengths);
        layout.validate(rank,world)?;
        if layout.global_len()? != full_master.dims()[0] {return Err(LBFGSShardError::Shape("destination changes complete master length"));}

        for original in &self.parameters {
            let pieces = coverage.get_mut(&original.id).ok_or(LBFGSShardError::Layout("complete master parameter has no owner"))?;
            if pieces.iter().any(Option::is_none) {
                if pieces.len() != 1 {return Err(LBFGSShardError::Layout("complete parameter ownership overlaps another slice"));}
                continue;
            }
            let axis = pieces[0].as_ref().expect("actual tensor interval").0;
            if pieces.iter().any(|piece|piece.as_ref().expect("actual tensor interval").0 != axis) {
                return Err(LBFGSShardError::Layout("one original tensor must use the same partition axis"));
            }
            pieces.sort_by_key(|piece|piece.as_ref().expect("actual tensor interval").1.start);
            let mut cursor = 0usize;
            for piece in pieces {
                let interval = &piece.as_ref().expect("actual tensor interval").1;
                if interval.start != cursor {return Err(LBFGSShardError::Layout("master tensor partition has overlap or missing coordinates"));}
                cursor = interval.end;
            }
            if cursor != original.shape[axis] {return Err(LBFGSShardError::Layout("master tensor partition is incomplete"));}
        }

        let destination = &placements[rank as usize];
        let parameters = rank_parameters.swap_remove(rank as usize);
        let convert = |value:Tensor<B,1>| {
            let mut pieces = Vec::new();
            for entry in destination {
                let (original,interval) = originals.get(&entry.source_parameter).expect("validated original parameter");
                let parameter = if interval.start == 0 && interval.end == value.dims()[0] {
                    value.clone()
                } else {value.clone().slice(interval.clone())};
                let local = if let Some(shard) = &entry.shard {
                    if shard.interval == (0..original.shape[shard.axis]) {parameter} else {
                        let (shape,length) = axis_view(&original.shape,shard).expect("validated exact tensor axis view");
                        parameter.reshape(shape)
                            .slice_dim(1,shard.interval.clone()).reshape([length])
                    }
                } else {parameter};
                pieces.push(local);
            }
            if pieces.len() == 1 {pieces.pop().expect("one actual destination parameter")} else {Tensor::cat(pieces,0)}
        };
        let state = LBFGSFp32MasterState {
            version:1,
            parameters,
            master:Some(convert(full_master.clone())),
            optimizer:LBFGSState {
                history_s:self.optimizer.history_s.iter().cloned().map(&convert).collect(),
                history_y:self.optimizer.history_y.iter().cloned().map(&convert).collect(),
                d:self.optimizer.d.clone().map(&convert),
                t:self.optimizer.t,
                prev_flat_grad:self.optimizer.prev_flat_grad.clone().map(&convert),
                prev_loss:self.optimizer.prev_loss,
                g_iter:self.optimizer.g_iter,
            },
            placement:Some((rank,layout)),
        };
        state.validate()?;Ok(state)
    }
}
