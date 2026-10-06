"""Committed sample cursors and epoch-preserving elastic data repartitioning."""
from __future__ import annotations

import copy
import torch
from torch.utils.data import Sampler


class StatefulShardSampler(Sampler):
    """DP-owned sample order, with committed progress separate from prefetch.

    commit(n) is called after actually consuming n dataset examples. Merely
    yielding indices to DataLoader workers never advances the saved checkpoint.
    tail='uneven' does not duplicate examples; tail='drop' explicitly omits the
    final incomplete DP round. TP/PP ranks can share a data coordinate via mesh.
    source_id identifies the caller's immutable dataset snapshot.
    """
    def __init__(self,length,*,source_id,rank=0,replicas=1,shuffle=False,seed=0,tail='uneven'):
        if type(length) is not int or length<0 or type(replicas) is not int or replicas<1 or type(rank) is not int or not 0<=rank<replicas:
            raise ValueError('invalid dataset length or data shard coordinates')
        if not isinstance(source_id,str) or not source_id or type(shuffle) is not bool or type(seed) is not int or not 0<=seed<1<<64:
            raise ValueError('identify a dataset snapshot and explicit shuffle/64-bit seed policy')
        if tail not in ('uneven','drop'):raise ValueError('tail must explicitly keep uneven shards or drop the last incomplete round')
        self.length,self.source_id,self.rank,self.replicas=length,source_id,rank,replicas
        self.shuffle,self.seed,self.tail=shuffle,seed,tail
        self.epoch=0
        self.limit=length if tail=='uneven' else length-length%replicas
        self.excluded=[]
        self.committed,self.issued=0,0
        self._indices,self._positions=None,None

    @classmethod
    def from_mesh(cls,length,mesh,**options):
        return cls(length,rank=mesh.coordinates[0],replicas=mesh.shape[0],**options)

    def _prepare(self):
        if self._positions is None:
            positions=range(self.rank,self.limit,self.replicas)
            self._positions=positions if not self.excluded else [position for position in positions
                if not any(position<span['stop'] and position%span['replicas']==span['rank'] for span in self.excluded)]
        if self._indices is None:
            if self.shuffle:
                generator=torch.Generator(device='cpu').manual_seed((self.seed+self.epoch)%(1<<64))
                self._indices=torch.randperm(self.length,generator=generator,device='cpu')
            else:self._indices=range(self.length)

    def __len__(self):
        self._prepare()
        return len(self._positions)

    @property
    def remaining(self):return len(self)-self.committed

    def __iter__(self):
        self._prepare()
        while self.issued<len(self._positions):
            position=self._positions[self.issued]
            self.issued+=1
            yield int(self._indices[position])

    def commit(self,count):
        if type(count) is not int or count<0 or self.committed+count>self.issued:
            raise ValueError('commit only examples already issued and actually consumed')
        self.committed+=count

    def rewind_uncommitted(self):
        """Explicitly replay prefetched but unconsumed indices with a NEW loader iterator."""
        self.issued=self.committed

    def set_epoch(self,epoch):
        if type(epoch) is not int or epoch<0:raise ValueError('epoch must be a nonnegative integer')
        self.epoch=epoch
        self.limit=self.length if self.tail=='uneven' else self.length-self.length%self.replicas
        self.excluded=[]
        self.committed,self.issued=0,0
        self._indices,self._positions=None,None

    def state_dict(self):
        self._prepare()
        stop=0 if not self.committed else self._positions[self.committed-1]+1
        return {'version':1,'source_id':self.source_id,'length':self.length,'rank':self.rank,'replicas':self.replicas,
                'shuffle':self.shuffle,'seed':self.seed,'tail':self.tail,'epoch':self.epoch,'limit':self.limit,
                'excluded':copy.deepcopy(self.excluded),'cursor':self.committed,'stop':stop,
                'order_version':str(torch.__version__) if self.shuffle else None}

    def load_state_dict(self,state):
        expected=self.state_dict()
        fixed=('version','source_id','length','rank','replicas','shuffle','seed','tail','order_version')
        if set(state)!=set(expected) or any(state[key]!=expected[key] for key in fixed):
            raise ValueError('sampler snapshot/source/order/topology differs')
        if type(state['epoch']) is not int or state['epoch']<0 or type(state['limit']) is not int or not 0<=state['limit']<=self.length:
            raise ValueError('invalid sampler epoch or original sample scope')
        spans=state['excluded']
        if not isinstance(spans,list) or any(set(span)!={'rank','replicas','stop'} or
                type(span['replicas']) is not int or span['replicas']<1 or type(span['rank']) is not int or not 0<=span['rank']<span['replicas'] or
                type(span['stop']) is not int or not 0<=span['stop']<=state['limit'] for span in spans):
            raise ValueError('invalid consumed sample spans')
        replacement=type(self)(self.length,source_id=self.source_id,rank=self.rank,replicas=self.replicas,
                               shuffle=self.shuffle,seed=self.seed,tail=self.tail)
        replacement.epoch,replacement.limit,replacement.excluded=state['epoch'],state['limit'],copy.deepcopy(spans)
        replacement._prepare()
        cursor=state['cursor']
        if type(cursor) is not int or not 0<=cursor<=len(replacement):raise ValueError('sampler committed cursor exceeds its shard')
        stop=0 if cursor==0 else replacement._positions[cursor-1]+1
        if state['stop']!=stop:raise ValueError('sampler cutoff does not match its committed cursor')
        self.epoch,self.limit,self.excluded=replacement.epoch,replacement.limit,replacement.excluded
        self.committed,self.issued=cursor,cursor
        self._indices,self._positions=replacement._indices,replacement._positions

    def validate_state_dict(self,state):
        clone=type(self)(self.length,source_id=self.source_id,rank=self.rank,replicas=self.replicas,
                         shuffle=self.shuffle,seed=self.seed,tail=self.tail)
        clone.load_state_dict(state)

    @staticmethod
    def consolidate(states):
        """CPU metadata operation; include all DP ranks, allowing identical TP copies."""
        states=list(states)
        if not states:raise ValueError('supply all data-rank sampler checkpoints')
        first=states[0]
        fixed=('version','source_id','length','replicas','shuffle','seed','tail','epoch','limit','excluded','order_version')
        ranks={}
        for state in states:
            if any(state[key]!=first[key] for key in fixed):raise ValueError('sampler rank snapshots describe different epochs/sources')
            sampler=StatefulShardSampler(state['length'],source_id=state['source_id'],rank=state['rank'],replicas=state['replicas'],
                                         shuffle=state['shuffle'],seed=state['seed'],tail=state['tail'])
            sampler.load_state_dict(state)
            if state['rank'] in ranks and ranks[state['rank']]!=state:raise ValueError('TP copies have different committed sample cursors')
            ranks[state['rank']]=state
        if set(ranks)!=set(range(first['replicas'])):raise ValueError('not every data rank supplied a committed sampler state')
        result={key:copy.deepcopy(first[key]) for key in fixed if key!='replicas'}
        result['excluded']+= [{'rank':rank,'replicas':first['replicas'],'stop':state['stop']}
                             for rank,state in sorted(ranks.items()) if state['cursor']]
        return result

    @classmethod
    def from_consolidated(cls,state,*,rank,replicas):
        """Redistribute only unconsumed original-epoch positions; no padding/repetition."""
        if state.get('version')!=1:raise ValueError('unsupported consolidated sampler state')
        result=cls(state['length'],source_id=state['source_id'],rank=rank,replicas=replicas,
                   shuffle=state['shuffle'],seed=state['seed'],tail=state['tail'])
        record=result.state_dict()
        record.update(epoch=state['epoch'],limit=state['limit'],excluded=state['excluded'],order_version=state['order_version'])
        result.load_state_dict(record)
        return result


def collate_varlen_causal_lm(samples,*,ignore_index=-100):
    """Pack explicit labels WITHOUT cross-document next-token targets or truncation.

    cu_seqlens is CPU metadata for varlen attention. position_ids reset for each
    document. labels at each document start are ignored: shifting packed rows
    must not train the preceding document to predict the next document's start.
    """
    ids,labels,positions,boundaries=[],[],[],[0]
    for sample in samples:
        tokens,targets=list(sample['input_ids']),list(sample['labels'])
        if not tokens or len(tokens)!=len(targets) or any(type(value) is not int or value<0 for value in tokens):
            raise ValueError('supply nonempty explicit token/label sequences')
        if any(type(value) is not int or value<0 and value!=ignore_index for value in targets):raise ValueError('invalid supervised token IDs')
        targets[0]=ignore_index
        ids.extend(tokens)
        labels.extend(targets)
        positions.extend(range(len(tokens)))
        boundaries.append(len(ids))
    if len(boundaries)==1:raise ValueError('packed batch must contain actual documents')
    return {'input_ids':torch.tensor(ids,dtype=torch.int64),'labels':torch.tensor(labels,dtype=torch.int64),
            'attention_mask':torch.ones(len(ids),dtype=torch.bool),
            'position_ids':torch.tensor(positions,dtype=torch.int64),'cu_seqlens':torch.tensor(boundaries,dtype=torch.int64),
            'max_seqlen':max(b-a for a,b in zip(boundaries,boundaries[1:]))}
