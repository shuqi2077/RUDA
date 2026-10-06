"""RUDA-native random filling with serializable seed/counter and dropout."""
from __future__ import annotations

import math
import secrets
import struct
from threading import RLock
import torch


class Generator:
    """Philox4x32-10 generator; four output slots are reserved per counter block.

    The byte state is RUDA-specific, not a PyTorch CUDA generator state. GPU
    sampling consumes only this explicit counter; no CPU random tensor is made.
    """
    def __init__(self, seed=None):
        self.lock = RLock()
        self.manual_seed(secrets.randbits(64) if seed is None else seed)

    def manual_seed(self, seed):
        if type(seed) is not int or not -(1<<63)<=seed<1<<64:
            raise ValueError('seed must fit a signed/unsigned 64-bit integer')
        with self.lock:
            self.seed_value, self.counter = seed%(1<<64), 0
        return self

    def initial_seed(self):
        return self.seed_value

    def get_state(self):
        with self.lock:
            return torch.tensor(list(struct.pack('<8sQQ',b'RUDARNG1',self.seed_value,self.counter)),dtype=torch.uint8,device='cpu')

    def set_state(self, state):
        if not isinstance(state,torch.Tensor) or state.device.type!='cpu' or state.dtype!=torch.uint8 or state.shape!=(24,):
            raise ValueError('RUDA RNG state must be a 24-byte CPU uint8 tensor')
        magic, seed, counter = struct.unpack('<8sQQ',bytes(state.tolist()))
        if magic!=b'RUDARNG1':
            raise ValueError('RUDA RNG state version differs')
        with self.lock:
            self.seed_value,self.counter=seed,counter
        return self

    def fill_(self, tensor, distribution, first, second):
        from . import _C, _random_available
        if not _random_available:
            raise RuntimeError('native random API 1 required; rebuild Rust and C++ libraries')
        if tensor.device.type!='ruda' or tensor.dtype not in (torch.float32,torch.float16,torch.bfloat16):
            raise ValueError('native random filling requires floating RUDA storage')
        if not math.isfinite(first) or not math.isfinite(second):
            raise ValueError('random distribution parameters must be finite')
        target = tensor if tensor.is_contiguous() else torch.empty(tensor.shape,device=tensor.device,dtype=tensor.dtype)
        blocks = (target.numel()+3)//4
        with self.lock:
            if self.counter+blocks >= 1<<64:
                raise OverflowError('RUDA random counter exhausted')
            counter = self.counter
            # Reservation precedes dispatch; a failed device operation is never
            # retried with reused random numbers and possibly partial writes.
            self.counter += blocks
            with torch.no_grad():
                _C.random_fill_(target,self.seed_value,counter,distribution,float(first),float(second))
                if target is not tensor:
                    tensor.copy_(target)
        return tensor


default_generator = Generator()


def _generator(generator):
    if generator is not None and not isinstance(generator,Generator):
        raise TypeError('use a RUDA Generator or the default generator, not a CPU/CUDA generator')
    return default_generator if generator is None else generator


def uniform_(tensor, from_=0., to=1., *, generator=None):
    if from_>to:
        raise ValueError('uniform bounds are reversed')
    return _generator(generator).fill_(tensor,0,from_,to)


def normal_(tensor, mean=0., std=1., *, generator=None):
    if std<0:
        raise ValueError('normal standard deviation must be nonnegative')
    return _generator(generator).fill_(tensor,1,mean,std)


def bernoulli_(tensor, p=.5, *, generator=None):
    if not 0<=p<=1:
        raise ValueError('Bernoulli probability must be in [0,1]')
    return _generator(generator).fill_(tensor,2,p,0.)


def native_dropout(input, p, train=True):
    if not 0<=p<=1:
        raise ValueError('dropout probability must be in [0,1]')
    if not train or p==0:
        return input.clone(),torch.ones_like(input,dtype=torch.bool)
    if p==1:
        mask=torch.zeros_like(input,dtype=torch.bool)
        return torch.where(mask,input,torch.zeros_like(input)),mask
    random = torch.empty(input.shape,dtype=torch.float32,device=input.device)
    uniform_(random)
    mask = random >= p
    return input*mask.to(input.dtype)/(1-p),mask


def register_random_ops():
    from ._ops import _registry
    _registry.impl('uniform_',lambda tensor,from_=0.,to=1.,generator=None: uniform_(tensor,from_,to,generator=generator))
    _registry.impl('normal_',lambda tensor,mean=0.,std=1.,generator=None: normal_(tensor,mean,std,generator=generator))
    _registry.impl('bernoulli_.float',lambda tensor,p=.5,generator=None: bernoulli_(tensor,p,generator=generator))
    def bernoulli_tensor(tensor,p,generator=None):
        if p.device!=tensor.device or not p.is_floating_point() or ((p<0)|(p>1)|torch.isnan(p)).any().item():
            raise ValueError('Bernoulli probabilities must be finite and in [0,1] on the output device')
        random=torch.empty(p.shape,dtype=torch.float32,device=p.device)
        uniform_(random,generator=generator)
        tensor.copy_((random<p).to(tensor.dtype))
        return tensor
    _registry.impl('bernoulli_.Tensor',bernoulli_tensor)
    _registry.impl('native_dropout',native_dropout)
    _registry.impl('native_dropout_backward',lambda grad,mask,scale: grad*mask.to(grad.dtype)*scale)
