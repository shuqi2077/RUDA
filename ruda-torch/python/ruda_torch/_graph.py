"""Native fixed-address subgraphs with opt-in first-order training support."""
from collections.abc import Mapping
from threading import RLock
import torch
from . import _C
from ._graph_spec import GraphOp, TensorSpec
from ._graph_opt import prepare_plan

class StaticGraph:
    """Prepare once, update input CONTENTS, replay into reused native outputs.

    Only contiguous ruda:0 FP32/FP16/BF16 tensors and explicit copy/add/mul/SiLU/
    last-axis RMSNorm and storage-rounded SiLU-mul nodes are accepted. No
    broadcasting, pointer rebinding or hidden CPU path. All scalars are
    fixed at build. optimize=True prunes unused nodes and fuses single-use SiLU
    on the left of mul. reuse_workspace=True reuses equal-specification scratch
    allocations after their final consumer. Requested outputs remain distinct.
    Both options are opt-in; this is not automatic model capture or a global pool.

    Replays must use the creation stream. Synchronization follows RUDA_TORCH_ASYNC
    (default unchanged). External concurrent writes must be ordered explicitly.
    training=True enables a custom autograd bridge for all accepted GraphOps:
    forward uses the native graph, backward recomputes with same-device PyTorch
    operations. Backward/optimizer steps are NOT native graph capture. Only
    first-order gradients are supported. Training returns independent results
    and snapshots inputs for each forward, using additional memory. Do not
    modify parameters or inputs until their backward pass has completed.

    In inference mode results alias buffers overwritten on the next run. Clone
    a result before reuse when persistence is required. close() waits for the
    owning queue; retained output tensors remain valid after close.
    """
    @classmethod
    def from_model(cls, model, **options):
        """Return a callable CompiledModel/CompiledFunction, not a fixed-input graph.

        PyTorch captures model forward/backward; eligible subgraphs use this
        class internally. Other operations stay on their original device.
        Call the result with new inputs normally, rather than using replay().
        """
        from .compiler import compile
        return compile(model, **options)

    def __init__(self, inputs, nodes, *, outputs=None, infer_dependencies=False,
                 track_completion=False, optimize=False, reuse_workspace=False, training=False):
        from . import _graph_available
        if not _graph_available:
            raise RuntimeError('native static graph API unavailable; rebuild Rust and C++ extensions')
        if not isinstance(inputs,Mapping): raise TypeError('inputs must be a name-to-tensor mapping')
        if type(training) is not bool: raise TypeError('training must be bool')
        metadata={}
        for name,t in inputs.items():
            if not isinstance(t,torch.Tensor): raise TypeError('inputs must contain torch.Tensor values')
            if t.device.type!='ruda' or t.device.index not in (None,0):
                raise ValueError('StaticGraph accepts native ruda:0 tensors, not CPU/CUDA tensors')
            if t.requires_grad and not training:
                raise ValueError('StaticGraph is inference-only unless training=True; requires_grad inputs need StaticGraph(training=True)')
            if t.layout!=torch.strided or not t.is_contiguous():
                raise ValueError('StaticGraph requires contiguous dense tensors')
            metadata[name]=TensorSpec(tuple(t.shape),str(t.dtype).removeprefix('torch.'))
        if type(infer_dependencies) is not bool or type(track_completion) is not bool:
            raise TypeError('graph options must be booleans')
        plan=prepare_plan(metadata,nodes,outputs,optimize=optimize,reuse_workspace=reuse_workspace)
        layout=plan.layout
        self._inputs=tuple(inputs.values())
        self._input_bindings=tuple(self._binding(t) for t in self._inputs)
        self._training=training
        self._replay_lock=RLock()
        # Native handles are fixed-address detached views, not new allocations.
        tensors=[t.detach() for t in self._inputs] if training else list(self._inputs)
        device=tensors[0].device
        for i in range(layout.inputs,len(layout.specs)):
            spec=layout.specs[i]
            root=plan.storage_roots[i]
            if root==i:
                tensors.append(torch.empty(spec.shape,dtype=getattr(torch,spec.dtype),device=device))
            else:
                tensors.append(tensors[root])
        self._plan=plan
        self._optimize=optimize
        self._reuse_workspace=reuse_workspace
        self._layout=layout
        self._tensors=tuple(tensors)
        self._native=_C.NativeStaticGraph(self._tensors,layout.inputs,layout.words,layout.scalars,
                                          track_completion,infer_dependencies)
        self._closed=False
        self._outputs={layout.names[i]:self._tensors[i] for i in layout.output_indices}
        self._track_completion=track_completion

    def _check(self):
        if self._closed: raise RuntimeError('StaticGraph is closed')

    @staticmethod
    def _binding(t):
        return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)

    def _check_bindings(self):
        if tuple(self._binding(t) for t in self._inputs) != self._input_bindings:
            raise RuntimeError('StaticGraph input binding changed (storage/shape/stride); rebuild the graph')

    def _run(self, eager):
        with self._replay_lock:
            self._check()
            self._check_bindings()
            if self._training:
                from ._graph_autograd import StaticGraphFunction
                result=StaticGraphFunction.apply(self,eager,*self._inputs)
                return dict(zip((self._layout.names[i] for i in self._layout.output_indices),
                                result, strict=True))
            self._native.run(eager)
            return dict(self._outputs)

    def replay(self):
        """Submit native forward; training results own storage and an autograd edge."""
        return self._run(False)

    def run_eager(self):
        """Preallocated control: same kernels/scalars, one final sync policy.

        For comparison, not a compatibility fallback. This enqueues every kernel
        normally instead of launching the graph; numerical expressions are equal.
        """
        return self._run(True)

    def synchronize(self): self._check(); self._native.synchronize()
    def query(self): self._check(); return self._native.query()
    def query_completion(self): self._check(); return self._native.query_completion()
    def wait_completion(self): self._check(); self._native.wait_completion()

    @property
    def info(self):
        return {'training':self._training,
                'backward_execution':'same-device-eager' if self._training else None,
                'nodes':len(self._layout.scalars),'edges':self._native.edge_count,
                'stream_id':self._native.stream_id,'workspace_bytes':self._plan.workspace_bytes,
                'logical_workspace_bytes':self._layout.workspace_bytes,
                'unoptimized_workspace_bytes':self._plan.original_workspace_bytes,
                'workspace_allocations':self._plan.workspace_allocations,
                'unoptimized_nodes':self._plan.original_nodes,
                'eliminated_outputs':self._plan.eliminated_outputs,
                'fused_activations':self._plan.fused_activations,
                'optimize':self._optimize,'reuse_workspace':self._reuse_workspace,
                'outputs':tuple(self._layout.names[i] for i in self._layout.output_indices),'tracked_completion':self._track_completion,
                'base_abi':10,'graph_api':3,'closed':self._closed}

    def close(self):
        with self._replay_lock:
            if not self._closed:
                self._native.close(); self._closed=True
                self._tensors=(); self._outputs={}; self._inputs=()

    def __enter__(self): self._check(); return self
    def __exit__(self,*_): self.close()
