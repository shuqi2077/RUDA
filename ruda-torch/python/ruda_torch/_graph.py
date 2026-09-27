"""Native fixed-address inference subgraphs, not torch.cuda.graph or model capture."""
from collections.abc import Mapping
import torch
from . import _C
from ._graph_spec import GraphOp, TensorSpec
from ._graph_opt import prepare_plan

class StaticGraph:
    """Prepare once, update input CONTENTS, replay into reused native outputs.

    Only contiguous ruda:0 FP32/FP16/BF16 tensors and explicit copy/add/mul/SiLU/
    last-axis RMSNorm and storage-rounded SiLU-mul nodes are accepted. No
    broadcasting, autograd, pointer rebinding or hidden CPU path. All scalars are
    fixed at build. optimize=True prunes unused nodes and fuses single-use SiLU
    on the left of mul. reuse_workspace=True reuses equal-specification scratch
    allocations after their final consumer. Requested outputs remain distinct.
    Both options are opt-in; this is not automatic model capture or a global pool.

    Replays must use the creation stream. Synchronization follows RUDA_TORCH_ASYNC
    (default unchanged). External concurrent writes must be ordered explicitly.
    Results alias plan-owned buffers and are overwritten on the next run. Clone
    a result before reuse when persistence is required. close() waits for the
    owning queue; retained output tensors remain valid after close.
    """
    def __init__(self, inputs, nodes, *, outputs=None, infer_dependencies=False,
                 track_completion=False, optimize=False, reuse_workspace=False):
        from . import _graph_available
        if not _graph_available:
            raise RuntimeError('native static graph API unavailable; rebuild Rust and C++ extensions')
        if not isinstance(inputs,Mapping): raise TypeError('inputs must be a name-to-tensor mapping')
        metadata={}
        for name,t in inputs.items():
            if not isinstance(t,torch.Tensor): raise TypeError('inputs must contain torch.Tensor values')
            if t.device.type!='ruda' or t.device.index not in (None,0):
                raise ValueError('StaticGraph accepts native ruda:0 tensors, not CPU/CUDA tensors')
            if t.requires_grad: raise ValueError('StaticGraph is inference-only')
            if t.layout!=torch.strided or not t.is_contiguous():
                raise ValueError('StaticGraph requires contiguous dense tensors')
            metadata[name]=TensorSpec(tuple(t.shape),str(t.dtype).removeprefix('torch.'))
        if type(infer_dependencies) is not bool or type(track_completion) is not bool:
            raise TypeError('graph options must be booleans')
        plan=prepare_plan(metadata,nodes,outputs,optimize=optimize,reuse_workspace=reuse_workspace)
        layout=plan.layout
        tensors=list(inputs.values())
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

    def replay(self):
        """Return reused output tensors after one native graph submission."""
        self._check(); self._native.run(False); return dict(self._outputs)

    def run_eager(self):
        """Preallocated control: same kernels/scalars, one final sync policy.

        For comparison, not a compatibility fallback. This enqueues every kernel
        normally instead of launching the graph; numerical expressions are equal.
        """
        self._check(); self._native.run(True); return dict(self._outputs)

    def synchronize(self): self._check(); self._native.synchronize()
    def query(self): self._check(); return self._native.query()
    def query_completion(self): self._check(); return self._native.query_completion()
    def wait_completion(self): self._check(); self._native.wait_completion()

    @property
    def info(self):
        return {'nodes':len(self._layout.scalars),'edges':self._native.edge_count,
                'stream_id':self._native.stream_id,'workspace_bytes':self._plan.workspace_bytes,
                'logical_workspace_bytes':self._layout.workspace_bytes,
                'unoptimized_workspace_bytes':self._plan.original_workspace_bytes,
                'workspace_allocations':self._plan.workspace_allocations,
                'unoptimized_nodes':self._plan.original_nodes,
                'eliminated_outputs':self._plan.eliminated_outputs,
                'fused_activations':self._plan.fused_activations,
                'optimize':self._optimize,'reuse_workspace':self._reuse_workspace,
                'outputs':tuple(self._layout.names[i] for i in self._layout.output_indices),'tracked_completion':self._track_completion,
                'base_abi':9,'graph_api':2,'closed':self._closed}

    def close(self):
        if not self._closed:
            self._native.close(); self._closed=True
            self._tensors=(); self._outputs={}

    def __enter__(self): self._check(); return self
    def __exit__(self,*_): self.close()
