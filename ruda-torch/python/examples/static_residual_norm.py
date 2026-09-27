"""Native PyTorch -> RUDA graph -> PTX example; no model download or CPU fallback.

Set RUDA_CUDA_COMPILER=ptx and device-supported RUDA_PTX_VERSION before running.
Must install/rebuild matching Rust library and C++ bridge. Not an entire model.
"""
import torch
import ruda_torch as r

with torch.inference_mode():
    x=torch.randn(1,4096).to('ruda')
    residual=torch.randn(1,4096).to('ruda')
    weight=torch.ones(4096).to('ruda')
    with r.StaticGraph({'x':x,'residual':residual,'weight':weight},[
        r.GraphOp.add('sum','x','residual'),
        r.GraphOp.rms_norm('normalized','sum','weight',eps=1e-5),
    ],outputs=('sum','normalized')) as graph:
        # In a model adapter, preceding RUDA kernels write to the same buffers.
        # Assigning x = another_tensor would NOT rebind this graph.
        for step in range(3):
            x.copy_(torch.full((1,4096),0.1*(step+1)))  # explicit host input transfer
            y=graph.replay()['normalized']
        graph.synchronize()
        print(graph.info)
        print('output',y.shape,y.device,'first values',y.cpu()[0,:4])
        print('executed paths',r.execution_stats())
