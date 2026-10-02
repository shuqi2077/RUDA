"""Real C++ validation/ABI tests. Callback records nodes, NEVER runs a GPU."""
import pytest
import torch
from test_static_graph_cpp import graph_bridge, bridge

UNARY=[3,9,10,11,12,13,17,19,20,21,22,23,24,25,27,35,36,37,38,39,40,43,45,108]
@pytest.mark.parametrize('op',UNARY)
def test_api3_unary_codes_reach_callback(graph_bridge,op):
    cpp,state=graph_bridge
    tensors=[torch.empty(2,3,device='ruda') for _ in range(2)]
    g=cpp.NativeStaticGraph(tensors,1,[op,0,0],[0.],False,False)
    assert list(state.graphs.values())[-1]['nodes']==[(op,0,0,0.)]
    g.close()

@pytest.mark.parametrize('op',[7,30])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_matrix_different_output_geometry(graph_bridge,op,dtype):
    cpp,state=graph_bridge;prefix=(2,) if op==30 else ()
    shapes=[prefix+(3,5),prefix+(5,7),prefix+(3,7)]
    tensors=[torch.empty(s,device='ruda',dtype=dtype) for s in shapes]
    g=cpp.NativeStaticGraph(tensors,2,[op,0,1],[0.],False,False)
    assert list(state.graphs.values())[-1]['nodes']==[(op,0,1,0.)]
    g.close()

@pytest.mark.parametrize('op',[102,103,106,107])
def test_softmax_and_backward_axis_contract(graph_bridge,op):
    cpp,state=graph_bridge
    inputs=2 if op>=106 else 1
    tensors=[torch.empty(2,3,4,device='ruda') for _ in range(inputs+1)]
    g=cpp.NativeStaticGraph(tensors,inputs,[op,0,inputs-1],[1.],False,False)
    g.close()

@pytest.mark.parametrize('op',[104,105])
@pytest.mark.parametrize('mask,shape',[(1,(1,3,4)),(2,(2,1,4)),(5,(1,3,1))])
def test_keepdim_mask_contract(graph_bridge,op,mask,shape):
    cpp,state=graph_bridge
    tensors=[torch.empty(2,3,4,device='ruda'),torch.empty(shape,device='ruda')]
    g=cpp.NativeStaticGraph(tensors,1,[op,0,0],[float(mask)],False,False)
    g.close()

@pytest.mark.parametrize('op',[109,110,111])
def test_scalar_contract(graph_bridge,op):
    cpp,state=graph_bridge
    tensors=[torch.empty(2,3,device='ruda') for _ in range(2)]
    g=cpp.NativeStaticGraph(tensors,1,[op,0,0],[2.5],False,False);g.close()

@pytest.mark.parametrize('op,scalar,shape',[(102,2.,(2,3)),(104,4.,(1,3)),(105,1.,(2,1)),(109,float('inf'),(2,3))])
def test_invalid_api3_contract_never_dispatches(graph_bridge,op,scalar,shape):
    cpp,state=graph_bridge;tensors=[torch.empty(2,3,device='ruda'),torch.empty(shape,device='ruda')]
    before=len(state.graph_calls)
    with pytest.raises(RuntimeError):cpp.NativeStaticGraph(tensors,1,[op,0,0],[scalar],False,False)
    assert len(state.graph_calls)==before
