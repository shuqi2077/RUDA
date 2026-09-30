"""New extension paired with legacy optional paged API must fail closed."""
import pytest
from test_v15_cpp import bridge
from test_paged_selected_cpp import selected,tensors,plan

def test_ordered_rejects_api1_before_allocation(selected):
    cpp,state=selected;a=tensors();p=plan(cpp,a[0]);calls=len(state.calls);allocs=state.next
    with pytest.raises(RuntimeError,match='API 2'):p.backward_selected(*a,.37,True,[True]*3,True)
    assert state.next==allocs and len(state.calls)==calls
