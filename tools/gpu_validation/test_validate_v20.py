"""Validator tests only: fabricated transcripts are never GPU evidence."""
import importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('v20_validator',Path(__file__).with_name('validate_v20.py'))
v=importlib.util.module_from_spec(s);s.loader.exec_module(v)

def valid():
    return ('RUDA_GRAPH_DAG_RUNTIME,CudaRuntime,direct-ptx\n'+
      '\n'.join(f'test {name} ... ok' for name in sorted(v.REQUIRED))+
      f'\ntest result: ok. {len(v.REQUIRED)} passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.01s\n')

def test_complete_log_is_accepted(): assert v.require_dag_success(valid())==len(v.REQUIRED)
@pytest.mark.parametrize('replacement',['','RUDA_GRAPH_DAG_RUNTIME,CpuRuntime,direct-ptx','RUDA_GRAPH_DAG_RUNTIME,CudaRuntime,nvrtc'])
def test_runtime_must_match(replacement):
    with pytest.raises(ValueError):v.require_dag_success(valid().replace('RUDA_GRAPH_DAG_RUNTIME,CudaRuntime,direct-ptx',replacement))
@pytest.mark.parametrize('name',sorted(v.REQUIRED))
def test_each_named_case_is_mandatory(name):
    with pytest.raises(ValueError):v.require_dag_success(valid().replace(name,'missing_case'))
@pytest.mark.parametrize('old,new',[(f'{len(v.REQUIRED)} passed','0 passed'),('0 failed','1 failed'),('0 ignored','1 ignored'),('test result: ok.','test result: FAILED.')])
def test_failure_zero_and_ignored_are_not_success(old,new):
    with pytest.raises(ValueError):v.require_dag_success(valid().replace(old,new))

def test_testlist_matches_real_rust_source():
    source=Path(__file__).resolve().parents[2]/'ruda-driver-cuda/tests/graph_dag.rs'
    import re
    names=set(re.findall(r'fn (graph_dag_\w+)\(',source.read_text()))-{'graph_dag_benchmark'}
    assert names==v.REQUIRED
