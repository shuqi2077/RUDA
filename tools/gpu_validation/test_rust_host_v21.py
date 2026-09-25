"""Failure-parser tests with fabricated logs, not production Rust results."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('host_v21',Path(__file__).with_name('rust_host_v21.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def valid():
    return '\n'.join('test tests::'+n+' ... ok' for n in sorted(m.REQUIRED))+f'\ntest result: ok. {len(m.REQUIRED)} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n'
def test_complete_named_result_accepted(): assert m.require_complete(valid())==52
@pytest.mark.parametrize('name', sorted(m.REQUIRED))
def test_each_production_case_required(name):
    with pytest.raises(ValueError):m.require_complete(valid().replace(name,'missing'))
@pytest.mark.parametrize('old,new',[('52 passed','0 passed'),('0 failed','1 failed'),('0 ignored','1 ignored'),('test result: ok.','test result: FAILED.')])
def test_failure_and_skip_rejected(old,new):
    with pytest.raises(ValueError):m.require_complete(valid().replace(old,new))
