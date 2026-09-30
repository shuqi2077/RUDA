"""Validator fixtures only; no hardware execution implied."""
import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import validate_paged_pruning as v
NAMES=[v.TESTS[0]+'::test_old',v.TESTS[1]+'::test_new[float32]']
MARKERS='\n'.join(v.MARKERS)

def xml(tmp,mode='ok'):
    text='<testsuite><testcase classname="old" name="test_old"/><testcase classname="new" name="test_new[float32]"/></testsuite>'
    if mode in ('failure','error','skipped'):text=text.replace('name="test_old"/>',f'name="test_old"><{mode}/></testcase>')
    if mode=='duplicate':text=text.replace('test_new[float32]','test_old')
    if mode=='empty':text='<testsuite/>'
    f=tmp/'out.xml';f.write_text(text);return f

def test_ok(tmp_path):assert v.check_result(xml(tmp_path),MARKERS,0,NAMES)==2
@pytest.mark.parametrize('mode',['failure','error','skipped','duplicate','empty'])
def test_bad_xml(tmp_path,mode):
    with pytest.raises(ValueError):v.check_result(xml(tmp_path,mode),MARKERS,0,NAMES)
@pytest.mark.parametrize('markers',['',v.MARKERS[0],v.MARKERS[1]])
def test_both_markers_required(tmp_path,markers):
    with pytest.raises(ValueError):v.check_result(xml(tmp_path),markers,0,NAMES)
@pytest.mark.parametrize('code',[1,2,124,-11])
def test_exit(tmp_path,code):
    with pytest.raises(ValueError):v.check_result(xml(tmp_path),MARKERS,code,NAMES)
def test_collection():assert v.collect_names('\n'.join(NAMES))==NAMES
@pytest.mark.parametrize('text',['','\n'.join(NAMES*2)])
def test_bad_collection(text):
    with pytest.raises(ValueError):v.collect_names(text)
def test_missing_build_tools(tmp_path,monkeypatch):
    monkeypatch.setattr(v.shutil,'which',lambda _:None)
    assert v.main(['--build','--output',str(tmp_path)])==2
    assert not json.loads((tmp_path/'summary.json').read_text())['gpu_validated']
