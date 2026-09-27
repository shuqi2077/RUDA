"""Host tests of report acceptance. Fake XML is NOT GPU evidence."""
import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).with_name('validate_static_graph.py')
spec=importlib.util.spec_from_file_location('static_graph_validator',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def xml(tmp_path,count=77,tag=''):
    p=tmp_path/'result.xml';p.write_text('<testsuite>'+''.join('<testcase name="t%d">%s</testcase>'%(i,tag if i==0 else '') for i in range(count))+'</testsuite>');return p

def test_full_report(tmp_path):assert m.check_report(xml(tmp_path),m.MARKER,0)==77
@pytest.mark.parametrize('count',[0,1,76,78])
def test_missing_or_extra_cases(tmp_path,count):
    with pytest.raises(ValueError):m.check_report(xml(tmp_path,count),m.MARKER,0)
@pytest.mark.parametrize('tag',['<skipped/>','<failure/>','<error/>'])
def test_nonpassing(tmp_path,tag):
    with pytest.raises(ValueError):m.check_report(xml(tmp_path,tag=tag),m.MARKER,0)
def test_no_marker(tmp_path):
    with pytest.raises(ValueError):m.check_report(xml(tmp_path),'all tests passed',0)
def test_nonzero_exit(tmp_path):
    with pytest.raises(ValueError):m.check_report(xml(tmp_path),m.MARKER,86)
@pytest.mark.parametrize('tool',['memcheck','initcheck','synccheck'])
def test_clean_sanitizer(tmp_path,tool):assert m.check_report(xml(tmp_path),m.MARKER+'\nERROR SUMMARY: 0 errors',0,tool)==77
@pytest.mark.parametrize('text',['','ERROR SUMMARY: 1 errors','ERROR SUMMARY: 0 errors\nERROR SUMMARY: 2 errors'])
def test_bad_sanitizer(tmp_path,text):
    with pytest.raises(ValueError):m.check_report(xml(tmp_path),m.MARKER+'\n'+text,0,'memcheck')
def test_racecheck_clean(tmp_path):assert m.check_report(xml(tmp_path),m.MARKER+'\nRACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)',0,'racecheck')==77
def test_racecheck_warning(tmp_path):
    with pytest.raises(ValueError):m.check_report(xml(tmp_path),m.MARKER+'\nRACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warnings)',0,'racecheck')
