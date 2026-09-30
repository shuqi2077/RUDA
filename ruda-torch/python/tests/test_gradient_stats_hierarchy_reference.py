"""Layout and FP32 mathematical tests, not production Rust/PTX execution."""
import numpy as np
import pytest
from gradient_stats_reference import layout, hierarchical, norm64, chunks

@pytest.mark.parametrize('rows',[1,31,32,33,1023,1024,1025,2048,32769,1048576,1048577,4194304])
def test_layout_exact_bounded_nonoverlapping(rows):
    plan=layout.statistics_plan(rows);cursor=0;previous=rows
    for stage in plan.stages:
        assert stage.input_rows==previous and stage.output_offset==cursor
        assert stage.output_rows==(previous+1023)//1024
        cursor+=stage.output_rows*3;previous=stage.output_rows
    assert cursor==plan.scratch_elements and previous==plan.final_rows<=1024
    assert cursor*4<=49200

@pytest.mark.parametrize('rows',[0,-1,4194305,2**64-1,True,1.,None,'1024'])
def test_invalid_layout(rows):
    with pytest.raises(ValueError):layout.statistics_plan(rows)

@pytest.mark.parametrize('rows',[1,31,32,33,1023,1024,1025,2049,32769,1048577])
@pytest.mark.parametrize('kind',['zero','uniform','wide'])
def test_numerical_hierarchy(rows,kind):
    rng=np.random.default_rng(27)
    stats=np.zeros((rows,3),dtype=np.float32)
    if kind=='uniform':stats[:,0]=3.;stats[:,1]=32.
    if kind=='wide':
        stats[:,0]=np.power(10.,rng.uniform(-30,30,size=rows)).astype(np.float32)
        stats[:,1]=rng.uniform(1,1024,size=rows).astype(np.float32)
    report=hierarchical(stats)
    assert report[0]==0 and np.all(np.isfinite(report))
    actual=float(report[1])*float(report[2])**.5
    assert actual==pytest.approx(norm64(stats),rel=2e-6,abs=1e-35)

@pytest.mark.parametrize('index',[0,31,32,1023,1024,1025,2048])
def test_bad_flag_propagates_even_when_all_scales_are_zero(index):
    stats=np.zeros((2049,3),np.float32);stats[index,2]=1
    report=hierarchical(stats)
    assert tuple(report)==(1.,0.,0.)

def test_maximum_rows_are_not_dropped():
    stats=np.zeros((4194304,3),np.float32)
    stats[:,0]=1;stats[:,1]=1;stats[-1,0]=2;stats[-1,2]=1
    report=hierarchical(stats)
    assert report[0]==1
    assert float(report[1])*float(report[2])**.5==pytest.approx((4194303+4)**.5,rel=2e-6)

def test_chunk_representation_is_not_report_order():
    stats=np.zeros((1025,3),np.float32);stats[0]=[2,1,1];stats[-1]=[4,1,0]
    out=chunks(stats)
    assert np.array_equal(out,np.array([[2,1,1],[4,1,0]],np.float32))
    assert np.array_equal(hierarchical(stats),np.array([1,4,1.25],np.float32))

def test_limits_and_single_stage_count():
    plan=layout.statistics_plan(4194304)
    assert [(s.input_rows,s.output_rows) for s in plan.stages]==[(4194304,4096),(4096,4)]
    assert plan.scratch_elements==12300
