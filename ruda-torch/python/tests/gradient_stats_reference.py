"""Independent NumPy FP32 test oracle, never imported by production.

Models lane assignment and padded tail handling, NOT PTX execution. Host layout
comes from the production Python helper; numerical expectations use float64.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'ruda_torch/_gradient_stats.py'
_spec=importlib.util.spec_from_file_location('ruda_stats_layout_reference', P)
layout=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=layout
_spec.loader.exec_module(layout)

def chunks(stats, fan_in=1024):
    n=len(stats);groups=(n+fan_in-1)//fan_in
    padded=np.zeros((groups*fan_in,3),dtype=np.float32)
    padded[:n]=stats
    data=padded.reshape(groups,fan_in//32,32,3)
    scale=data[:,:,:,0].max(axis=(1,2))
    bad=data[:,:,:,2].max(axis=(1,2))
    sums=np.zeros((groups,32),dtype=np.float32)
    for offset in range(fan_in//32):
        ratio=np.divide(data[:,offset,:,0],scale[:,None],
                        out=np.zeros_like(sums),where=scale[:,None]>0)
        term=(data[:,offset,:,1]*ratio)*ratio
        sums=np.add(sums,term,dtype=np.float32)
    # Pairwise warp reduction; exact compiler lowering is not asserted here.
    for delta in (16,8,4,2,1):
        sums[:,:delta]=sums[:,:delta]+sums[:,delta:2*delta]
    return np.stack((scale,sums[:,0],bad),axis=1)

def hierarchical(stats):
    plan=layout.statistics_plan(len(stats))
    values=np.asarray(stats,dtype=np.float32)
    for stage in plan.stages:
        assert len(values)==stage.input_rows
        values=chunks(values)
        assert len(values)==stage.output_rows
    # The final reduction operates on at most 1024 rows and changes field order.
    scale,squares,bad=chunks(values)[0]
    return np.array([bad,scale,squares],dtype=np.float32)

def norm64(stats):
    values=np.asarray(stats,dtype=np.float64)
    largest=values[:,0].max()
    return float(largest*np.sqrt(np.sum(values[:,1]*(values[:,0]/largest)**2))) if largest else 0.
