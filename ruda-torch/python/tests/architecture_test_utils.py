"""Load production Python modules without pretending the native runtime exists."""
import importlib
from pathlib import Path
import sys
import types
import torch

NAME = 'ruda_architecture_cpu_reference'
if NAME not in sys.modules:
    package = types.ModuleType(NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'ruda_torch')]
    package._graph_available = False
    sys.modules[NAME] = package
ops = importlib.import_module(NAME+'._architecture_ops')
mhc = importlib.import_module(NAME+'.mhc')
sparse = importlib.import_module(NAME+'.sparse_attention')
optim = importlib.import_module(NAME+'.optim')
model = importlib.import_module(NAME+'.hybrid_model')
compiler = importlib.import_module(NAME+'.compiler')

def close(a,b,**kwargs):
    torch.testing.assert_close(a,b,**kwargs)
