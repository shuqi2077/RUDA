import ctypes
import importlib.util
from pathlib import Path
import sys

import torch

root = Path(__file__).resolve().parents[1]
extension = next(p for p in root.glob('build/lib*/ruda_torch/_C*') if p.suffix in ('.so', '.pyd'))
package = extension.parent
spec = importlib.util.spec_from_file_location('_C', extension)
cpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp)
assert cpp.abi_version == 10 and cpp.factory_api_version == 1
for name in ('set_.source_Storage', 'set_.source_Storage_storage_offset', 'resize_',
             'normal_', 'arange.start_out'):
    assert torch._C._dispatch_has_kernel_for_dispatch_key('aten::'+name, 'PrivateUse1'), name

library = package / ('ruda_torch_native.dll' if sys.platform == 'win32' else 'libruda_torch_native.so')
native = ctypes.CDLL(str(library))
for name, version in (('abi', 10), ('factory_api', 1)):
    function = getattr(native, 'ruda_torch_'+name+'_version')
    function.restype = ctypes.c_uint32
    assert function() == version
for name in ('arange', 'normal'):
    assert getattr(native, 'ruda_torch_'+name)

generator = torch.Generator(device='privateuseone').manual_seed(1729)
state = generator.get_state()
generator.manual_seed(42)
generator.set_state(state)
assert generator.initial_seed() == 1729
assert torch.equal(generator.clone_state().get_state(), state)
print('C++ registrations, native factory exports and generator state passed; no GPU execution')
