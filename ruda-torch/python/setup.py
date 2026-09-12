import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="ruda-torch", version="0.1.0", packages=["ruda_torch"],
    package_data={"ruda_torch": ["ruda_torch_native.dll"]},
    ext_modules=[CppExtension("ruda_torch._C", ["ruda_torch/csrc/backend.cpp"],
        extra_compile_args=["/std:c++20", "/O2", "/EHsc"] if sys.platform == "win32" else ["-std=c++20", "-O2"])],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
