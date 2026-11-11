import sys
import subprocess
from pathlib import Path

import torch
from setuptools import setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parents[2]


class TorchBuildExtension(BuildExtension):
    def build_extensions(self):
        for extension in self.extensions:
            extension.depends.append(str(ROOT / 'ruda-torch/python/ruda_torch/csrc/qwen_graph.inc'))
            extension.depends.append(str(ROOT / 'ruda-torch/python/ruda_torch/csrc/eager_arithmetic.inc'))
            extension.depends.append(str(ROOT / 'ruda-torch/python/ruda_torch/csrc/eager_selection.inc'))
        if self.compiler.compiler_type == "unix" and "GCC" in torch.__config__.show():
            version = subprocess.check_output(self.compiler.compiler_cxx + ["--version"], text=True)
            if "clang" in version.lower():
                for extension in self.extensions:
                    extension.extra_compile_args.append("-fclang-abi-compat=17")
        super().build_extensions()


class BuildPy(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "ruda_torch" / "licenses"
        destination.mkdir(parents=True, exist_ok=True)
        for source in ("LICENSE", "THIRD_PARTY_NOTICES.md", "ruDNN/src/kda/LICENSE.fla"):
            self.copy_file(str(ROOT / source), str(destination / Path(source).name))


setup(
    name="ruda-torch", version="0.3.0", packages=["ruda_torch"],
    description="Native RUDA backend for PyTorch",
    long_description=(ROOT / "ruda-torch/README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/shuqi2077/RUDA",
    project_urls={"Documentation": "https://shuqi2077.github.io/RUDA/en/docs/native-pytorch/"},
    license="Apache-2.0 AND MIT",
    python_requires=f">={sys.version_info.major}.{sys.version_info.minor},<{sys.version_info.major}.{sys.version_info.minor + 1}",
    install_requires=[f"torch=={torch.__version__}"],
    package_data={"ruda_torch": ["ruda_torch_native.dll", "libruda_torch_native.so", "csrc/*.cpp", "csrc/*.inc"]},
    ext_modules=[CppExtension("ruda_torch._C", ["ruda_torch/csrc/backend.cpp"],
        depends=["ruda_torch/csrc/fused_autograd_dispatch.inc", "ruda_torch/csrc/arithmetic_dispatch.inc", "ruda_torch/csrc/matmul_dispatch.inc", "ruda_torch/csrc/collective_program.inc", "ruda_torch/csrc/collective_program_bindings.inc", "ruda_torch/csrc/ring_attention.inc", "ruda_torch/csrc/ring_attention_bindings.inc", "ruda_torch/csrc/operation_graph.inc", "ruda_torch/csrc/dense_attention.inc", "ruda_torch/csrc/dense_attention_bindings.inc", "ruda_torch/csrc/varlen_attention.inc", "ruda_torch/csrc/varlen_attention_bindings.inc", "ruda_torch/csrc/float8.inc", "ruda_torch/csrc/float8_bindings.inc", "ruda_torch/csrc/sorting.inc", "ruda_torch/csrc/ordering.inc", "ruda_torch/csrc/statistics.inc", "ruda_torch/csrc/static_graph.inc", "ruda_torch/csrc/training.inc", "ruda_torch/csrc/router.inc", "ruda_torch/csrc/nf4.inc", "ruda_torch/csrc/sequence_training.inc", "ruda_torch/csrc/kda.inc", "ruda_torch/csrc/packed.inc", "ruda_torch/csrc/random.inc", "ruda_torch/csrc/random_extra.inc", "ruda_torch/csrc/fused_optim.inc", "ruda_torch/csrc/foreach_unary.inc", "ruda_torch/csrc/foreach_binary.inc", "ruda_torch/csrc/foreach_ternary.inc", "ruda_torch/csrc/foreach_reduce.inc", "ruda_torch/csrc/storage.inc", "ruda_torch/csrc/generator.inc", "ruda_torch/csrc/factories.inc", "ruda_torch/csrc/rank_one.inc", "ruda_torch/csrc/indexing.inc", "ruda_torch/csrc/scientific_bridge.inc", "ruda_torch/csrc/scientific_bridge_bindings.inc", "ruda_torch/csrc/blas_bridge.inc", "ruda_torch/csrc/blas_bridge_bindings.inc", "ruda_torch/csrc/linalg_bridge.inc", "ruda_torch/csrc/linalg_bridge_bindings.inc", "ruda_torch/csrc/fft_bridge.inc", "ruda_torch/csrc/fft_bridge_bindings.inc", "ruda_torch/csrc/vendor_bridge.inc", "ruda_torch/csrc/vendor_bridge_bindings.inc"],
        extra_compile_args=["/O2", "/EHsc", "/utf-8"] if sys.platform == "win32" else ["-O2"])],
    cmdclass={"build_ext": TorchBuildExtension.with_options(use_ninja=False), "build_py": BuildPy},
)
