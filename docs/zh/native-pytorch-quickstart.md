# 原生 PyTorch 快速上手

[文档首页](README.md) · [原生 API](native-pytorch-api.md) · [English](../en/native-pytorch-quickstart.md)

`ruda-torch` 通过 `ruda:0` 设备在单张 NVIDIA GPU 上执行 PyTorch 运算。创建 RUDA 张量前先导入 `ruda_torch`。它与 `torch.cuda` 是不同的后端，CUDA 张量和 RUDA 张量不能直接互换。

## 1. 构建与安装

无需本地编译 Rust／C++ 时，可使用 [v0.3.0 Release wheel](../../ruda-torch/README.md#release-wheels)：CPython 3.13，Linux x86_64 搭配 PyTorch `2.10.0+cu126`（Ubuntu 24.04 构建），或 Windows x86_64 搭配 `2.13.0+cu130`。安装匹配的 wheel／PyTorch 后，设置下方编译器环境即可；需要源码构建时再执行以下步骤。

该版本的 Python 包为 `ruda-torch==0.3.0`，内含 release 优化构建的 Rust 库、C++ 扩展及许可证。下载对应 wheel、`SHA256SUMS` 和 `manifest.json`，核对 SHA-256 后安装：

```sh
# Linux x86_64 / CPython 3.13
python -m pip install ./ruda_torch-0.3.0-cp313-cp313-linux_x86_64.whl --index-url https://download.pytorch.org/whl/cu126
```

```powershell
# Windows x86_64 / CPython 3.13
python -m pip install ./ruda_torch-0.3.0-cp313-cp313-win_amd64.whl --index-url https://download.pytorch.org/whl/cu130
```

包元数据约束 Python `>=3.13,<3.14` 和对应 PyTorch 的完整版本（含 CUDA 后缀）。Linux wheel 未声明 manylinux 兼容，不保证旧发行版可用。升级旧版 wheel 后重启 Python 进程，并移除指向旧库的 `RUDA_TORCH_LIBRARY` 设置；NVIDIA 驱动与所选编译器路径的运行时依赖仍需自行准备。

准备 [NVIDIA 构建环境](getting-started.md)、已安装 PyTorch 与 setuptools 的 Python，以及 C++20 编译器。Windows 下使用 x64 MSVC 开发者终端。在仓库根目录执行：

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --locked -p ruda-torch-native
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python
```

运行 Python 前选择编译器。PowerShell 下选择直接 PTX：

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

Bash 下：

```sh
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

选择 GPU 与驱动支持的 [PTX 版本](ptx.md)。改用 `RUDA_CUDA_COMPILER=nvrtc` 可选择 CUDA C++／NVRTC 路径。

加载器优先寻找包内原生库，其次寻找仓库的 debug 库。使用单独构建的 release 库或其他路径时，在导入前设置 `RUDA_TORCH_LIBRARY`；wheel 已包含对应原生库。Rust 库与 C++ 扩展都必须使用 ABI 10。Linux／Windows wheel 及独立的 Linux／Colab 预编译包见[安装指南](../../ruda-torch/README.md#build-and-install)，按其中指定的 Python、PyTorch 和平台版本安装。

## 2. 上传数据并读取结果

```python
import torch
import ruda_torch

device = torch.device("ruda:0")
host = torch.arange(4, dtype=torch.float32)
x = host.to(device)
y = x + x
ruda_torch.synchronize()
result = y.cpu()
torch.testing.assert_close(result, host + host)
print(result)
print(ruda_torch.execution_stats())
```

结果为 `[0., 2., 4., 6.]`。`.to(device)` 上传输入，`.cpu()` 将输出回读到主机。`synchronize()` 等待设备执行完成。`execution_stats()` 返回累计运算／传输计数，不是耗时；`is_available()` 表示集成已初始化，不表示支持所有算子或 GPU 指令。

## 3. 执行一步 FP32 训练

在主机初始化模型和数据，再显式移动到 RUDA。这个小例子使用一阶自动微分与 RUDA AdamW：

```python
import torch
import ruda_torch

torch.manual_seed(7)
device = torch.device("ruda:0")
model = torch.nn.Sequential(
    torch.nn.Linear(8, 16),
    torch.nn.SiLU(),
    torch.nn.Linear(16, 4),
).to(device)
x = torch.ones(3, 8, dtype=torch.float32).to(device)
target = torch.zeros(3, 4, dtype=torch.float32).to(device)
optimizer = ruda_torch.AdamW(model.parameters(), lr=1e-3)

optimizer.zero_grad(set_to_none=True)
loss = (model(x) - target).square().mean()
loss.backward()
optimizer.step()
ruda_torch.synchronize()
print("loss:", loss.detach().cpu().item())
```

在下一个梯度累积窗口前清空梯度。低精度存储、FP32 优化器主参数、损失缩放、裁剪和优化器状态各有独立契约，见[原生 API 参考](native-pytorch-api.md)。改为 FP16／BF16 或恢复优化器 checkpoint 前，先查阅该页。

## 4. 编译模型或微调本地权重

- [模型编译器](model-compiler.md)：使用 `ruda_torch.compile` 包装 Module 或 callable，选择原生覆盖策略，在所有反向计算完成后释放缓存。
- [静态子图](static-pytorch-graphs.md)：显式构造固定地址的 `GraphOp` 计划，管理 replay 输出。
- [LoRA／NF4 微调](finetuning.md)：准备本地权重、准确选择目标、按 token 加权累积梯度、保存与恢复运行状态。
- [架构组件](architecture-training.md)：注意力变体、压缩 KV 缓存、mHC 和 Python Muon。

## 5. 定位安装与执行错误

| 现象 | 检查内容 |
| --- | --- |
| 原生库无法加载 | 检查 `RUDA_TORCH_LIBRARY` 指向的文件及依赖库。Windows 下 `RUDA_TORCH_DLL_DIR` 接受以分号分隔的依赖目录。 |
| ABI 不匹配或缺少可选 API | 从同一份源码重新构建 Rust 库和 C++ 扩展；仅匹配基础 ABI 10 并不能提供所有可选功能。 |
| 已注册其他 PrivateUse1 后端 | 使用尚未被其他后端占用 PrivateUse1 的 Python 进程。 |
| 算子不支持 | 检查[原生 API](native-pytorch-api.md)及该算子的 shape／dtype／device 要求。eager 后端没有通用 CPU fallback。 |
| 编译模型拒绝某个操作 | 检查[编译选项与覆盖范围](model-compiler.md)。`native='auto'` 在适用情况下保留原设备分发，不会转移到 CPU。 |

设备错误会向上传播；仅导入成功或看到 launch 计数不能证明结果正确。编译器和缓存问题见[调试指南](debugging.md)。
