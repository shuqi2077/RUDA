# 通用 PyTorch 模型入口

[文档目录](README.md) · [原生 API](native-pytorch-api.md) · [English](../en/model-compiler.md)

## 解决的限制

此前 `StaticGraph(training=True)` 只能执行手写 `GraphOp`，反向也只有几种预先实现的梯度公式。现在增加 `ruda_torch.compile(model)`：模型仍然是普通 `torch.nn.Module`，前向和反向由 PyTorch 的 AOTAutograd 生成。AOTAutograd 是 PyTorch 用来生成可编译反向计算图的组件，不需要为每个模型重写梯度。

**通用模型接入不等于全部设备算子已经实现，也不等于整步训练都进入一个原生 GPU 图。** 本版本消除了模型必须手写 GraphOp 的限制，但模型实际用到的前向、反向、随机数和自定义操作仍须有 RUDA 设备实现。

实现与主机验证基于 PyTorch **2.10.0+cpu**、Python **3.13.5**。原生设备验收单独执行；没有声称全部 PyTorch 版本兼容。

## 基本使用

先准备普通模型并明确搬到设备，然后编译。优化器可以在包装模型之前创建；包装不会替换参数对象。

```python
import torch
import ruda_torch

model = torch.nn.Sequential(
    torch.nn.Linear(8, 16),
    torch.nn.SiLU(),
    torch.nn.Linear(16, 4),
).to('ruda:0')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
compiled = ruda_torch.compile(model)

x = torch.randn(3, 8).to('ruda:0')
target = torch.zeros(3, 4).to('ruda:0')
optimizer.zero_grad(set_to_none=True)
loss = (compiled(x) - target).square().mean()
loss.backward()
optimizer.step()

print(compiled.info)
state = compiled.state_dict()  # 保留原模型的键名
compiled.close()              # 所有反向完成之后关闭
```

`StaticGraph.from_model(model, **options)` 是同一入口，返回可调用的 `CompiledModel`，而不是固定输入地址的 `StaticGraph`。调用 `compiled(x)`，不要调用 `replay()`。

普通函数也可传入，或用作装饰器。嵌套字典/列表/元组、关键字参数、多个输出和非张量返回值由 PyTorch 保留，不人为展平成固定模板。

```python
@ruda_torch.compile(native='auto', fullgraph=False)
def block(x, residual):
    return {'hidden': torch.nn.functional.silu(x) + residual}
```

## 执行方式

`capture='aot'` 为默认：使用 PyTorch 捕获前向和反向。`fullgraph=False` 允许不能捕获的 Python 代码形成图间断点，分段执行；`fullgraph=True` 拒绝断点。输入形状和 Python 状态由 PyTorch 的条件检查决定是否重新编译，`dynamic` 原样传给 `torch.compile`。

本实现不修改全局编译器开关。包装入口拒绝 `torch._dynamo.config.suppress_errors=True`，避免编译失败被全局选项转成隐式普通执行。直接使用 `make_backend()` 后，外部 `torch.compile` 的后续配置仍由调用者负责。

`native` 决定捕获后的操作如何执行：

完整入口为 `compile(model=None, capture='aot', native='auto', device_type='ruda', fullgraph=False, dynamic=None, min_native_ops=2, cache_size=4, decompositions=None)`。`model=None` 返回装饰器；`device_type` 为设备类型而非 `ruda:0` 索引。`min_native_ops` 为 1–256，auto 模式默认连续两个合格操作才形成区域；required 模式最少一个。`cache_size` 为每区域 1–64，默认 4。`decompositions` 须为准确 PyTorch overload 到可调用分解的映射；eager 模式不接受非空分解、fullgraph／dynamic 选项或 required 策略。

| 参数 | 行为 |
| --- | --- |
| `auto` | 对满足条件的连续操作建立原生图；其他操作在原设备正常执行。 |
| `off` | 保留前向/反向图捕获，但全部操作按原设备的 PyTorch 调度执行。适合隔离图捕获与原生分段问题。 |
| `required` | 已捕获图中只要存在非原生操作，或运行时形状/类型不满足原生条件，就报错。**不承诺图外代码已捕获**；审查整图还须使用 `fullgraph=True`。 |

自动分段按准确的 ATen 操作重载识别下列操作，不根据名称相似性替换算子：

| 操作 | 可进入原生区域的重载 |
| --- | --- |
| 复制与一元逐点操作 | `aten.clone.default`；`relu`、`silu`、`sigmoid`、`tanh`、`exp`、`log`、`sqrt`、`rsqrt` 等一元操作的 `.default`，完整列表见 [`UNARY_CODES`](../../ruda-torch/python/ruda_torch/_graph_spec.py)。 |
| 张量与标量算术 | `aten.add/sub/mul/div.Tensor` 和 `.Scalar`；张量二元操作要求形状与类型相同，不广播、不做混合类型提升。 |
| 矩阵乘 | `aten.mm.default`、`aten.bmm.default`；分别要求同类型的二维、三维输入，内维和批次匹配，不做批次广播。 |
| 保留维度的归约 | `aten.sum.dim_IntList`、`aten.mean.dim`，要求 `keepdim=True`，不指定额外的 `dtype` 转换。 |
| Softmax 与反向 | `aten._softmax.default`、`aten._log_softmax.default`、`aten._softmax_backward_data.default`、`aten._log_softmax_backward_data.default`；不做半精度到 FP32 的隐式提升，反向类型须匹配。 |
| 激活反向 | `aten.silu_backward.default`、`aten.sigmoid_backward.default`、`aten.tanh_backward.default`。 |

运行时仍检查 `ruda:0` 设备、稠密布局、FP32/FP16/BF16 类型、非空的 1–8 维形状和节点数限制；输入复制到连续暂存存储，操作输出须为连续布局。标量须能表示为有限 FP32 值。完整分段规则见 [`_compile_native.py`](../../ruda-torch/python/ruda_torch/_compile_native.py)。显式 `GraphOp` 的 RMSNorm/SiLU-mul 节点不代表 AOT 会自动识别对应复合操作。广播、视图、就地修改、随机操作及其他未支持操作在 `native='auto'` 下留在原设备普通执行，在 `native='required'` 下报错。

不能把 `fullgraph=True` 理解为“全部操作原生捕获”：它只约束 PyTorch 图是否出现断点。`native='required'` 也只审查捕获到的操作。

`capture='eager'` 是**明确选择的不编译模式**：按原模型执行，不生成图、不做原生分段，也不声明加速。可用于不能追踪的 Python 模型，或需要高阶导数的模型。高阶导数仍取决于每个设备算子的实现；AOT 模式不支持二阶反向。

## 整步训练函数

可以把损失计算、反向和优化器步骤放入一个函数，再交给入口：

```python
def train_step(x, target):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()

compiled_step = ruda_torch.compile(train_step, fullgraph=False)
loss = compiled_step(x, target)
```

主机测试覆盖了含动量 SGD 更新的整步函数。**这不是把任意优化器、`.backward()`、日志和 Python 副作用全部捕获为一个原生图**：PyTorch 可能在这些位置分段；包含主机读取的优化器也不会被本入口自动变成无主机同步的实现。

## 参数、状态与梯度

`CompiledModel` 保留原始参数对象，直接在包装器上调用 `state_dict` / `load_state_dict` 时保留原模型键名。将包装器作为另一个模型的子模块保存时，父模型的检查点保留 `_original` 结构前缀，以确保递归加载与模块版本信息可以正确往返。`train()` / `eval()` 递归作用于原模型，`.original` 可取回原模型。模型参数与调用输入必须明确位于 `device_type` 指定的设备上，默认 `ruda`；包装不会自行移动张量。

已经验证的 CPU 语义包括：共享嵌入权重、冻结/未使用参数、梯度累积、多次前向后统一反向、保留计算图再次反向、BatchNorm 状态、Dropout 随机数状态、反向钩子、非重入式激活检查点、嵌套输出、输出别名和输入就地修改。

这些是**通用编译通路的 CPU 对照结果，不是所有对应 RUDA 算子的硬件结果**。例如原包仍有未支持的 `erf` 设备操作；随机操作也不能因为 Dropout 的 CPU 测试通过而视为 RUDA 已支持。

## 原生图的内存和流

追踪阶段只处理张量形状、类型等信息，不读取假张量（FakeTensor）的数据，也不启动原生内核。第一次真实调用才创建固定地址的暂存输入，随后把新输入内容复制进去再重放。

每个区域按形状、步长、类型、设备、执行流和 inference-mode 状态缓存，`cache_size` 默认 4，范围 1–64，**按区域计算，不是全模型总数上限**。缓存满时关闭最近最少使用的图。单个原生区域最多 256 个操作、512 个张量；不满足条件的区域不强行捕获。

输出在返回前复制到独立存储，避免下一次重放覆盖已返回的结果或反向保存的中间值。这需要额外设备内存和同设备复制，不应预设提速。不同流分开缓存；区域内部串行保护复制、重放和结果保存。外部跨流张量依赖仍由调用者正确排序。

`close()` 在所有反向之后调用；关闭后不允许继续前向或惰性生成反向图。释放失败的句柄会保留，以便再次显式 `close()`。缓存逐出若关闭失败，不丢弃旧句柄。

## 覆盖报告与错误

`compiled.info` 可 JSON 序列化，包含每个前向/反向图的操作列表、计划分段、实际原生重放次数、普通执行原因、缓存命中/逐出和错误计数。

`planned_native_ops` 是**计划值**，不是执行成功的证明。须结合 `native_replays`、`native_nodes_executed` 与 `reference_reasons` 查看实际路径。报告中没有图间断点的完整目录；需审查断点时使用 PyTorch 的编译日志。

设备内核缺失、原生构建失败、显存分配失败或重放异常会向上抛出，并保留异常原因；不会捕获后重新运行整个模型，也不会转到 CPU。仅对预先确定的原生能力/元数据限制，在任何原生调用发生前选择原设备普通执行。

## 自定义算子与扩展

自定义算子需要设备实现、供追踪使用的假张量/元信息规则，以及训练所需的自动微分规则。已有 `torch.autograd.Function` 可由 PyTorch 按自身规则捕获；本入口不假设任意外部代码都可追踪。

可通过 `decompositions={operator_overload: callable}` 提供操作分解，让 AOT 把自定义或复合操作展开成已支持的操作。这是用户显式指定的数学变换，正确性由调用者负责；入口不会擅自用近似激活函数替换模型里的精确形式。

也可直接使用后端：

```python
backend = ruda_torch.make_backend(native='auto')
compiled = torch.compile(model, backend=backend, fullgraph=False)
# 正常调用 compiled，然后读取 backend.info；全部反向结束后 backend.close()。
```

## 验证命令

主机测试隔离导入纯 Python 编译代码，不加载真实 RUDA 运行库。`device_type='cpu'` 仅用于明确的编译器参考执行；标准 `import ruda_torch` 本身仍要求原生运行库可加载。

```bash
python -m pytest -q \
  ruda-torch/python/tests/test_model_compile.py \
  ruda-torch/python/tests/test_compile_native.py

python ruda-torch/tools/check_cpp_bridge.py \
  --compiler clang++ --static-graph --output ./model-cpp
RUDA_CPP_TEST_LIBRARY="$(find ./model-cpp -name '_C*.so' -print -quit)" \
  python -m pytest -q ruda-torch/python/tests/test_model_compile_cpp.py
```

真实设备验收须先准备匹配的 Rust 和 C++ 组件，可按仓库说明从源码构建，或使用 [Linux/Colab 预编译包](../../ruda-torch/README.md#linuxcolab-precompiled-bundle)：

```bash
RUDA_CUDA_COMPILER=ptx python ruda-torch/tools/validate_model_compile.py \
  --output ./model-gpu-result.json --steps 3
```

该验收要求真实 RUDA 内核执行、捕获的前向和反向被调用、门控模型至少使用一个原生图区域，并检查训练中没有新增主机传输。缺少设备/运行库或者计算不一致会以非零状态退出，不将跳过当作通过。主机 CI 不替代这项验收。

## 上游接口资料

PyTorch 官方文档：[自定义后端与 AOTAutograd](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html)、[设备算子注册](https://docs.pytorch.org/docs/stable/accelerator/operators.html)。本实现用到的部分接口位于带下划线的 PyTorch 模块，升级 PyTorch 时应重新运行测试，而非仅凭接口名称相同就认定兼容。
