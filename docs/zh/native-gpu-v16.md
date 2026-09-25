# v16：公共 FFT 语义补充与 GPU 启动路径优化

## 1. 真正的指定长度设备 FFT

新增 `rufft::tensor::{rfft_exact, irfft_exact}`，支持 F32 实数输入、一个变换轴和批次维。
六点输入得到六点离散傅里叶变换的四个独立频点，不把频率网格偷偷改成八点。
旧 `rfft/irfft` 仍保留原有补到二次幂的公开约定。更改其默认语义会影响其他后端、图形状和自动求导，
本轮没有只修改 GPU 一侧就破坏这些约定。新接口尚未接成 PyTorch `torch.fft` 的自动注册替代。

非二次幂采用 Bluestein 算法：把长度 N 的变换转为长度 M 的卷积，其中 M 是不小于 2N-1 的二次幂。
信号的准备、复数 FFT、频域乘法和输出处理都使用 RUDA 设备内核；没有信号数据回读或 CPU 计算回退。
该实现复用上游的 radix-2 和 four-step 复数 FFT，不新增 CUDA C++、NVCC 或 cuFFT 依赖。
它不是 cuFFT 全功能替代：只支持 F32、单轴、分离的实部/虚部；不提供多维复数变换、回调或多卡 FFT。

新增 `RealFftPlan` 保存 GPU 相位表、卷积核频谱和批次工作区。批次形状不变时复用；变更时更换工作区。
四步 FFT 的额外暂存也由计划持有，避免每次执行再申请。初始化阶段仍有临时分配。
计划固定创建时的执行队列，避免复用工作区时随着线程局部流变化而发生跨队列访问。
这不是 CUDA Graph 捕获，也不是全局显存池。创建临时计划的便利接口不会获得跨调用的计划复用收益。

显存计算（F32）：表占 8*(N+M) 字节；批次工作区占 8*batch*M；当 M>4096 时四步暂存再占 8*batch*M。
例如 N=1009、M=2048、batch=32：表 24,456 字节，工作区 524,288 字节，总保留 548,744 字节。
该数不含输入、输出、驱动、模块和初始化暂存，不能表述为模型显存或节省比例。
精确变换可能比旧补零变换更贵，因为它们计算的问题本来就不同。

## 2. 兼容接口也减少补零和复制

旧 tensor::rfft 不再先构造补齐的输入张量，转而向内核传入有效输入长度；
旧 tensor::irfft 不再先补齐/截断频谱副本。旧逆变换在要求长度小于二次幂时仍需要裁剪输出。
同样的读边界检查覆盖共享内存、打包实数和四步路径。

修正空输入虚拟补零时仍可能读取第 0 个元素的问题。逆变换明确不读取实数专用频点的虚部：
直流频点，以及偶数 N 的奈奎斯特频点。即使这些应忽略的虚部是 NaN，也不污染有效实数结果。
一元素正/逆变换现在有设备入口。原先代码接受 F64 标记却固定启动 F32 内核的入口改为明确拒绝，
与既有文档的“设备计算仅 F32”一致，而不是把 64 位数据重新解释成 32 位数据。

## 3. 驱动内核加载与重复启动

`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` 从每次启动前设置，改为模块加载时配置一次。
共享内存为零时不设置该属性。直接 PTX 和已编译 PTX 缓存命中都经过相同验证/配置入口。
驱动报告配置错误会中止模块发布，不会静默切到 CPU 或旧内核。

同时校验缓存 PTX 的终止符、内嵌零字节、入口名和共享内存边界；错误信息带入口和配置值。
内核函数查询/配置失败时卸载尚未使用的模块；成功模块跟随运行时持有。
运行时销毁时先等待 GPU 完成，再卸载模块。如果等待失败，不冒险卸载仍可能被执行的代码，而是记录错误。
此等待仅发生在销毁阶段，不是每算子同步。没有更改异步默认值、流/事件 ABI 或用户设置。

## 4. 验证分层

主机数学参考使用独立 Python/NumPy 以及模拟 float32 butterfly；它没有执行 Rust 内核。
新增 Rust 单元测试覆盖加载参数检查；新增真实设备测试覆盖六点、质数、奇数、批次、中间轴、
截断、补零、空源、一元素、四步卷积、计划复用和实数频点 NaN；它们必须在硬件上执行才算通过。

硬件验收脚本将缺少工具、缺设备、零用例、忽略或失败的必测用例视为失败。

```bash
# 主机数学与验收脚本测试（不是 GPU 验收）
python -m pytest -q ruFFT/tests/host tools/gpu_validation

# 在目标 GPU 机器上，显式设置适合实际驱动的 PTX 版本
export RUDA_PTX_VERSION=8.0   # 示例，不是所有设备/驱动的通用选择
python tools/gpu_validation/validate_v16.py --output ./v16-hardware-validation

# 可选：真实 GPU 内存检测，依赖已经安装的 Compute Sanitizer
python tools/gpu_validation/validate_v16.py --sanitizer memcheck --output ./v16-memcheck

# 可选：精确 N 点计划复用 vs 重新建计划，包含 CPU 提交和 GPU 完成时间
python tools/gpu_validation/validate_v16.py --benchmark --output ./v16-benchmark
```

不能拿六点精确 DFT 和八点补零 FFT 作相同算子的速度比较。计时脚本只比较相同 N、批次和精度。
此基准不包含厂商库对照，不会预先给出任何性能倍数。

官方接口语义参考：
- https://docs.nvidia.com/cuda/cufft/index.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

这些参考用于区分功能边界，不是 RUDA 已通过厂商认证或性能对等的证据。
