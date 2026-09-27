# v22 直接 PTX 验收与续跑

本次候选基于用户实际跑通的 7501b85；不要把旧 GPU 结果自动算到新源码。

## 沿用已跑通的 T4 环境示例

```bash
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
export CARGO_BUILD_JOBS=1
export RUST_MIN_STACK=16777216

# 在源码根目录运行。分批上限只是控制本次新执行任务数量。
python tools/gpu_validation/validate_v22.py \
  --sanitizers all --benchmark --max-cases 100 \
  --output /path/to/persistent/v22-results

# 相同代码/GPU/配置续跑；不要使用 v21 结果目录。
python tools/gpu_validation/validate_v22.py \
  --sanitizers all --benchmark --resume \
  --output /path/to/persistent/v22-results
```

PTX 8.0 是上传 T4/580.82.07 环境使用的值，不是所有设备/驱动的通用承诺。编译需要已有 Rust 工具链和可取得的 Cargo.lock 依赖。

退出码 0 表示全部请求的项目通过；1 表示用例失败；2 表示前置条件、编译、清单或其他操作被阻断；3 表示达到分批上限仍未完成。结果按任务写入 `result.json`。不要把分批的退出码 3 当成 GPU 数值失败，也不能当成全通过。

默认六组：graph-replay、graph-update、graph-batch、graph-dag、graph-infer、fft-exact。87 项普通测试；`--sanitizers all` 再逐项运行四种工具（348 项），总计 435 个用例×工具任务。基准不计入正确性通过数。`--groups fft-exact` 可单独检查 FFT，但结果只代表该选择范围。

## 新 FFT 优化

`RealFftPlan::set_spectrum_fusion(true)` 显式启用候选融合，默认关闭。新增测试自动覆盖开关开启、单点/二次幂不变、批次、非最后轴、截断、虚拟补零、空输入及四步边界。它在设备内核的读入阶段计算频谱乘积；不是 CPU 执行。

基准 `exact_fft_spectrum_fusion_benchmark` 和 `graph_replay_balanced_benchmark` 会先预热两路，交替顺序，按配置记录 7 对样本，并在计时外核对输出。程序计算每对普通/候选比值，再提供中位数、范围和原始数据；不把小型内核链解释为 tokens/s。

## 完成判据

脚本先构建、从 Cargo 产物清单确定二进制、发现测试名称，再逐项执行。不存在为缺失运行时标记开后门、包住 Cargo 代替仪器化二进制，或者复用另一个 GPU 的旧结果。只有全部请求通过才设置 `all_requested_passed=true`。

硬件检查范围仍是驱动图与 FFT，不等于整个 PyTorch、大模型、训练、多卡或其他 ISA 验收。
