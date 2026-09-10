# ruLLM

**English** | [简体中文](docs/zh/README.md)

LLM inference, model loading, and text generation on Ruda.

ruLLM combines tokenization, model loading, cached autoregressive generation, and request scheduling with Ruda tensors and compute libraries.

- Cargo package: `ruLLM`
- Rust crate: `rullm`

## Quick Start

Set up the [NVIDIA environment](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md) and prepare a local [Qwen3.5-0.8B model directory](https://github.com/shuqi2077/RUDA/blob/main/docs/en/model-inference.md#prepare-a-local-model).

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

Replace `./models/qwen35` with your model directory. The final arguments select the maximum new tokens and number of runs. The example defaults to direct PTX when `RUDA_CUDA_COMPILER` is unset. Set it to `nvrtc` to use the CUDA C++ compilation path instead. See [PTX configuration](https://github.com/shuqi2077/RUDA/blob/main/docs/en/ptx.md) for target-version selection.

## Documentation

- [Model loading and inference](https://github.com/shuqi2077/RUDA/blob/main/docs/en/model-inference.md)
- [Cargo features](Cargo.toml) · [Module exports](src/lib.rs)
