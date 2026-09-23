"""Actual driver PTX JIT/launch tests. Skips are NOT GPU validation successes."""
import os
import pytest
import torch
import torch.nn.functional as F
from ruda_ptx import Executor, compile_exported
from ruda_ptx.nvidia_driver import DriverError, NvidiaDriverRuntime

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def driver():
    try:
        runtime = NvidiaDriverRuntime()
    except DriverError as exc:
        if os.environ.get("RUDA_REQUIRE_GPU") == "1":
            pytest.fail(f"GPU required but unavailable: {exc}")
        pytest.skip(str(exc))
    yield runtime
    runtime.close()


class Binary(torch.nn.Module):
    def __init__(self, op):
        super().__init__(); self.op = op
    def forward(self, x, y):
        if self.op == "add": return x + y
        if self.op == "mul": return x * y
        if self.op == "silu_mul": return F.silu(x) * y
        return x @ y


class Silu(torch.nn.Module):
    def forward(self, x):
        return F.silu(x)


class TinyMLP(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.norm = torch.nn.RMSNorm(17, eps=1e-6, dtype=dtype)
        self.g = torch.nn.Linear(17, 33, dtype=dtype)
        self.u = torch.nn.Linear(17, 33, dtype=dtype)
        self.o = torch.nn.Linear(33, 17, dtype=dtype)
    def forward(self, x):
        y = self.norm(x)
        return self.o(F.silu(self.g(y)) * self.u(y)) + x


def tolerance(dtype):
    return 5e-4 if dtype == torch.float32 else (4e-3 if dtype == torch.float16 else 3e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("op", ["add", "mul", "silu", "silu_mul", "rms_norm", "matmul", "linear", "mlp"])
def test_ptx_end_to_end(driver, dtype, op):
    if dtype == torch.bfloat16 and driver.sm < 80:
        pytest.skip("This BF16 PTX variant declares sm_80")
    torch.manual_seed(728)
    x = torch.randn(3, 17, dtype=dtype) * 0.3
    if op in {"add", "mul", "silu_mul"}:
        model, args = Binary(op), (x, torch.randn_like(x))
    elif op == "matmul":
        model, args = Binary(op), (x, torch.randn(17, 35, dtype=dtype))
    elif op == "silu":
        model, args = Silu(), (x,)
    elif op == "rms_norm":
        model, args = torch.nn.RMSNorm(17, eps=1e-6, dtype=dtype), (x,)
    elif op == "linear":
        model, args = torch.nn.Linear(17, 35, dtype=dtype), (x,)
    else:
        model, args = TinyMLP(dtype), (x,)
    model.eval()
    plan = compile_exported(torch.export.export(model, args))
    with Executor(plan, driver) as run, torch.inference_mode():
        expected = model(*args)
        for _ in range(3):  # Reused buffers/kernels must remain correct.
            actual = run(*args)
            torch.testing.assert_close(actual, expected, rtol=tolerance(dtype), atol=tolerance(dtype))
        assert run.stats["calls"] == 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 255, 257, 1031])
def test_rmsnorm_tail_and_strided_reduction(driver, dtype, width):
    if dtype == torch.bfloat16 and driver.sm < 80:
        pytest.skip("This BF16 PTX variant declares sm_80")
    torch.manual_seed(829)
    model = torch.nn.RMSNorm(width, eps=1e-6, dtype=dtype).eval()
    x = torch.randn(2, width, dtype=dtype)
    plan = compile_exported(torch.export.export(model, (x,)))
    with Executor(plan, driver) as run, torch.inference_mode():
        torch.testing.assert_close(run(x), model(x), rtol=tolerance(dtype), atol=tolerance(dtype))
