import ctypes
import gc
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ruda_torch


class NativeGpuTests(unittest.TestCase):
    def gpu(self, data):
        return torch.as_tensor(data, dtype=torch.float32).to("ruda")

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-5, atol=2e-5, equal_nan=True)

    def test_storage_is_cuda_device_memory(self):
        tensor = self.gpu([1, 2, 3])
        cuda = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        cuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
        cuda.cuPointerGetAttribute.restype = ctypes.c_int
        memory_type = ctypes.c_uint()
        self.assertEqual(cuda.cuPointerGetAttribute(ctypes.byref(memory_type), 2, tensor.data_ptr()), 0)
        self.assertEqual(memory_type.value, 2)  # CU_MEMORYTYPE_DEVICE, not HOST or UNIFIED.
        self.assertEqual(tensor.device, torch.device("ruda:0"))

    def test_transfer_and_strided_alias(self):
        source = torch.arange(35, dtype=torch.float32).view(5, 7)
        tensor = source.to("ruda")
        view = tensor[1:4, 1:6:2].t()
        self.assertClose(view, source[1:4, 1:6:2].t())
        view.fill_(9)
        source[1:4, 1:6:2] = 9
        self.assertClose(tensor, source)

    def test_noncontiguous_host_transfer(self):
        source = torch.arange(30, dtype=torch.float32).view(5, 6).t()
        tensor = source.to("ruda")
        self.assertClose(tensor, source)
        output = torch.empty(5, 6).t()
        output.copy_(tensor)
        torch.testing.assert_close(output, source)

    def test_copy_broadcast(self):
        output = torch.empty((3, 5), device="ruda")
        output.copy_(self.gpu([1, 2, 3, 4, 5]))
        self.assertClose(output, torch.tensor([[1, 2, 3, 4, 5]] * 3, dtype=torch.float32))

    def test_view_retains_allocation(self):
        original = self.gpu([[1, 2], [3, 4]])
        view = original.t()
        del original
        gc.collect()
        pressure = [torch.empty((128,), device="ruda") for _ in range(20)]
        self.assertClose(view + 1, torch.tensor([[2, 4], [3, 5]], dtype=torch.float32))

    def test_pointwise_broadcast_and_tail(self):
        a = torch.linspace(-2, 2, 257).reshape(1, 257)
        b = torch.tensor([[0.5], [1.25], [-0.5]])
        x, y = a.to("ruda"), b.to("ruda")
        self.assertClose(torch.relu(x + 2 * y) * x, torch.relu(a + 2 * b) * a)

    def test_special_values(self):
        a = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, -1, 1])
        self.assertClose(torch.relu(a.to("ruda")), torch.relu(a))
        self.assertClose((a.to("ruda") + a.to("ruda")), a + a)

    def test_mm_strides_and_empty(self):
        for m, k, n in [(3, 7, 5), (17, 13, 19), (0, 4, 3), (3, 0, 5), (3, 4, 0)]:
            a = torch.arange(k * m, dtype=torch.float32).reshape(k, m).t() / 19
            b = torch.arange(n * k, dtype=torch.float32).reshape(n, k).t() / 23
            self.assertClose(a.to("ruda") @ b.to("ruda"), a @ b)

    def test_sum_axes_empty_and_scalar(self):
        for shape in [(3, 4, 5), (2, 0, 3), ()]:
            a = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
            x = a.to("ruda")
            for dim in [None, [0], [-1], []]:
                for keepdim in [False, True]:
                    self.assertClose(x.sum(dim=dim, keepdim=keepdim), a.sum(dim=dim, keepdim=keepdim))

    def test_no_host_transfers_during_forward_backward(self):
        a = self.gpu([[1, 2, -3], [4, -5, 6]]).requires_grad_()
        w = self.gpu([[1, -2], [3, 4], [-1, 2]]).requires_grad_()
        before = ruda_torch.execution_stats()
        loss = torch.relu(a @ w).square().sum()
        loss.backward()
        after = ruda_torch.execution_stats()
        self.assertGreater(after["kernel_launches"], before["kernel_launches"])
        for name in ["host_to_device_bytes", "device_to_host_bytes"]:
            self.assertEqual(before[name], after[name], name)
        ca = a.detach().cpu().requires_grad_()
        cw = w.detach().cpu().requires_grad_()
        reference = torch.relu(ca @ cw).square().sum()
        reference.backward()
        self.assertClose(loss, reference)
        self.assertClose(a.grad, ca.grad)
        self.assertClose(w.grad, cw.grad)

    def test_eager_model_sgd_matches_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.pt"
            subprocess.run([sys.executable, str(Path(__file__).with_name("cuda_reference.py")), str(path)], check=True)
            reference = torch.load(path, weights_only=True)
        model = torch.nn.Sequential(torch.nn.Linear(7, 9), torch.nn.ReLU(), torch.nn.Linear(9, 3))
        model.load_state_dict(reference["initial"])
        model = model.to("ruda")
        data, target = reference["data"], reference["target"]
        x, y = data.to("ruda"), target.to("ruda")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, foreach=False)
        before = ruda_torch.execution_stats()
        for _ in range(3):
            optimizer.zero_grad()
            loss = (model(x) - y).square().sum()
            loss.backward()
            optimizer.step()
        after = ruda_torch.execution_stats()
        self.assertEqual(before["host_to_device_bytes"], after["host_to_device_bytes"])
        self.assertEqual(before["device_to_host_bytes"], after["device_to_host_bytes"])
        self.assertClose(loss, reference["loss"])
        for actual, expected, grad in zip(model.parameters(), reference["parameters"], reference["gradients"]):
            self.assertClose(actual, expected)
            self.assertClose(actual.grad, grad)

    def test_unsupported_operator_never_falls_back(self):
        tensor = self.gpu([1, 2])
        before = ruda_torch.execution_stats()
        with self.assertRaisesRegex(NotImplementedError, "CPU fallback is disabled"):
            torch.sin(tensor)
        self.assertEqual(before, ruda_torch.execution_stats())

    def test_overlap_rejected(self):
        tensor = self.gpu([1, 2, 3])
        with self.assertRaises(RuntimeError):
            tensor.expand(3, 3).add_(1)

    def test_bmm_forward_backward_strides(self):
        a = torch.linspace(-1, 1, 2 * 7 * 5).reshape(2, 7, 5).transpose(1, 2).requires_grad_()
        b = torch.linspace(0.1, 1, 2 * 3 * 7).reshape(2, 3, 7).transpose(1, 2).requires_grad_()
        x, y = a.detach().to("ruda").requires_grad_(), b.detach().to("ruda").requires_grad_()
        output, expected = torch.bmm(x, y), torch.bmm(a, b)
        output.square().mean().backward(); expected.square().mean().backward()
        self.assertClose(output, expected)
        self.assertClose(x.grad, a.grad); self.assertClose(y.grad, b.grad)
        for shape in [(0, 3, 4, 5), (2, 0, 4, 5), (2, 3, 0, 5), (2, 3, 4, 0)]:
            batch, m, k, n = shape
            a, b = torch.ones(batch, m, k), torch.ones(batch, k, n)
            self.assertClose(torch.bmm(a.to("ruda"), b.to("ruda")), torch.bmm(a, b))

    def test_softmax_and_log_softmax_axes_backward(self):
        for shape in [(2, 3, 7), (), (2, 0, 3)]:
            data = torch.linspace(-80, 80, math.prod(shape)).reshape(shape)
            if data.ndim:
                data = data.transpose(0, -1)
            for dim in range(-max(data.ndim, 1), max(data.ndim, 1)):
                for function in (torch.softmax, torch.log_softmax):
                    a = data.detach().requires_grad_()
                    x = data.to("ruda").requires_grad_()
                    output, expected = function(x, dim), function(a, dim)
                    weights = torch.linspace(0.2, 1, data.numel()).reshape(data.shape)
                    output.backward(weights.to("ruda")); expected.backward(weights)
                    self.assertClose(output, expected)
                    self.assertClose(x.grad, a.grad)

    def test_softmax_special_values(self):
        a = torch.tensor([[float("inf"), 1, 2], [-float("inf")] * 3,
                          [float("nan"), 0, 1], [10000, 9999, -10000]])
        for function in (torch.softmax, torch.log_softmax):
            self.assertClose(function(a.to("ruda"), -1), function(a, -1))

    def test_activations_and_division_backward(self):
        data = torch.linspace(-12, 12, 257)
        for function in (torch.sigmoid, torch.tanh, torch.nn.functional.silu):
            a = data.clone().requires_grad_()
            x = data.to("ruda").requires_grad_()
            output, expected = function(x), function(a)
            output.square().mean().backward(); expected.square().mean().backward()
            self.assertClose(output, expected); self.assertClose(x.grad, a.grad)
        a = torch.linspace(0.1, 3, 257).requires_grad_()
        x = a.detach().to("ruda").requires_grad_()
        output = (torch.exp(x / 3).log() + torch.sqrt(x) + torch.rsqrt(x)).mean()
        expected = (torch.exp(a / 3).log() + torch.sqrt(a) + torch.rsqrt(a)).mean()
        output.backward(); expected.backward()
        self.assertClose(output, expected); self.assertClose(x.grad, a.grad)

    def test_layer_norm_forward_backward(self):
        for shape, norm in [((2, 3, 7), (7,)), ((2, 3, 7), (3, 7)), ((3, 7), (3, 7)), ((0, 3, 7), (7,))]:
            for affine in (False, True):
                a = torch.linspace(-2, 3, math.prod(shape)).reshape(shape).requires_grad_()
                x = a.detach().to("ruda").requires_grad_()
                w = torch.linspace(0.5, 1.5, math.prod(norm)).reshape(norm).requires_grad_() if affine else None
                b = torch.linspace(-1, 1, math.prod(norm)).reshape(norm).requires_grad_() if affine else None
                rw = w.detach().to("ruda").requires_grad_() if affine else None
                rb = b.detach().to("ruda").requires_grad_() if affine else None
                output = torch.nn.functional.layer_norm(x, norm, rw, rb)
                expected = torch.nn.functional.layer_norm(a, norm, w, b)
                upstream = torch.linspace(0.1, 1, math.prod(shape)).reshape(shape)
                output.backward(upstream.to("ruda")); expected.backward(upstream)
                self.assertClose(output, expected); self.assertClose(x.grad, a.grad)
                if affine:
                    self.assertClose(rw.grad, w.grad); self.assertClose(rb.grad, b.grad)

    def test_attention_training_matches_cuda_without_transfers(self):
        self._check_attention_training(False)

    def test_builtin_multihead_attention_training(self):
        self._check_attention_training(True)

    def _check_attention_training(self, builtin, dtype=torch.float32):
        from attention_model import AttentionBlock, BuiltinAttentionBlock
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.pt"
            command = [sys.executable, str(Path(__file__).with_name("cuda_reference.py")), str(path), "--attention"]
            if builtin:
                command.append("--builtin")
            if dtype != torch.float32:
                command.append("--" + str(dtype).removeprefix("torch."))
            subprocess.run(command, check=True)
            reference = torch.load(path, weights_only=True)
        model = BuiltinAttentionBlock() if builtin else AttentionBlock()
        model.load_state_dict(reference["initial"])
        model = model.to(device="ruda", dtype=dtype)
        x, y = reference["data"].to(device="ruda", dtype=dtype), reference["target"].to(device="ruda", dtype=dtype)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, foreach=False)
        before = ruda_torch.execution_stats()
        for _ in range(3):
            optimizer.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            optimizer.step()
        after = ruda_torch.execution_stats()
        self.assertEqual(before["host_to_device_bytes"], after["host_to_device_bytes"])
        self.assertEqual(before["device_to_host_bytes"], after["device_to_host_bytes"])
        compare = self.assertClose if dtype == torch.float32 else self.assertLowClose
        compare(loss, reference["loss"])
        for actual, expected, grad in zip(model.parameters(), reference["parameters"], reference["gradients"]):
            compare(actual, expected); compare(actual.grad, grad)

    def assertLowClose(self, actual, expected):
        dtype = expected.dtype
        rtol, atol = (0.003, 3e-5) if dtype == torch.float16 else (0.025, 3e-4) if dtype == torch.bfloat16 else (2e-5, 2e-5)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol, equal_nan=True)

    def test_low_precision_storage_and_conversion(self):
        cuda = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        cuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
        cuda.cuPointerGetAttribute.restype = ctypes.c_int
        for dtype in (torch.float16, torch.bfloat16):
            bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(128, 512)
            source = bits.view(dtype)
            value = source.to("ruda")
            memory_type = ctypes.c_uint()
            self.assertEqual(cuda.cuPointerGetAttribute(ctypes.byref(memory_type), 2, value.data_ptr()), 0)
            self.assertEqual(memory_type.value, 2)
            self.assertEqual(value.element_size(), 2)
            self.assertEqual(value.untyped_storage().nbytes(), source.numel() * 2)
            self.assertTrue(torch.equal(value.cpu().view(torch.int16), bits))
            self.assertTrue(torch.equal(value.t().clone().cpu().view(torch.int16), bits.t()))
            self.assertTrue(torch.equal(value[:, 1::2].cpu().view(torch.int16), bits[:, 1::2]))
            before = ruda_torch.execution_stats()
            widened = value.float()
            crossed = value.to(torch.bfloat16 if dtype == torch.float16 else torch.float16)
            after = ruda_torch.execution_stats()
            self.assertEqual(before["host_to_device_bytes"], after["host_to_device_bytes"])
            self.assertEqual(before["device_to_host_bytes"], after["device_to_host_bytes"])
            torch.testing.assert_close(widened.cpu(), source.float(), rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(crossed.cpu(), source.to(crossed.dtype), rtol=0, atol=0, equal_nan=True)
            f32 = torch.tensor([0., -0., 2**-25, 2**-24, 2**-133, 2**-134,
                                1 + 2**-11, 1 + 3 * 2**-11, 1 + 2**-8, 1 + 3 * 2**-8,
                                65504., 65520., float("inf"), -float("inf"), float("nan")])
            for converted in (f32.to("ruda").to(dtype), f32.to(device="ruda", dtype=dtype)):
                torch.testing.assert_close(converted.cpu(), f32.to(dtype), rtol=0, atol=0, equal_nan=True)
                self.assertEqual(converted[1].float().cpu().view(torch.int32).item(), -2147483648)
            target = torch.empty((5, 3), device="ruda", dtype=dtype).t()
            target.copy_(torch.arange(15, dtype=torch.float32).reshape(3, 5))
            self.assertEqual(target[1, 1].item(), 6.)
            host = torch.empty((5, 3), dtype=torch.float32).t()
            host.copy_(target)
            torch.testing.assert_close(host, torch.arange(15, dtype=torch.float32).reshape(3, 5))

    def _check_low_precision_operators(self, dtype):
        from low_precision_reference import prepare
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.pt"
            subprocess.run([sys.executable, str(Path(__file__).with_name("low_precision_reference.py")),
                            str(path), str(dtype).removeprefix("torch.")], check=True)
            expected = torch.load(path, weights_only=True)
        run = prepare("ruda", dtype)
        before = ruda_torch.execution_stats()
        actual = run()
        after = ruda_torch.execution_stats()
        for name in ("host_to_device_bytes", "device_to_host_bytes"):
            self.assertEqual(before[name], after[name])
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            with self.subTest(result=name):
                self.assertLowClose(actual[name], expected[name])

    def test_float16_operators_backward(self):
        self._check_low_precision_operators(torch.float16)

    def test_bfloat16_operators_backward(self):
        self._check_low_precision_operators(torch.bfloat16)

    def test_float16_multihead_attention_training(self):
        self._check_attention_training(True, torch.float16)

    def test_bfloat16_multihead_attention_training(self):
        self._check_attention_training(True, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
