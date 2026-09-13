import math
import unittest

import torch
import torch.nn.functional as F
import ruda_torch


class MaxPoolGpuTests(unittest.TestCase):
    def compare(self, source, kwargs, *, cuda_reference=False):
        reference_device = "cuda" if cuda_reference else "cpu"
        x = source.detach().to(reference_device).requires_grad_()
        a = source.detach().to("ruda").requires_grad_()
        before = ruda_torch.execution_stats()["device_to_host_bytes"]
        actual, indices = F.max_pool2d(a, return_indices=True, **kwargs)
        self.assertEqual(ruda_torch.execution_stats()["device_to_host_bytes"], before)
        expected, expected_indices = F.max_pool2d(x, return_indices=True, **kwargs)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(indices.cpu(), expected_indices.cpu(), rtol=0, atol=0)
        self.assertEqual(indices.dtype, torch.int64)
        grad = torch.linspace(-0.7, 0.9, actual.numel()).reshape(actual.shape).to(actual.dtype)
        expected.backward(grad.to(reference_device))
        device_grad = grad.to("ruda")
        before = ruda_torch.execution_stats()["device_to_host_bytes"]
        actual.backward(device_grad)
        self.assertEqual(ruda_torch.execution_stats()["device_to_host_bytes"], before)
        torch.testing.assert_close(a.grad.cpu(), x.grad.cpu(), equal_nan=True)

    def test_shapes_dilation_ceil_and_noncontiguous_inputs(self):
        cases = [((2, 3, 7, 9), dict(kernel_size=(3, 2), stride=(2, 2), padding=(1, 1), dilation=(2, 1))),
                 ((3, 2, 2), dict(kernel_size=3, stride=2)),
                 ((2, 3, 2, 2), dict(kernel_size=2, stride=3, padding=1))]
        for shape, kwargs in cases:
            source = torch.sin(torch.arange(math.prod(shape), dtype=torch.float32)).reshape(shape).transpose(-1, -2)
            with self.subTest(shape=shape):
                self.compare(source, dict(kwargs, ceil_mode=True))

    def test_nan_ties_infinities_and_signed_zero(self):
        source = torch.tensor([[[[float("-inf"), float("-inf"), float("nan"), 2],
                                 [float("-inf"), float("-inf"), 3, float("nan")],
                                 [-0.0, 0.0, 5, 5], [0.0, -0.0, 5, 5]]]])
        self.compare(source, dict(kernel_size=2, stride=2))
        values, indices = F.max_pool2d(source.to("ruda"), 2, 2, return_indices=True)
        torch.testing.assert_close(indices.cpu(), torch.tensor([[[[0, 7], [8, 10]]]]))
        self.assertTrue(torch.signbit(values.cpu()[0, 0, 1, 0]).item())

    def test_dilated_empty_window(self):
        self.compare(torch.tensor([[[[2.0]]]]), dict(kernel_size=2, stride=1, padding=1, dilation=2))

    def test_channels_last_and_low_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            source = torch.sin(torch.arange(280, dtype=torch.float32)).reshape(2, 4, 5, 7)
            source = source.to(dtype).contiguous(memory_format=torch.channels_last)
            with self.subTest(dtype=dtype):
                self.compare(source, dict(kernel_size=3, stride=2, padding=1, ceil_mode=True), cuda_reference=True)
                values, indices = F.max_pool2d(source.to("ruda"), 3, return_indices=True)
                self.assertTrue(values.is_contiguous(memory_format=torch.channels_last))
                self.assertTrue(indices.is_contiguous(memory_format=torch.channels_last))

    def test_channels_last_negative_infinity_indices(self):
        source = torch.full((1, 4, 4, 4), float("-inf")).contiguous(memory_format=torch.channels_last)
        self.compare(source, dict(kernel_size=2, stride=2), cuda_reference=True)

    def test_one_dimensional_composite_forward_and_backward(self):
        x = torch.tensor([[[1.0, 3, 2, 4, 4, -1, 2]]], requires_grad=True)
        a = x.detach().to("ruda").requires_grad_()
        y, indices = F.max_pool1d(a, 3, 2, 1, ceil_mode=True, return_indices=True)
        expected, expected_indices = F.max_pool1d(x, 3, 2, 1, ceil_mode=True, return_indices=True)
        torch.testing.assert_close(y.cpu(), expected)
        torch.testing.assert_close(indices.cpu(), expected_indices)
        expected.sum().backward()
        y.sum().backward()
        torch.testing.assert_close(a.grad.cpu(), x.grad)

    def test_empty_batch_and_invalid_parameters(self):
        a = torch.empty((0, 3, 5, 7), device="ruda", requires_grad=True)
        y, indices = F.max_pool2d(a, 2, return_indices=True)
        self.assertEqual(y.shape, (0, 3, 2, 3))
        self.assertEqual(indices.shape, y.shape)
        y.sum().backward()
        self.assertEqual(a.grad.shape, a.shape)
        a = torch.ones((1, 1, 3, 3)).to("ruda")
        for kwargs in (dict(kernel_size=0), dict(kernel_size=2, stride=0),
                       dict(kernel_size=2, padding=2), dict(kernel_size=2, dilation=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                F.max_pool2d(a, **kwargs)


if __name__ == "__main__":
    unittest.main()
