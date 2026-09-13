import copy
import math
import unittest

import torch
import torch.nn.functional as F
import ruda_torch


class SpatialGpuTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), equal_nan=True)

    def _convolution(self, rank, transposed, groups):
        shape = (2, 4, *((5,) * rank))
        kernel = (3,) * rank
        weight_shape = (4, 3, *kernel) if transposed else (6, 4 // groups, *kernel)
        channels = 3 * groups if transposed else 6
        x = torch.linspace(-0.9, 1.1, math.prod(shape)).reshape(shape).requires_grad_()
        w = torch.linspace(-0.4, 0.7, math.prod(weight_shape)).reshape(weight_shape).requires_grad_()
        bias = torch.linspace(-0.2, 0.2, channels).requires_grad_()
        inputs = [value.detach().to("ruda").requires_grad_() for value in (x, w, bias)]
        function = getattr(F, f"conv{'_transpose' if transposed else ''}{rank}d")
        kwargs = dict(stride=2, padding=1, dilation=2, groups=groups)
        if transposed:
            kwargs["output_padding"] = 1
        expected = function(x, w, bias, **kwargs)
        actual = function(*inputs, **kwargs)
        self.close(actual, expected)
        grad = torch.linspace(-0.5, 0.8, expected.numel()).reshape(expected.shape)
        expected.backward(grad)
        actual.backward(grad.to("ruda"))
        for source, device in zip((x, w, bias), inputs):
            self.close(device.grad, source.grad)

    def test_convolution_dimensions_groups_and_transpose(self):
        for rank in (1, 2, 3):
            for transposed in (False, True):
                for groups in (1, 2):
                    with self.subTest(rank=rank, transposed=transposed, groups=groups):
                        self._convolution(rank, transposed, groups)

    def test_transposed_output_padding_larger_than_stride(self):
        x = torch.linspace(-0.7, 0.9, 10).reshape(1, 2, 5).requires_grad_()
        w = torch.linspace(-0.3, 0.4, 12).reshape(2, 2, 3).requires_grad_()
        a, b = [v.detach().to("ruda").requires_grad_() for v in (x, w)]
        expected = F.conv_transpose1d(x, w, stride=1, dilation=3, output_padding=2)
        actual = F.conv_transpose1d(a, b, stride=1, dilation=3, output_padding=2)
        self.close(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        self.close(a.grad, x.grad)
        self.close(b.grad, w.grad)

    def test_convolution_empty_batch(self):
        x = torch.empty((0, 2, 5, 5), device="ruda", requires_grad=True)
        w = torch.ones((3, 2, 3, 3)).to("ruda").requires_grad_()
        y = F.conv2d(x, w)
        self.assertEqual(y.shape, (0, 3, 3, 3))
        y.sum().backward()
        self.close(w.grad, torch.zeros_like(w.cpu()))
        self.assertEqual(x.grad.shape, x.shape)

    def test_avg_pool_ceil_padding_divisor_and_strides(self):
        cases = [((2, 3, 7, 9), (3, 2), (2, 2), (1, 1)),
                 ((2, 3, 2, 2), (3, 3), (2, 2), (0, 0)),
                 ((3, 2, 2), (2, 2), (3, 3), (1, 1))]
        for shape, kernel, stride, padding in cases:
            for include_pad in (False, True):
                for divisor in (None, 5, -3):
                    with self.subTest(shape=shape, include_pad=include_pad, divisor=divisor):
                        x = torch.linspace(-1, 1, math.prod(shape)).reshape(shape).transpose(-1, -2).requires_grad_()
                        a = x.detach().to("ruda").requires_grad_()
                        kwargs = dict(kernel_size=kernel, stride=stride, padding=padding,
                                      ceil_mode=True, count_include_pad=include_pad, divisor_override=divisor)
                        expected = F.avg_pool2d(x, **kwargs)
                        actual = F.avg_pool2d(a, **kwargs)
                        self.close(actual, expected)
                        grad = torch.linspace(-0.2, 0.7, expected.numel()).reshape(expected.shape)
                        expected.backward(grad)
                        actual.backward(grad.to("ruda"))
                        self.close(a.grad, x.grad)

    def test_batch_norm_training_inference_and_running_stats(self):
        for training in (False, True):
            for affine in (False, True):
                with self.subTest(training=training, affine=affine):
                    host = torch.nn.BatchNorm2d(3, affine=affine, momentum=0.2).train(training)
                    device = copy.deepcopy(host).to("ruda")
                    x = torch.linspace(-2, 3, 210).reshape(2, 3, 5, 7).transpose(2, 3).requires_grad_()
                    a = x.detach().to("ruda").requires_grad_()
                    expected, actual = host(x), device(a)
                    self.close(actual, expected)
                    self.close(device.running_mean, host.running_mean)
                    self.close(device.running_var, host.running_var)
                    grad = torch.linspace(-0.7, 0.9, x.numel()).reshape(x.shape)
                    expected.backward(grad)
                    actual.backward(grad.to("ruda"))
                    self.close(a.grad, x.grad)
                    if affine:
                        self.close(device.weight.grad, host.weight.grad)
                        self.close(device.bias.grad, host.bias.grad)

    def test_batch_norm_no_running_stats_and_zero_variance(self):
        x = torch.ones((2, 3, 4), device="ruda", requires_grad=True)
        y, mean, invstd = torch.ops.aten.native_batch_norm(x, None, None, None, None, True, 0.1, 0.0)
        self.close(y, torch.zeros(2, 3, 4))
        self.close(mean, torch.ones(3))
        self.close(invstd, torch.zeros(3))
        y.sum().backward()
        self.close(x.grad, torch.zeros(2, 3, 4))

    def test_conv_batch_norm_pool_training_chain(self):
        torch.manual_seed(5)
        host = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.BatchNorm2d(4),
                                   torch.nn.ReLU(), torch.nn.AvgPool2d(2))
        device = copy.deepcopy(host).to("ruda")
        x = torch.linspace(-1, 1, 256).reshape(2, 2, 8, 8)
        expected, actual = host(x), device(x.to("ruda"))
        self.close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for a, b in zip(device.parameters(), host.parameters()):
            self.close(a.grad, b.grad)


if __name__ == "__main__":
    unittest.main()
