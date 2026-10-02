#!/usr/bin/env python3
"""Strict RUDA device acceptance: no CPU substitute, no successful skipped tests.

Build the Rust runtime and C++ Python extension first. This script separately
compares outputs, gradients and optimizer updates against CPU reference models.
CPU-generated inputs are transferred explicitly *before* transfer counters start.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('steps must be positive')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='JSON validation report')
    parser.add_argument('--steps', type=positive, default=3)
    args = parser.parse_args(argv)
    report = {'status': 'failed', 'gpu_executed': False, 'cpu_fallback': False,
              'rust_built_by_this_script': False, 'cases': []}
    code = 2
    try:
        import torch
        report['torch'] = torch.__version__
        source = Path(__file__).resolve().parents[1] / 'python'
        sys.path.insert(0, str(source))
        import ruda_torch
        if not ruda_torch.is_available() or not ruda_torch._graph_available:
            raise RuntimeError('RUDA runtime and native static graph API are required')

        class GatedBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Linear(8, 8)
                self.up = torch.nn.Linear(8, 8)
            def forward(self, x):
                gate = self.gate(x)
                up = self.up(x)
                return torch.nn.functional.silu(gate) * up + x

        class ConvBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.output = torch.nn.Linear(4, 2)
            def forward(self, x):
                return self.output(torch.relu(self.conv(x)).mean((2, 3)))

        class AttentionCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.key = torch.nn.Parameter(torch.randn(2, 4, 5) * .1)
                self.value = torch.nn.Parameter(torch.randn(2, 5, 6) * .1)
            def forward(self, x):
                scores = torch.bmm(x, self.key) / 2.
                probabilities = torch.softmax(scores, -1)
                return torch.bmm(probabilities, self.value).tanh().mean(-1, keepdim=True)

        for name, constructor, shape in (
            ('gated_dense', GatedBlock, (3, 8)),
            ('convolution', ConvBlock, (2, 3, 5, 5)),
            ('native_attention_core', AttentionCore, (2, 3, 4)),
        ):
            torch.random.default_generator.manual_seed(672)
            reference = constructor()
            model = copy.deepcopy(reference).to('ruda:0')
            compiled = ruda_torch.compile(model, min_native_ops=1, dynamic=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=.002, foreach=False)
            expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.002, foreach=False)
            case = {'name': name, 'passed': False, 'steps': 0}
            report['cases'].append(case)
            try:
                for step in range(args.steps):
                    host = torch.randn(shape, requires_grad=True)
                    x = host.detach().to('ruda:0').requires_grad_()
                    expected_optimizer.zero_grad(set_to_none=True)
                    expected = reference(host)
                    expected_loss = expected.square().sum()
                    expected_loss.backward()
                    expected_optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    before = ruda_torch.execution_stats()
                    actual = compiled(x)
                    loss = actual.square().sum()
                    loss.backward()
                    optimizer.step()
                    ruda_torch.synchronize()
                    after = ruda_torch.execution_stats()
                    if after['kernel_launches'] <= before['kernel_launches']:
                        raise RuntimeError('no real RUDA kernels were executed')
                    report['gpu_executed'] = True
                    for counter in ('host_to_device_bytes', 'device_to_host_bytes'):
                        if after[counter] != before[counter]:
                            raise RuntimeError(f'unexpected host transfer during training: {counter}')
                    torch.testing.assert_close(actual.detach().cpu(), expected.detach(), rtol=5e-4, atol=5e-5)
                    torch.testing.assert_close(x.grad.cpu(), host.grad, rtol=7e-4, atol=7e-5)
                    for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(parameter.detach().cpu(), expected_parameter.detach(), rtol=5e-4, atol=5e-5)
                        torch.testing.assert_close(parameter.grad.cpu(), expected_parameter.grad, rtol=7e-4, atol=7e-5)
                    case['steps'] = step + 1
                case['compiler'] = compiled.info
                if compiled.info['forward_calls'] < args.steps or compiled.info['backward_calls'] < args.steps:
                    raise RuntimeError('the test did not execute captured forward AND backward graphs')
                if name in ('gated_dense', 'native_attention_core') and compiled.info['native_replays'] == 0:
                    raise RuntimeError(f'{name} executed no native static graph region')
                if name == 'native_attention_core':
                    operators = [op for graph in compiled.info['graphs'] for op in graph['operators']]
                    for target in ('aten.bmm.default', 'aten._softmax.default'):
                        if not any(op['target'] == target and op['execution'] == 'guarded-native-region'
                                   for op in operators):
                            raise RuntimeError(f'attention native mapping missing: {target}')
                case['passed'] = True
            finally:
                case.setdefault('compiler', compiled.info)
                compiled.close()
        report['status'] = 'passed'
        code = 0
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
