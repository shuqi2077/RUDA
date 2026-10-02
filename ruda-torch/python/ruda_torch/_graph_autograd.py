"""Native fixed-address forward with a same-device, first-order backward.

Backward recomputes from per-forward snapshots; neither CPU transfers nor a
hidden CPU execution fallback are used. This is not native backward capture.
"""
import torch
from torch.autograd.function import once_differentiable
from ._graph_spec import NO_WEIGHT


def _silu(x):
    return x / (1.0 + (-x).exp())


def forward_values(layout, inputs):
    """Storage-rounded expressions for backward recomputation, in node order."""
    values = list(inputs)
    for i, scalar in enumerate(layout.scalars):
        op, a, b = layout.words[3*i:3*i+3]
        left = values[a]
        x = left.float()
        y = None if b == NO_WEIGHT else values[b].float()
        if op == 0:
            out = left.clone()
        elif op == 1:
            out = (x + scalar*y).to(left.dtype)
        elif op == 2:
            out = (x*y).to(left.dtype)
        elif op == 14:
            out = _silu(x).to(left.dtype)
        elif op == 101:
            activation = _silu(x).to(left.dtype)
            out = (activation.float()*y).to(left.dtype)
        elif op == 100:
            norm = x*torch.rsqrt((x*x).mean(-1, keepdim=True)+scalar)
            out = (norm if y is None else norm*y).to(left.dtype)
        else:
            raise RuntimeError(f'unsupported training graph opcode: {op}')
        values.append(out)
    return values


def backward_values(layout, inputs, grad_outputs):
    """First-order vector-Jacobian product, including fan-out and repeated edges.

    Intermediate gradients accumulate in their corresponding storage dtype.
    In particular, fused SiLU-mul keeps the original low-precision cast boundary.
    """
    values = forward_values(layout, inputs)
    gradients = [None]*len(values)

    def add(index, value):
        value = value.to(values[index].dtype)
        previous = gradients[index]
        gradients[index] = value if previous is None else previous+value

    for index, gradient in zip(layout.output_indices, grad_outputs, strict=True):
        if gradient is not None:
            add(index, gradient)
    for i in range(len(layout.scalars)-1, -1, -1):
        gradient = gradients[layout.inputs+i]
        if gradient is None:
            continue
        op, a, b = layout.words[3*i:3*i+3]
        scalar = layout.scalars[i]
        g, x = gradient.float(), values[a].float()
        if op == 0:
            add(a, g)
        elif op == 1:
            add(a, g)
            add(b, g*scalar)
        elif op == 2:
            add(a, g*values[b].float())
            add(b, g*x)
        elif op in (14, 101):
            sigmoid = 1.0/(1.0+(-x).exp())
            derivative = sigmoid*(1.0+x*(1.0-sigmoid))
            if op == 14:
                add(a, g*derivative)
            else:
                activation = _silu(x).to(values[a].dtype)
                activation_grad = (g*values[b].float()).to(values[a].dtype).float()
                add(a, activation_grad*derivative)
                add(b, g*activation.float())
        elif op == 100:
            inv = torch.rsqrt((x*x).mean(-1, keepdim=True)+scalar)
            weighted = g if b == NO_WEIGHT else g*values[b].float()
            dot = (weighted*x).mean(-1, keepdim=True)
            add(a, inv*(weighted-x*inv*inv*dot))
            if b != NO_WEIGHT:
                weight_grad = g*x*inv
                axes = tuple(range(x.ndim-1))
                if axes:
                    weight_grad = weight_grad.sum(axes)
                add(b, weight_grad)
        else:
            raise RuntimeError(f'unsupported training graph opcode: {op}')
    return tuple(gradients[:layout.inputs])


class StaticGraphFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, eager, *inputs):
        graph._check()
        graph._check_bindings()
        ctx.layout = graph._layout
        ctx.input_count = len(inputs)
        ctx.set_materialize_grads(False)
        # Originals provide version guards; snapshots protect against subsequent
        # native workspace replays. Do not mutate a leaf before its backward pass.
        snapshots = tuple(t.detach().clone() for t in inputs)
        ctx.save_for_backward(*inputs, *snapshots)
        graph._native.run(eager)
        # Outputs must not alias buffers overwritten by the next native replay.
        return tuple(graph._tensors[i].clone() for i in graph._layout.output_indices)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        saved = ctx.saved_tensors  # Also checks the original input versions.
        gradients = backward_values(ctx.layout, saved[ctx.input_count:], grad_outputs)
        needed = ctx.needs_input_grad[2:]
        return (None, None, *(g if use else None for g, use in zip(gradients, needed, strict=True)))
