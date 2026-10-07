"""Native fixed-address forward with a same-device, first-order backward.

Backward recomputes from per-forward snapshots; neither CPU transfers nor a
hidden CPU execution fallback are used. This is not native backward capture.
"""
import torch
from torch.autograd.function import once_differentiable
from ._graph_spec import NO_WEIGHT, UNARY_CODES


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
        elif op in UNARY_CODES.values():
            name = next(name for name, code in UNARY_CODES.items() if code == op)
            out = getattr(torch.ops.aten, name).default(x).to(left.dtype)
        elif op in (7, 30):
            out = (torch.mm(x, y) if op == 7 else torch.bmm(x, y)).to(left.dtype)
        elif op == 8: out = (x / y).to(left.dtype)
        elif op in (15, 16, 18):
            name = {15:'silu_backward', 16:'sigmoid_backward', 18:'tanh_backward'}[op]
            out = getattr(torch.ops.aten, name).default(x, y).to(left.dtype)
        elif op in (109,110,111):
            out = ({109: lambda: x+scalar, 110: lambda: x*scalar,
                    111: lambda: x/scalar}[op]()).to(left.dtype)
        elif op in (102,103):
            out = (torch.softmax(x, int(scalar)) if op == 102 else
                   torch.log_softmax(x, int(scalar))).to(left.dtype)
        elif op in (106,107):
            fn = torch.ops.aten._softmax_backward_data if op == 106 else torch.ops.aten._log_softmax_backward_data
            out = fn.default(x, y, int(scalar), x.dtype).to(left.dtype)
        elif op in (104,105):
            dims = tuple(d for d in range(x.ndim) if int(scalar) & (1 << d))
            out = (x.sum(dims, keepdim=True) if op == 104 else x.mean(dims, keepdim=True)).to(left.dtype)
        elif op == 112:
            out = left.reshape(layout.specs[layout.inputs+i].shape).clone()
        elif op == 113:
            axes = tuple((int(scalar)>>(3*d))&7 for d in range(left.ndim))
            out = left.permute(axes).contiguous().clone()
        elif op == 114:
            out = left.to(getattr(torch,layout.specs[layout.inputs+i].dtype),copy=True)
        elif op == 115:
            out = left.expand(layout.specs[layout.inputs+i].shape).contiguous().clone()
        elif op in (116,117,118):
            out = (x+scalar*y if op == 116 else x*y if op == 117 else x/y).to(left.dtype)
        elif op in (119,120):
            dims = tuple(d for d in range(x.ndim) if int(scalar)&(1<<d))
            out = (x.sum(dims) if op == 119 else x.mean(dims)).to(left.dtype)
        elif op in (57,121,122,123,124):
            if op == 121: out = torch.pow(x,y)
            elif op == 122: out = torch.div(x,y,rounding_mode='floor')
            elif op == 57: out = torch.div(x,y,rounding_mode='trunc')
            elif op == 123: out = torch.fmod(x,y)
            else: out = torch.remainder(x,y)
            out = out.to(left.dtype)
        elif op in (129,130,131,132,133):
            stored = torch.full((),scalar,dtype=left.dtype,device=left.device).float()
            if op == 129:
                if scalar == 0: out = torch.ones_like(x)
                elif scalar == 1: out = x.clone()
                elif scalar == 2: out = x*x
                elif scalar == 3: out = x*x*x
                elif scalar == 0.5: out = x.sqrt()
                elif scalar == -0.5: out = x.rsqrt()
                elif scalar == -1: out = x.reciprocal()
                else: out = torch.pow(x,stored)
            elif op == 130: out = torch.div(x,stored,rounding_mode='floor')
            elif op == 131: out = torch.fmod(x,stored)
            elif op == 132: out = torch.remainder(x,stored)
            else: out = torch.div(x,stored,rounding_mode='trunc')
            out = out.to(left.dtype)
        else:
            raise RuntimeError(f'unsupported training graph opcode: {op}')
        values.append(out)
    return values


def backward_values(layout, inputs, grad_outputs):
    """First-order vector-Jacobian product, including fan-out and repeated edges.

    Intermediate gradients accumulate in their corresponding storage dtype.
    In particular, fused SiLU-mul keeps the original low-precision cast boundary.
    """
    if any(op not in (0,1,2,14,100,101) for op in layout.words[::3]):
        # Legacy graphs keep their tested storage-rounded hand derivatives.
        # New operations use same-device autograd; the model compiler separately
        # captures these derivatives through AOTAutograd, not this bridge.
        with torch.enable_grad():
            leaves = tuple(value.detach().requires_grad_(True) for value in inputs)
            values = forward_values(layout, leaves)
            selected = [(values[i], grad) for i, grad in
                        zip(layout.output_indices, grad_outputs, strict=True)
                        if grad is not None and values[i].requires_grad]
            if not selected: return (None,) * len(leaves)
            return torch.autograd.grad(tuple(v for v,g in selected), leaves,
                tuple(g for v,g in selected), allow_unused=True)
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
