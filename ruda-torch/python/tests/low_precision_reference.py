import sys
import torch


def prepare(device, dtype):
    a = torch.linspace(-2, 3, 42).reshape(2, 3, 7).transpose(0, 1).to(device=device, dtype=dtype).requires_grad_()
    w = torch.linspace(0.5, 1.5, 7).to(device=device, dtype=dtype).requires_grad_()
    b = torch.linspace(-0.5, 0.5, 7).to(device=device, dtype=dtype).requires_grad_()
    grad = torch.linspace(-0.5, 0.7, 42).reshape(a.shape).to(device=device, dtype=dtype)
    large = torch.full((2, 17), 60000., dtype=dtype).to(device)
    matrix = torch.linspace(-1, 1, 35).reshape(5, 7).to(device=device, dtype=dtype)

    def run():
        results = {}
        operations = {
            "scalar_add": lambda x: x + 0.333333333,
            "scalar_mul": lambda x: x * 0.333333333,
            "scalar_div": lambda x: x / 3.14159265,
            "relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh,
            "silu": torch.nn.functional.silu,
            "softmax": lambda x: x.softmax(-1), "log_softmax": lambda x: x.log_softmax(0),
            "softmax_float": lambda x: x.softmax(-1, dtype=torch.float32),
            "power3": lambda x: x.pow(3),
            "layer_norm": lambda x: torch.nn.functional.layer_norm(x, (7,), w, b),
            "mm": lambda x: x.reshape(6, 7) @ matrix.t(),
        }
        for name, operation in operations.items():
            a.grad = w.grad = b.grad = None
            output = operation(a)
            upstream = grad if output.shape == a.shape else grad.reshape(6, 7)[:, :5]
            output.backward(upstream.to(output.dtype))
            results[name] = output.detach()
            results[name + "_grad"] = a.grad.detach()
            if name == "layer_norm":
                results["weight_grad"] = w.grad.detach()
                results["bias_grad"] = b.grad.detach()
        output, mean, rstd = torch.native_layer_norm(a, (7,), w, b, 1e-5)
        results.update(norm_output=output.detach(), norm_mean=mean.detach(), norm_rstd=rstd.detach(),
                       mean_large=large.mean(-1), sum_large=large.sum(-1),
                       sum_float=large.sum(-1, dtype=torch.float32))
        return results
    return run


if __name__ == "__main__":
    dtype = getattr(torch, sys.argv[2])
    result = prepare("cuda", dtype)()
    torch.save({name: value.cpu() for name, value in result.items()}, sys.argv[1])
