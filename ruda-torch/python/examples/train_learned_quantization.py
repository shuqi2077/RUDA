"""Train input and quantization scale on RUDA; requires the built native runtime.

Fake quantization returns floating tensors and does NOT provide packed inference.
No implicit CPU fallback is used. Keep scale parameters in FP32.
"""
import torch
import ruda_torch


def main():
    if not ruda_torch.is_available():
        raise RuntimeError("Build and load the RUDA Rust runtime and C++ bridge first")
    x = torch.tensor([-3., -.25, .25, 3.], device="ruda:0", requires_grad=True)
    quantizer = ruda_torch.LearnedFakeQuantize(
        torch.tensor([.5], dtype=torch.float32, device="ruda:0"), bits=4
    )
    optimizer = torch.optim.SGD([x, *quantizer.parameters()], lr=.01, foreach=False)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        y = quantizer(x)
        loss = y.square().mean()
        loss.backward()
        optimizer.step()
        # Explicit reporting copies, not an execution fallback.
        print({"step": step, "loss": loss.detach().cpu().tolist(),
               "scales": quantizer.scales.detach().cpu().tolist()})


if __name__ == "__main__":
    main()
