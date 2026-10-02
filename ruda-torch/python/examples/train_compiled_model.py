"""A caller-owned model/optimizer; requires the real RUDA native extension.

Construct on CPU, transfer explicitly, then compile. No CUDA-specific compiler
or CPU fallback is installed. Use validate_model_compile.py for acceptance.
"""
import json
import torch
import ruda_torch


class GatedModel(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.gate = torch.nn.Linear(width, width)
        self.up = torch.nn.Linear(width, width)

    def forward(self, x):
        gate = self.gate(x)
        up = self.up(x)
        return torch.nn.functional.silu(gate) * up + x


def main():
    torch.random.default_generator.manual_seed(41)
    model = GatedModel().to('ruda:0')
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, foreach=False)
    x = torch.randn(3, 8).to('ruda:0')
    target = torch.zeros(3, 8).to('ruda:0')
    with ruda_torch.compile(model, native='auto') as compiled:
        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = (compiled(x) - target).square().mean()
            loss.backward()
            optimizer.step()
            # This is an EXPLICIT user-requested readback for logging.
            print(f'step={step}, loss={loss.detach().cpu().item():.6f}')
        print(json.dumps(compiled.info, ensure_ascii=False, indent=2))
        # Checkpoint keys and parameter objects remain those of the original.
        torch.save(compiled.state_dict(), 'compiled-model-state.pt')


if __name__ == '__main__':
    main()
