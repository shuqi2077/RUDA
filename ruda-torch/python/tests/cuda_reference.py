import sys
import torch


def main(path, attention=False, builtin=False, dtype=torch.float32):
    torch.manual_seed(12)
    if builtin:
        from attention_model import BuiltinAttentionBlock
        model = BuiltinAttentionBlock()
    elif attention:
        from attention_model import AttentionBlock
        model = AttentionBlock()
    else:
        model = torch.nn.Sequential(torch.nn.Linear(7, 9), torch.nn.ReLU(), torch.nn.Linear(9, 3))
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    model = model.to(device="cuda", dtype=dtype)
    data, target = (torch.randn(2, 5, 8), torch.randn(2, 5, 8)) if attention else (torch.randn(5, 7), torch.randn(5, 3))
    x, y = data.to(device="cuda", dtype=dtype), target.to(device="cuda", dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, foreach=False)
    for _ in range(3):
        optimizer.zero_grad()
        error = (model(x) - y).square()
        loss = error.mean() if attention else error.sum()
        loss.backward()
        optimizer.step()
    torch.save(dict(initial=initial, data=data, target=target, loss=loss.detach().cpu(),
        parameters=[p.detach().cpu() for p in model.parameters()],
        gradients=[p.grad.cpu() for p in model.parameters()]), path)


if __name__ == "__main__":
    dtype = torch.float16 if "--float16" in sys.argv else torch.bfloat16 if "--bfloat16" in sys.argv else torch.float32
    main(sys.argv[1], "--attention" in sys.argv[2:], "--builtin" in sys.argv[2:], dtype)
