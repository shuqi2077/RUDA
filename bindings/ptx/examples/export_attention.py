"""Export static GQA decode to direct PTX; GPU execution is an explicit opt-in."""
import argparse
import json
import torch
import torch.nn.functional as F
from ruda_ptx import compile_exported, Executor


class DecodeAttention(torch.nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value, enable_gqa=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-dir", default="./ptx-attention")
    parser.add_argument("--run-driver", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(801)
    model = DecodeAttention().eval()
    tensors = (torch.randn(1,4,1,64)*0.3, torch.randn(1,2,127,64)*0.3, torch.randn(1,2,127,64))
    plan = compile_exported(torch.export.export(model, tensors), decode_partitions=4)
    plan.write(args.emit_dir)
    print(json.dumps(plan.report(), indent=2))
    if args.run_driver:
        from ruda_ptx.nvidia_driver import NvidiaDriverRuntime
        with NvidiaDriverRuntime() as rt, Executor(plan, rt) as execute, torch.inference_mode():
            torch.testing.assert_close(execute(*tensors), model(*tensors), rtol=8e-4, atol=8e-4)
            print("Actual driver PTX JIT and numerical comparison passed for this example only.")


if __name__ == "__main__":
    main()
