"""Explicit native-driver example; synthetic inputs, no model weights required.

Uploads only new Q/K/V per step for this demonstration. In a connected RUDA
model, pass DeviceTensor outputs of prior projections instead of uploading.
"""
import torch
import torch.nn.functional as F
from ruda_ptx import DeviceTensor, TensorSpec, StaticKVCache, Executor
from ruda_ptx.nvidia_driver import NvidiaDriverRuntime


def main():
    torch.manual_seed(802)
    with NvidiaDriverRuntime() as rt, StaticKVCache(rt, batch=1, kv_heads=2, capacity=32,
                                                 head_dim=64, dtype="float32") as cache:
        decode = cache.prepare_decode(4, partitions=4)
        storage = []
        def make(shape):
            spec = TensorSpec(shape)
            b = rt.allocate(spec.nbytes)
            storage.append(b)
            return DeviceTensor(b, spec)
        qd, kd, vd = make((1,4,1,64)), make((1,2,1,64)), make((1,2,1,64))
        keys, values = [], []
        try:
            for _ in range(4):
                q, k, v = torch.randn(1,4,1,64)*0.3, torch.randn(1,2,1,64)*0.3, torch.randn(1,2,1,64)
                for dst, src in ((qd,q),(kd,k),(vd,v)):
                    rt.write(dst.buffer, Executor._tensor_bytes(src))
                cache.append(kd, vd)
                out = decode.run(qd)
                keys.append(k); values.append(v)
                # Readback is explicit here for correctness verification, not
                # performed by cache.append or DecodeSession.run.
                actual = torch.frombuffer(bytearray(rt.read(out.buffer, out.spec.nbytes)), dtype=torch.float32).reshape(out.spec.shape)
                expected = F.scaled_dot_product_attention(q, torch.cat(keys,2), torch.cat(values,2), enable_gqa=True)
                torch.testing.assert_close(actual, expected, rtol=8e-4, atol=8e-4)
            print({"length":cache.length, "cache_bytes":cache.nbytes, "stats":cache.stats,
                   "decode_workspace_bytes":decode.program.workspace_bytes})
        finally:
            rt.synchronize()
            for b in reversed(storage):
                rt.free(b)


if __name__ == "__main__":
    main()
