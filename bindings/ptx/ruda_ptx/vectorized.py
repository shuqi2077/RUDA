"""Opt-in 4-wide FP32 elementwise PTX. Scalar tail and alignment-safe path.

No extra tensor/workspace. A vector instruction reduces issued memory
instructions, not the number of bytes read/written. Not always faster on small
inputs; retain scalar emitter as the default pending GPU measurements.
"""
import math
from .emitter import Kernel, TensorSpec, _start, _hex


def elementwise4(name: str, operation: str, spec: TensorSpec) -> Kernel:
    if spec.dtype != "float32":
        raise ValueError("4-wide PTX candidate supports float32 only")
    if operation not in ("add", "mul", "silu", "silu_mul"):
        raise ValueError("Unsupported vector elementwise operation")
    binary = operation != "silu"
    params = ("x", "y", "out") if binary else ("x", "out")
    body, sm = _start(name, spec.dtype, params)
    body += f"""    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd2, [out];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    mul.lo.u32 %r3, %r3, 4;
    setp.ge.u32 %p0, %r3, {spec.numel};
    @%p0 bra DONE;
    or.b64 %rd6, %rd0, %rd2;
"""
    if binary:
        body += "    ld.param.u64 %rd1, [y];\n    or.b64 %rd6, %rd6, %rd1;\n"
    body += f"""    and.b64 %rd6, %rd6, 15;
    setp.ne.u64 %p1, %rd6, 0;
    @%p1 bra TAIL;
    add.u32 %r4, %r3, 3;
    setp.ge.u32 %p0, %r4, {spec.numel};
    @%p0 bra TAIL;
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.v4.f32 {{%f0, %f1, %f2, %f3}}, [%rd4];
"""
    if binary:
        body += "    add.u64 %rd4, %rd1, %rd3;\n    ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd4];\n"

    def compute(x, y):
        text = ""
        if operation in ("silu", "silu_mul"):
            text += f"""    neg.f32 %f8, {x};
    mul.f32 %f8, %f8, {_hex(math.log2(math.e))};
    ex2.approx.f32 %f8, %f8;
    add.f32 %f8, %f8, {_hex(1.0)};
    div.rn.f32 {x}, {x}, %f8;
"""
        if operation != "silu":
            op = "add" if operation == "add" else "mul"
            text += f"    {op}.f32 {x}, {x}, {y};\n"
        return text

    for j in range(4):
        body += compute(f"%f{j}", f"%f{j+4}")
    body += "    add.u64 %rd4, %rd2, %rd3;\n    st.global.v4.f32 [%rd4], {%f0, %f1, %f2, %f3};\n    bra DONE;\nTAIL:\n"
    for j in range(4):
        body += f"""    add.u32 %r4, %r3, {j};
    setp.ge.u32 %p0, %r4, {spec.numel};
    @%p0 bra DONE;
    mul.wide.u32 %rd3, %r4, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
"""
        if binary:
            body += "    add.u64 %rd4, %rd1, %rd3;\n    ld.global.f32 %f4, [%rd4];\n"
        body += compute("%f0", "%f4")
        body += "    add.u64 %rd4, %rd2, %rd3;\n    st.global.f32 [%rd4], %f0;\n"
    body += "DONE:\n    ret;\n}\n"
    return Kernel(name, operation, body, params, ((spec.numel+1023)//1024, 1, 1), (256, 1, 1), sm)
