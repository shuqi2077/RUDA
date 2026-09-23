"""TEST ONLY: generated LLVM arithmetic/control flow re-targeted to host CPU.
Not AMD machine-code emulation, GPU validation, or a production fallback.
"""
import ctypes as C
import numpy as np
from llvmlite import binding as llvm


def execute_ir(ir, entry, inputs, numel, vector_width):
    llvm.initialize_all_targets(); llvm.initialize_all_asmprinters()
    text = ir.replace('target triple = "amdgcn-amd-amdhsa"', f'target triple = "{llvm.get_default_triple()}"')
    text = text.replace("amdgpu_kernel ", "").replace("ptr addrspace(1)", "ptr")
    text = text.replace(" !reqd_work_group_size !0", "").replace(' "amdgpu-flat-work-group-size"="256,256"', "")
    for which in ("workitem", "workgroup"):
        text = text.replace("llvm.amdgcn."+which+".id.x", "debug_"+which)
        text = text.replace(f"declare i32 @debug_{which}()", f"""@id_{which} = global i32 0
        define i32 @debug_{which}() {{
          %v = load i32, ptr @id_{which}
          ret i32 %v
        }}""")
    libm = C.CDLL("libm.so.6")
    llvm.add_symbol("exp2f", C.cast(libm.exp2f, C.c_void_p).value)
    module = llvm.parse_assembly(text); module.verify()
    target = llvm.Target.from_default_triple().create_target_machine(opt=2)
    module.data_layout = str(target.target_data)
    engine = llvm.create_mcjit_compiler(module, target)
    engine.finalize_object()
    out = np.full(numel+8, -12345., dtype=np.float32)
    arrays = [np.asarray(v, dtype=np.float32).copy() for v in inputs] + [out]
    assert all(a.ctypes.data % 16 == 0 for a in arrays)
    function = C.CFUNCTYPE(None, *([C.c_void_p]*len(arrays)))(engine.get_function_address(entry))
    tid = C.c_int32.from_address(engine.get_global_value_address("id_workitem"))
    bid = C.c_int32.from_address(engine.get_global_value_address("id_workgroup"))
    try:
        for group in range((numel+256*vector_width-1)//(256*vector_width)):
            bid.value = group
            for thread in range(256):
                tid.value = thread
                function(*(a.ctypes.data for a in arrays))
        assert (out[numel:] == -12345.).all(), "out-of-range store overwrote canary"
        return out[:numel].copy()
    finally:
        engine.close()
