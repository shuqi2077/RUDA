"""Isolated LLVM worker: native compiler failures must not abort the caller."""
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("gfx90a", "gfx942", "gfx1100"):
        raise SystemExit("Expected a supported AMD target and build directory")
    from llvmlite import binding as llvm
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    directory = Path(sys.argv[2])
    module = llvm.parse_assembly((directory/"kernel.ll").read_text())
    module.verify()
    machine = llvm.Target.from_triple("amdgcn-amd-amdhsa").create_target_machine(
        cpu=sys.argv[1], opt=3, reloc="pic", codemodel="small")
    module.data_layout = str(machine.target_data)
    tuning = llvm.create_pipeline_tuning_options(speed_level=3, size_level=0)
    passes = llvm.create_pass_builder(machine, tuning)
    manager = passes.getModulePassManager()
    manager.run(module, passes)
    module.verify()
    # LLVM's target-codegen pass mutates IR. Running it twice on the same module
    # can attempt to lower AMD control-flow intrinsics twice and abort LLVM.
    optimized = str(module)
    asm_module = llvm.parse_assembly(optimized)
    obj_module = llvm.parse_assembly(optimized)
    (directory/"kernel.s").write_text(machine.emit_assembly(asm_module))
    (directory/"kernel.o").write_bytes(machine.emit_object(obj_module))


if __name__ == "__main__":
    main()
