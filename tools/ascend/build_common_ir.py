#!/usr/bin/env python3
"""Build the actual shared Rust IR compiler -> CCE -> linked Ascend artifact.

No replacement Python emitter and no external algorithm template. --emit-only
still compiles/runs the real Rust compiler and never publishes kernel.ruda.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ROW_OPS = ("row_sum", "row_mean", "row_max", "softmax", "log_softmax", "rms_norm",
           "softmax_backward", "log_softmax_backward", "rms_norm_input_backward")
OPS = ("copy", "add", "mul", "silu", "silu_mul", "silu_backward", "silu_mul_backward")


def run(cmd: list[str], log: Path, timeout: int = 1800) -> str:
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd!r}; see {log}")
    return p.stdout


def parse_contract(text: str) -> dict[str, str]:
    if len(text) > 131072:
        raise ValueError("oversized build contract")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            raise ValueError("malformed build contract")
        k, v = line.split("=", 1)
        if not k or not v or k in result:
            raise ValueError("empty/duplicate build field")
        result[k] = v
    row_mode = result.get("schema") == "ruda.ascend.common-row.v1"
    schema = "ruda.ascend.common-row.v1" if row_mode else "ruda.ascend.common-map.v1"
    for k, v in {"schema": schema, "source_language": "ruda-kernel-ir",
                 "lowering": "ascendc-vector", "soc": "Ascend950DT", "arch": "dav-c310"}.items():
        if result.get(k) != v:
            raise ValueError(f"unsupported common IR contract {k}")
    count = int(result.get("bindings", "-1"))
    fixed = {"schema", "source_language", "lowering", "soc", "arch", "kernel_name", "elements",
             "block_dim", "tile_elements", "ub_bytes", "bindings"}
    if row_mode:
        fixed |= {"row_width", "logical_plane"}
    if not 1 <= count <= 8 or set(result) != fixed | {f"binding_{i}" for i in range(count)}:
        raise ValueError("unknown/incomplete contract fields")
    n = int(result["elements"])
    if not 0 <= n <= 2**32 - 1 or not 1 <= int(result["block_dim"]) <= 32:
        raise ValueError("invalid compiled launch domain")
    tile = int(result["tile_elements"])
    if tile < 8 or tile > 4096 or tile % 8 or not 0 < int(result["ub_bytes"]) <= 131072:
        raise ValueError("invalid local memory contract")
    sizes = {n * 4}
    if row_mode:
        width = int(result["row_width"])
        if not 32 <= width <= 4096 or width % 32 or n % width or result["logical_plane"] != "32" or tile != width:
            raise ValueError("invalid row/plane contract")
        sizes |= {width * 4, (n // width) * 4}
    name = result["kernel_name"]
    if not name.isascii() or not name.isidentifier() or not name[0].isalpha() or len(name) > 128:
        raise ValueError("invalid entrypoint")
    ids = set()
    reads = writes = 0
    for i in range(count):
        idx, visibility, size = result[f"binding_{i}"].split(",")
        number = int(idx)
        if not 0 <= number < 2**32 or number in ids or visibility not in ("r", "w") or int(size) not in sizes:
            raise ValueError("invalid binding")
        ids.add(number)
        reads += visibility == "r"
        writes += visibility == "w"
    if reads > 4 or not 1 <= writes <= 4:
        raise ValueError("invalid queue count")
    return result


def validate_args(op: str, elements: int, tile: int, cores: int,
                  row_width: int | None = None, epsilon: float = 1e-5) -> None:
    allowed = OPS if row_width is None else ROW_OPS
    if op not in allowed or not 0 <= elements < 2**32 or tile < 8 or tile > 4096 or tile % 8 or not 1 <= cores <= 32:
        raise ValueError("invalid op/elements/tile/cores")
    if row_width is not None:
        if not 32 <= row_width <= 4096 or row_width % 32 or elements % row_width:
            raise ValueError("row width must be 32..4096, divisible by 32; integral row count required")
        if not math.isfinite(epsilon) or not 0 < epsilon <= 3.4028234663852886e38 or epsilon < 1.401298464324817e-45:
            raise ValueError("epsilon must be a positive representable FP32 value")


def emit(out: Path, op: str, elements: int, tile: int = 256, cores: int = 32,
         row_width: int | None = None, epsilon: float = 1e-5) -> None:
    validate_args(op, elements, tile, cores, row_width, epsilon)
    if out.exists():
        raise ValueError("refusing to overwrite output")
    cargo = shutil.which("cargo")
    if not cargo or not shutil.which("rustc"):
        raise ValueError("missing cargo/rustc: generation must execute production Rust")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".common-ir-", dir=out.parent) as td:
        work = Path(td)
        cmd = [cargo, "run", "--locked", "--release", "-p", "ruda-compiler", "--no-default-features",
               "--features", "ascend", "--example", "ascend_ir", "--", "--op", op,
               "--elements", str(elements), "--tile", str(tile), "--cores", str(cores),
               "--out", str(work / "generated")]
        if row_width is not None:
            cmd += ["--row-width", str(row_width), "--epsilon", repr(epsilon)]
        log = run(cmd, work / "rust-emit.log")
        if "RUDA_ASCEND_IR_EMITTED" not in log:
            raise RuntimeError("production Rust emitter did not confirm output")
        generated = work / "generated"
        contract = parse_contract((generated / "kernel.contract").read_text())
        if contract["elements"] != str(elements) or contract["kernel_name"] != f"ruda_cann_{op}":
            raise ValueError("emitted kernel does not match request")
        if row_width is not None and contract.get("row_width") != str(row_width):
            raise ValueError("emitted row layout differs from request")
        text = (generated / "kernel.asc").read_text()
        if "Generated by RUDA AscendCompiler" not in text or "deep_gemm" in text or "aclnn" in text:
            raise ValueError("unexpected compiler output")
        shutil.copy2(work / "rust-emit.log", generated / "rust-emit.log")
        (generated / "commands.json").write_text(json.dumps([cmd], indent=2) + "\n")
        generated.rename(out)


def includes(toolkit: Path) -> list[Path]:
    # SDK layouts vary across x86/aarch64 installations. Require actual headers;
    # do not silently select a downloaded or bundled fake SDK.
    roots = [toolkit / arch / "asc/include" for arch in ("aarch64-linux", "x86_64-linux")]
    roots += [toolkit / "include", toolkit / "compiler/tikcpp/tikcfw"]
    found = [p for p in roots if (p / "kernel_operator.h").is_file()]
    if not found:
        raise ValueError("installed kernel_operator.h not found in CANN SDK")
    base = found[0]
    return [p for p in (base, base / "interface", base / "impl", base / "adv_api") if p.is_dir()]


def build(out: Path, toolkit: Path, op: str, elements: int, tile: int = 256, cores: int = 32,
          row_width: int | None = None, epsilon: float = 1e-5) -> None:
    validate_args(op, elements, tile, cores, row_width, epsilon)
    if out.exists():
        raise ValueError("refusing to overwrite existing artifact")
    cc, ld = toolkit / "bin/bisheng", toolkit / "bin/ld.lld"
    for exe in (cc, ld):
        if not exe.is_file() or not os.access(exe, os.X_OK):
            raise ValueError(f"missing CANN compiler/linker: {exe}")
    inc = includes(toolkit)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".common-build-", dir=out.parent) as td:
        work = Path(td)
        generated = work / "generated"
        emit(generated, op, elements, tile, cores, row_width, epsilon)
        src, rel, obj = generated / "kernel.asc", generated / "kernel.rel.o", generated / "kernel.o"
        contract = parse_contract((generated / "kernel.contract").read_text())
        compile_cmd = [str(cc), "-x", "cce", "-std=c++20", "-O2", "--cce-aicore-only",
                       "--cce-aicore-arch=dav-c310", "-ffp-contract=off"]
        for path in inc:
            compile_cmd += ["-I", str(path)]
        compile_cmd += ["-c", str(src), "-o", str(rel)]
        link_cmd = [str(ld), "-m", "aicorelinux", "-Ttext", "0", "--no-mmap-output-file",
                    str(rel), "-o", str(obj)]
        run(compile_cmd, generated / "compile.log")
        run(link_cmd, generated / "link.log")
        # Reuse only the ELF parser, never v38's independent matrix generator.
        from build_deepgemm import kernel_name
        name = kernel_name(obj.read_bytes())
        if name != contract["kernel_name"]:
            raise ValueError("linked object metadata entry does not match common IR")
        manifest = dict(contract, object="kernel.o", source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                        object_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(),
                        compiler_sha256=hashlib.sha256(cc.read_bytes() + ld.read_bytes()).hexdigest())
        (generated / "kernel.ruda").write_text("".join(f"{k}={v}\n" for k, v in manifest.items()))
        commands = json.loads((generated / "commands.json").read_text()) + [compile_cmd, link_cmd]
        (generated / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        generated.rename(out)  # Publish loadable metadata only after successful compile/link.


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=OPS + ROW_OPS, default="silu_mul")
    p.add_argument("--elements", type=int, default=1025)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--cores", type=int, default=32)
    p.add_argument("--row-width", type=int, help="enable 32-lane common row lowering; elements=rows*width")
    p.add_argument("--epsilon", type=float, default=1e-5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--toolkit", type=Path)
    p.add_argument("--emit-only", action="store_true")
    a = p.parse_args()
    try:
        if a.emit_only:
            emit(a.out.absolute(), a.op, a.elements, a.tile, a.cores, a.row_width, a.epsilon)
        else:
            toolkit = a.toolkit or (Path(os.environ["ASCEND_HOME_PATH"]) if "ASCEND_HOME_PATH" in os.environ else None)
            if toolkit is None:
                raise ValueError("set ASCEND_HOME_PATH or --toolkit")
            build(a.out.absolute(), toolkit.resolve(), a.op, a.elements, a.tile, a.cores, a.row_width, a.epsilon)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        p.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
