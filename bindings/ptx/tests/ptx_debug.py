"""TEST ONLY: executable interpreter for this package's scalar PTX subset.

Not a PTX assembler, not an NVIDIA emulator, not a production runtime/fallback.
Executes emitted text to catch indexing, predication, uninitialized reads and
reduction errors. ex2/sqrt use NumPy rather than hardware approximations; warp
scheduling is deterministic, not a complete data-race or memory-model proof.
"""
from __future__ import annotations
import itertools
import re
import struct
import numpy as np
from ruda_ptx import Buffer


class DebugRuntime:
    name = "test_ptx_subset_interpreter_NOT_GPU"

    def __init__(self):
        self.regions, self.buffers, self.modules, self.calls = {}, {}, {}, []
        self.next = 0x10000

    def allocate(self, nbytes):
        p = self.next
        self.next += ((nbytes + 255)//256 + 1)*256
        b = Buffer(p, nbytes, self)
        self.buffers[p] = b
        self.regions[p] = (bytearray(nbytes), np.zeros(nbytes, dtype=bool))
        self.calls.append(("allocate", nbytes))
        return b

    def validate_buffer(self, b):
        if not isinstance(b, Buffer) or b.owner is not self or self.buffers.get(b.handle) is not b:
            raise ValueError("foreign/freed buffer")

    def free(self, b):
        self.validate_buffer(b)
        del self.regions[b.handle], self.buffers[b.handle]
        self.calls.append(("free", b.nbytes))

    def write(self, b, data):
        self.validate_buffer(b)
        if len(data) > b.nbytes:
            raise ValueError("oversized upload")
        raw, init = self.regions[b.handle]
        raw[:len(data)] = data
        init[:len(data)] = True
        self.calls.append(("write", len(data)))

    def read(self, b, n):
        self.validate_buffer(b)
        raw, init = self.regions[b.handle]
        assert n <= b.nbytes and init[:n].all(), "uninitialized/out-of-bounds readback"
        self.calls.append(("read", n))
        return bytes(raw[:n])

    def load(self, kernel):
        self.modules[kernel.digest] = _parse(kernel.ptx)
        self.calls.append(("load", kernel.operation))
        return kernel.digest

    def launch(self, loaded, kernel, buffers):
        assert loaded == kernel.digest
        for b in buffers:
            self.validate_buffer(b)
        assert len(buffers) == len(kernel.parameters)
        program = self.modules[loaded]
        assert np.prod(kernel.grid) <= 256, "test interpreter grid too large"
        params = dict(zip(kernel.parameters, (b.handle for b in buffers)))
        for z, y, x in itertools.product(range(kernel.grid[2]), range(kernel.grid[1]), range(kernel.grid[0])):
            _block(program, kernel, params, self.regions, (x, y, z))
        self.calls.append(("launch", kernel.operation))

    def launch_many(self, calls):
        for loaded, kernel, buffers in calls:
            self.launch(loaded, kernel, buffers)

    def synchronize(self):
        self.calls.append(("synchronize",))


def _parse(text):
    text = re.sub(r"//[^\n]*", "", text)
    body = text.split("{", 1)[1].rsplit("}", 1)[0]
    shared = re.findall(r"\.shared\s+\.align\s+\d+\s+\.b8\s+(\w+)\[(\d+)\];", body)
    body = re.sub(r"(?m)^\s*\.(?:reg|shared)\b[^;]*;", "", body)
    labels, ops = {}, []
    for part in body.split(";"):
        part = part.strip()
        if not part:
            continue
        while re.match(r"^\w+:", part):
            label, part = part.split(":", 1)
            labels[label.strip()] = len(ops)
            part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(?:@(!?%\w+)\s+)?([\w.]+)(?:\s+(.*))?", part, re.S)
        assert match, part
        predicate, opcode, args = match.groups()
        args = tuple(a.strip() for a in re.findall(r"\{[^}]*\}|\[[^]]*\]|[^,\s{}][^,{}]*", args or "") if a.strip())
        ops.append((predicate, opcode, args))
    return ops, labels, shared


def _block(program, kernel, params, global_regions, cta):
    ops, labels, shared_defs = program
    n = int(np.prod(kernel.block))
    tids = np.arange(n, dtype=np.int64)
    pc = np.zeros(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    regs, initialized = {}, {}
    shared_regions, symbols = {}, {}
    base = 1 << 48
    for symbol, count in shared_defs:
        count = int(count)
        symbols[symbol] = base
        shared_regions[base] = (bytearray(count), np.zeros(count, dtype=bool))
        base += ((count + 255)//256+1)*256
    specials = {"%tid.x": tids % kernel.block[0],
                "%tid.y": (tids // kernel.block[0]) % kernel.block[1],
                "%tid.z": tids // (kernel.block[0]*kernel.block[1]),
                "%ntid.x": kernel.block[0], "%ntid.y": kernel.block[1], "%ntid.z": kernel.block[2],
                "%ctaid.x": cta[0], "%ctaid.y": cta[1], "%ctaid.z": cta[2]}

    def val(token, lanes):
        token = token.strip()
        if token in specials:
            x = specials[token]
            return x[lanes] if isinstance(x, np.ndarray) else np.full(len(lanes), x, dtype=np.int64)
        if token in symbols:
            return np.full(len(lanes), symbols[token], dtype=np.int64)
        if token.startswith("%"):
            assert token in regs and initialized[token][lanes].all(), f"uninitialized register {token} at {ops[current]}"
            return regs[token][lanes]
        if token.startswith("0f"):
            return np.full(len(lanes), struct.unpack(">f", bytes.fromhex(token[2:]))[0], dtype=np.float32)
        return np.full(len(lanes), int(token, 0), dtype=np.int64)

    def put(token, lanes, value):
        dtype = np.float32 if token.startswith("%f") else (bool if token.startswith("%p") else np.int64)
        if token not in regs:
            regs[token] = np.zeros(n, dtype=dtype)
            initialized[token] = np.zeros(n, dtype=bool)
        arr = np.asarray(value)
        if token.startswith("%h"):
            arr = arr.astype(np.int64) & 0xffff
        elif token.startswith("%r") and not token.startswith("%rd"):
            arr = arr.astype(np.int64) & 0xffffffff
        regs[token][lanes] = arr
        initialized[token][lanes] = True

    def cast_signed(x):
        return x.astype(np.uint32).view(np.int32).astype(np.int64)

    def memory(address, lanes, dtype, space, write=None, offset=0):
        addresses = val(address[1:-1], lanes) + offset
        fmt = {"f32": "<f", "u32": "<I", "b32": "<I", "s32": "<i", "b16": "<H", "u16": "<H"}[dtype]
        width = struct.calcsize(fmt)
        regions = global_regions if space == "global" else shared_regions
        output = []
        for i, addr in enumerate(addresses.tolist()):
            found = False
            for start, (raw, init) in regions.items():
                off = addr - start
                if 0 <= off and off + width <= len(raw):
                    assert off % width == 0, "misaligned memory access"
                    if write is None:
                        assert init[off:off+width].all(), f"uninitialized {space} load at {ops[current]}"
                        output.append(struct.unpack_from(fmt, raw, off)[0])
                    else:
                        v = write[i].item() if isinstance(write[i], np.generic) else write[i]
                        if fmt in ("<I", "<H"):
                            v = int(v) & ((1 << (width * 8)) - 1)
                        struct.pack_into(fmt, raw, off, v)
                        init[off:off+width] = True
                    found = True
                    break
            assert found, f"out-of-bounds {space} access {addr} at {ops[current]}"
        return np.asarray(output, dtype=np.float32 if dtype == "f32" else np.int64)

    steps = 0
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        while alive.any():
            steps += 1
            assert steps < 200000, "debug instruction budget exhausted"
            current = int(pc[alive].min())
            assert current < len(ops), "fell off PTX entry"
            lanes = np.where(alive & (pc == current))[0]
            pred, op, args = ops[current]
            pc[lanes] += 1
            if pred:
                positive = not pred.startswith("!")
                token = pred if positive else pred[1:]
                keep = val(token, lanes).astype(bool)
                lanes = lanes[keep if positive else ~keep]
                if not len(lanes):
                    continue
            a = lambda i: val(args[i], lanes)
            parts = op.split(".")
            baseop, dtype = parts[0], parts[-1]
            if baseop == "bra":
                pc[lanes] = labels[args[0]]
            elif baseop == "ret":
                alive[lanes] = False
            elif baseop == "bar":
                assert len(lanes) == n, "divergent CTA barrier in debug schedule"
            elif baseop == "ld":
                if parts[1] == "param":
                    put(args[0], lanes, np.full(len(lanes), params[args[1][1:-1]], dtype=np.int64))
                elif "v4" in parts:
                    assert (val(args[1][1:-1], lanes) % 16 == 0).all(), "misaligned vector load"
                    targets = [r.strip() for r in args[0][1:-1].split(",")]
                    assert len(targets) == 4 and dtype == "f32"
                    for j, reg in enumerate(targets):
                        put(reg, lanes, memory(args[1], lanes, dtype, parts[1], offset=j*4))
                else:
                    put(args[0], lanes, memory(args[1], lanes, dtype, parts[1]))
            elif baseop == "st":
                if "v4" in parts:
                    assert (val(args[0][1:-1], lanes) % 16 == 0).all(), "misaligned vector store"
                    sources = [r.strip() for r in args[1][1:-1].split(",")]
                    assert len(sources) == 4 and dtype == "f32"
                    for j, reg in enumerate(sources):
                        memory(args[0], lanes, dtype, parts[1], val(reg, lanes), offset=j*4)
                else:
                    memory(args[0], lanes, dtype, parts[1], a(1))
            elif baseop == "mov":
                value = a(1)
                if dtype == "b32":
                    if args[0].startswith("%f") and not args[1].startswith("%f"):
                        value = value.astype(np.uint32).view(np.float32)
                    elif not args[0].startswith("%f") and args[1].startswith("%f"):
                        value = value.astype(np.float32).view(np.uint32).astype(np.int64)
                put(args[0], lanes, value)
            elif baseop == "shfl":
                assert parts[2] == "bfly"
                offset = int(args[2], 0)
                source = (lanes // 32) * 32 + ((lanes % 32) ^ offset)
                assert np.isin(source, lanes).all(), "partial-warp shuffle"
                put(args[0], lanes, val(args[1], source))
            elif baseop == "cvt":
                dst, src = parts[-2:]
                value = a(1)
                if (dst, src) == ("f32", "f16"):
                    result = value.astype(np.uint16).view(np.float16).astype(np.float32)
                elif (dst, src) == ("f16", "f32"):
                    result = value.astype(np.float16).view(np.uint16).astype(np.int64)
                elif (dst, src) == ("bf16", "f32"):
                    bits = value.astype(np.float32).view(np.uint32)
                    result = ((bits.astype(np.uint64) + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.int64)
                elif dst == "f32":
                    result = value.astype(np.float32)
                else:
                    result = value.astype(np.int64)
                put(args[0], lanes, result)
            elif baseop == "setp":
                x, y = a(1), a(2)
                if dtype == "s32":
                    x, y = cast_signed(x), cast_signed(y)
                function = {"eq": np.equal, "ne": np.not_equal, "lt": np.less, "le": np.less_equal,
                            "gt": np.greater, "ge": np.greater_equal}[parts[1]]
                put(args[0], lanes, function(x, y))
            elif baseop == "selp":
                put(args[0], lanes, np.where(a(3), a(1), a(2)))
            elif baseop in {"add", "sub", "mul", "div", "rem", "max", "min", "and", "or", "xor", "shl", "shr"}:
                x, y = a(1), a(2)
                if dtype == "s32":
                    x, y = cast_signed(x), cast_signed(y)
                functions = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                             "div": np.divide if dtype.startswith("f") else np.floor_divide,
                             "rem": np.remainder, "max": np.fmax, "min": np.fmin,
                             "and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor,
                             "shl": np.left_shift, "shr": np.right_shift}
                put(args[0], lanes, functions[baseop](x, y))
            elif baseop in {"mad", "fma"}:
                if dtype == "f32":
                    value = a(1).astype(np.float64)*a(2).astype(np.float64) + a(3).astype(np.float64)
                else:
                    value = a(1)*a(2)+a(3)
                put(args[0], lanes, value)
            elif baseop in {"neg", "sqrt", "ex2"}:
                put(args[0], lanes, {"neg": np.negative, "sqrt": np.sqrt, "ex2": np.exp2}[baseop](a(1)))
            else:
                raise NotImplementedError(f"Debug interpreter does not implement {op}")
