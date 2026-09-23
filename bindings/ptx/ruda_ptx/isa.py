"""Native GPU code-object contract. Not a PTX-to-all-vendors translator.

Only pointer arguments are supported. Code objects are executable code: use a
trusted, private cache. SHA256 detects accidental corruption, not malicious code.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import cached_property
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
import threading
from typing import Callable

MAX_IMAGE_BYTES = 64 * 1024 * 1024
SCHEMA = "ruda-native-image-v1"
# LLVM AMDGPU ELF EF_AMDGPU_MACH codes; deliberately a small tested codegen set.
AMD_TARGETS = {"gfx90a": 0x3f, "gfx942": 0x4c, "gfx1100": 0x41}


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _symbol(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value) is not None


def elf_info(code: bytes) -> dict:
    """Validate bounded ELF64 little-endian headers/tables, return defined symbols.

    This is structural validation, NOT full executable validation. The device
    loader must also accept the code object and its ABI before it can be run.
    """
    if not isinstance(code, bytes) or not 64 <= len(code) <= MAX_IMAGE_BYTES:
        raise ValueError("Native code must be an ELF image of 64 bytes to 64 MiB")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", code)
    ident, typ, machine, version, entry, phoff, shoff, flags, ehsize, phsize, phnum, shsize, shnum, shstr = header
    if ident[:7] != b"\x7fELF\x02\x01\x01" or version != 1 or ehsize != 64:
        raise ValueError("Only ELF64 little-endian native objects are supported")
    if typ not in (2, 3):
        raise ValueError("Native image must be linked ET_EXEC/ET_DYN, not relocatable .o")
    if (not 0 < shnum <= 8192 or shsize != 64 or shoff < 64
            or shoff + shsize * shnum > len(code) or shstr >= shnum):
        raise ValueError("Invalid or unsupported ELF section table")
    if phnum and (phsize != 56 or phoff < 64 or phoff + phsize * phnum > len(code)):
        raise ValueError("Invalid ELF program header table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", code, shoff + i * shsize) for i in range(shnum)]
    for section in sections:
        _, stype, _, _, off, size, _, _, _, _ = section
        if stype != 8 and (off > len(code) or size > len(code) - off):
            raise ValueError("ELF section exceeds the image")
    names = set()
    for section in sections:
        _, stype, _, _, off, size, link, _, _, entsize = section
        if stype not in (2, 11):
            continue
        if entsize != 24 or size % 24 or link >= shnum or sections[link][1] != 3:
            raise ValueError("Invalid ELF symbol/string table")
        string_section = sections[link]
        strings = code[string_section[4]:string_section[4] + string_section[5]]
        for pos in range(off, off + size, 24):
            name, _, _, shndx, _, _ = struct.unpack_from("<IBBHQQ", code, pos)
            if name >= len(strings):
                raise ValueError("Invalid ELF symbol name offset")
            end = strings.find(b"\0", name)
            if end == -1:
                raise ValueError("Unterminated ELF symbol name")
            if shndx:
                names.add(strings[name:end].decode("ascii", errors="replace"))
    return {"machine": machine, "type": typ, "flags": flags,
            "osabi": ident[7], "abi_version": ident[8], "symbols": names}


@dataclass(frozen=True)
class NativeImage:
    isa: str
    target: str
    entry: str
    parameters: tuple[str, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    source_digest: str
    compiler_id: str
    code: bytes
    required_bytes: tuple[int, ...] = ()
    pointer_alignment: int = 1
    shared_bytes: int = 0

    def __post_init__(self):
        for field in ("parameters", "grid", "block", "required_bytes"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.isa not in ("nvidia-sass", "amdgcn"):
            raise ValueError("Unsupported ISA: no implicit PTX, CUDA, CPU, or vendor fallback")
        if not _symbol(self.entry) or not self.parameters or not all(map(_symbol, self.parameters)):
            raise ValueError("Invalid kernel entry/parameter ABI")
        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError("Duplicate parameter name")
        for dims, maximum in ((self.grid, (2**31 - 1, 65535, 65535)), (self.block, (1024, 1024, 64))):
            if len(dims) != 3 or any(type(v) is not int or not 0 < v <= m for v, m in zip(dims, maximum)):
                raise ValueError("Invalid native launch dimensions")
        if math.prod(self.block) > 1024:
            raise ValueError("Native block exceeds 1024 threads")
        if not isinstance(self.source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_digest):
            raise ValueError("Missing/invalid source identity")
        if not isinstance(self.compiler_id, str) or not 0 < len(self.compiler_id) <= 4096:
            raise ValueError("Missing/oversized compiler identity")
        if self.required_bytes and (len(self.required_bytes) != len(self.parameters)
                                   or any(type(n) is not int or n <= 0 for n in self.required_bytes)):
            raise ValueError("Invalid pointer buffer size requirements")
        if type(self.pointer_alignment) is not int or self.pointer_alignment not in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError("Invalid pointer alignment")
        if type(self.shared_bytes) is not int or not 0 <= self.shared_bytes < 2**32:
            raise ValueError("Invalid dynamic shared memory size")
        info = elf_info(self.code)
        if self.entry not in info["symbols"]:
            raise ValueError("Native entry is not defined in the ELF image")
        if self.isa == "amdgcn":
            if self.target not in AMD_TARGETS:
                raise ValueError(f"Unimplemented AMD target; supported codegen targets: {tuple(AMD_TARGETS)}")
            if (info["machine"] != 224 or info["type"] != 3 or info["osabi"] != 64
                    or info["flags"] & 0xff != AMD_TARGETS[self.target]):
                raise ValueError("AMD target/ELF mismatch; ET_DYN AMDHSA is required")
        elif (not isinstance(self.target, str) or not re.fullmatch(r"sm_[1-9][0-9]{1,2}", self.target)
              or info["machine"] != 190):
            raise ValueError("NVIDIA target/ELF mismatch; a native cubin is required")

    @cached_property
    def code_digest(self):
        return hashlib.sha256(self.code).hexdigest()

    def metadata(self):
        fields = {key: val for key, val in asdict(self).items() if key != "code"}
        return {"schema": SCHEMA, **fields, "code_sha256": self.code_digest}

    @cached_property
    def digest(self):
        return digest_json(self.metadata())

    def to_bytes(self):
        return json.dumps({**self.metadata(), "code_base64": base64.b64encode(self.code).decode("ascii")},
                          sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, raw: bytes):
        if not isinstance(raw, bytes) or len(raw) > 2 * MAX_IMAGE_BYTES:
            raise ValueError("Oversized native image container")
        try:
            data = json.loads(raw)
            if data.pop("schema") != SCHEMA:
                raise ValueError("Unsupported native image schema")
            expected = data.pop("code_sha256")
            code = base64.b64decode(data.pop("code_base64"), validate=True)
            if hashlib.sha256(code).hexdigest() != expected:
                raise ValueError("Native image checksum mismatch")
            return cls(code=code, **data)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed native image container") from exc

    def write(self, path):
        """Create a new artifact; never overwrite an existing user file."""
        with Path(path).open("xb") as out:
            out.write(self.to_bytes())

    @classmethod
    def read(cls, path):
        with Path(path).open("rb") as inp:
            raw = inp.read(2 * MAX_IMAGE_BYTES + 1)
        return cls.from_bytes(raw)


class NativeCache:
    """Opt-in private disk cache, atomic per-entry writes and process locking.

    A malformed entry is an error, never silently compiled around. This cache
    holds code, not weights or user tensors. No pickle or executable Python.
    """
    def __init__(self, directory):
        self.directory = Path(directory).absolute()
        if self.directory.is_symlink():
            raise ValueError("Native cache may not be a symlink")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.stat()
        if not self.directory.is_dir() or info.st_mode & 0o022 or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
            raise ValueError("Use an owned, non-group/world-writable native cache directory")
        self._lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "compiles": 0}

    def get_or_compile(self, identity: dict, compile_fn: Callable[[], NativeImage]) -> NativeImage:
        import fcntl  # Reference native runtimes and process lock are Linux-only.
        key = digest_json(identity)
        path = self.directory / (key + ".risa")
        lockpath = self.directory / (key + ".lock")
        with self._lock:
            fd = os.open(lockpath, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                if path.is_symlink():
                    raise ValueError("Refusing a symlink cache entry")
                if path.exists():
                    image = NativeImage.read(path)
                    # The complete compiler-specific identity is the filename;
                    # cross-check critical fields so moving entries cannot alias.
                    for k in ("source_digest", "target", "isa", "compiler_id"):
                        if k in identity and getattr(image, k) != identity[k]:
                            raise ValueError(f"Native cache identity mismatch: {k}")
                    self.stats["hits"] += 1
                    return image
                self.stats["misses"] += 1
                image = compile_fn()
                if not isinstance(image, NativeImage):
                    raise TypeError("ISA compiler did not return NativeImage")
                for k in ("source_digest", "target", "isa", "compiler_id"):
                    if k in identity and getattr(image, k) != identity[k]:
                        raise ValueError(f"Compiled artifact identity mismatch: {k}")
                self.stats["compiles"] += 1
                tmp = None
                try:
                    with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".writing-", delete=False) as out:
                        tmp = Path(out.name)
                        out.write(image.to_bytes()); out.flush(); os.fsync(out.fileno())
                    os.replace(tmp, path)
                finally:
                    if tmp is not None:
                        tmp.unlink(missing_ok=True)
                return image
            finally:
                os.close(fd)
