#!/usr/bin/env python3
"""Independent numerical/selection/cache references and source guards, NOT Rust execution."""
from __future__ import annotations
import itertools, math, re, statistics, struct, tomllib, unittest
from decimal import Decimal, localcontext
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
MASK = (1 << 64) - 1

def digest(data: bytes) -> str:
    a, b = 0xcbf29ce484222325, 0x84222325cbf29ce4
    for v in data:
        a = ((a ^ v) * 0x100000001b3) & MASK
        b = ((b ^ (v + 1)) * 0x100000001b3) & MASK
    return f'{a:016x}{b:016x}'

def fields(parts: tuple[str, ...] | list[str]) -> str:
    return ''.join(f'{len(x.encode())}:{x}' for x in parts)

def parse_fields(text: str) -> tuple[str, ...]:
    raw, pos, out = text.encode(), 0, []
    while pos < len(raw):
        end = raw.index(b':', pos); n = int(raw[pos:end]); pos = end + 1
        if pos + n > len(raw): raise ValueError('truncated field')
        out.append(raw[pos:pos+n].decode()); pos += n
    return tuple(out)

def encode(key='known/测试', winner='reference-plan', scope=1, ratio=.8) -> bytes:
    b = bytearray(b'RUDATN01')
    for value in (key, winner):
        value = value.encode(); b += struct.pack('<I',len(value)) + value
    b += struct.pack('<QQQdBB',1700000000,1000000,800000,ratio,1,scope)
    return bytes(b) + digest(b).encode()

def decode(raw: bytes) -> dict:
    if not 83 <= len(raw) <= 512*1024 or raw[:8] != b'RUDATN01': raise ValueError('header')
    data, checksum = raw[:-32], raw[-32:]
    if digest(data).encode() != checksum: raise ValueError('checksum')
    pos, values = 8, []
    for limit in (384*1024,4096):
        if pos + 4 > len(data): raise ValueError('length')
        n, = struct.unpack_from('<I',data,pos); pos += 4
        if n > limit or pos + n > len(data): raise ValueError('field')
        values.append(data[pos:pos+n].decode()); pos += n
    if len(data)-pos != 34: raise ValueError('tail')
    created, reference, winner_time, ratio, checked, scope = struct.unpack_from('<QQQdBB',data,pos)
    if not values[1] or not reference or not winner_time or not math.isfinite(ratio) or ratio <= 0 or checked > 1 or scope > 2:
        raise ValueError('invalid metadata')
    return dict(key=values[0],winner=values[1],created=created,reference=reference,winner_time=winner_time,ratio=ratio,checked=bool(checked),scope=scope)

def score(pairs, samples=7, max_mad=.15):
    if len(pairs) < samples or any(a<=0 or b<=0 for a,b in pairs): return None
    ratios=[b/a for a,b in pairs]
    if not all(math.isfinite(x) and x>0 for x in ratios): return None
    center=statistics.median(ratios)
    mad=statistics.median(abs(x-center) for x in ratios)/center
    return None if mad>max_mad else (center,mad)

def finite_equal(a,b,atol,rtol):
    if not math.isfinite(a) or not math.isfinite(b): return False
    scale=max(abs(a),abs(b),1.)
    return not ((atol==rtol==0 and a!=b) or abs(a/scale-b/scale)>atol/scale+rtol*(abs(a)/scale))

def balanced_rust(source: str) -> None:
    """Delimiter guard only. Does not claim to parse, type-check or compile Rust."""
    stack, i, n = [], 0, len(source)
    while i<n:
        if source.startswith('//',i):
            j=source.find('\n',i); i=n if j<0 else j+1; continue
        if source.startswith('/*',i):
            depth=1; i+=2
            while i<n and depth:
                if source.startswith('/*',i): depth+=1; i+=2
                elif source.startswith('*/',i): depth-=1; i+=2
                else: i+=1
            if depth: raise AssertionError('unclosed block comment')
            continue
        raw=re.match(r'(?:b|c)?r(#{0,32})"',source[i:])
        if raw:
            end='"'+raw[1]; j=source.find(end,i+raw.end())
            if j<0: raise AssertionError('unclosed raw string')
            i=j+len(end); continue
        if source[i]=='"':
            i+=1
            while i<n:
                if source[i]=='\\': i+=2
                elif source[i]=='"': i+=1; break
                else: i+=1
            else: raise AssertionError('unclosed string')
            continue
        if source[i]=="'":
            char=re.match(r"'(?:[^'\\\n]|\\(?:.|u\{[0-9A-Fa-f_]+\}))'",source[i:])
            if char: i+=char.end(); continue
        ch=source[i]
        if ch in '([{': stack.append((ch,i))
        elif ch in ')]}':
            if not stack or '([{'.index(stack.pop()[0])!=')]}'.index(ch): raise AssertionError(f'unbalanced delimiter at {i}')
        i+=1
    if stack: raise AssertionError(f'unclosed delimiter {stack[-1]}')

class References(unittest.TestCase):
    def test_01_length_delimited_keys(self):
        atoms=['','a','bc',':','3:a','中文','δ',';dtype=bf16','\0','x:y']
        seen={}
        for n in range(1,5):
            for parts in itertools.product(atoms,repeat=n):
                key=fields(parts); self.assertEqual(parse_fields(key),parts)
                self.assertNotIn(key,seen); seen[key]=parts
        self.assertEqual(len(seen),11110)
    def test_02_paired_scale_invariance(self):
        count=0
        for ref in range(1,101):
            for candidate in range(1,81):
                pairs=[(ref*x,candidate*x) for x in (1,2,3,4,5,7,11)]
                ratio,mad=score(pairs)
                self.assertAlmostEqual(ratio,candidate/ref); self.assertAlmostEqual(mad,0.)
                count+=1
        self.assertEqual(count,8000)
    def test_03_noise_and_failed_samples(self):
        self.assertIsNone(score([(100,1),(100,100),(100,10000)],3))
        self.assertIsNone(score([(100,50)]*2,3)); self.assertIsNone(score([(0,50)]*7))
        self.assertEqual(score([(100,50)]*6+[(100,1000000)]),(0.5,0.0))
    def test_04_wrong_candidate_cannot_win_reference_model(self):
        for times in itertools.product((10,50,98,100,150),repeat=3):
            for valid in itertools.product((False,True),repeat=2):
                choices=[(1.,0)]+[(times[i]/times[0],i) for i in (1,2) if valid[i-1] and times[i]*1.05<=times[0]]
                _, winner=min(choices)
                self.assertTrue(winner==0 or valid[winner-1])
    def test_05_cache_golden_and_corruption(self):
        raw=encode(); fixture=ROOT/'ruda/src/runtime/tune/stack/testdata/cache_v1.fixture'
        self.assertEqual(fixture.read_bytes(),raw); self.assertEqual(decode(raw)['key'],'known/测试')
        for i in range(len(raw)):
            with self.assertRaises((ValueError,UnicodeDecodeError)): decode(raw[:i])
            b=bytearray(raw); b[i]^=1
            with self.assertRaises((ValueError,UnicodeDecodeError)): decode(bytes(b))
    def test_06_cache_structural_limits(self):
        for raw in (encode(scope=255),encode(ratio=float('nan')),encode(ratio=float('inf')),encode(ratio=0),encode(winner='')):
            with self.assertRaises(ValueError): decode(raw)
        b=bytearray(encode()[:-32]); b[8:12]=struct.pack('<I',384*1024+1)
        with self.assertRaises(ValueError): decode(bytes(b)+digest(b).encode())
        raw=encode(); self.assertNotEqual(decode(raw)['key'],'other-key')
    def test_07_finite_extremes_and_decimal_oracle(self):
        maximum=float.fromhex('0x1.fffffffffffffp+1023'); tiny=float.fromhex('0x0.0000000000001p-1022')
        values=[-maximum,-1e300,-1e-300,-tiny,0.,tiny,1e-300,1.,1e300,maximum]
        with localcontext() as ctx:
            ctx.prec=800
            for a,b,atol,rtol in itertools.product(values,values,(0.,1e-6),(0.,1e-3)):
                oracle=abs(Decimal(a)-Decimal(b))<=Decimal(atol)+Decimal(rtol)*abs(Decimal(a))
                self.assertEqual(finite_equal(a,b,atol,rtol),oracle,(a,b,atol,rtol))
        self.assertFalse(finite_equal(float('nan'),float('nan'),1,1))
        self.assertFalse(finite_equal(float('inf'),float('inf'),1,1))
    def test_08_integration_hooks_are_in_real_dispatch_paths(self):
        for file in ('ruBLAS/src/tensor_matmul/tune/base.rs','ruDNN/src/attention/tensor/tune.rs',
                     'ruDNN/src/convolution/tensor/forward/tune.rs','ruda-fusion/src/device/optim/matmul/tune.rs'):
            self.assertIn('.with_stack_tuning(', (ROOT/file).read_text())
        adapter=(ROOT/'ruda/src/runtime/tune/stack/runtime_adapter.rs').read_text()
        self.assertIn('let batch = plan.next(None)',adapter); self.assertIn('expected.validate_for_tuning(',adapter)
        self.assertIn('super::engine::DepthGuard::enter()',adapter)
        model=(ROOT/'ruLLM/src/autotune.rs').read_text()
        self.assertIn('Scope::Pipeline',model); self.assertIn('generate_greedy_packed_with_mode(',model)
        self.assertIn('tuner.lower_level_fingerprint()',model)
    def test_09_cache_hits_do_not_force_gpu_synchronization(self):
        src=(ROOT/'ruda/src/runtime/tune/stack/runtime_adapter.rs').read_text().split('pub fn try_execute_stack',1)[1]
        src=src.split('fn autotune_error',1)[0]
        self.assertNotIn('complete(client)',src); self.assertNotIn('client.sync(',src); self.assertNotIn('client.profile(',src)
        self.assertIn('.execute(input)',src); self.assertIn('tuner.invalidate(&decision, true)',src)
    def test_10_layout_preservation_and_relative_graph_ids(self):
        src=(ROOT/'ruBLAS/src/tensor_matmul/tune/base.rs').read_text().split('/// Executes autotune',1)[0]
        self.assertNotIn('out.copy()',src); self.assertIn('handle.offset_start = out.handle.offset_start',src)
        self.assertIn('handle.offset_end = out.handle.offset_end',src)
        graph=(ROOT/'ruda-fusion/src/device/tune.rs').read_text().split('pub(crate) fn autotune_context_signature',1)[1].split('/// Read-only access',1)[0]
        self.assertIn('get_handle_ref(&tensor.id)',graph); self.assertIn('relative_id={id:?}',graph)
        self.assertNotIn('tensor={tensor:?}',graph); self.assertIn('shapes_relative2global',graph)
    def test_11_manifests_and_new_features(self):
        files=list(ROOT.rglob('Cargo.toml')); self.assertGreater(len(files),40)
        for p in files: tomllib.loads(p.read_text())
        fusion=tomllib.loads((ROOT/'ruda-fusion/Cargo.toml').read_text())
        self.assertIn('device-tensor',fusion['features']['device-stack-autotune'])
        self.assertNotIn('device-autotune-checks',fusion['features']['device-stack-autotune'])
        llm=tomllib.loads((ROOT/'ruLLM/Cargo.toml').read_text())
        self.assertIn('stack-autotune',llm['features'])
        for example in llm['example']:
            self.assertTrue((ROOT/'ruLLM'/example.get('path',f"examples/{example['name']}.rs")).exists())
    def test_12_legacy_dispatch_and_default_methods(self):
        local=(ROOT/'ruda/src/runtime/tune/local.rs').read_text()
        self.assertIn('operations.stack_reference().is_some()',local); self.assertIn('tuner.check_tune',local)
        operation=(ROOT/'ruda/src/runtime/tune/operation.rs').read_text()
        legacy=operation.split('pub fn compute_checksum',1)[1].split('pub fn stack_checksum',1)[0]
        self.assertIn('checksum += &tune.function.name',legacy)
        self.assertIn('{ Ok(false) }',(ROOT/'ruda/src/runtime/tune/tune_benchmark.rs').read_text())
        backend=(ROOT/'ruda/src/runtime/backend.rs').read_text()
        self.assertRegex(backend,r'fn autotune_driver_fingerprint[\s\S]*?\{\s*None\s*\}')
    def test_13_new_rust_delimiters_and_no_unsafe_core(self):
        paths=list((ROOT/'ruda/src/runtime/tune/stack').glob('*.rs'))+[
            ROOT/'ruLLM/src/autotune.rs',ROOT/'ruda/src/runtime/tune/validation.rs',ROOT/'ruda/tests/runtime/stack_autotune.rs',
            ROOT/'ruLLM/tests/stack_autotune_gpu.rs',ROOT/'ruLLM/examples/support/qwen2_autotune.rs']
        for p in paths:
            with self.subTest(path=str(p.relative_to(ROOT))): balanced_rust(p.read_text())
        for p in (ROOT/'ruda/src/runtime/tune/stack').glob('*.rs'):
            self.assertNotRegex(p.read_text(),r'\bunsafe\s*\{')
    def test_14_persistent_identity_is_conservative(self):
        source=(ROOT/'ruda/src/runtime/tune/stack/runtime_adapter.rs').read_text()
        self.assertIn('driver.is_some() && env!("RUDA_STACK_BUILD_ID") != "unavailable"',source)
        self.assertIn('Some(fields(&[&probed, &extra]))',source)
        for key in ('RUDA_AUTOTUNE_DRIVER_TAG','RUDA_AUTOTUNE_BUILD_TAG','RUDA_AUTOTUNE_CONTEXT_TAG','runtime_options'):
            self.assertIn(key,source)
        engine=(ROOT/'ruda/src/runtime/tune/stack/engine.rs').read_text()
        self.assertIn('if *scope != Scope::Pipeline',engine); self.assertIn('record.is_fresh(now_seconds()',engine)

if __name__=='__main__':
    unittest.main(verbosity=2)
