import ast
import ctypes
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / 'ruDNN/examples/sol_export'


def runtime():
    namespace = {'ctypes': ctypes, '_aligned': lambda tensor: tensor,
                 '_sm_count': lambda _: 148, '_launch': Mock(),
                 '_tmap': Mock(return_value=(None, 0))}
    library = EXPORT / 'sol_lib.py'
    nodes = [node for node in ast.parse(library.read_text(encoding='utf-8')).body
             if (isinstance(node, ast.FunctionDef) and node.name in ('_gemm', '_gemm_variant', '_flash'))
             or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id in ('_BIG', '_KIND_BYTES') for target in node.targets))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(library), 'exec'), namespace)
    return namespace


class Tcgen05TmaHostTests(unittest.TestCase):
    def test_tensor_map_swizzle_encoding(self):
        source = (EXPORT / 'sol_launcher.py').read_text(encoding='utf-8')
        functions = [node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name in ('_tmap', '_check')]
        self.assertEqual(len(functions), 2)
        driver = SimpleNamespace(cuTensorMapEncodeTiled=Mock(return_value=0))
        namespace = {'ctypes': ctypes, '_driver': lambda: driver, '_MAP_TYPES': {'f32': 7}}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<launcher>', 'exec'), namespace)
        tensor = SimpleNamespace(dtype='f32', data_ptr=lambda: 4096)
        for atomic32, expected in ((False, 3), (True, 4)):
            with self.subTest(atomic32=atomic32):
                namespace['_tmap'](tensor, [32, 8], [128], [32, 8], atomic32=atomic32)
                self.assertEqual(driver.cuTensorMapEncodeTiled.call_args.args[-3], expected)

    def test_gemm_uses_atomic32_only_for_tf32_mn_operands(self):
        namespace = runtime()
        tensor = SimpleNamespace(device='cpu', data_ptr=lambda: 4096)
        operand = SimpleNamespace(tensor=tensor, dims=[128, 1, 128], strides=[512, 512], coords=[0] * 7)
        for kind, a_mn, b_mn in itertools.product('hbteg', (0, 1), (0, 1)):
            with self.subTest(kind=kind, a_mn=a_mn, b_mn=b_mn):
                maps = namespace['_tmap']
                maps.reset_mock()
                namespace['_gemm'](f'gemm_{kind}f_256_{a_mn}{b_mn}_00000',
                                   operand, operand, tensor, 128, 128, 128)
                self.assertEqual([call.kwargs['atomic32'] for call in maps.call_args_list],
                                 [kind == 't' and bool(a_mn), kind == 't' and bool(b_mn)])

    def test_attention_uses_atomic32_only_for_tf32_values(self):
        namespace = runtime()
        for kind, bhld in itertools.product('hbt', (0, 1)):
            with self.subTest(kind=kind, bhld=bhld):
                size = 4 if kind == 't' else 2
                query = SimpleNamespace(shape=(2, 4, 7, 128) if bhld else (2, 7, 4, 128),
                                        element_size=lambda: size)
                kv = SimpleNamespace(shape=(2, 2, 9, 128) if bhld else (2, 9, 2, 128))
                maps = namespace['_tmap']
                maps.reset_mock()
                namespace['_flash'](f'flash_{kind}f_128_64_2_{bhld}00001', query, kv, kv, query, lse=query)
                self.assertEqual([call.kwargs['atomic32'] for call in maps.call_args_list],
                                 [False, False, kind == 't'])


if __name__ == '__main__':
    unittest.main()
