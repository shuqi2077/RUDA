import ast
from pathlib import Path

import torch


root = Path(__file__).resolve().parents[1] / "ruda_torch"
for filename in ("_spatial.py", "_batch_norm.py", "_max_pool.py"):
    tree = ast.parse((root / filename).read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in tree.body:
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
            continue
        mapping = node.iter.func.value if isinstance(node.iter.func, ast.Attribute) else None
        if not isinstance(mapping, ast.Dict):
            continue
        for key, value in zip(mapping.keys, mapping.values):
            name = ast.literal_eval(key)
            operator, _, overload = name.partition(".")
            schema = getattr(getattr(torch.ops.aten, operator), overload or "default")._schema
            implementation = functions[value.id].args
            required = len(implementation.args) - len(implementation.defaults)
            positional = [arg for arg in schema.arguments if not arg.kwarg_only]
            assert required <= len(positional) <= len(implementation.args), (name, schema)
            print(f"{filename}: {schema}")
