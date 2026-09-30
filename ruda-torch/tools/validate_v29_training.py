#!/usr/bin/env python3
import os, shutil, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
missing=[x for x in ('cargo','rustc') if shutil.which(x) is None]
if missing:
    print('REFUSED: missing '+','.join(missing), file=sys.stderr); raise SystemExit(2)
if os.environ.get('RUDA_REQUIRE_GPU')!='1':
    print('REFUSED: set RUDA_REQUIRE_GPU=1 for strict hardware validation', file=sys.stderr); raise SystemExit(2)
cmd=['python','-m','pytest','-q',str(root/'python/tests/test_v29_gpu.py')]
raise SystemExit(subprocess.call(cmd,cwd=root))
