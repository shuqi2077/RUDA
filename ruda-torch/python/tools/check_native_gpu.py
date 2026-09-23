"""Fail-closed direct-PTX smoke check. No CPU fallback, no skipped success."""
import argparse
import json
import os
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ptx-version',required=True,help='PTX major.minor supported by your driver')
    parser.add_argument('--matrix-path',choices=['auto','rublas','scalar'],default='auto')
    args=parser.parse_args()
    os.environ['RUDA_CUDA_COMPILER']='ptx'
    os.environ['RUDA_PTX_VERSION']=args.ptx_version
    os.environ['RUDA_TORCH_MATMUL']=args.matrix_path
    try:
        import torch
        import ruda_torch
        a=torch.eye(32).to('ruda')
        b=torch.ones(32,32).to('ruda')
        before=ruda_torch.execution_stats()
        result=torch.mm(a,b)
        after=ruda_torch.execution_stats()
        ruda_torch.synchronize()
        torch.testing.assert_close(result.cpu(),torch.ones(32,32))
        for field in ['host_to_device_bytes','device_to_host_bytes']:
            assert before[field]==after[field], f'unexpected transfer: {field}'
        print(json.dumps({'status':'gpu_smoke_passed','torch':torch.__version__,
                          'compiler':os.environ['RUDA_CUDA_COMPILER'],
                          'ptx_version':args.ptx_version,'matrix_path':args.matrix_path,
                          'dispatch_delta':{k:after[k]-before[k] for k in before}},indent=2))
        return 0
    except Exception as error:
        print(json.dumps({'status':'gpu_smoke_failed','error_type':type(error).__name__,
                          'error':str(error),'cpu_fallback':False},indent=2),file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
