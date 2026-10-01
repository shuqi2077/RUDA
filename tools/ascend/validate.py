#!/usr/bin/env python3
"""Strict real-Ascend acceptance. Missing prerequisites and incomplete runs fail."""
from __future__ import annotations
import argparse,ctypes,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest_tree(root:Path):
    h=hashlib.sha256()
    for base in [root/'ruda-driver-cann',root/'ruda-ascend-kernels',root/'tools/ascend']:
        for p in sorted(base.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
                h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()
def preflight(toolkit:Path|None):
    missing=[]
    for t in ['cargo','rustc','g++']:
        if not shutil.which(t):missing.append(t)
    if toolkit is None:missing.append('ASCEND_HOME_PATH/--toolkit')
    else:
        for rel in ['bin/bisheng','bin/ld.lld','aarch64-linux/asc/include/adv_api']:
            if not (toolkit/rel).exists():missing.append(str(toolkit/rel))
    try:ctypes.CDLL(os.environ.get('RUDA_CANN_LIBRARY','libascendcl.so'))
    except OSError as e:missing.append('AscendCL: '+str(e))
    return missing

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--toolkit',type=Path);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_TOOLKIT_HOME')
    toolkit=a.toolkit or (Path(env) if env else None)
    report={'source_sha256':digest_tree(ROOT),'status':'not_run','npu_passed':False,'stages':[]}
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    report['missing']=preflight(toolkit);save()
    if report['missing']:
        report['status']='preflight_failed';save();print('\n'.join(report['missing']),file=sys.stderr);return 1
    if a.preflight_only:
        report['status']='preflight_only_not_validated';save();return 3
    def run(label,cmd,needle=None):
        started=time.monotonic()
        try:
            r=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=3600)
            code=r.returncode;text=r.stdout
        except subprocess.TimeoutExpired as e:code=124;text=str(e)
        (out/(label+'.log')).write_text(text)
        accepted=code==0 and (needle is None or needle in text)
        report['stages'].append({'stage':label,'command':cmd,'exit_code':code,'seconds':time.monotonic()-started,'accepted':accepted});save()
        if not accepted:raise RuntimeError(label+' failed; see result log')
    try:
        run('sdk_abi',['g++','-std=c++20','-fsyntax-only','-I',str(toolkit/'include'),'-I',str(toolkit/'aarch64-linux/include'),str(ROOT/'tools/ascend/check_acl_abi.cpp')])
        run('rust_device_program_tests',['cargo','test','--locked','-p','ruda-ascend-kernels'])
        run('rust_tests',['cargo','test','--locked','-p','ruda-driver-cann','--lib'])
        run('rublas_check',['cargo','check','--locked','-p','rublas','--no-default-features','--features','cann'])
        run('rudnn_check',['cargo','check','--locked','-p','ruDNN','--no-default-features','--features','cann'])
        run('rutensor_check',['cargo','check','--locked','-p','ruTENSOR','--no-default-features','--features','cann'])
        run('build_kernels',[sys.executable,str(ROOT/'tools/ascend/build_deepgemm.py'),'--toolkit',str(toolkit),'--out',str(out/'kernels')])
        run('npu_numeric',['cargo','run','--locked','-p','ruda-driver-cann','--release','--example','deepgemm_validate','--',str(out/'kernels')],'RUDA_ASCEND_REAL_DEVICE_OK cases=20 launches=36 modules=18')
        report['status']='passed';report['npu_passed']=True;save();return 0
    except (RuntimeError,OSError) as e:report['status']='failed';report['error']=str(e);save();return 1
if __name__=='__main__':raise SystemExit(main())
