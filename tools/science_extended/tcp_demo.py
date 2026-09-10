#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the compiled Rust/ruCCL distributed-CG example as real OS processes.

Loopback CPU/TCP correctness demonstration, not GPU/RDMA performance. Server
uses a demonstration fixed identity, binds loopback only. No remote code download.
"""
from __future__ import annotations
import argparse,json,math,os,queue,signal,subprocess,threading,time
from pathlib import Path


def terminate(p):
    if p.poll()is not None:return
    if os.name=='nt':p.terminate()
    else:os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name=='nt':p.kill()
        else:os.killpg(p.pid,signal.SIGKILL)
        p.wait(timeout=2)


def exercise(binary,world,order,output,timeout):
    output.mkdir(parents=True,exist_ok=True);processes=[];streams=[];report={'status':'running','execution':'actual compiled Rust example','world':world,'order':order,'transport':'host-memory TCP loopback','performance_measured':False}
    start=time.monotonic()
    def remaining():
        left=timeout-(time.monotonic()-start)
        if left<=0:raise TimeoutError('TCP demo deadline')
        return left
    try:
        server=subprocess.Popen([str(binary),'server','127.0.0.1:0',str(world)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=(os.name!='nt'))
        processes.append(server);lines=queue.Queue()
        def reader():
            with(output/'server.log').open('w')as log:
                for line in server.stdout:log.write(line);log.flush();lines.put(line)
                lines.put(None)
        t=threading.Thread(target=reader,daemon=True);t.start();address=None
        while address is None:
            line=lines.get(timeout=remaining())
            if line is None:raise RuntimeError('server stopped before rendezvous')
            if line.startswith('LISTEN '):address=line.split()[1]
        workers=[]
        for rank in range(world):
            f=(output/f'rank-{rank}.log').open('w');streams.append(f)
            proc=subprocess.Popen([str(binary),'rank',address,str(world),str(rank),str(order)],stdout=f,stderr=subprocess.STDOUT,text=True,start_new_session=(os.name!='nt'));processes.append(proc);workers.append(proc)
        codes=[p.wait(timeout=remaining())for p in workers]
        for f in streams:f.flush()
        logs=[(output/f'rank-{r}.log').read_text()for r in range(world)]
        if any(c!=0 for c in codes)or any('PASS 'not in s for s in logs):raise RuntimeError(f'worker failures/status missing: {codes}')
        report.update(status='passed',rank_returncodes=codes,rank_logs=logs)
    except (OSError,RuntimeError,TimeoutError,subprocess.TimeoutExpired,queue.Empty)as error:report.update(status='failed',error=repr(error))
    finally:
        for proc in reversed(processes):terminate(proc)
        for f in streams:f.close()
        report['seconds']=time.monotonic()-start;(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--world',type=int,default=2);p.add_argument('--order',type=int,default=32);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=120);a=p.parse_args()
    if not a.binary.is_file():p.error('compiled binary missing; build the actual Rust example first')
    if not 1<=a.world<=16 or not 1<=a.order<=100000 or not math.isfinite(a.timeout)or a.timeout<=0:p.error('invalid demo size/deadline')
    if a.output.exists()and any(a.output.iterdir()):p.error('use a new output directory')
    r=exercise(a.binary.resolve(),a.world,a.order,a.output.resolve(),a.timeout);print(json.dumps(r,indent=2));return 0 if r['status']=='passed'else 1
if __name__=='__main__':raise SystemExit(main())
