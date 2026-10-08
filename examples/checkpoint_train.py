#!/usr/bin/env python3
"""CPU linear regression continued across real worker exits under mock leases."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'integrations'))
from mlus_checkpoint import atomic_json_checkpoint, yield_checkpoint
from mlus_report import write_report

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--chunk-steps',type=int,default=50)
p.add_argument('--steps',type=int,default=200)
p.add_argument('--resume-vram-mib',type=int,default=7000)
p.add_argument('--seconds',type=float,default=0)
a=p.parse_args()
if a.chunk_steps<=0 or a.steps<=0 or a.seconds<0:p.error('positive step counts and nonnegative seconds required')
if os.environ.get('MLUS_BACKEND')!='mock' or os.environ.get('MLUS_COOPERATIVE')!='1':
    p.error('this CPU/synthetic example requires a --mock daemon and --cooperative')
checkpoint=a.checkpoint.absolute()
resume=os.environ.get('MLUS_CHECKPOINT_PATH')
if resume:
    if Path(resume)!=checkpoint:raise RuntimeError('resume path does not match application checkpoint')
    state=json.loads(Path(resume).read_text())
else:
    if checkpoint.exists():raise RuntimeError('use a fresh checkpoint path; existing application state is preserved')
    state=dict(step=0,weight=0.0,bias=0.0)
start=state['step']
xs=[-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]
for _ in range(min(a.chunk_steps,a.steps-start)):
    errors=[state['weight']*x+state['bias']-(2*x+1) for x in xs]
    state['weight']-=.1*sum(e*x for e,x in zip(errors,xs))/len(xs)
    state['bias']-=.1*sum(errors)/len(xs)
    state['step']+=1
loss=sum((state['weight']*x+state['bias']-(2*x+1))**2 for x in xs)/len(xs)
print(json.dumps(dict(kind='cpu_checkpoint_training',attempt=int(os.environ['MLUS_ATTEMPT']),start_step=start,step=state['step'],loss=loss,gpu_measurement=False)),flush=True)
time.sleep(a.seconds)
final=state['step']==a.steps
if final:assert loss<1e-12
# Explicit synthetic memory samples for scheduler/profile checks, not CPU stats.
if 'MLUS_REPORT_PATH' in os.environ:
    write_report(allocated_mib=512,reserved_mib=1024,outcome='success' if final else 'checkpointed',measurement='simulation')
atomic_json_checkpoint(checkpoint,state)
if not final:yield_checkpoint(checkpoint,step=state['step'],resume_vram_mib=a.resume_vram_mib)
