#!/usr/bin/env python3
"""Optional real CUDA checkpoint example. Unverified without a NVIDIA GPU."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'integrations'))
from mlus_checkpoint import yield_checkpoint
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--steps',type=int,default=50)
p.add_argument('--chunk-steps',type=int,default=10)
p.add_argument('--resume-vram-mib',type=int,default=1024)
a=p.parse_args()
if a.steps<=0 or a.chunk_steps<=0:p.error('positive step counts required')
if os.environ.get('MLUS_COOPERATIVE')!='1' or os.environ.get('MLUS_BACKEND')!='nvidia':
    p.error('requires a real NVIDIA daemon and --cooperative')
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise SystemExit('one CUDA GPU is required; no CPU fallback')
checkpoint=a.checkpoint.absolute()
torch.manual_seed(123)
model=torch.nn.Linear(32,16).cuda()
optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
step=0
resume=os.environ.get('MLUS_CHECKPOINT_PATH')
if resume:
    if Path(resume)!=checkpoint:raise RuntimeError('application checkpoint path changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'],device=0)
    step=saved['step']
elif checkpoint.exists():
    raise RuntimeError('use a fresh checkpoint path; existing state is preserved')
start=step
for _ in range(min(a.chunk_steps,a.steps-step)):
    x=torch.randn(32,32).cuda()
    target=x[:,:16]*.5
    optimizer.zero_grad()
    loss=torch.nn.functional.mse_loss(model(x),target)
    loss.backward()
    optimizer.step()
    step+=1
torch.cuda.synchronize()
print(f'CUDA checkpoint example: attempt={os.environ["MLUS_ATTEMPT"]} steps={start}->{step} loss={loss.item()}',flush=True)

def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value
saved=dict(step=step,model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0))
checkpoint.parent.mkdir(parents=True,exist_ok=True)
temporary=None
try:
    with tempfile.NamedTemporaryFile(mode='wb',dir=checkpoint.parent,prefix='.mlus-torch-',delete=False) as f:
        temporary=Path(f.name);torch.save(saved,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,checkpoint);temporary=None
finally:
    if temporary is not None:temporary.unlink(missing_ok=True)
if step<a.steps:yield_checkpoint(checkpoint,step=step,resume_vram_mib=a.resume_vram_mib)
