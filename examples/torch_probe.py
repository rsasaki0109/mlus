#!/usr/bin/env python3
"""Opt-in real GPU validation. Requires separately installed CUDA PyTorch."""
import argparse, json, time
import torch
p=argparse.ArgumentParser();p.add_argument('--mib',type=int,default=256);p.add_argument('--seconds',type=float,default=3);a=p.parse_args()
if a.mib<=0 or a.seconds<0:p.error('positive MiB and nonnegative seconds required')
if not torch.cuda.is_available():raise SystemExit('CUDA GPU unavailable; no measurement performed')
torch.cuda.reset_peak_memory_stats()
x=torch.empty(a.mib*1024*1024,device='cuda',dtype=torch.uint8)
x.fill_(1);torch.cuda.synchronize();time.sleep(a.seconds)
print(json.dumps({'device':torch.cuda.get_device_name(0),'visible_devices':torch.cuda.device_count(),'tensor_bytes':x.numel(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}))
