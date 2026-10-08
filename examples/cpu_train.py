#!/usr/bin/env python3
"""Tiny deterministic CPU ML workload. Not a GPU or framework benchmark."""
import json
xs=[-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]
ys=[2*x+1 for x in xs]
w,b=0.0,0.0
for step in range(200):
    errors=[w*x+b-y for x,y in zip(xs,ys)]
    w-=.05*2*sum(e*x for e,x in zip(errors,xs))/len(xs)
    b-=.05*2*sum(errors)/len(xs)
loss=sum((w*x+b-y)**2 for x,y in zip(xs,ys))/len(xs)
assert loss<1e-12, f'training failed: {loss}'
print(json.dumps({'kind':'cpu_linear_regression','steps':200,'weight':w,'bias':b,'loss':loss,'gpu_measurement':False}))
