#!/usr/bin/env python3
"""Record a real CPU training restart demo on mock GPU reservations."""
import json
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',18)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    server=subprocess.Popen([str(ROOT/'target/debug/mlus'),'serve','--mock','--port',str(port),'--state-dir',str(root/'state')],stdout=subprocess.DEVNULL)
    def api(payload=None):
        return json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/'+('jobs' if payload is not None else 'status'),data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'}),timeout=3))
    snapshots=[];frames=[];high=None
    try:
        for _ in range(100):
            try:api();break
            except OSError:time.sleep(.05)
        else:raise RuntimeError('daemon startup failed')
        # Reserve GPU-1, so the priority workload waits for training on GPU-0.
        api(dict(vram_mib=11000,command=['python3','-c','import time; time.sleep(6)']))
        training=api(dict(vram_mib=7000,cooperative=True,profile='demo:cpu-linear',command=['python3',str(ROOT/'examples/checkpoint_train.py'),'--checkpoint',str(root/'model.json'),'--seconds','.65']))['id']
        for _ in range(100):
            state=api();train=next(j for j in state['jobs'] if j['id']==training)
            if train['state']=='running' and high is None:
                high=api(dict(vram_mib=7000,priority=5,command=['python3','-c','import time; time.sleep(1.2)']))['id']
            snapshots.append(state)
            image=Image.new('RGB',(1100,480),'#101827');draw=ImageDraw.Draw(image)
            draw.text((25,20),'MLus | cooperative checkpoint / priority / resume',font=font,fill='#67e8f9')
            draw.text((25,55),'Real CPU training and process exits; GPU reservations are simulated.',font=font,fill='#cbd5e1')
            for i,gpu in enumerate(state['gpus']):
                reserved=sum(j['reservation_mib'] for j in state['jobs'] if j['state']=='running' and j['gpu']==gpu['uuid'])
                draw.text((25,105+i*45),f"{gpu['uuid']} | reserved {reserved} MiB",font=font,fill='#dbeafe')
            for i,j in enumerate(state['jobs']):
                saved=j['checkpoint']['step'] if j['checkpoint'] else 0
                draw.text((25,230+i*45),f"job {j['id']} | {j['state']:14} | attempt {j['attempts']} | saved step {saved}",font=font,fill='#86efac' if j['state']=='succeeded' else '#dbeafe')
            frames.append(image)
            if train['state']=='succeeded' and high is not None and all(j['state']=='succeeded' for j in state['jobs']):break
            time.sleep(.1)
        assert train['state']=='succeeded' and train['attempts']==4
        assert any(next(j for j in s['jobs'] if j['id']==training)['state']=='waiting_resume' for s in snapshots)
        records=[json.loads(line) for line in (Path(state['log_directory'])/f'job-{training}.log').read_text().splitlines()]
        assert records[-1]['step']==200 and records[-1]['loss']<1e-12
        snapshots[-1]['demo_training_result']=records[-1]
        frames[0].save(ROOT/'assets/checkpoint-demo.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
        (ROOT/'assets/checkpoint-demo-states.json').write_text(json.dumps(snapshots,indent=2)+'\n')
        print('Recorded checkpoint demo: waiting_resume observed; 4 workers, 200 steps, loss',records[-1]['loss'])
    finally:
        server.send_signal(signal.SIGTERM);server.wait(timeout=5)
