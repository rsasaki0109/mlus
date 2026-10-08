#!/usr/bin/env python3
"""Record a real daemon restart and CPU model continuation; GPU leases are mock."""
import json
from pathlib import Path
import tempfile
import time
from PIL import Image, ImageDraw, ImageFont
from recovery_smoke import Harness, gate, ROOT

font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',18)
frames=[]; snapshots=[]
def record(state, phase):
    snapshots.append(dict(state,demo_phase=phase))
    picture=Image.new('RGB',(1120,480),'#101827'); draw=ImageDraw.Draw(picture)
    draw.text((25,20),'MLus 0.4 | persistent queue / checkpoint / daemon restart',font=font,fill='#67e8f9')
    draw.text((25,55),'Real CPU model and processes; mock GPU leases, no CUDA measurement.',font=font,fill='#cbd5e1')
    draw.text((25,100),phase,font=font,fill='#fbbf24')
    draw.text((25,140),f"daemon PID {state['daemon_pid']} | recovery required: {state['recovery_required']}",font=font,fill='#dbeafe')
    for i,j in enumerate(state['jobs']):
        step=j['checkpoint']['step'] if j['checkpoint'] else 0
        draw.text((25,210+i*48),f"job {j['id']} | {j['state']:14} | attempt {j['attempts']} | checkpoint step {step}",font=font,fill='#86efac' if j['state']=='succeeded' else '#dbeafe')
    frames.append(picture)

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); h=Harness(root)
    try:
        h.start()
        other=h.submit(11000,gate(root/'other'))
        h.wait(lambda s:h.job(s,other)['state']=='running')
        train=h.submit(7000,['python3',str(ROOT/'examples/checkpoint_train.py'),'--checkpoint',str(root/'model.json'),'--seconds','.5'],cooperative=True,profile='recovery-demo:cpu-linear')
        h.wait(lambda s:h.job(s,train)['state']=='running')
        high=h.submit(7000,gate(root/'high'),priority=10)
        for _ in range(80):
            s=h.api();record(s,'Before restart: checkpoint worker yields to a priority job')
            if h.job(s,train)['state']=='waiting_resume':break
            time.sleep(.05)
        else:raise AssertionError('handoff not observed')
        for _ in range(10):record(h.api(),'Before restart: checkpoint wait is durable');time.sleep(.05)
        old_pid=s['daemon_pid']; logs=s['log_directory']
        h.stop()
        doc=json.loads((h.state/'mock-jobs.json').read_text())
        assert h.job(doc,train)['state']=='waiting_resume'
        persisted=dict(s,jobs=doc['jobs'])
        for _ in range(12):record(persisted,'Daemon stopped: direct children cancelled; checkpoint wait retained')
        h.start()
        for _ in range(100):
            s=h.api();record(s,'New daemon: same job ID and logs; checkpoint model continues')
            if h.job(s,train)['state']=='succeeded':break
            time.sleep(.05)
        assert h.job(s,train)['state']=='succeeded' and h.job(s,train)['attempts']==4
        assert s['daemon_pid']!=old_pid and s['log_directory']==logs
        records=[json.loads(line) for line in (Path(logs)/f'job-{train}.log').read_text().splitlines()]
        assert [r['start_step'] for r in records]==[0,50,100,150] and records[-1]['loss']<1e-12
        for _ in range(16):record(s,'Complete: four CPU workers, 200 steps; history survived restart')
        snapshots[-1]['demo_training_result']=records[-1]
        frames[0].save(ROOT/'assets/recovery-demo.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
        (ROOT/'assets/recovery-demo-states.json').write_text(json.dumps(snapshots,indent=2)+'\n')
        print('Recorded daemon restart: same job/logs, four CPU workers, 200 steps, loss',records[-1]['loss'])
    finally:h.close()
