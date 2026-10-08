#!/usr/bin/env python3
"""Record live mock reservations and real parent/child/grandchild cleanup."""
import json
import os
from pathlib import Path
import tempfile
import time
from PIL import Image, ImageDraw, ImageFont
from recovery_smoke import Harness, ROOT, gate
from process_smoke import tree, gone

font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',18)
frames=[];snapshots=[]
def record(state,phase,pids):
    alive=[pid for pid in pids if not gone(pid)]
    snapshots.append(dict(state,demo_phase=phase,demo_family_pids=pids,demo_alive_pids=alive))
    image=Image.new('RGB',(1120,500),'#101827');d=ImageDraw.Draw(image)
    d.text((25,20),'MLus 0.5 | worker process group / descendants / VRAM lease',font=font,fill='#67e8f9')
    d.text((25,55),'Real processes and reaping; GPU reservations are simulated.',font=font,fill='#cbd5e1')
    d.text((25,105),phase,font=font,fill='#fbbf24')
    d.text((25,145),f'Original parent/child/grandchild still present: {len(alive)} / 3',font=font,fill='#dbeafe')
    for i,g in enumerate(state['gpus']):
        reserved=sum(j['reservation_mib'] for j in state['jobs'] if j['state']=='running' and j['gpu']==g['uuid'])
        d.text((25,195+i*40),f"{g['uuid']} | reserved {reserved} MiB",font=font,fill='#dbeafe')
    for i,j in enumerate(state['jobs']):
        d.text((25,300+i*45),f"job {j['id']} | {j['state']:12} | {j['gpu'] or 'waiting':8} | {j['request']['vram_mib']} MiB",font=font,fill='#86efac' if j['state']=='succeeded' else '#dbeafe')
    frames.append(image)

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);h=Harness(root)
    try:
        h.start();other=h.submit(11000,gate(root/'other'))
        h.wait(lambda s:h.job(s,other)['state']=='running')
        command,ready,release,paths=tree(root,'demo')
        family=h.submit(7000,command)
        s=h.wait(lambda s:ready.exists() and h.job(s,family)['state']=='running')
        pids=[h.job(s,family)['pid'],*[int(p.read_text()) for p in paths]]
        check=f"import os,time\nfor pid in {pids!r}:\n try:os.kill(pid,0)\n except ProcessLookupError:continue\n raise RuntimeError('prior job was not fully reaped')\ntime.sleep(.6)"
        queued=h.submit(7000,['python3','-c',check])
        for _ in range(14):record(h.api(),'Live family: next job waits for the lease',pids);time.sleep(.05)
        assert h.job(h.api(),queued)['state']=='queued'
        release.touch()
        for _ in range(100):
            s=h.api();record(s,'Parent exits: group cleanup precedes the next admission',pids)
            if h.job(s,queued)['state']=='succeeded':break
            time.sleep(.05)
        assert h.job(s,queued)['state']=='succeeded' and all(gone(pid) for pid in pids)
        for _ in range(18):record(s,'Complete: all 3 original processes are reaped',pids)
        frames[0].save(ROOT/'assets/process-demo.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
        (ROOT/'assets/process-demo-states.json').write_text(json.dumps(snapshots,indent=2)+'\n')
        print('Recorded real process tree cleanup; next worker verified all three prior PIDs absent')
    finally:h.close()
