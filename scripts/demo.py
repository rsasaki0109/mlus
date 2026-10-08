#!/usr/bin/env python3
"""Record real mock-daemon states and render a GIF; optional Pillow required."""
import argparse, json, os, signal, socket, subprocess, tempfile, time, urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
parser=argparse.ArgumentParser();parser.add_argument('--profiles',action='store_true');args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
font_path='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
font=ImageFont.truetype(font_path,18)
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
state_dir=tempfile.TemporaryDirectory()
server=subprocess.Popen([str(ROOT/'target/debug/mlus'),'serve','--mock','--port',str(port),'--state-dir',state_dir.name],stdout=subprocess.DEVNULL)
def api(payload=None):
    return json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/'+('jobs' if payload else 'status'),data=json.dumps(payload).encode() if payload else None,headers={'Content-Type':'application/json'}),timeout=3))
frames=[];snapshots=[]
try:
    for attempt in range(100):
        try:api();break
        except OSError:time.sleep(.05)
    else:raise RuntimeError('daemon did not start')
    if args.profiles:
        for need,peak in [(7000,7500),(11000,11500)]:
            api({'vram_mib':need,'profile':'demo:fixed-shape','command':['python3',str(ROOT/'examples/profile_simulation.py'),'--peak-mib',str(peak),'--seconds','1']})
    else:
        for need,duration,priority in [(7000,2,0),(11000,2,0),(7000,1,0),(7000,1,5)]:
            api({'vram_mib':need,'command':['python3','-c',f'import time; time.sleep({duration})'],'priority':priority})
    feedback_submitted=False
    for frame in range(55):
        state=api()
        if args.profiles and not feedback_submitted and len(state['jobs'])==2 and all(j['state']=='succeeded' for j in state['jobs']):
            for peak in (7500,11500):
                api({'vram_mib':7000,'profile':'demo:fixed-shape','command':['python3',str(ROOT/'examples/profile_simulation.py'),'--peak-mib',str(peak),'--seconds','1.5']})
            feedback_submitted=True
            state=api()
        snapshots.append(state)
        image=Image.new('RGB',(960,440),'#101827');draw=ImageDraw.Draw(image)
        draw.text((25,20),'MLus | '+('profile feedback (synthetic)' if args.profiles else 'live mock-daemon recording'),font=font,fill='#67e8f9')
        draw.text((25,55),'Simulation: real child processes, no GPU / VRAM measurement',font=font,fill='#cbd5e1')
        for i,gpu in enumerate(state['gpus']):
            reserved=sum(j['reservation_mib'] for j in state['jobs'] if j['state']=='running' and j['gpu']==gpu['uuid'])
            y=105+i*55
            draw.text((25,y),f"{gpu['uuid']}  reserved {reserved:5} / {gpu['total_mib']:5} MiB",font=font,fill='#dbeafe')
            draw.rectangle((615,y+3,920,y+25),fill='#334155')
            if reserved:draw.rectangle((615,y+3,615+int(305*reserved/gpu['total_mib']),y+25),fill='#22d3ee')
        for i,job in enumerate(state['jobs']):
            draw.text((25,235+i*38),f"job {job['id']} | declared {job['request']['vram_mib']:5} / reserved {job['reservation_mib']:5} | {job['state']:9}",font=font,fill='#86efac' if job['state']=='succeeded' else '#dbeafe')
        frames.append(image);time.sleep(.1)
    assert len(snapshots[-1]['jobs'])==4
    assert all(j['state']=='succeeded' for j in snapshots[-1]['jobs'])
    name='profile-demo' if args.profiles else 'demo'
    frames[0].save(ROOT/f'assets/{name}.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
    (ROOT/f'assets/{name}-states.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    print(f'Recorded 55 live snapshots; all four jobs completed; assets/{name}.gif written')
finally:
    server.send_signal(signal.SIGINT);server.wait(timeout=5);state_dir.cleanup()
