#!/usr/bin/env python3
"""Check real process handoff, pinned resume and CPU model continuity; no GPU."""
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
BIN=Path(os.environ.get('MLUS_BIN',str(ROOT/'target/debug/mlus')))


def main():
    with tempfile.TemporaryDirectory() as tmp,tempfile.TemporaryFile(mode='w+') as log:
        root=Path(tmp)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        server=subprocess.Popen([str(BIN),'serve','--mock','--port',str(port),'--state-dir',str(root/'state')],stdout=log,stderr=log)
        def api(payload=None):
            return json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/'+('jobs' if payload is not None else 'status'),data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'}),timeout=3))
        def job(s,jid):return next(j for j in s['jobs'] if j['id']==jid)
        def wait(predicate):
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                try:
                    s=api()
                    for gpu in s['gpus']:
                        total=sum(j['reservation_mib'] for j in s['jobs'] if j['state']=='running' and j['gpu']==gpu['uuid'])
                        assert total<=gpu['total_mib']-s['margin_mib']
                    if predicate(s):return s
                except OSError:pass
                time.sleep(.025)
            raise AssertionError('checkpoint state timeout')
        def submit(need,command,cooperative=False,priority=0,profile=None):
            return api(dict(vram_mib=need,command=command,cooperative=cooperative,priority=priority,profile=profile))['id']
        def finished(jid):return wait(lambda s:job(s,jid)['state'] in ('succeeded','failed','rejected'))
        def gated(path):return ['python3','-c',f"from pathlib import Path; import time; p=Path({str(path)!r}); until=time.monotonic()+12\nwhile not p.exists() and time.monotonic()<until:time.sleep(.02)"]
        try:
            wait(lambda s:s['backend']=='mock')
            gate1=root/'gpu1-release';other=submit(11000,gated(gate1));wait(lambda s:job(s,other)['state']=='running')
            gate0=root/'yield-release';announced=root/'announced';checkpoint=root/'cooperative.json'
            worker=root/'worker.py'
            worker.write_text(f"""import os,sys,time,json
from pathlib import Path
sys.path.insert(0,{str(ROOT/'integrations')!r})
from mlus_checkpoint import atomic_json_checkpoint,yield_checkpoint
path=Path({str(checkpoint)!r})
if int(os.environ['MLUS_ATTEMPT'])==1:
    atomic_json_checkpoint(path,{{'value':42,'step':1}})
    try:yield_checkpoint(path,step=1,resume_vram_mib=7000)
    except SystemExit:
        Path({str(announced)!r}).touch()
        until=time.monotonic()+12
        while not Path({str(gate0)!r}).exists() and time.monotonic()<until:time.sleep(.02)
        raise
else:
    assert os.environ['MLUS_MOCK_GPU']=='MOCK-0'
    assert json.loads(Path(os.environ['MLUS_CHECKPOINT_PATH']).read_text())['value']==42
    print('checkpoint-loaded')
""")
            coop=submit(7000,['python3',str(worker)],True)
            s=wait(lambda s:announced.exists() and job(s,coop)['state']=='running')
            original_pid=job(s,coop)['pid'];assert job(s,coop)['reservation_mib']==7000
            high_gate=root/'high-release';high=submit(7000,gated(high_gate),priority=5)
            s=wait(lambda s:job(s,high)['state']=='queued')
            assert job(s,coop)['state']=='running' # report alone did not release
            gate0.touch();s=wait(lambda s:job(s,high)['state']=='running' and job(s,coop)['state']=='waiting_resume')
            assert job(s,coop)['reservation_mib']==0 and job(s,coop)['pid'] is None
            try:os.kill(original_pid,0)
            except ProcessLookupError:pass
            else:raise AssertionError('worker was not reaped before lease release')
            gate1.touch();finished(other)
            assert job(api(),coop)['state']=='waiting_resume' # free GPU-1 is not used
            high_gate.touch();s=finished(coop)
            assert job(s,coop)['state']=='succeeded' and job(s,coop)['attempts']==2
            assert job(s,coop)['gpu']=='MOCK-0'
            # Train a model over four process lifetimes and compare with uninterrupted run.
            model_checkpoint=root/'model.json'
            command=['python3',str(ROOT/'examples/checkpoint_train.py'),'--checkpoint',str(model_checkpoint),'--seconds','.05']
            cli=subprocess.run([str(BIN),'submit','--port',str(port),'--cooperative','--profile','cpu-linear:fixed-shape','--vram-mib','7000','--',*command],capture_output=True,text=True,check=True)
            model=json.loads(cli.stdout)['id'];s=finished(model);j=job(s,model)
            assert j['state']=='succeeded' and j['attempts']==4
            records=[json.loads(line) for line in (Path(s['log_directory'])/f'job-{model}.log').read_text().splitlines()]
            assert [r['start_step'] for r in records]==[0,50,100,150]
            assert [r['step'] for r in records]==[50,100,150,200]
            assert len({r['attempt'] for r in records})==4
            assert records[-1]['loss']<1e-12
            baseline=json.loads(subprocess.check_output(['python3',str(ROOT/'examples/cpu_train.py')],text=True))
            saved=json.loads(model_checkpoint.read_text())
            assert math.isclose(saved['weight'],baseline['weight'],abs_tol=1e-12)
            assert math.isclose(saved['bias'],baseline['bias'],abs_tol=1e-12)
            profile=s['profiles']['cpu-linear:fixed-shape'][j['gpu']]
            assert profile['checkpoint_samples']==3 and profile['successful_samples']==1
            # Plain exit-75 never grants a restart, even on an opted-in job without report.
            for opted in (False,True):
                bad=submit(1,['python3','-c','raise SystemExit(75)'],opted)
                assert job(finished(bad),bad)['state']=='failed' and job(api(),bad)['attempts']==1
            # Wrong attempt/backend, missing checkpoint, repeated progress and absurd demand.
            for kind in ('wrong-attempt','wrong-backend','missing-file','same-step','too-large'):
                path=root/f'{kind}.json'
                code=f"""import os,json
from pathlib import Path
p=Path({str(path)!r});p.write_text('{{}}')
r=dict(schema_version=1,measurement='simulation',attempt=int(os.environ['MLUS_ATTEMPT']),step=1,checkpoint_path=str(p),resume_vram_mib=7000)
"""
                if kind=='wrong-attempt':code+="r['attempt']+=1\n"
                elif kind=='wrong-backend':code+="r['measurement']='cuda'\n"
                elif kind=='missing-file':code+="p.unlink()\n"
                elif kind=='too-large':code+="r['resume_vram_mib']=20000\n"
                code+="Path(os.environ['MLUS_CHECKPOINT_REPORT_PATH']).write_text(json.dumps(r));raise SystemExit(75)\n"
                bad=submit(7000,['python3','-c',code],True);s=finished(bad)
                assert job(s,bad)['state']==('rejected' if kind=='too-large' else 'failed')
                assert job(s,bad)['attempts']==(2 if kind=='same-step' else 1)
            print('PASS: no release before worker exit, priority takeover, pinned resume, state loading, real CPU model continuity')
            print('PASS: 4 attempts / 200 steps match uninterrupted training; checkpoint profiles; invalid handoffs rejected')
        finally:
            server.send_signal(signal.SIGTERM);server.wait(timeout=5)
            if server.returncode!=0:
                log.seek(0);print(log.read());raise AssertionError('daemon failed')


if __name__=='__main__':main()
