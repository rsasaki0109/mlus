#!/usr/bin/env python3
"""Real daemon + synthetic cooperative reports. No GPU measurements."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('MLUS_BIN', str(ROOT/'target/debug/mlus')))


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]


def main():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode='w+') as log:
        root=Path(tmp);state=root/'state';port=free_port();server=None
        def start():
            return subprocess.Popen([str(BIN),'serve','--mock','--port',str(port),'--state-dir',str(state)],stdout=log,stderr=log)
        def api(payload=None):
            return json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/'+('jobs' if payload is not None else 'status'),data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'}),timeout=3))
        def wait(predicate):
            deadline=time.monotonic()+12
            while time.monotonic()<deadline:
                try:
                    s=api()
                    for gpu in s['gpus']:
                        reserved=sum(j['reservation_mib'] for j in s['jobs'] if j['state']=='running' and j['gpu']==gpu['uuid'])
                        assert reserved <= gpu['total_mib']-s['margin_mib']
                    if predicate(s):return s
                except OSError:pass
                time.sleep(.03)
            raise AssertionError('profile workflow timeout')
        def submit(need,command,key=None):
            return api(dict(vram_mib=need,command=command,profile=key))['id']
        def sim(peak,outcome='success',seconds=0):
            return ['python3',str(ROOT/'examples/profile_simulation.py'),'--peak-mib',str(peak),'--outcome',outcome,'--seconds',str(seconds)]
        def job(s,jid):return next(j for j in s['jobs'] if j['id']==jid)
        def finished(jid):return wait(lambda s:job(s,jid)['state'] in ('succeeded','failed','rejected'))
        def gated(path):
            return ['python3','-c',f"from pathlib import Path; import time; p=Path({str(path)!r}); deadline=time.monotonic()+10\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.02)"]
        try:
            server=start();wait(lambda s:s['backend']=='mock')
            key='decoder:bs8:fp16'
            seed=submit(11000,sim(11500),key)
            s=finished(seed);assert job(s,seed)['memory_report']['measurement']=='simulation'
            assert s['profiles'][key]['MOCK-1']['peak_reserved_mib']==11500
            assert 'MOCK-0' not in s['profiles'][key]
            gate0=root/'release0';gate1=root/'release1'
            a=submit(7000,gated(gate0));wait(lambda s:job(s,a)['state']=='running')
            b=submit(750,gated(gate1));wait(lambda s:job(s,b)['state']=='running')
            c=submit(11000,sim(11000),key)
            s=wait(lambda s:job(s,c)['state']=='queued')
            assert job(s,b)['gpu']=='MOCK-1'
            # Declaration 11000 would fit beside 750; learned 11500+128 cannot.
            assert 11000+750 <= 12288-512 < 11500+128+750
            gate1.touch();s=finished(c)
            assert job(s,c)['reservation_mib']==11628
            assert s['profiles'][key]['MOCK-1']['peak_reserved_mib']==11500
            assert s['profiles'][key]['MOCK-1']['successful_samples']==2
            gate0.touch();finished(a)
            # Report collected before scheduling queued jobs changes eligibility.
            oversize_key='model:capacity-change'
            seed=submit(11000,sim(11700,seconds=.8),oversize_key)
            following=submit(11000,sim(1),oversize_key)
            s=finished(following)
            assert job(s,seed)['state']=='succeeded' and job(s,following)['state']=='rejected'
            try:submit(11000,sim(1),oversize_key);raise AssertionError('oversize learned demand accepted')
            except urllib.error.HTTPError as e:assert e.code==400
            # Valid OOM feedback raises the floor beyond the attempted reservation.
            oom=submit(6000,sim(1000,'oom'),'oom-model')
            s=finished(oom);assert job(s,oom)['state']=='failed'
            assert s['profiles']['oom-model']['MOCK-0']['oom_floor_mib']==6000
            retry=submit(1,sim(1000),'oom-model');s=finished(retry)
            assert job(s,retry)['reservation_mib']==6128
            assert s['profiles']['oom-model']['MOCK-0']['oom_samples']==1
            # Malformed, oversized, forged-backend and inconsistent-exit reports.
            good=dict(schema_version=1,measurement='simulation',outcome='success',peak_allocated_mib=1,peak_reserved_mib=2)
            invalids=[json.dumps({**good,'measurement':'cuda'}),json.dumps({**good,'outcome':'oom'}),json.dumps({**good,'peak_reserved_mib':99999}),'invalid-json',' '*65537]
            for index,content in enumerate(invalids):
                name=f'invalid-{index}'
                expression=repr(content) if len(content)<8192 else "' ' * 65537"
                command=['python3','-c',f"import os; from pathlib import Path; Path(os.environ['MLUS_REPORT_PATH']).write_text({expression})"]
                jid=submit(1,command,name);s=finished(jid)
                assert job(s,jid)['state']=='succeeded' and job(s,jid)['report_error']
                assert name not in s['profiles']
            nonregular=submit(1,['python3','-c',"import os; from pathlib import Path; Path(os.environ['MLUS_REPORT_PATH']).mkdir()"],'directory-report');s=finished(nonregular)
            assert job(s,nonregular)['report_error']=='memory report must be a regular file'
            missing=submit(1,['true'],'missing');s=finished(missing)
            assert job(s,missing)['report_error'] and 'missing' not in s['profiles']
            for name in ('','x'*129,'model\n'):
                try:submit(1,['true'],name);raise AssertionError('invalid profile key accepted')
                except urllib.error.HTTPError as e:assert e.code==400
            cli_seed=subprocess.run([str(BIN),'submit','--port',str(port),'--vram-mib','7000','--profile','cli-model','--',*sim(4000)],capture_output=True,text=True,check=True)
            cid=json.loads(cli_seed.stdout)['id'];s=finished(cid)
            assert s['profiles']['cli-model']['MOCK-0']['peak_reserved_mib']==4000
            for flags in [['submit','--vram-mib','1','--profile','--','true'],['submit','--vram-mib','1','--profle','key','--','true']]:
                invalid_cli=subprocess.run([str(BIN),*flags],capture_output=True,text=True,timeout=5)
                assert invalid_cli.returncode==1
            # The CUDA wrapper refuses mock jobs rather than inventing data.
            wrapper=submit(1,['python3',str(ROOT/'integrations/pytorch_profile.py'),str(ROOT/'examples/cpu_train.py')],'no-cuda')
            s=finished(wrapper);assert job(s,wrapper)['exit_code']==2 and 'no-cuda' not in s['profiles']
            profiles=s['profiles']
            duplicate=subprocess.run([str(BIN),'serve','--mock','--port',str(free_port()),'--state-dir',str(state)],capture_output=True,text=True,timeout=5)
            assert duplicate.returncode==1 and 'already in use' in duplicate.stderr
            server.send_signal(signal.SIGTERM);server.wait(timeout=5);server=None
            persisted=json.loads((state/'mock-profiles.json').read_text())
            assert persisted['profiles']==profiles
            server=start();s=wait(lambda s:s['profiles']==profiles)
            assert len(s['jobs']) == len(json.loads((state/'mock-jobs.json').read_text())['jobs']) and s['jobs']
            assert all(j['state'] != 'running' for j in s['jobs'])
            # Force a write failure using our own temporary state files.
            (state/'mock-profiles.json').rename(state/'saved-profiles.json')
            (state/'mock-profiles.json').mkdir()
            seed=submit(1,sim(100),'write-failure');s=finished(seed)
            assert s['profile_store_error'] is not None
            queued=submit(1,sim(100),'write-failure')
            s=wait(lambda s:job(s,queued)['state']=='queued')
            assert 'persistence failed' in job(s,queued)['message']
            plain=submit(1,['true']);assert job(finished(plain),plain)['state']=='succeeded'
            print('PASS: per-GPU learning, learned reservation wait/release, queued re-evaluation, capacity rejection, OOM floor')
            print('PASS: bad/missing reports, backend separation, CLI wrapper refusal, locking, restart persistence, write failure')
        finally:
            if server is not None:
                server.send_signal(signal.SIGTERM);server.wait(timeout=5)
                if server.returncode!=0:
                    log.seek(0);print(log.read());raise AssertionError('profile daemon failed')


if __name__=='__main__':main()
