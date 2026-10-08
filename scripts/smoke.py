#!/usr/bin/env python3
"""Functional checks of real daemon/HTTP/child lifecycles against mock GPUs."""
import json, os, signal, socket, subprocess, tempfile, time, urllib.request, urllib.error
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('MLUS_BIN', str(ROOT / 'target/debug/mlus')))

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]

def run():
    port = free_port()
    with tempfile.TemporaryDirectory() as state_dir, tempfile.TemporaryFile(mode='w+') as log:
        server = subprocess.Popen([str(BIN), 'serve', '--mock', '--port', str(port), '--state-dir', state_dir], stdout=log, stderr=log)
        def api(path='/api/status', payload=None):
            data = None if payload is None else json.dumps(payload).encode()
            return json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data, headers={'Content-Type':'application/json'}), timeout=3))
        def wait(predicate):
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                try:
                    status = api()
                    for gpu in status['gpus']:
                        reserved=sum(j['reservation_mib'] for j in status['jobs'] if j['state']=='running' and j['gpu']==gpu['uuid'])
                        assert reserved <= gpu['total_mib'] - status['margin_mib']
                    if predicate(status): return status
                except (OSError, urllib.error.URLError): pass
                time.sleep(.04)
            raise AssertionError('readiness/state timeout')
        def submit(need, command, priority=0):
            return api('/api/jobs', {'vram_mib':need,'command':command,'priority':priority})['id']
        try:
            wait(lambda s:s['backend']=='mock')
            html = urllib.request.urlopen(f'http://127.0.0.1:{port}/',timeout=3).read()
            assert b'GPU admission control' in html
            a = submit(7000, ['python3','-c','import time; time.sleep(1.4)'])
            b = submit(11000, ['python3','-c','import time; time.sleep(1.4)'])
            s = wait(lambda s:len([j for j in s['jobs'] if j['state']=='running'])==2)
            assert {j['gpu'] for j in s['jobs']} == {'MOCK-0','MOCK-1'}
            low = submit(7000, ['python3','-c','import time; time.sleep(.4)'], 0)
            high = submit(7000, ['python3','-c','import time; time.sleep(.4)'], 5)
            s = api(); assert all(s['jobs'][i-1]['state']=='queued' for i in (low,high))
            wait(lambda s:s['jobs'][high-1]['state']=='running')
            # Every observed running set must fit the logical mock capacity.
            s=wait(lambda s:all(j['state']=='succeeded' for j in s['jobs']))
            bad = submit(1, ['/mlus/nonexistent-executable'])
            exit7 = submit(1,['python3','-c','raise SystemExit(7)'])
            output = submit(1,['python3','-c',"import os; print('real-child-output'); assert os.environ['MLUS_MOCK_GPU'].startswith('MOCK-')"])
            s=wait(lambda s: all(s['jobs'][i-1]['state'] in ('failed','succeeded') for i in (bad,exit7,output)))
            assert s['jobs'][bad-1]['state']=='failed'
            assert s['jobs'][exit7-1]['exit_code']==7
            assert s['jobs'][output-1]['state']=='succeeded'
            assert 'real-child-output' in (Path(s['log_directory'])/f'job-{output}.log').read_text()
            for payload in [{'vram_mib':20000,'command':['true']},{'vram_mib':0,'command':['true']},{'vram_mib':1,'command':[]}]:
                try: api('/api/jobs', payload); raise AssertionError('invalid request accepted')
                except urllib.error.HTTPError as e: assert e.code==400
            cli = subprocess.run([str(BIN),'status','--port',str(port)],capture_output=True,text=True,check=True)
            assert json.loads(cli.stdout)['backend']=='mock'
            cli_submit = subprocess.run([str(BIN),'submit','--port',str(port),'--vram-mib','1','--','true'],capture_output=True,text=True,check=True)
            last=json.loads(cli_submit.stdout)['id'];wait(lambda s:s['jobs'][last-1]['state']=='succeeded')
            model=submit(1,['python3',str(ROOT/'examples/cpu_train.py')])
            s=wait(lambda s:s['jobs'][model-1]['state']=='succeeded')
            trained=json.loads((Path(s['log_directory'])/f'job-{model}.log').read_text())
            assert trained['loss']<1e-12 and trained['gpu_measurement'] is False
            for headers in [{'Content-Type':'text/plain'},{'Content-Type':'application/json','Origin':'http://untrusted.invalid'}]:
                req=urllib.request.Request(f'http://127.0.0.1:{port}/api/jobs', data=json.dumps({'vram_mib':1,'command':['true']}).encode(),headers=headers)
                try:urllib.request.urlopen(req,timeout=3);raise AssertionError('browser/simple submission accepted')
                except urllib.error.HTTPError as e:assert e.code==400
            print('PASS: CPU model training and browser-submission rejection')
            print('PASS: dashboard, CLI, two-GPU placement, queue/resume, priority, child exit, logs, invalid requests')
        finally:
            server.send_signal(signal.SIGINT)
            try: server.wait(timeout=5)
            except subprocess.TimeoutExpired: server.kill();server.wait();raise
            if server.returncode != 0:
                log.seek(0);print(log.read());raise AssertionError('daemon failed')
if __name__ == '__main__': run()

def telemetry_failure_and_shutdown():
    """Exercise NVIDIA adapter with a deterministic executable fixture, no GPU."""
    port=free_port()
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode='w+') as log:
        folder=Path(tmp);mode=folder/'mode';mode.write_text('busy')
        fake=folder/'nvidia-smi'
        fake.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\nimport sys\nmode=Path("+repr(str(mode))+ ").read_text()\nif mode=='error': raise SystemExit(1)\nif '--query-compute-apps' in sys.argv[1]: print('')\nelse: print('GPU-fixture, 8192, '+('7900' if mode=='busy' else '0')+', 0')\n")
        fake.chmod(0o755)
        server=subprocess.Popen([str(BIN),'serve','--port',str(port),'--state-dir',str(folder/'state')],stdout=log,stderr=log,env={**os.environ,'PATH':str(folder)+os.pathsep+os.environ['PATH']})
        def api(payload=None):
            return json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/'+('jobs' if payload else 'status'),data=json.dumps(payload).encode() if payload else None,headers={'Content-Type':'application/json'}),timeout=3))
        def wait(predicate):
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                try:
                    s=api()
                    if predicate(s):return s
                except OSError:pass
                time.sleep(.04)
            raise AssertionError('fixture timeout')
        child_pid=None
        try:
            wait(lambda s:s['backend']=='nvidia')
            api({'vram_mib':7000,'command':['python3','-c','import time; time.sleep(30)']})
            assert wait(lambda s:len(s['jobs'])==1)['jobs'][0]['state']=='queued'
            mode.write_text('error')
            s=wait(lambda s:s['telemetry_error'] is not None)
            assert s['jobs'][0]['state']=='queued'
            mode.write_text('free')
            s=wait(lambda s:s['jobs'][0]['state']=='running');child_pid=s['jobs'][0]['pid']
        finally:
            server.send_signal(signal.SIGTERM);server.wait(timeout=5)
            assert server.returncode==0
        assert child_pid is not None
        try:os.kill(child_pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError('child survived graceful shutdown')
        print('PASS: NVIDIA CSV adapter fixture, external VRAM wait, fail-closed telemetry, recovery, direct-child SIGTERM shutdown')
if __name__=='__main__': telemetry_failure_and_shutdown()
