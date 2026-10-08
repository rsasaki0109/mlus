#!/usr/bin/env python3
"""Real child processes and disk faults, synthetic GPUs. No CUDA measurements."""
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

class Harness:
    def __init__(self, root):
        self.root = root
        self.state = root/'state'
        self.work = root/'work'
        self.work.mkdir()
        self.log = tempfile.TemporaryFile(mode='w+')
        self.server = None
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); self.port = sock.getsockname()[1]
    def start(self, cwd=None):
        self.server = subprocess.Popen([str(BIN), 'serve', '--mock', '--port', str(self.port), '--state-dir', str(self.state)], cwd=cwd or self.work, stdout=self.log, stderr=self.log)
        return self.wait(lambda s: True)
    def stop(self, abrupt=False):
        self.server.send_signal(signal.SIGKILL if abrupt else signal.SIGTERM)
        self.server.wait(timeout=5)
        assert self.server.returncode == (-signal.SIGKILL if abrupt else 0)
        self.server = None
    def api(self, path='status', payload=None):
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}/api/{path}', data=json.dumps(payload).encode() if payload is not None else None, headers={'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=3))
    def wait(self, predicate):
        until = time.monotonic()+12
        while time.monotonic()<until:
            try:
                s = self.api()
                if predicate(s): return s
            except OSError: pass
            if self.server is not None and self.server.poll() is not None:
                self.log.seek(0); raise AssertionError(self.log.read())
            time.sleep(.025)
        raise AssertionError('recovery timeout')
    def submit(self, need, command, **extra):
        return self.api('jobs', dict(vram_mib=need, command=command, **extra))['id']
    @staticmethod
    def job(s, jid): return next(j for j in s['jobs'] if j['id']==jid)
    def close(self):
        if self.server is not None and self.server.poll() is None: self.stop()
        self.log.close()

def gate(path):
    return ['python3', '-c', f"from pathlib import Path; import time; p=Path({str(path)!r})\nwhile not p.exists(): time.sleep(.02)"]

def expect_http_error(fn, status):
    try: fn()
    except urllib.error.HTTPError as e:
        assert e.code == status
        return json.load(e)
    raise AssertionError('request unexpectedly accepted')

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        h = Harness(root)
        survivor = None
        try:
            h.start()
            # Fill both devices; acknowledged queue and command cwd must survive graceful restart.
            a = h.submit(11000, gate(root/'a'))
            b = h.submit(7000, gate(root/'b'))
            h.wait(lambda s: all(h.job(s, j)['state']=='running' for j in (a,b)))
            (h.work/'relative.py').write_text("from pathlib import Path\nPath('ran-relative').write_text('ok')\n")
            q = h.submit(7000, ['python3', 'relative.py'], priority=4)
            disk = json.loads((h.state/'mock-jobs.json').read_text())
            assert h.job(disk, q)['state']=='queued' # response follows durable save
            logs = Path(h.api()['log_directory'])
            assert logs.is_relative_to(h.state)
            h.stop()
            disk = json.loads((h.state/'mock-jobs.json').read_text())
            assert all(h.job(disk, j)['state']=='cancelled' for j in (a,b))
            h.start(cwd=ROOT)
            s = h.wait(lambda s: h.job(s,q)['state']=='succeeded')
            assert (h.work/'ran-relative').read_text()=='ok'
            assert s['working_directory']==str(h.work) and s['log_directory']==str(logs)
            assert not s['recovery_required'] and h.submit(1,['true'])==q+1
            h.wait(lambda s: s['jobs'][-1]['state']=='succeeded')
            # A surviving child after SIGKILL must block every new launch, across repeated restarts.
            marker = root/'child-started'
            code = f"from pathlib import Path; import time; Path({str(marker)!r}).write_text('once')\nwhile True:time.sleep(.1)"
            active = h.submit(7000,['python3','-c',code])
            s = h.wait(lambda s: h.job(s,active)['state']=='running' and marker.exists())
            survivor = h.job(s,active)['pid']
            h.stop(abrupt=True)
            os.kill(survivor,0) # this is our known test child, not a PID loaded from disk
            h.start()
            s = h.api(); assert s['recovery_required'] and h.job(s,active)['state']=='interrupted'
            blocked_marker = root/'must-wait'
            blocked = h.submit(1,['python3','-c',f"from pathlib import Path;Path({str(blocked_marker)!r}).touch()"])
            time.sleep(.25); assert h.job(h.api(),blocked)['state']=='queued' and not blocked_marker.exists()
            expect_http_error(lambda:h.api('recover', {'confirm_cleanup':False}),400)
            cli = subprocess.run([str(BIN),'recover','--port',str(h.port)],capture_output=True,text=True)
            assert cli.returncode==1 and 'confirm-cleanup' in cli.stderr
            h.stop(); h.start(); assert h.api()['recovery_required']
            os.kill(survivor,signal.SIGKILL); survivor = None
            subprocess.run([str(BIN),'recover','--port',str(h.port),'--confirm-cleanup'],check=True,capture_output=True)
            s=h.wait(lambda s:h.job(s,blocked)['state']=='succeeded')
            assert h.job(s,active)['attempts']==1 and h.job(s,active)['state']=='interrupted'
            assert not s['recovery_required'] and blocked_marker.exists()
            # Failed acceptance save returns 503 and never launches until explicit repair.
            target = h.state/'mock-jobs.json'
            target.rename(h.state/'backup-jobs.json'); target.mkdir()
            failed_marker = root/'after-repair'
            reply = expect_http_error(lambda:h.api('jobs',dict(vram_mib=1,command=['python3','-c',f"from pathlib import Path;Path({str(failed_marker)!r}).touch()"])),503)
            assert reply['persistence']=='unconfirmed'
            time.sleep(.25); assert not failed_marker.exists() and h.api()['job_store_error']
            expect_http_error(lambda:h.api('recover',{'confirm_cleanup':True}),503)
            target.rmdir()
            h.api('recover',{'confirm_cleanup':True})
            h.wait(lambda s:h.job(s,reply['id'])['state']=='succeeded')
            assert failed_marker.exists()
            h.stop()
            # Starting intent also requires cleanup, even without a PID.
            doc = json.loads(target.read_text())
            doc['jobs'][-1].update(state='starting',pid=None,reservation_mib=1)
            target.write_text(json.dumps(doc))
            h.start(); assert h.api()['recovery_required']
            assert h.api()['jobs'][-1]['state']=='interrupted'
            h.api('recover',{'confirm_cleanup':True}); h.stop()
            # Corrupt/mismatched/duplicate records are preserved and refused.
            valid = target.read_bytes()
            for broken in (b'{bad', json.dumps(dict(doc,measurement='cuda')).encode(), json.dumps(dict(doc,jobs=[doc['jobs'][0],doc['jobs'][0]])).encode()):
                target.write_bytes(broken)
                result=subprocess.run([str(BIN),'serve','--mock','--port',str(h.port),'--state-dir',str(h.state)],capture_output=True,text=True,timeout=5)
                assert result.returncode==1 and target.read_bytes()==broken
            target.write_bytes(valid)
            print('PASS: durable acceptance, queued/history/ID/log/cwd restore, graceful cancellation')
            print('PASS: surviving child and starting intent block restart; explicit recovery; save failure; corrupt state preserved')
        finally:
            if survivor is not None:
                try:os.kill(survivor,signal.SIGKILL)
                except ProcessLookupError:pass
            h.close()
    checkpoint_restart()

def checkpoint_restart():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); h=Harness(root)
        try:
            h.start()
            other=h.submit(11000,gate(root/'other'))
            h.wait(lambda s:h.job(s,other)['state']=='running')
            worker=root/'worker.py'; checkpoint=root/'model.json'; ready=root/'ready'; release=root/'release'
            worker.write_text(f'''import os,sys,time,json
from pathlib import Path
sys.path.insert(0,{str(ROOT/'integrations')!r})
from mlus_checkpoint import atomic_json_checkpoint,yield_checkpoint
p=Path({str(checkpoint)!r})
if int(os.environ['MLUS_ATTEMPT'])==1:
    atomic_json_checkpoint(p,{{'weight':2,'step':50}})
    Path({str(ready)!r}).touch()
    while not Path({str(release)!r}).exists():time.sleep(.02)
    yield_checkpoint(p,step=50,resume_vram_mib=7000)
else:
    assert json.loads(Path(os.environ['MLUS_CHECKPOINT_PATH']).read_text())['weight']==2
    print('restored-step-50')
''')
            coop=h.submit(7000,['python3',str(worker)],cooperative=True)
            h.wait(lambda s:ready.exists() and h.job(s,coop)['state']=='running')
            high=h.submit(7000,gate(root/'high'),priority=10)
            release.touch()
            s=h.wait(lambda s:h.job(s,coop)['state']=='waiting_resume' and h.job(s,high)['state']=='running')
            assert h.job(s,coop)['checkpoint']['step']==50
            logs=Path(s['log_directory']); h.stop()
            doc=json.loads((h.state/'mock-jobs.json').read_text())
            assert h.job(doc,coop)['state']=='waiting_resume'
            h.start(); s=h.wait(lambda s:h.job(s,coop)['state']=='succeeded')
            assert h.job(s,coop)['attempts']==2 and h.job(s,coop)['gpu']=='MOCK-0'
            assert 'restored-step-50' in (logs/f'job-{coop}.log').read_text()
            print('PASS: accepted checkpoint wait survives daemon restart and resumes state on the same GPU')
        finally:h.close()

if __name__=='__main__':main()
