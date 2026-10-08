#!/usr/bin/env python3
"""Real Linux process trees; synthetic GPU leases, no CUDA measurements."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from recovery_smoke import Harness, ROOT, gate, expect_http_error


def gone(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return True
    return False


def tree(root, name):
    grand = root/f'{name}-grand.pid'
    middle = root/f'{name}-middle.pid'
    ready = root/f'{name}-ready'
    release = root/f'{name}-release'
    child_code = f"import os,time;from pathlib import Path;Path({str(grand)!r}).write_text(str(os.getpid()))\nwhile True:time.sleep(.02)"
    middle_code = f"import os,subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-c',{child_code!r}]);Path({str(middle)!r}).write_text(str(os.getpid()))\nwhile True:time.sleep(.02)"
    worker = root/f'{name}.py'
    worker.write_text(f"import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-c',{middle_code!r}])\nwhile not Path({str(grand)!r}).exists():time.sleep(.02)\nPath({str(ready)!r}).touch()\nwhile not Path({str(release)!r}).exists():time.sleep(.02)\n")
    return ['python3', str(worker)], ready, release, (middle, grand)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); h = Harness(root); escaped = None
        try:
            s = h.start(); assert s['process_management']['subreaper']
            other = h.submit(11000, gate(root/'gpu1-release'))
            h.wait(lambda s: h.job(s,other)['state']=='running')
            # Parent exits with live child and grandchild. Queue must wait for full cleanup.
            command, ready, release, paths = tree(root, 'normal')
            family = h.submit(7000, command)
            s = h.wait(lambda s: ready.exists() and h.job(s,family)['state']=='running')
            leader = h.job(s,family)['pid']; descendants = [int(p.read_text()) for p in paths]
            assert all(os.getpgid(pid)==leader for pid in [leader,*descendants])
            assert os.getpgid(h.server.pid)!=leader
            check = f"import os\npids={descendants!r}\nfor pid in pids:\n try:os.kill(pid,0)\n except ProcessLookupError:continue\n raise RuntimeError('descendant still alive or unreaped')\nprint('clean-before-admission')"
            queued = h.submit(7000, ['python3','-c',check])
            assert h.job(h.api(),queued)['state']=='queued'
            release.touch()
            s = h.wait(lambda s:h.job(s,queued)['state']=='succeeded')
            assert h.job(s,family)['state']=='succeeded' and all(gone(p) for p in [leader,*descendants])
            assert not s['process_management']['cleanup_error']
            print('PASS: distinct job group; real child/grandchild killed and reaped before next admission')
            # Checkpoint handoff also waits for descendant cleanup before another job/resume.
            worker = root/'cooperative.py'; pidfile=root/'cooperative-child.pid'
            ready=root/'cooperative-ready'; release=root/'cooperative-release'; checkpoint=root/'model.json'
            child_code=f"import os,time;from pathlib import Path;Path({str(pidfile)!r}).write_text(str(os.getpid()))\nwhile True:time.sleep(.02)"
            worker.write_text(f'''import os,sys,subprocess,time,json
from pathlib import Path
sys.path.insert(0,{str(ROOT/'integrations')!r})
from mlus_checkpoint import atomic_json_checkpoint,yield_checkpoint
if int(os.environ['MLUS_ATTEMPT'])==1:
    subprocess.Popen([sys.executable,'-c',{child_code!r}])
    while not Path({str(pidfile)!r}).exists():time.sleep(.02)
    atomic_json_checkpoint({str(checkpoint)!r},{{'step':1,'value':42}})
    Path({str(ready)!r}).touch()
    while not Path({str(release)!r}).exists():time.sleep(.02)
    yield_checkpoint({str(checkpoint)!r},step=1,resume_vram_mib=7000)
else:
    assert json.loads(Path(os.environ['MLUS_CHECKPOINT_PATH']).read_text())['value']==42
''')
            coop=h.submit(7000,['python3',str(worker)],cooperative=True)
            h.wait(lambda s:ready.exists() and h.job(s,coop)['state']=='running')
            descendant=int(pidfile.read_text()); highgate=root/'priority-release'
            high=h.submit(7000,gate(highgate),priority=5)
            release.touch()
            s=h.wait(lambda s:h.job(s,coop)['state']=='waiting_resume' and h.job(s,high)['state']=='running')
            assert gone(descendant)
            highgate.touch();s=h.wait(lambda s:h.job(s,coop)['state']=='succeeded')
            assert h.job(s,coop)['attempts']==2
            print('PASS: checkpoint handoff and resume occur after group cleanup')
            # Graceful daemon shutdown terminates every inherited group member.
            command, ready, release, paths=tree(root,'shutdown')
            family=h.submit(7000,command);s=h.wait(lambda s:ready.exists() and h.job(s,family)['state']=='running')
            descendants=[int(p.read_text()) for p in paths];leader=h.job(s,family)['pid']
            h.stop();assert all(gone(p) for p in [leader,*descendants])
            doc=json.loads((h.state/'mock-jobs.json').read_text())
            assert h.job(doc,family)['state']=='cancelled' and not doc['recovery_required']
            h.start()
            # A fatal preparation error must use the same group cleanup guard.
            command, ready, release, paths=tree(root,'fatal')
            family=h.submit(1,command);s=h.wait(lambda s:ready.exists() and h.job(s,family)['state']=='running')
            descendants=[int(p.read_text()) for p in paths];leader=h.job(s,family)['pid']
            bad_id=len(s['jobs'])+1; badlog=Path(s['log_directory'])/f'job-{bad_id}.log';badlog.mkdir()
            assert h.submit(1,['true'])==bad_id
            h.server.wait(timeout=5);assert h.server.returncode==1;h.server=None
            assert all(gone(p) for p in [leader,*descendants])
            badlog.rmdir();s=h.start();assert s['recovery_required'] and h.job(s,family)['state']=='interrupted'
            h.api('recover',{'confirm_cleanup':True});h.wait(lambda s:h.job(s,bad_id)['state']=='succeeded')
            print('PASS: graceful and fatal-error cleanup reap process trees; durable interrupted records require recovery')
            # Escaped groups are outside containment: detect adopted live children and fail closed.
            marker=root/'escaped.pid';ready=root/'escape-ready';release=root/'escape-release'
            code=f"import os,time;from pathlib import Path;Path({str(marker)!r}).write_text(str(os.getpid()))\nwhile True:time.sleep(.02)"
            worker=root/'escaped-parent.py'
            worker.write_text(f"import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-c',{code!r}],start_new_session=True)\nwhile not Path({str(marker)!r}).exists():time.sleep(.02)\nPath({str(ready)!r}).touch()\nwhile not Path({str(release)!r}).exists():time.sleep(.02)\n")
            family=h.submit(7000,['python3',str(worker)])
            s=h.wait(lambda s:ready.exists() and h.job(s,family)['state']=='running')
            escaped=int(marker.read_text());assert os.getpgid(escaped)!=h.job(s,family)['pid']
            release.touch();s=h.wait(lambda s:s['recovery_required'] and escaped in s['process_management']['escaped_children'])
            assert not gone(escaped) and s['process_management']['cleanup_error']
            blocked=h.submit(1,['true']);expect_http_error(lambda:h.api('recover',{'confirm_cleanup':True}),409)
            assert h.job(h.api(),blocked)['state']=='queued'
            # Only kill the real test child that this script launched and observed, never journal PIDs.
            os.kill(escaped,signal.SIGKILL)
            h.wait(lambda s:not s['process_management']['escaped_children'] and not s['process_management']['cleanup_error'])
            assert gone(escaped);escaped=None
            h.api('recover',{'confirm_cleanup':True})
            h.wait(lambda s:h.job(s,blocked)['state']=='succeeded')
            print('PASS: setsid escape detected on adoption; admissions blocked; recover refused until actual cleanup')
        finally:
            if escaped is not None:
                try:os.kill(escaped,signal.SIGKILL)
                except ProcessLookupError:pass
            h.close()

if __name__=='__main__':main()
