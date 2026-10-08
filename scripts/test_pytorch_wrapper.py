#!/usr/bin/env python3
"""Test wrapper behavior against an explicit torch API fixture; no actual CUDA."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
TORCH_FIXTURE='''
import os
class OutOfMemoryError(MemoryError): pass
class Cuda:
    OutOfMemoryError=OutOfMemoryError
    def is_available(self): return os.environ.get('MLUS_FIXTURE_CUDA','yes')=='yes'
    def device_count(self): return 1
    def set_device(self,n): assert n==0
    def init(self): pass
    def synchronize(self): pass
    def reset_peak_memory_stats(self,n): assert n==0
    def max_memory_allocated(self,n): return 1048577
    def max_memory_reserved(self,n): return 3*1048576
cuda=Cuda()
'''


class WrapperTest(unittest.TestCase):
    def invoke(self,source,backend='nvidia',cuda='yes',cooperative=False):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'torch.py').write_text(TORCH_FIXTURE)
            script_dir=folder/'application';script_dir.mkdir()
            (script_dir/'helper.py').write_text('value=42\n')
            script=script_dir/'train.py';script.write_text(source)
            report=folder/'report.json'
            env={**os.environ,'PYTHONPATH':str(folder),'MLUS_REPORT_PATH':str(report),'MLUS_BACKEND':backend,'MLUS_FIXTURE_CUDA':cuda,'MLUS_COOPERATIVE':'1' if cooperative else '0'}
            p=subprocess.run(['python3',str(ROOT/'integrations/pytorch_profile.py'),str(script),'--port','9999'],env=env,capture_output=True,text=True,timeout=5)
            return p,json.loads(report.read_text()) if report.exists() else None
    def test_script_arguments_sibling_imports_and_ceiling(self):
        p,r=self.invoke("import sys; from helper import value; assert value==42; assert sys.argv[1:]==['--port','9999']; print('existing-code-output')")
        self.assertEqual(p.returncode,0);self.assertIn('existing-code-output',p.stdout)
        self.assertEqual(r['peak_allocated_mib'],2);self.assertEqual(r['peak_reserved_mib'],3);self.assertEqual(r['outcome'],'success')
    def test_system_exit(self):
        for code,outcome in [(0,'success'),(7,'error')]:
            p,r=self.invoke(f'raise SystemExit({code})')
            self.assertEqual(p.returncode,code);self.assertEqual(r['outcome'],outcome)
    def test_checkpoint_exit(self):
        for opted,outcome in [(False,'error'),(True,'checkpointed')]:
            p,r=self.invoke('raise SystemExit(75)',cooperative=opted)
            self.assertEqual(p.returncode,75);self.assertEqual(r['outcome'],outcome)
    def test_oom(self):
        p,r=self.invoke("import torch; raise torch.cuda.OutOfMemoryError('fixture OOM')")
        self.assertEqual(p.returncode,1);self.assertEqual(r['outcome'],'oom')
    def test_non_oom_error(self):
        p,r=self.invoke("raise RuntimeError('fixture application error')")
        self.assertEqual(p.returncode,1);self.assertEqual(r['outcome'],'error')
    def test_no_mock_or_cpu_fallback(self):
        for backend,cuda in [('mock','yes'),('nvidia','no')]:
            p,r=self.invoke('print("must not execute")',backend,cuda)
            self.assertEqual(p.returncode,2);self.assertIsNone(r);self.assertNotIn('must not execute',p.stdout)


if __name__=='__main__':unittest.main()
