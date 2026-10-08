#!/usr/bin/env python3
"""Explicit synthetic feedback for mock scheduler checks, not CUDA measurement."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integrations'))
from mlus_report import write_report
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--peak-mib', type=int, required=True)
p.add_argument('--seconds', type=float, default=0)
p.add_argument('--outcome', choices=['success', 'oom', 'error'], default='success')
a = p.parse_args()
time.sleep(a.seconds)
write_report(allocated_mib=a.peak_mib // 2, reserved_mib=a.peak_mib,
             outcome=a.outcome, measurement='simulation')
print(f'SYNTHETIC profile: {a.peak_mib} MiB, {a.outcome}; no GPU measurement')
raise SystemExit(0 if a.outcome == 'success' else 1)
