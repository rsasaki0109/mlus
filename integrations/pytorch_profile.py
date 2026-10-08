#!/usr/bin/env python3
"""Run an existing single-GPU Python script and report PyTorch allocator peaks.

Usage: mlus submit --profile KEY --vram-mib N -- python3 /path/to/pytorch_profile.py /path/to/train.py ARGS...
This does not measure all CUDA allocations or transparently offload memory.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys
from mlus_report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script', type=Path)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.environ.get('MLUS_BACKEND') != 'nvidia' or 'MLUS_REPORT_PATH' not in os.environ:
        parser.exit(2, 'pytorch_profile requires a real NVIDIA MLus job submitted with --profile\n')
    try:
        import torch
    except ImportError:
        parser.exit(2, 'CUDA-enabled PyTorch is required in the job Python environment\n')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.exit(2, 'exactly one visible CUDA GPU is required; no measurement performed\n')
    script = args.script.resolve(strict=True)
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    original_argv, original_path = sys.argv, sys.path.copy()
    sys.argv = [str(script), *args.args]
    sys.path.insert(0, str(script.parent))
    outcome = 'success'
    try:
        runpy.run_path(str(script), run_name='__main__')
    except torch.cuda.OutOfMemoryError:
        outcome = 'oom'
        raise
    except SystemExit as exc:
        if exc.code == 75 and os.environ.get('MLUS_COOPERATIVE') == '1':
            outcome = 'checkpointed'
        else:
            outcome = 'success' if exc.code is None or exc.code == 0 else 'error'
        raise
    except BaseException:
        outcome = 'error'
        raise
    finally:
        pending_exception = sys.exc_info()[0] is not None
        sys.argv, sys.path = original_argv, original_path
        try:
            if outcome == 'success':
                torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated(0)
            reserved = torch.cuda.max_memory_reserved(0)
            write_report(allocated_mib=(allocated + 1048575) // 1048576,
                         reserved_mib=(reserved + 1048575) // 1048576,
                         outcome=outcome, measurement='cuda')
        except Exception:
            print('MLus memory report failed; allocator observation is unavailable', file=sys.stderr)
            if not pending_exception or outcome == 'success':
                raise


if __name__ == '__main__':
    main()
