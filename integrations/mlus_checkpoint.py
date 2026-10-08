"""Opt-in checkpoint-and-exit contract. Never frees a live CUDA allocation."""
import json
import os
from pathlib import Path
import tempfile

CHECKPOINT_EXIT_CODE = 75


def atomic_json_checkpoint(path, state):
    """Write application-owned JSON state atomically (only trusted local data)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.mlus-checkpoint-', delete=False) as f:
            temporary = Path(f.name)
            json.dump(state, f)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        temporary = None
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def yield_checkpoint(path, *, step, resume_vram_mib):
    """Declare an already saved checkpoint, then exit the worker with code 75.

    The daemon releases the reservation only after observing process exit.
    Application code must not catch this SystemExit and keep using the GPU.
    Checkpoint loading and CUDA state reconstruction remain application-owned.
    """
    if os.environ.get('MLUS_COOPERATIVE') != '1':
        raise RuntimeError('job must be submitted with --cooperative')
    if type(step) is not int or step < 0:
        raise ValueError('checkpoint step must be a nonnegative integer')
    if type(resume_vram_mib) is not int or resume_vram_mib <= 0:
        raise ValueError('positive integer resume VRAM is required')
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError('checkpoint must be an existing regular file')
    measurement = {'mock': 'simulation', 'nvidia': 'cuda'}.get(os.environ.get('MLUS_BACKEND'))
    if measurement is None:
        raise RuntimeError('MLus backend is required for checkpoint handoff')
    report = dict(schema_version=1, measurement=measurement,
                  attempt=int(os.environ['MLUS_ATTEMPT']), step=step,
                  checkpoint_path=str(path), resume_vram_mib=resume_vram_mib)
    atomic_json_checkpoint(os.environ['MLUS_CHECKPOINT_REPORT_PATH'], report)
    raise SystemExit(CHECKPOINT_EXIT_CODE)
