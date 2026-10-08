"""Versioned cooperative report writer; no framework dependencies."""
import json
import os
import tempfile
from pathlib import Path


def write_report(*, allocated_mib, reserved_mib, outcome, measurement):
    """Atomically publish one final report to the daemon-provided path."""
    if outcome not in ('success', 'oom', 'error', 'checkpointed'):
        raise ValueError('invalid report outcome')
    expected = {'mock': 'simulation', 'nvidia': 'cuda'}.get(os.environ.get('MLUS_BACKEND'))
    if measurement != expected:
        raise ValueError('measurement does not match MLus backend')
    if any(type(n) is not int or n < 0 for n in (allocated_mib, reserved_mib)):
        raise ValueError('memory peaks must be nonnegative integer MiB')
    if allocated_mib > reserved_mib:
        raise ValueError('allocated peak exceeds reserved peak')
    target = Path(os.environ['MLUS_REPORT_PATH'])
    report = dict(schema_version=1, measurement=measurement, outcome=outcome,
                  peak_allocated_mib=allocated_mib, peak_reserved_mib=reserved_mib)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, prefix='.mlus-report-', delete=False) as f:
            temporary = Path(f.name)
            json.dump(report, f)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
