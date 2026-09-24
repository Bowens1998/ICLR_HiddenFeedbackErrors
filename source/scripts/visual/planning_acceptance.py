"""Explicit acceptance policy selection; never silently fall back from staging."""
import hashlib
import json
from pathlib import Path


def load_acceptance(run, staged_root=None):
    run = Path(run)
    summary = json.loads((run / 'summary.json').read_text())
    if staged_root is not None and summary['score_space'] == 'state':
        path = Path(staged_root) / run.name / 'staged_acceptance.json'
        report = json.loads(path.read_text())
        assert report['verification_policy'] == 'staged_fp32_direct_state_v1'
        assert report['original_summary_sha256'] == hashlib.sha256((run / 'summary.json').read_bytes()).hexdigest()
    else:
        path = run / 'acceptance.json'
        report = json.loads(path.read_text())
    return report, path
