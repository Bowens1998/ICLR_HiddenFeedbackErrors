"""Shared integrity and split-role guards. Confirmation is fail closed."""
import hashlib
import json
import os
from pathlib import Path

TRAIN_ROLES = {'head_train', 'continuation_train', 'basis_train'}
VALIDATION_ROLES = {'head_validation', 'continuation_validation'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def namespace_seed(root, namespace):
    return int.from_bytes(hashlib.sha256(f'{root}:{namespace}'.encode()).digest()[:4], 'big')


def require_role(manifest, allowed):
    if manifest.get('role') not in set(allowed):
        raise ValueError(f"Forbidden data role: {manifest.get('role')}; allowed={sorted(allowed)}")
    if not manifest.get('parent_manifest_sha256'):
        raise ValueError('Missing parent lineage binding')
    return manifest


def require_confirmation_lock(path, expected_sha):
    if not path or not expected_sha or sha(path) != expected_sha:
        raise ValueError('Confirmation requires an existing immutable protocol lock and matching hash')

