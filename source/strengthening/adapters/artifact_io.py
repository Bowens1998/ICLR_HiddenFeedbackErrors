"""Atomic arrays and canonical array-content fingerprints."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps([str(value.dtype), value.shape]).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def state_hash(model):
    """Same tensor digest as the legacy verifier, without importing its model stack."""
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
