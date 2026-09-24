"""Load the unchanged upstream PushT module without registering unrelated tasks."""
import importlib.util
from pathlib import Path
import sys


def load_native_pusht(source):
    path = Path(source) / 'env/pusht/pusht_env.py'
    name = '_strengthening_upstream_native_pusht'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise ValueError('Cannot mix native simulator source paths in one process')
        return module.PushTEnv
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.PushTEnv
