"""Fail-closed input construction under a design lock; never authorizes evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from contracts import sha

FORMAL_ROLES = {'confirmation_A_C', 'donor_bank_A_C', 'confirmation_B', 'donor_bank_B'}
INPUT_STAGES = {
    'prepare_AC_development.py': ('simulator-v1', {'confirmation_A_C', 'donor_bank_A_C'}),
    'plan_AC_development.py': ('reacher-planning-v1', {'confirmation_A_C', 'donor_bank_A_C'}),
    'replay_AC_development.py': ('simulator-v1', {'confirmation_A_C', 'donor_bank_A_C'}),
    'prepare_B_development.py': ('native-simulator-v1', {'confirmation_B', 'donor_bank_B'}),
}


def add_design_arguments(parser):
    parser.add_argument('--design-lock')
    parser.add_argument('--design-sha256')


def read_design(path, expected_sha, root):
    if not path or not expected_sha or sha(path) != expected_sha:
        raise ValueError('A matching scientific design lock is required')
    design = json.loads(Path(path).read_text())
    if design.get('status') != 'FROZEN_SCIENTIFIC_DESIGN_INPUT_CONSTRUCTION_ONLY':
        raise ValueError('Not an input-construction design lock')
    if design.get('confirmation_evaluation_authorized') is not False:
        raise ValueError('Design lock must not authorize outcome evaluation')
    for name, digest in design['input_sources'].items():
        if sha(Path(root) / name) != digest:
            raise ValueError('Frozen input source changed: ' + name)
    for name, digest in design.get('external_input_sources', {}).items():
        if sha(name) != digest:
            raise ValueError('Frozen external input source changed: ' + name)
    return design


def verify_environment(design, name):
    row = next(x for x in design['environment_inventory']['environments'] if x['name'] == name)
    if Path(sys.prefix) != Path(row['metadata']['prefix']):
        raise ValueError('Wrong environment for input construction')
    if sha(sys.executable) != row['python_binary_sha256']:
        raise ValueError('Python binary changed')
    frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'])
    if hashlib.sha256(frozen).hexdigest() != row['freeze_sha256']:
        raise ValueError('Installed package inventory changed')


def input_context(args, role, root, source, *, verify_runtime=True):
    """Development remains usable; formal roles require both immutable lock arguments."""
    if role not in FORMAL_ROLES:
        if not role.startswith(('diagnostic_development_', 'donor_development_')):
            raise ValueError('Unrecognized input role')
        if args.design_lock or args.design_sha256:
            raise ValueError('Do not relabel development data with a confirmation design')
        draft = json.loads((Path(root) / 'strengthening/configs/bank_design.draft.json').read_text())
        return draft['banks'][role], {}
    design = read_design(args.design_lock, args.design_sha256, root)
    stage = Path(source).name
    if stage not in INPUT_STAGES or role not in INPUT_STAGES[stage][1]:
        raise ValueError('This lock authorizes input construction stages only')
    if verify_runtime:
        verify_environment(design, INPUT_STAGES[stage][0])
    spec = design['banks'][role]
    if spec['count'] != 256 or spec['role'] != role:
        raise ValueError('Frozen case roster violated')
    if hasattr(args, 'base') and Path(args.base) != Path(design['base']):
        raise ValueError('Wrong source asset root')
    if hasattr(args, 'bank') and Path(args.bank) != Path(spec['output']):
        raise ValueError('Wrong locked bank path')
    key = {'plan_AC_development.py':'actions_output',
           'replay_AC_development.py':'physics_output'}.get(stage, 'output')
    if hasattr(args, 'output') and Path(args.output) != Path(spec[key]):
        raise ValueError('Wrong locked input output path')
    if stage == 'replay_AC_development.py' and Path(args.actions) != Path(spec['actions_output']):
        raise ValueError('Wrong fixed action archive')
    if stage == 'prepare_B_development.py':
        path = Path(args.environment) / 'report.json'
        if str(path) not in design['verified_assets'] or sha(path) != design['verified_assets'][str(path)]:
            raise ValueError('Native simulator acceptance is not bound')
    return spec, dict(scientific_design_sha256=args.design_sha256,
                      phase='confirmation_input_construction', expected_cases=spec['count'])


def check_parent_design(role_manifest, binding):
    if role_manifest.get('scientific_design_sha256') != binding.get('scientific_design_sha256'):
        raise ValueError('Parent input belongs to another scientific design')
