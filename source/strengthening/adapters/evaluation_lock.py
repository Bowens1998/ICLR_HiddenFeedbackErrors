"""Final protocol gate and fixed goal partitions for confirmation evaluators."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

from contracts import sha
from input_lock import FORMAL_ROLES, read_design, verify_environment

ENVIRONMENTS = {
    'cache_AC_development.py':'reacher-planning-v1',
    'cache_B_development.py':'native-dinowm-v1',
    'project_development_feedback.py':'fiber-projection-v1',
    'rollout_AC_development.py':'reacher-planning-v1',
    'rollout_B_development.py':'native-dinowm-v1',
}


def add_protocol_arguments(parser, *, partitioned=False):
    parser.add_argument('--protocol-lock')
    parser.add_argument('--protocol-sha256')
    if partitioned:
        parser.add_argument('--task-index',type=int)


def read_protocol(path, expected_sha, root):
    if not path or not expected_sha or sha(path) != expected_sha:
        raise ValueError('Confirmation evaluation requires the final protocol and exact hash')
    protocol=json.loads(Path(path).read_text())
    if protocol.get('status') != 'LOCKED_CONFIRMATION_READY':
        raise ValueError('Input design alone cannot authorize evaluation')
    design=read_design(protocol['design_path'],protocol['design_sha256'],root)
    for name,digest in protocol['evaluation_sources'].items():
        if sha(Path(root)/name) != digest:
            raise ValueError('Frozen evaluator source changed: '+name)
    return protocol,design


def evaluation_context(args, role, root, source, *, verify_runtime=True):
    if role not in FORMAL_ROLES:
        if role not in {'diagnostic_development_A_C','donor_development_A_C',
                        'diagnostic_development_B','donor_development_B'}:
            raise ValueError('Unknown evaluation role')
        if args.protocol_lock or args.protocol_sha256:
            raise ValueError('Development data cannot be relabeled confirmation')
        spec=json.loads((Path(root)/'strengthening/configs/diagnostic_development_checks.json').read_text())
        return SimpleNamespace(formal=False,count=64,binding={},spec=spec,role=role,protocol=None,design=None)
    protocol,design=read_protocol(args.protocol_lock,args.protocol_sha256,root)
    stage=Path(source).name
    if stage not in ENVIRONMENTS:
        raise ValueError('Unrecognized confirmation evaluator')
    if verify_runtime:verify_environment(design,ENVIRONMENTS[stage])
    spec=dict(root_seed=design['root_seed'],AC_streams=design['streams'],
        A_models_per_group=design['objectives'],dual_groups=design['dual_groups'],
        C_groups=design['C_groups'],C_objectives=['decoded_teacher','physical_labels'],C_conditions=design['C_conditions'])
    ctx=SimpleNamespace(formal=True,count=design['banks'][role]['count'],role=role,protocol=protocol,design=design,
        binding=dict(scientific_design_sha256=protocol['design_sha256'],protocol_sha256=args.protocol_sha256,
                     phase='confirmation_evaluation'),spec=spec)
    if ctx.count!=256:raise ValueError('Changed confirmation population')
    if hasattr(args,'bank'):
        if Path(args.bank)!=Path(design['banks'][role]['output']):raise ValueError('Wrong confirmation bank')
        bound_input(Path(args.bank)/'manifest.json',ctx)
        bound_input(Path(args.bank)/'role.json',ctx)
    for key in ['base','assets','environment','head_cache']:
        if hasattr(args,key) and str(getattr(args,key))!=protocol['cli_assets'][key]:
            raise ValueError('Unbound evaluation asset '+key)
    return ctx


def bound_input(path,ctx):
    if not ctx.formal:return
    path=Path(path)
    if str(path) not in ctx.protocol['input_files'] or sha(path)!=ctx.protocol['input_files'][str(path)]:
        raise ValueError('Input is not in the frozen complete roster: '+str(path))


def check_evaluation_parent(report,ctx):
    if ctx.formal:
        for key,value in ctx.binding.items():
            if report.get(key)!=value:raise ValueError('Wrong evaluator parent '+key)
        if report.get('role') not in ctx.design['banks']:
            raise ValueError('Missing confirmation role')


def require_output(path,ctx,key):
    if ctx.formal and Path(path)!=Path(ctx.protocol['outputs'][key]):
        raise ValueError('Wrong fixed output path: '+key)


def partition(args,ctx,branch):
    if not ctx.formal:
        if getattr(args,'task_index',None) is not None:raise ValueError('Formal partition supplied for development')
        group=(int(os.environ['SLURM_ARRAY_TASK_ID']) if args.group is None else args.group) if branch=='AC' else 0
        return group,None,ctx.spec['closedloop_qc_case_indices']
    index=args.task_index if args.task_index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    rows=ctx.protocol['partitions'][branch]
    if not 0<=index<len(rows):raise ValueError('Unplanned shard index')
    row=rows[index]
    if row['task_index']!=index:raise ValueError('Malformed fixed shard roster')
    if hasattr(args,'group') and args.group is not None and args.group!=row['group']:
        raise ValueError('Conflicting group and task index')
    return row['group'],row['shard'],row['cases']


def donor_assignment(ctx,branch,group,stream):
    if not ctx.formal:raise ValueError('Development assignment remains in the development specification')
    return ctx.design['donor_assignments'][f'{branch}/{group}/{stream}']['donor_indices']
