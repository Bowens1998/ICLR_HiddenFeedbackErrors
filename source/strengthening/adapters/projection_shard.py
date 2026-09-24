"""Content-bound QP shards, durable failures, and complete-family resume checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from contracts import atomic_json, sha
from projection import project, match_family, ProjectionFailure
from verifier import verify_token


def binding_digest(binding):
    return hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def diagnostic_json(value):
    if isinstance(value,dict):return {k:diagnostic_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [diagnostic_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return 'NONFINITE:'+str(value)
    return value


def run_shard(output, binding, families, projector=project):
    """families: tokens, guidance, head_sets, reference_heads, member_ids.

    Caller binds the role, all source arrays/heads, fixed protocol and roster.
    Interrupted shards may reuse only complete, hash-verified families. Numerical
    failures remain blocking artifacts; never reinterpret them as a zero token.
    """
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    effective={'inputs':binding,'family_count':len(families),
        'code':{name:sha(Path(__file__).with_name(name)) for name in ['projection_shard.py','projection.py','verifier.py']}}
    digest=binding_digest(effective);bp=out/'binding.json'
    if bp.exists():
        if json.loads(bp.read_text())['sha256']!=digest:raise ValueError('Resume input, protocol or implementation binding changed')
    else:atomic_json(bp,dict(sha256=digest,binding=effective))
    if (out/'FAILURE.json').exists():raise ProjectionFailure('Prior numerical failure requires recorded engineering disposition',json.loads((out/'FAILURE.json').read_text()))
    rows=[]
    with (out/'solver.jsonl').open('a') as log:
        def event(value):
            log.write(json.dumps(diagnostic_json(value),allow_nan=False)+'\n');log.flush()
        for fi,family in enumerate(families):
            prefix=out/f'family_{fi:04d}';npz=prefix.with_suffix('.npz');rp=prefix.with_suffix('.json')
            try:
                if rp.exists():
                    row=json.loads(rp.read_text());assert row['binding_sha256']==digest and row['arrays_sha256']==sha(npz)
                    with np.load(npz) as z:corrected=z['corrected']
                    assert row['member_ids']==family['member_ids'] and len(corrected)==len(family['tokens'])
                    for token,replacement,heads,ref in zip(family['tokens'],corrected,family['head_sets'],family['reference_heads']):
                        assert verify_token(token,replacement,heads,ref,expected_norm=row['matching']['effective_norm'])['accepted']
                    rows.append(row);event(dict(event='verified_resume',family=fi));continue
                directions=[]
                for mi,(token,guide,heads,ref) in enumerate(zip(family['tokens'],family['guidance'],family['head_sets'],family['reference_heads'])):
                    direction=projector(heads,token,guide,reference_head=ref);directions.append(direction)
                    event(dict(event='solver',family=fi,member=mi,member_id=family['member_ids'][mi],**direction.solver))
                corrected,matching=match_family(family['tokens'],directions,family['head_sets'],family['reference_heads'])
                temporary=prefix.with_suffix('.partial')
                with temporary.open('wb') as f:np.savez_compressed(f,corrected=corrected,full_standardized_directions=np.stack([d.delta for d in directions]))
                temporary.replace(npz)
                row=dict(family=fi,binding_sha256=digest,member_ids=family['member_ids'],arrays_sha256=sha(npz),matching=matching)
                atomic_json(rp,row);rows.append(row);event(dict(event='accepted_family',family=fi,norm=matching['effective_norm']))
            except Exception as exc:
                failure=dict(status='FAILED_SHARD_NOT_ZERO_DISPLACEMENT',family=fi,completed_families=len(rows),
                    binding_sha256=digest,exception=repr(exc),details=getattr(exc,'detail',{}))
                (out/'DONE').unlink(missing_ok=True)
                atomic_json(out/'FAILURE.json',diagnostic_json(failure));event(dict(event='failure',family=fi,exception=repr(exc)));raise
    report=dict(status='PASS_COMPLETE_PROJECTION_SHARD',binding_sha256=digest,expected_families=len(families),
                accepted_families=len(rows),legitimate_zero_families=sum(r['matching']['legitimate_zero_norm'] for r in rows),
                family_report_sha256=[sha((out/f'family_{i:04d}').with_suffix('.json')) for i in range(len(rows))])
    atomic_json(out/'report.json',report);(out/'DONE').write_text('accepted\n');return report
