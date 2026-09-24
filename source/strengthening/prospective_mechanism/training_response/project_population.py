"""Draft complete S2 T0 projection producer; no scientific execution authority.

Authentic future frozen contracts and all six complete latent caches are required
before the first QP. Only construction g_A and insertion-time T0/observed tokens
enter projection. q_g, response models, decoded predictions and errors are never
deserialized or calculated. Source closure entries may be byte-hashed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import accept_probe_population as gate
import projection_four as four

ROOT, PHASE = gate.ROOT, gate.PHASE
INPUT_STATUS = 'S2_PROJECTION_INPUT_BINDING_FROZEN'
POOL_STATUS = 'PASS_COMPLETE_DRAFT_S2_T0_FOUR_FAMILY_PROJECTION'
POPULATION_STATUS = 'PASS_COMPLETE_S2_PROJECTION_POPULATION'


class ProjectionPopulationBlocked(RuntimeError):
    """Complete predetermined technical-failure inventory was retained."""


def file_pair(path):
    return dict(path=str(Path(path).resolve()), sha256=gate.sha(path))


def require(value, message):
    if not value:
        raise ValueError(message)


def configuration(binding, *, require_unused_output=True):
    """Validate real upstream-only contracts, not fictitious downstream reports."""
    gate._exact(set(binding), {'status', 'split', 'protocol', 'sources', 'raw_acceptance',
                             'caches', 'donor_assignment', 'output_root'}, 'projection input fields')
    gate._exact(binding['status'], INPUT_STATUS, 'projection input binding status')
    split = binding['split']; n = gate._size(split)
    out = Path(binding['output_root'])
    require(out.is_absolute(), 'An actual bound absolute projection output root is required')
    if require_unused_output:
        require(not out.exists() and not out.with_name(out.name+'.intent.json').exists(),
                'Existing output or durable intent is retained; no retry/overwrite')
    protocol = gate._json(binding['protocol'])
    gate._exact(protocol.get('status'), 'S2_SCIENTIFIC_PROTOCOL_FROZEN', 'S2 scientific freeze')
    cfg = protocol['probe_population']
    expected = dict(axes=gate.ROLLOUT_AXES, native_population=300, history_tokens=3,
        prefix_steps=10, insertion_relative_step=5, shrink_factors=list(four.SHRINK_FACTORS),
        tolerance=four.TOLERANCE, receipt_atol=four.RECEIPT_ATOL)
    for key, value in expected.items():
        gate._exact(cfg.get(key), value, 'projection protocol/' + key)
    gate._exact(set(cfg['donor_assignment_seeds']), {'calibration', 'test'}, 'both frozen donor seeds')
    sources = gate._json(binding['sources'])
    gate._exact(sources.get('status'), 'S2_PROBE_POPULATION_SOURCES_FROZEN', 'probe source freeze')
    gate._exact(sources.get('protocol_sha256'), binding['protocol']['sha256'], 'source/protocol binding')
    gate._exact(set(sources['producers']), {'projection', 'rollout'}, 'both real producing wrappers')
    gate._exact(sources['producers']['projection'], file_pair(__file__), 'actual projection wrapper identity')
    closure = sources['files']; actual = {}
    for name, digest in closure.items():
        path = Path(name); path = path if path.is_absolute() else ROOT/path
        gate._pair(dict(path=str(path), sha256=digest)); actual[str(path.resolve())] = digest
    required = [Path(__file__), Path(gate.__file__), Path(four.__file__), HERE/'model_runtime.py',
        HERE/'cache_population.py', HERE/'raw_inputs.py', HERE/'accept_raw_inputs.py',
        *gate.model_runtime.METADATA.values(), PHASE/'scripts/s1_projection.py',
        PHASE/'scripts/s1_common.py', PHASE/'scripts/s1_readout.py',
        ROOT/'strengthening/adapters/verifier.py']
    for pair in sources['producers'].values():
        path = gate._pair(pair)
        require(path.suffix == '.py', 'Actual Python producing wrapper source required')
        required.append(path)
    require(all(str(p.resolve()) in actual for p in required), 'Missing actual direct projection/runtime/geometry source')
    raw_source = gate._json(sources['raw_sources']); cache_source = gate._json(sources['cache_sources'])
    for source, status in ((raw_source,'S2_RAW_INPUT_SOURCES_FROZEN'),(cache_source,'S2_LATENT_CACHE_SOURCES_FROZEN')):
        gate._exact(source.get('status'), status, 'upstream source freeze')
        gate._exact(source.get('protocol_sha256'), binding['protocol']['sha256'], 'upstream protocol')
        for name, digest in source['files'].items():
            gate._exact(closure.get(name), digest, 'complete inherited raw/cache source closure')
    gate._exact(cache_source.get('raw_input_sources'), sources['raw_sources'], 'cache/raw source chain')
    gate._exact(cache_source.get('raw_population_acceptance'), binding['raw_acceptance'], 'cache/raw complete acceptance')
    heads = sources['heads_A']
    require(len(heads) == 3 and {(h['pool'],h['group']) for h in heads} == {(p,2*p) for p in range(3)}
            and all(type(h['pool']) is int and type(h['group']) is int for h in heads),
            'All three fixed construction heads required exactly once')
    caches = binding['caches']
    require(len(caches) == 6 and {(r['pool'],r['role']) for r in caches} ==
            {(p,r) for p in range(3) for r in ('recipient','donor')} and all(type(r['pool']) is int for r in caches),
            'All six complete recipient/donor caches required exactly once')
    return split, n, cfg, sources, out


def validate_cache_values(recipient, donor, *, split):
    """Complete pre-QP numeric admission, including unused saved horizons."""
    n = gate._size(split)
    gate._exact(set(recipient), {'free','observed_history','reset','observed','truth',
        'prefix_actions','selected_actions','selected_index','seeds'}, 'complete recipient cache keys')
    gate._exact(set(donor), {'observed5','seeds'}, 'donor encoder-only cache keys')
    for name in ('free','observed_history','reset'):
        gate._array(recipient[name], (2,n,5,192), np.dtype('float32'), name)
    gate._array(recipient['observed'], (n,5,192), np.dtype('float32'), 'all observed tokens')
    gate._array(recipient['truth'], (n,5,6), np.dtype('float64'), 'complete physical truth')
    gate._array(recipient['prefix_actions'], (n,10,2), np.dtype('float32'), 'all prefixes')
    gate._array(recipient['selected_actions'], (n,25,2), np.dtype('float32'), 'all selected controls')
    gate._array(recipient['selected_index'], (n,), np.dtype('int64'), 'all selected indices')
    require(np.all((0 <= recipient['selected_index']) & (recipient['selected_index'] < 300)),
            'Selected index outside complete native300 population')
    gate._array(donor['observed5'], (n,192), np.dtype('float32'), 'all observed donor5 tokens')
    for name, data in (('recipient',recipient),('donor',donor)):
        seeds = gate._array(data['seeds'], (n,), np.dtype('int64'), name+' parent IDs')
        require(len(np.unique(seeds)) == n and np.all((0 <= seeds) & (seeds < 2**32)),
                'Incomplete unique actual parent roster')
    require(not np.intersect1d(recipient['seeds'],donor['seeds']).size, 'Recipient/donor parent overlap')
    gate._equal(recipient['observed_history'][:,:,0],recipient['free'][:,:,0],'observed-history first prediction')
    gate._equal(recipient['reset'][:,:,0],np.broadcast_to(recipient['observed'][None,:,0],(2,n,192)),
                'complete actual reset insertion')


def load_context(binding_path, binding_sha256, *, require_unused_output=True, retain_caches=False):
    pair = dict(path=str(Path(binding_path).resolve()), sha256=binding_sha256)
    binding = gate._json(pair)
    split, n, cfg, sources, out = configuration(binding,require_unused_output=require_unused_output)
    raw, parents, admissions = gate._raw_admission(binding, sources, split, n)
    assignment_data = gate._arrays(binding['donor_assignment'])
    gate._exact(set(assignment_data), {'donor_assignment','recipient_ids','donor_ids'}, 'assignment archive keys')
    assignment = assignment_data['donor_assignment']
    gate.validate_assignment(assignment, split=split, seed=cfg['donor_assignment_seeds'][split])
    for name, role in (('recipient_ids','_recipient'),('donor_ids','_donor')):
        gate._array(assignment_data[name], (n,), np.dtype('int64'), name)
        gate._equal(assignment_data[name],np.asarray(parents[split+role],np.int64),'exact accepted parent order')
    pools = []
    # Authenticate every raw/cache/head input before even one projection call.
    for pool in range(3):
        pairs = {r['role']:r['report'] for r in binding['caches'] if r['pool'] == pool}
        rr, recipient = gate._cache(pairs['recipient'],binding,sources,raw,admissions,
            split=split,pool=pool,donor=False)
        dr, donor = gate._cache(pairs['donor'],binding,sources,raw,admissions,
            split=split,pool=pool,donor=True)
        validate_cache_values(recipient,donor,split=split)
        gate._equal(recipient['seeds'],assignment_data['recipient_ids'],'all-pool recipient order')
        gate._equal(donor['seeds'],assignment_data['donor_ids'],'all-pool donor order')
        gate._exact(rr['models'][0]['spec'],dr['models'][0]['spec'],'recipient/donor original encoder lineage')
        head_pair = next(h['file'] for h in sources['heads_A'] if h['pool'] == pool)
        for model in rr['models']:
            gate._exact(model['spec']['head_A_sha256'],head_pair['sha256'],'actual old construction g_A')
        head = gate._arrays(head_pair); four._head(head)
        # Retain only insertion-time inputs in the computational interface.
        pools.append(dict(pool=pool,group=2*pool,recipient_cache=pairs['recipient'],donor_cache=pairs['donor'],
            head_A_file=head_pair,head_A=head,predicted=recipient['free'][:,:,0].copy(),
            actual=recipient['observed'][:,0].copy(),donor=donor['observed5'].copy()))
        if retain_caches:
            pools[-1].update(recipient_report=rr,donor_report=dr,
                             recipient_arrays=recipient,donor_arrays=donor)
    gate._pair(pair)
    return dict(binding=pair,input=binding,split=split,count=n,sources=sources,output=out,
        assignment=assignment.copy(),recipient_ids=assignment_data['recipient_ids'].copy(),
        donor_ids=assignment_data['donor_ids'].copy(),pools=pools,
        admission_status='ALL_SIX_COMPLETE_CACHES_THREE_HEADS_RAW_AND_ASSIGNMENT_AUTHENTICATED')


def _failure_detail(value):
    """JSON-safe diagnostics only; nonfinite failures are never imputed tokens."""
    if isinstance(value,np.ndarray):
        return _failure_detail(value.tolist())
    if isinstance(value,np.generic):
        return _failure_detail(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return {'nonfinite_diagnostic':repr(value)}
    if isinstance(value,dict):
        return {str(k):_failure_detail(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_failure_detail(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    return repr(value)


def _append(stream, value):
    stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
    stream.flush();os.fsync(stream.fileno())


def produce(ctx):
    """Attempt the full predetermined split, retaining zeros and every failure."""
    out=Path(ctx['output']);n=gate._size(ctx['split']);binding=ctx['input']
    require(ctx.get('admission_status')=='ALL_SIX_COMPLETE_CACHES_THREE_HEADS_RAW_AND_ASSIGNMENT_AUTHENTICATED',
            'Complete pre-QP input authentication is mandatory')
    require(ctx['count']==n and len(ctx['pools'])==3 and [p['pool'] for p in ctx['pools']]==[0,1,2],
            'Full fixed split and all ordered pools required')
    require(not out.exists(),'Existing projection output retained; no resume or overwrite')
    intent=out.with_name(out.name+'.intent.json')
    require(not intent.exists(),'Prior projection intent blocks automatic retry')
    gate._pair(ctx['binding'])
    gate._write(intent,dict(status='S2_COMPLETE_PROJECTION_ATTEMPT_STARTED',input_binding=ctx['binding'],
        count=n,split=ctx['split'],pool_count=3,source_sha256=gate.sha(__file__),
        q_g_weights_deserialized=False,response_models_deserialized=False))
    out.mkdir(parents=True,exist_ok=False)
    reports=[];total_failed=0;total_zero=0;started=time.monotonic()
    for data in ctx['pools']:
        pool=data['pool'];folder=out/f'pool_{pool}';folder.mkdir(exist_ok=False)
        replacements=[];directions=[];families=[];failures=[];zero=0
        journal=folder/'families.jsonl'
        with journal.open('x') as stream:
            for goal in range(n):
                donor_index=int(ctx['assignment'][goal])
                identity=dict(goal_index=goal,recipient_seed=int(ctx['recipient_ids'][goal]),
                    donor_index=donor_index,donor_seed=int(ctx['donor_ids'][donor_index]))
                predicted={o:data['predicted'][i,goal] for i,o in enumerate(four.OBJECTIVES)}
                guides=dict(actual=data['actual'][goal],donor=data['donor'][donor_index])
                try:
                    replacement,direction,family=four.project_t0_four_family(predicted,guides,data['head_A'],
                        goal_index=goal,donor_index=donor_index)
                    accepted=four.accept_t0_four_family(predicted,guides,data['head_A'],replacement,direction,family,
                        goal_index=goal,donor_assignment=ctx['assignment'])
                    # Serializability is checked before admitting this family.
                    entry=dict(status='PASS_S2_PROJECTED_FAMILY',**identity,family=family,
                        replacements=replacement.tolist(),directions=direction.tolist(),independent_numeric_check=accepted)
                    json.dumps(entry,allow_nan=False)
                except Exception as exc:
                    failed=dict(status='FAILED_S2_PROJECTED_FAMILY',**identity,error_type=type(exc).__name__,
                        error=str(exc),detail=_failure_detail(getattr(exc,'detail',{})))
                    _append(stream,failed);failures.append(failed)
                    continue
                _append(stream,entry)
                replacements.append(np.array(replacement,copy=True));directions.append(np.array(direction,copy=True))
                families.append(family);zero += accepted['effective_norm']==0
        common=dict(split=ctx['split'],pool=pool,group=2*pool,count=n,model_role='T0',
            protocol_sha256=binding['protocol']['sha256'],probe_sources_sha256=binding['sources']['sha256'],
            source_sha256=gate.sha(__file__),recipient_cache=data['recipient_cache'],donor_cache=data['donor_cache'],
            donor_assignment=binding['donor_assignment'],head_A=data['head_A_file'],axes=gate.PROJECTION_AXES,
            input_binding=ctx['binding'],journal=file_pair(journal),attempted_families=n,
            completed_families=len(families),failed_families=len(failures),zero_families=int(zero),
            q_g_weights_deserialized=False,q_g_outputs_computed=False,response_models_deserialized=False,
            later_horizon_values_used_for_projection=False,rows_removed=0)
        if failures:
            report=dict(status='BLOCKED_COMPLETE_S2_PROJECTION_POOL',**common,failures=failures,
                scope='Every predetermined family attempted. Failed families have no substitute arrays. '
                      'Successful and failed records retained in the journal; this pool cannot authorize rollout/scoring.')
        else:
            require(len(families)==n and [r['goal_index'] for r in families]==list(range(n)),
                    'Complete ordered family population required')
            array_path=folder/'arrays.npz'
            with array_path.open('xb') as stream:
                np.savez_compressed(stream,replacements=np.stack(replacements),directions=np.stack(directions))
                stream.flush();os.fsync(stream.fileno())
            report=dict(status=POOL_STATUS,**common,arrays=file_pair(array_path),families=families,
                scope='Complete T0-only four-direction production using original g_A and insertion-time guides. '
                      'Independent dense numeric family check reused; QP optimality is not independently proved.')
        gate._write(folder/'report.json',report)
        if not failures:
            with (folder/'DONE').open('x') as stream:stream.write('complete projected pool\n')
        reports.append(dict(pool=pool,report=file_pair(folder/'report.json'),status=report['status']))
        total_failed+=len(failures);total_zero+=zero
    gate._pair(ctx['binding'])
    result=dict(status='BLOCKED_COMPLETE_S2_PROJECTION_POPULATION' if total_failed else POPULATION_STATUS,
        input_binding=ctx['binding'],protocol_sha256=binding['protocol']['sha256'],
        probe_sources_sha256=binding['sources']['sha256'],source_sha256=gate.sha(__file__),
        split=ctx['split'],count=n,pool_count=3,attempted_families=3*n,completed_families=3*n-total_failed,
        failed_families=total_failed,zero_families=int(total_zero),projections=reports,
        q_g_weights_deserialized=False,q_g_outputs_computed=False,response_models_deserialized=False,
        elapsed_seconds=time.monotonic()-started,rows_removed=0,
        scope='Fixed whole-split projection production. This is not the post-rollout whole-population '
              'acceptance and does not authorize q_g scoring or response-model access.')
    gate._write(out/'report.json',result)
    if total_failed:
        raise ProjectionPopulationBlocked(f'{total_failed} of {3*n} predetermined families failed; complete inventory retained')
    with (out/'DONE').open('x') as stream:stream.write('all three complete projection pools\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',required=True);parser.add_argument('--binding-sha256',required=True)
    args=parser.parse_args()
    result=produce(load_context(args.binding,args.binding_sha256))
    print(json.dumps({k:result[k] for k in ('status','split','count','attempted_families','failed_families')}))


if __name__=='__main__':main()
