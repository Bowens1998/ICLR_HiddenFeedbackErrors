"""DRAFT complete S2 T0 probe/observed-donor cache producer.

Requires real future frozen contracts and complete raw-population acceptance.
No source/seed/protocol defaults, response models, readouts or submission API.
Importing this module does not import Torch or open data/model files.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

import raw_inputs as raw
import model_runtime as runtime

PHASE, ROOT = raw.PHASE, raw.ROOT
OBJECTIVES = ('decoded_teacher', 'physical_labels')
AXES = dict(objectives=list(OBJECTIVES), horizons=[5,10,15,20,25], latent_dim=192)
RUNTIME_ROLE = {'calibration_recipient':'probe_calibration', 'test_recipient':'probe_test',
                'calibration_donor':'probe_donor_calibration', 'test_donor':'probe_donor_test'}
ALL_ADMISSIONS = set(RUNTIME_ROLE.values()) | {'calibration_response','test_response'}
POSE_SOURCE = ROOT / 'strengthening/presubmission_v8_20260922/independent_readout/scripts/common.py'


def require(value, message):
    if not value: raise ValueError(message)


def pair(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'}, 'Exact actual path/hash pair required')
    require(Path(value['path']).is_absolute(), 'Actual absolute input path required')
    # checked_json/file_sha enforce actual bytes before content access.
    require(isinstance(value['sha256'],str) and len(value['sha256'])==64 and
            all(c in '0123456789abcdef' for c in value['sha256']), 'Actual SHA256 required')
    return value


def read_pair(value):
    value=pair(value);return raw.checked_json(value['path'],value['sha256'])


def read_arrays(value):
    value=pair(value)
    require(raw.sha(value['path'])==value['sha256'],'Changed admitted raw case bytes')
    with np.load(value['path'],allow_pickle=False) as data:
        require(len(data.files)==len(set(data.files)),'Duplicate raw array key')
        return {name:data[name] for name in data.files}


def role_info(role,pool):
    require(role in RUNTIME_ROLE and isinstance(pool,int) and not isinstance(pool,bool) and pool in range(3),
            'Only fixed S2 probe bank roles and three pools are supported')
    return dict(role=role,split=role.split('_')[0],pool=pool,group=2*pool,policy_index=8*pool,
                count=raw.COUNTS[role],runtime_role=RUNTIME_ROLE[role],donor=role.endswith('_donor'))


def validate_population(population, *, protocol_sha256,raw_sources_sha256):
    require(population.get('status')=='PASS_S2_COMPLETE_RAW_INPUT_POPULATION' and
            population.get('protocol_sha256')==protocol_sha256 and
            population.get('sources_sha256')==population.get('raw_sources_sha256')==raw_sources_sha256,
            'Complete raw-population acceptance or source identity is missing')
    admissions=population['admissions'];banks=population['banks']
    require(len(admissions)==6 and {r['role'] for r in admissions}==ALL_ADMISSIONS,
            'All six independent role/stream admissions are required')
    require(len(banks)==4 and {b['bank_role'] for b in banks}==set(raw.ROLES),'All four full raw banks are required')
    used=set()
    for bank in banks:
        role=bank['bank_role'];ids=bank['parent_ids'];n=raw.COUNTS[role]
        require(bank['count']==len(ids)==n and len(set(ids))==n and
                all(isinstance(x,int) and not isinstance(x,bool) and 0<=x<2**32 for x in ids),'Incomplete raw parent roster')
        require(not used.intersection(ids),'Raw bank roles overlap');used.update(ids)
        pair(bank['manifest']);pair(bank['role_metadata'])
    for row in admissions: pair({k:row[k] for k in ('path','sha256')})
    pair(population['content_lineage'])


def validate_manifest(manifest, admission, info, protocol_sha256,raw_sources_sha256):
    expected=dict(status='S2_COMPLETE_ROLE_STREAM_INPUT_MANIFEST',protocol_sha256=protocol_sha256,
        sources_sha256=raw_sources_sha256,bank_role=info['role'],stream_role=admission['stream_role'],count=info['count'])
    require(all(manifest.get(k)==v for k,v in expected.items()),'Wrong complete role/stream input manifest')
    ids=list(admission['parent_ids']);require(manifest.get('parent_ids')==ids,'Accepted input parent order changed')
    routes=manifest['routes'];require(len(routes)==3 and {r['pool'] for r in routes}=={0,1,2},'All three probe pools are required')
    for route in routes:
        p=route['pool'];require(route['group']==2*p and route['policy_index']==8*p,'Response/reindexed stream in probe manifest')
        pair(route['actions_report']);pair(route['physics_report'])
        rows=route['cases'];require(len(rows)==info['count'] and [r['index'] for r in rows]==list(range(info['count'])),
                                   'Missing/reordered complete per-case input evidence')
        require([r['seed'] for r in rows]==ids,'Case seeds differ from admitted parents')
        for row in rows:
            for name in ('bank','actions','physics'):pair(row[name])
            require(isinstance(row['selected_index'],int) and not isinstance(row['selected_index'],bool) and
                    row['selected_index'] in range(300),'Invalid native selected candidate')
            require(isinstance(row['selected_iteration'],int) and not isinstance(row['selected_iteration'],bool) and
                    row['selected_iteration'] in range(30),'Invalid original search iteration')
    return next(route for route in routes if route['pool']==info['pool'])


def context(protocol,protocol_sha256,sources,sources_sha256,role,pool):
    info=role_info(role,pool);source=raw.checked_json(sources,sources_sha256)
    require(source.get('status')=='S2_LATENT_CACHE_SOURCES_FROZEN' and source.get('protocol_sha256')==protocol_sha256 and
            source.get('source_root')==str(ROOT) and source.get('authorized_roles')==list(raw.ROLES),'Wrong cache source freeze')
    raw_source=pair(source['raw_input_sources'])
    ctx=raw.context(protocol,protocol_sha256,raw_source['path'],raw_source['sha256'],role)
    required=(Path(__file__),Path(raw.__file__),Path(runtime.__file__),POSE_SOURCE,PHASE/'stage2/accept_raw_inputs.py')
    files=source['files'];resolved={}
    for name,digest in files.items():
        path=raw.resolve(ctx,name);require(raw.sha(path)==digest,'Changed cache source/runtime closure: '+str(path))
        resolved[str(path.resolve())]=digest
    require(all(str(p.resolve()) in resolved for p in required),'Missing cache/direct-admitter/runtime source')
    require(all(files.get(name)==digest for name,digest in ctx['sources']['files'].items()),'Raw input source closure changed')
    require(set(source['runtime_contracts'])==set(RUNTIME_ROLE.values()),'Cache runtime contract roster must be probe/donor only')
    runtime_pair=pair(source['runtime_contracts'][info['runtime_role']])
    contract,admission,models=runtime._runtime_admission(runtime_pair['path'],runtime_pair['sha256'],info['runtime_role'])
    require(contract['protocol']['sha256']==protocol_sha256 and all(m['condition']=='T0' for m in models),
            'Cache runtime must expose only bound T0 models')
    for name,digest in contract['files'].items():
        require(resolved.get(str(raw.resolve(ctx,name).resolve()))==digest,'Cache/runtime source closure mismatch')
    population_pair=pair(source['raw_population_acceptance']);population=read_pair(population_pair)
    validate_population(population,protocol_sha256=protocol_sha256,raw_sources_sha256=raw_source['sha256'])
    population_admission=next(row for row in population['admissions'] if row['role']==info['runtime_role'])
    admission_pair={k:population_admission[k] for k in ('path','sha256')}
    require(contract['input_admission']==admission_pair,'Runtime did not bind the accepted raw population admission')
    input_manifest=read_pair(admission['input_manifest'])
    route=validate_manifest(input_manifest,admission,info,protocol_sha256,raw_source['sha256'])
    bank=next(b for b in population['banks'] if b['bank_role']==role)
    require(bank['parent_ids']==list(admission['parent_ids']) and bank['manifest']==admission['bank_manifest'],
            'Runtime bank roster/manifest differs from complete raw population')
    content=read_pair(population['content_lineage'])
    require(content.get('status')=='PASS_S2_RETAINED_PIXEL_ISOLATION' and content.get('protocol_sha256')==protocol_sha256 and
            content.get('sources_sha256')==raw_source['sha256'],'Raw content admission is unbound or failed')
    outroot=Path(source['output_root']);require(outroot.is_absolute(),'Actual cache output root required')
    ctx.update(cache_source=source,cache_sources_sha256=sources_sha256,info=info,admission=admission,
        input_admission=admission_pair,input_manifest=admission['input_manifest'],runtime_contract=runtime_pair,
        raw_population_acceptance=population_pair,content_lineage=population['content_lineage'],route=route,
        output=outroot/role/f'pool_{pool}',models=models)
    ctx['cache_binding']=dict(protocol_sha256=protocol_sha256,cache_sources_sha256=sources_sha256,
        raw_sources_sha256=raw_source['sha256'],source_sha256=raw.sha(__file__),
        raw_population_acceptance=population_pair,input_admission=admission_pair,input_manifest=admission['input_manifest'],
        runtime_contract=runtime_pair,content_lineage=population['content_lineage'])
    return ctx


def authenticate_reports(ctx):
    """Verify complete selected probe reports and every file hash before weights."""
    bank,role,bm=raw._bank(ctx);route=ctx['route'];info=ctx['info'];n=info['count']
    require(ctx['admission']['bank_manifest']==dict(path=str((bank/'manifest.json').resolve()),sha256=raw.sha(bank/'manifest.json')),
            'Role admission changed actual raw bank')
    actions=raw._path(ctx,'actions')/f"route_{info['policy_index']}"
    physics=raw._path(ctx,'physics')/f"route_{info['policy_index']}"
    require(route['actions_report']['path']==str((actions/'report.json').resolve()) and route['physics_report']['path']==str((physics/'report.json').resolve()),
            'Selected probe reports have wrong canonical policy paths')
    ar,pr=read_pair(route['actions_report']),read_pair(route['physics_report'])
    for doc,status,path in ((ar,'PASS_COMPACT_FIXED_REFERENCE_SEARCH',actions),(pr,'PASS_COMPLETE_SELECTED_PHYSICS',physics)):
        require(doc.get('status')==status and len(doc['cases'])==n and (path/'DONE').is_file(),'Incomplete selected raw route')
        require(all(doc['binding'].get(k)==v for k,v in ctx['binding'].items()),'Selected raw route belongs to another source/role')
    require(ar['binding']['row']==raw._reference(ctx,info['policy_index']),'Original probe reference policy changed')
    require(ar['binding']['bank_manifest_sha256']==raw.sha(bank/'manifest.json') and
            pr['binding']['parent_manifest_sha256']==raw.sha(bank/'manifest.json') and
            pr['binding']['action_report_sha256']==route['actions_report']['sha256'],'Broken physics/action/bank report ancestry')
    require(ar['binding_sha256']==raw.sha(actions/'binding.json') and pr.get('binding')==json.loads((physics/'binding.json').read_text()),
            'Changed action/physics producer binding')
    cases=[]
    for i,(c,a,p,evidence) in enumerate(zip(bm['cases'],ar['cases'],pr['cases'],route['cases'],strict=True)):
        require(c['index']==a['index']==p['index']==evidence['index']==i and
                c['seed']==a['seed']==p['seed']==evidence['seed'],'Changed whole-population index/parent order')
        expected=dict(bank=dict(path=str((bank/f'case_{i:03d}.npz').resolve()),sha256=c['sha256']),
            actions=dict(path=str((actions/f'case_{i:03d}.npz').resolve()),sha256=a['file_sha256']),
            physics=dict(path=str((physics/f'case_{i:03d}.npz').resolve()),sha256=p['file_sha256']))
        require(a['file']==p['file']==f'case_{i:03d}.npz' and all(evidence[k]==v for k,v in expected.items()),
                'Per-case accepted raw identity changed')
        require(a['selected_index']==evidence['selected_index'] and a['selected_iteration']==evidence['selected_iteration'] and
                a.get('population_replay_exact') is True and a.get('scored_candidates')==9000 and
                p.get('all_two_replays_exact') is True,'Incomplete original candidate or double-physics replay')
        require(a['input_sha256']==p['input_sha256']==c['sha256'] and p['action_file_sha256']==a['file_sha256'] and
                a['binding_sha256']==ar['binding_sha256'] and p['binding_sha256']==raw.sha(physics/'binding.json'),
                'Broken per-case producer ancestry')
        for item in expected.values():require(raw.sha(item['path'])==item['sha256'],'Changed accepted raw case bytes')
        cases.append(dict(index=i,seed=c['seed'],parent_id=c['seed'],**expected,
            selected_index=a['selected_index'],selected_iteration=a['selected_iteration'],
            bank_manifest=ctx['admission']['bank_manifest'],actions_report=route['actions_report'],physics_report=route['physics_report']))
    return cases


def case_inputs(evidence):
    bank,action,physics=[read_arrays(evidence[key]) for key in ('bank','actions','physics')]
    for item in (bank,action,physics):
        seed=np.asarray(item['seed'])
        require(seed.shape==() and seed.dtype.kind in 'iu' and int(seed)==evidence['seed'],'Changed raw array parent')
    selected=np.asarray(action['selected_index']);iteration=np.asarray(action['selected_iteration'])
    require(selected.shape==iteration.shape==() and selected.dtype.kind in 'iu' and iteration.dtype.kind in 'iu' and
            int(selected)==evidence['selected_index'] and int(iteration)==evidence['selected_iteration'],'Changed selected search identity')
    runtime.validate_native_inputs(bank['history_pixels'],bank['goal_pixels'],bank['prefix'],
        action['population_actions'],int(selected),physics['pixels'][3:])
    require(physics['pixels'].dtype==np.uint8 and physics['pixels'].shape==(8,224,224,3),'Incomplete rendered future history')
    require(physics['states'].shape==(36,7) and physics['states'].dtype.kind=='f' and np.isfinite(physics['states']).all(),
            'Incomplete finite physical states')
    require(physics['actions'].shape==(35,2) and physics['actions'].dtype==np.float32 and
            action['selected_actions'].shape==(25,2) and action['selected_actions'].dtype==np.float32,'Incomplete selected controls')
    require(action['cost_trace'].shape==(30,300) and action['cost_trace'].dtype==np.float64 and np.isfinite(action['cost_trace']).all() and
            action['population_costs'].shape==(300,) and action['population_costs'].dtype==np.float64 and np.isfinite(action['population_costs']).all(),
            'Complete FP64 original pose-encoded search cost archive required')
    require(np.unravel_index(np.argmin(action['cost_trace']),action['cost_trace'].shape)==(int(iteration),int(selected)),
            'Changed original selected minimum; no outcome-dependent action substitution')
    for left,right in ((action['population_actions'][int(selected)],action['selected_actions']),
        (action['population_costs'],action['cost_trace'][int(iteration)]),
        (physics['actions'][:10],bank['prefix']),(physics['actions'][10:],action['selected_actions']),
        (physics['pixels'][:3],bank['history_pixels']),(physics['states'][[0,5,10]],bank['history_states'])):
        np.testing.assert_array_equal(left,right)
    return dict(history_pixels=bank['history_pixels'],goal_pixels=bank['goal_pixels'],prefix_actions=bank['prefix'],
        population_actions=action['population_actions'],selected_index=int(selected),observed_pixels=physics['pixels'][3:],
        selected_actions=action['selected_actions'],states=physics['states'])


def physical_truth(states):
    """Use the existing v8 FP64 physical-pose target mapping, unchanged."""
    spec=importlib.util.spec_from_file_location('_s2_bound_physical_pose',POSE_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=np.stack([module.physical_pose(states[i]) for i in (15,20,25,30,35)])
    require(result.shape==(5,6) and result.dtype==np.float64 and np.isfinite(result).all(),'Invalid complete physical target')
    return result


def validate_token_result(result,info):
    arrays=result['arrays']
    require(set(arrays)=={'free','observed','observed_history','reset'},'Cache may contain only complete uncorrected T0 probe branches')
    for value in arrays.values():require(value.shape==(5,192) and value.dtype==np.float32 and np.isfinite(value).all(),'Nonfinite/incomplete latent trajectory')
    checks=result['checks']
    for key in ('native_endpoint_exact','identity_exact','observed_history_first_exact','model_unchanged','actions_and_normalization_unchanged'):
        require(checks.get(key) is True,'Native runtime parity or fixed-state gate failed')
    require(result['access_receipt']['role']==info['runtime_role'],'Runtime output belongs to another input role')
    np.testing.assert_array_equal(arrays['observed_history'][0],arrays['free'][0])
    np.testing.assert_array_equal(arrays['reset'][0],arrays['observed'][0])


def model_evidence(handle,info,objective):
    spec=handle.spec
    require(spec['condition']=='T0' and spec['pool']==info['pool'] and spec['group']==info['group'] and
            spec['objective']==objective and handle.access_receipt['role']==info['runtime_role'],'Wrong fixed probe model identity')
    require(tuple(handle.admission['parent_ids'])==tuple(info['parent_ids']),'Model runtime admitted another parent population')
    return dict(objective=objective,spec=spec,spec_sha256=runtime.value_sha(spec),normalization_sha256=handle.normalization_sha256,
                model_state_sha256=handle.model_sha256,access_receipt=handle.access_receipt)


def produce(ctx):
    """Produce one complete split/pool role; any case failure retains partial output."""
    info=dict(ctx['info'],parent_ids=list(ctx['admission']['parent_ids']));out=ctx['output'];n=info['count']
    require(not out.exists(),'Existing cache/partial output is retained; never resume or overwrite')
    intent=out.parent/(out.name+'.intent.json')
    require(not intent.exists(),'Prior cache intent blocks retry')
    cases=authenticate_reports(ctx)  # Complete actual file/provenance audit before any model load.
    require(len(cases)==n and [c['seed'] for c in cases]==info['parent_ids'],'Complete ordered input population required')
    raw.write_exclusive(intent,dict(**ctx['cache_binding'],role=info['role'],pool=info['pool'],count=n,output=str(out)))
    out.mkdir(parents=True,exist_ok=False)
    arrays=dict(seeds=np.asarray(info['parent_ids'],dtype=np.int64));models=[];start=time.monotonic()
    rp=ctx['runtime_contract'];loader_kwargs=dict(runtime_contract=rp['path'],runtime_contract_sha256=rp['sha256'],split=info['split'])
    if info['donor']:
        arrays['observed5']=np.empty((n,192),np.float32)
        handle=runtime.load_donor_encoder(info['pool'],'decoded_teacher',**loader_kwargs)
        models.append(model_evidence(handle,info,'decoded_teacher'))
        for i,evidence in enumerate(cases):
            inputs=case_inputs(evidence)
            result=runtime.encode_donor_observations(handle,parent_id=evidence['seed'],observed_pixels=inputs['observed_pixels'])
            token=result['observed']
            require(token.shape==(5,192) and token.dtype==np.float32 and np.isfinite(token).all(),'Incomplete finite observed donor encoding')
            require(result['model_binding_sha256']==models[0]['spec_sha256'] and result['access_receipt']==handle.access_receipt,
                    'Donor token/model admission mismatch')
            arrays['observed5'][i]=token[0]
        del handle;gc.collect()
    else:
        for name in ('free','observed_history','reset'):arrays[name]=np.empty((2,n,5,192),np.float32)
        arrays.update(observed=np.empty((n,5,192),np.float32),truth=np.empty((n,5,6),np.float64),
            prefix_actions=np.empty((n,10,2),np.float32),selected_actions=np.empty((n,25,2),np.float32),
            selected_index=np.empty(n,np.int64))
        for oi,objective in enumerate(OBJECTIVES):
            handle=runtime.load_probe_model(info['pool'],objective,**loader_kwargs)
            binding=model_evidence(handle,info,objective)
            if models:
                for key in ('original','summary','config'):
                    require(binding['spec'][key]==models[0]['spec'][key],'Objectives do not share original encoder/config lineage')
                require(binding['normalization_sha256']==models[0]['normalization_sha256'],'Objectives use different action normalization')
            models.append(binding)
            for i,evidence in enumerate(cases):
                inputs=case_inputs(evidence)
                result=runtime.native_rollouts(handle,parent_id=evidence['seed'],
                    **{k:inputs[k] for k in ('history_pixels','goal_pixels','prefix_actions','population_actions','selected_index','observed_pixels')})
                validate_token_result(result,info)
                require(result['model_binding_sha256']==binding['spec_sha256'],'Latent outputs from wrong T0 model')
                require(result['access_receipt']==handle.access_receipt,'Latent outputs have wrong complete input admission')
                for name in ('free','observed_history','reset'):arrays[name][oi,i]=result['arrays'][name]
                if oi==0:
                    arrays['observed'][i]=result['arrays']['observed'];arrays['truth'][i]=physical_truth(inputs['states'])
                    arrays['prefix_actions'][i]=inputs['prefix_actions'];arrays['selected_actions'][i]=inputs['selected_actions']
                    arrays['selected_index'][i]=inputs['selected_index']
                else:
                    np.testing.assert_array_equal(arrays['observed'][i],result['arrays']['observed'])
                    np.testing.assert_array_equal(arrays['truth'][i],physical_truth(inputs['states']))
                    np.testing.assert_array_equal(arrays['prefix_actions'][i],inputs['prefix_actions'])
                    np.testing.assert_array_equal(arrays['selected_actions'][i],inputs['selected_actions'])
                    require(arrays['selected_index'][i]==inputs['selected_index'],'Action identity changed between objectives')
            del handle;gc.collect()
    require(all(np.isfinite(v).all() for v in arrays.values()),'Incomplete/nonfinite population; no case skipped')
    with (out/'data.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    report=dict(status='PASS_COMPLETE_S2_LATENT_CACHE',**ctx['cache_binding'],
        **{k:info[k] for k in ('role','split','pool','group','policy_index','count')},axes=AXES,
        arrays=dict(path=str(out/'data.npz'),sha256=raw.sha(out/'data.npz')),
        array_schema={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in arrays.items()},
        cases=cases,models=models,elapsed_seconds=time.monotonic()-start,
        whole_population_retained=True,response_models_deserialized=False,readout_weights_deserialized=False,
        readout_outputs_computed=False,
        donor_model_predictions_generated=False,native300_identity_and_endpoint_exact=not info['donor'],
        observed_shared_encoder_exact=not info['donor'],frozen_model_state_unchanged=True,
        action_normalization_unchanged=True,donor_encoder_only=info['donor'],
        scope='Complete fixed T0 probe predictions or separately admitted donor observations only. '
              'Recipient truth is raw simulator pose, never a readout/error. Candidate300 controls remain in bound raw files. '
              'No corrected projection, q_g, T1/T2, response prediction or scientific effect is computed here.')
    raw.write_exclusive(out/'report.json',report)
    with (out/'DONE').open('x') as f:f.write('Complete S2 latent cache produced; independent population acceptance still required\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','protocol-sha256','sources','sources-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--role',choices=list(RUNTIME_ROLE),required=True);parser.add_argument('--pool',type=int,required=True)
    args=parser.parse_args();ctx=context(args.protocol,args.protocol_sha256,args.sources,args.sources_sha256,args.role,args.pool)
    print(json.dumps(produce(ctx),indent=2,allow_nan=False))


if __name__=='__main__':main()
