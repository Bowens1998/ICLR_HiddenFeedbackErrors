"""Draft complete response LATENTS with distinct calibration/sealed-test paths.

No q_g decoder, response error, scalar response or scientific execution authority.
All actual future hashes/roles must be supplied by frozen contracts. Importing
this module does not load model weights or Torch. Saved-data verification below
authenticates provenance/consistency; it does not independently replay a model.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import accept_probe_population as gate
import cache_population as cache
import finalize_regression as finalizer
import regression

runtime, raw = cache.runtime, cache.raw
INPUT_STATUS = 'S2_RESPONSE_LATENT_INPUT_BINDING_FROZEN'
SOURCE_STATUS = 'S2_RESPONSE_LATENT_SOURCES_FROZEN'
STATUS = 'PASS_COMPLETE_S2_RESPONSE_LATENT_POPULATION'
POOL_STATUS = 'PASS_COMPLETE_S2_RESPONSE_LATENT_POOL'
CONDITIONS = ('T0', 'T1')
AXES = dict(objectives=list(runtime.OBJECTIVES), conditions=list(CONDITIONS),
            horizons=[5,10,15,20,25], latent_dim=192,
            pose=['agent_x','agent_y','block_x','block_y','sin_theta','cos_theta'])
ADMITTED = 'ALL_RESPONSE_SOURCE_RUNTIME_AND_RAW_CASES_AUTHENTICATED'
TEST_FIELDS = {'prediction_lock','response_contract','expected_bindings','orchestration_seal'}


def require(value, message):
    if not value: raise ValueError(message)


def pair(path): return dict(path=str(Path(path).resolve()), sha256=gate.sha(path))


def _hex(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _seal(binding):
    """Verify actual outer/inner seal and all five original input hashes."""
    expected=binding['expected_bindings']
    require(set(expected)==set(regression.REQUIRED_BINDINGS) and all(_hex(v) for v in expected.values()),
            'The exact five actual expected prediction bindings are required')
    outer=gate._json(binding['orchestration_seal'])
    ctx=finalizer.load_context(outer['input_binding'])
    gate._exact(ctx['binding']['protocol'],binding['protocol'],'sealed response protocol')
    gate._exact(ctx['binding']['response_contract'],binding['response_contract'],'predeclared sealed response contract')
    gate._exact(ctx['expected_bindings'],expected,'actual five finalizer input bindings')
    lock=finalizer.verify_seal(ctx,binding['orchestration_seal'])
    gate._exact(lock,binding['prediction_lock'],'actual inner prediction lock, not outer seal')
    return ctx['binding']['expected_parent_ids']['test_recipient_ids']


def _model_metadata(spec):
    """Authenticate fixed metadata and bytes, without tensor deserialization."""
    for name in ('original','checkpoint','summary','report','config'):gate._pair(spec[name])
    summary=gate._json(spec['summary']); report=gate._json(spec['report'])
    expected=dict(status='PASS_FIXED_C_CONTINUATION',pool=spec['pool'],group=spec['group'],
        objective=spec['objective'],condition=spec['condition'],updates=2100,
        weights_sha256=spec['checkpoint']['sha256'],initial_A_checkpoint_sha256=spec['initial_A_checkpoint_sha256'],
        head_A_sha256=spec['head_A_sha256'])
    for key,value in expected.items():gate._exact(report.get(key),value,'fixed response training metadata/'+key)
    gate._exact(summary['config_sha256'],spec['config']['sha256'],'response original config')
    gate._exact(summary['seed'],spec['seed'],'response original seed')
    return runtime.value_sha(runtime._normalization(summary['normalization']))


def validate_response_manifest(manifest, admission, *, split, protocol_sha256, raw_sources_sha256):
    """Response policies are 1/9/17; probe-route receipts cannot be substituted."""
    n=gate._size(split);ids=list(admission['parent_ids'])
    expected=dict(status='S2_COMPLETE_ROLE_STREAM_INPUT_MANIFEST',protocol_sha256=protocol_sha256,
        sources_sha256=raw_sources_sha256,bank_role=split+'_recipient',stream_role='response',count=n,parent_ids=ids)
    for key,value in expected.items():gate._exact(manifest.get(key),value,'response manifest/'+key)
    routes=manifest['routes']
    require(len(routes)==3 and {r['pool'] for r in routes}=={0,1,2} and
            all(type(r['pool']) is int for r in routes),'All three complete response routes required')
    for route in routes:
        pool=route['pool'];gate._exact(route['group'],2*pool,'response coordinate group')
        gate._exact(route['policy_index'],8*pool+1,'original response policy row')
        gate._pair(route['actions_report']);gate._pair(route['physics_report'])
        cases=route['cases']
        require(len(cases)==n and [r['index'] for r in cases]==list(range(n)) and
                [r['seed'] for r in cases]==ids,'Complete ordered response cases required')
        for case in cases:
            require(type(case['selected_index']) is int and case['selected_index'] in range(300) and
                    type(case['selected_iteration']) is int and case['selected_iteration'] in range(30),
                    'Original native300 response selection is invalid')
            for name in ('bank','actions','physics'):gate._pair(case[name])
    return sorted(routes,key=lambda r:r['pool'])


def load_context(binding_path, binding_sha256, *, require_unused_output=True):
    bp=dict(path=str(Path(binding_path).resolve()),sha256=binding_sha256);b=gate._json(bp)
    split=b['split'];n=gate._size(split);role=split+'_response'
    expected={'status','split','protocol','sources','output_root'} | (TEST_FIELDS if split=='test' else set())
    gate._exact(set(b),expected,'response input fields and explicit split access path')
    gate._exact(b['status'],INPUT_STATUS,'response input status')
    output=Path(b['output_root']);require(output.is_absolute(),'Actual absolute response output root required')
    if require_unused_output:
        require(not output.exists() and not output.with_name(output.name+'.intent.json').exists(),
                'Existing response output/intent retained; no overwrite or resume')
    sealed_ids=_seal(b) if split=='test' else None
    protocol=gate._json(b['protocol'])
    gate._exact(protocol.get('status'),'S2_SCIENTIFIC_PROTOCOL_FROZEN','response protocol freeze')
    source=gate._json(b['sources'])
    gate._exact(set(source),{'status','protocol_sha256','source_root','files','raw_sources','raw_acceptance','runtime_contracts'},
                'forward-only response source contract')
    gate._exact(source['status'],SOURCE_STATUS,'response source status')
    gate._exact(source['protocol_sha256'],b['protocol']['sha256'],'response source protocol')
    gate._exact(source['source_root'],str(runtime.ROOT),'actual deployed response source root')
    gate._exact(set(source['runtime_contracts']),{'calibration_response','test_response'},'two explicit response runtime roles')
    ctx=raw.context(b['protocol']['path'],b['protocol']['sha256'],source['raw_sources']['path'],source['raw_sources']['sha256'],split+'_recipient')
    resolved={}
    for name,digest in source['files'].items():
        path=raw.resolve(ctx,name);gate._pair(dict(path=str(path),sha256=digest));resolved[str(path.resolve())]=digest
    required=[Path(__file__),Path(cache.__file__),Path(raw.__file__),Path(runtime.__file__),Path(gate.__file__),
        Path(finalizer.__file__),Path(regression.__file__),HERE/'independent_regression.py',HERE/'accept_raw_inputs.py',
        HERE/'projection_four.py',cache.POSE_SOURCE,*runtime.METADATA.values()]
    require(all(str(p.resolve()) in resolved for p in required),'Incomplete response wrapper/runtime/raw/seal source closure')
    for name,digest in ctx['sources']['files'].items():
        gate._exact(resolved.get(str(raw.resolve(ctx,name).resolve())),digest,'full inherited raw source closure')
    population,parents,admissions=gate._raw_admission(dict(protocol=b['protocol'],raw_acceptance=source['raw_acceptance']),
        dict(raw_sources=source['raw_sources']),split,n)
    # Metadata admission for both declared response roles prevents a partial or
    # swapped runtime allowlist; only the chosen role's public loader may execute.
    runtime_values={}
    for runtime_role,rp in source['runtime_contracts'].items():
        gate._pair(rp);contract,admission,models=runtime._runtime_admission(rp['path'],rp['sha256'],runtime_role)
        gate._exact(contract['protocol'],b['protocol'],'response runtime protocol')
        gate._exact(contract['input_admission'],admissions[runtime_role][0],'actual raw response role admission')
        for name,digest in contract['files'].items():
            gate._exact(resolved.get(str(raw.resolve(ctx,name).resolve())),digest,'full response runtime source closure')
        runtime_values[runtime_role]=(rp,contract,admission,models)
    rp,contract,admission,models=runtime_values[role]
    require(len(models)==12 and [(m['pool'],m['objective'],m['condition']) for m in models]==
            [(p,o,c) for p in range(3) for o in runtime.OBJECTIVES for c in CONDITIONS],
            'All twelve fixed T0/T1 models required in exact order')
    gate._exact(list(admission['parent_ids']),parents[split+'_recipient'],'complete response parent order')
    if split=='test':
        gate._exact(list(admission['parent_ids']),sealed_ids,'response parents from actual sealed predictions')
        response=gate._json(b['response_contract'])
        for key,value in dict(status=runtime.RESPONSE_STATUS,role='test_response',count=512,
                protocol_sha256=b['protocol']['sha256'],runtime_contract=rp,input_admission=contract['input_admission'],
                response_latent_sources=b['sources']).items():
            gate._exact(response.get(key),value,'predeclared test response/'+key)
    manifest=gate._json(admission['input_manifest'])
    routes=validate_response_manifest(manifest,admission,split=split,protocol_sha256=b['protocol']['sha256'],
                                      raw_sources_sha256=source['raw_sources']['sha256'])
    pools=[]
    for route in routes:
        pool=route['pool'];info=dict(role=split+'_recipient',split=split,pool=pool,group=2*pool,
                                    policy_index=8*pool+1,count=n,runtime_role=role)
        rawctx=dict(ctx,admission=admission,route=route,info=info)
        # This existing parser uses the supplied original policy index. It does
        # not invent a probe admission or change the frozen planner/replay body.
        cases=cache.authenticate_reports(rawctx)
        require(len(cases)==n and [r['seed'] for r in cases]==parents[split+'_recipient'],'Complete authenticated response route')
        specs=[m for m in models if m['pool']==pool]
        normalizers=[_model_metadata(m) for m in specs]
        pools.append(dict(pool=pool,cases=cases,models=specs,normalizers=normalizers))
    gate._pair(bp)
    return dict(binding=bp,input=b,split=split,role=role,count=n,output=output,sources=source,
        protocol=b['protocol'],producer_sources=b['sources'],runtime_contract=rp,
        raw_acceptance=source['raw_acceptance'],role_admission=contract['input_admission'],
        input_manifest=admission['input_manifest'],content_lineage=population['content_lineage'],
        parent_ids=parents[split+'_recipient'],pools=pools,admission_status=ADMITTED,
        prediction_seal_verified_before_response_inference=split=='test')


def _access(ctx):
    access=dict(role=ctx['role'],runtime_contract_sha256=ctx['runtime_contract']['sha256'],input_admission=ctx['role_admission'])
    if ctx['split']=='test':
        access.update(prediction_lock_sha256=ctx['input']['prediction_lock']['sha256'],
                      response_contract_sha256=ctx['input']['response_contract']['sha256'])
    return access


def _model_evidence(handle,ctx,spec,normalizer):
    gate._exact(handle.spec,spec,'exact fixed response model specification')
    gate._exact(list(handle.admission['parent_ids']),ctx['parent_ids'],'complete admitted response model parents')
    gate._exact(handle.admission['stream_role'],'response','response model stream')
    gate._exact(handle.access_receipt,_access(ctx),'actual response loader access evidence')
    gate._exact(handle.normalization_sha256,normalizer,'model own action normalization')
    require(_hex(handle.model_sha256),'Actual loaded model-state digest required')
    return dict(spec=spec,spec_sha256=runtime.value_sha(spec),normalization_sha256=normalizer,
                model_state_sha256=handle.model_sha256,access_receipt=handle.access_receipt)


def _native(result,model):
    gate._exact(set(result['arrays']),{'free'},'response API free-only latent outputs')
    value=gate._array(result['arrays']['free'],(5,192),np.dtype('float32'),'complete response free rollout')
    for key in ('native_endpoint_exact','identity_exact','model_unchanged','actions_and_normalization_unchanged'):
        require(result['checks'].get(key) is True,'Native response exact parity/frozen state failed: '+key)
    require(result['checks'].get('observed_history_first_exact') is False,'Probe observed-history receipt in response output')
    gate._exact(result['model_binding_sha256'],model['spec_sha256'],'response token model binding')
    gate._exact(result['access_receipt'],model['access_receipt'],'response token access binding')
    return value


def _save(path,arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())


def _common(ctx):
    value=dict(split=ctx['split'],role=ctx['role'],count=ctx['count'],protocol=ctx['protocol'],
        producer_sources=ctx['producer_sources'],runtime_contract=ctx['runtime_contract'],raw_acceptance=ctx['raw_acceptance'],
        role_admission=ctx['role_admission'],input_manifest=ctx['input_manifest'],content_lineage=ctx['content_lineage'],
        input_binding=ctx['binding'],protocol_sha256=ctx['protocol']['sha256'],
        producer_sources_sha256=ctx['producer_sources']['sha256'],source_sha256=gate.sha(__file__),
        parent_ids=ctx['parent_ids'],axes=AXES,native_population=300,history_tokens=3,prefix_steps=10,
        prediction_seal_verified_before_response_inference=ctx['split']=='test',
        native_identity_and_endpoint_exact=True,frozen_model_state_unchanged=True,action_normalization_unchanged=True,
        q_g_weights_deserialized=False,q_g_outputs_computed=False,response_errors_computed=False,
        complete_population_retained=True,rows_removed=0)
    if ctx['split']=='test':value.update({key:ctx['input'][key] for key in TEST_FIELDS})
    return value


def produce(ctx):
    require(ctx.get('admission_status')==ADMITTED,'Complete response source/raw/runtime admission required')
    n=gate._size(ctx['split']);out=Path(ctx['output'])
    require(ctx['count']==n and [p['pool'] for p in ctx['pools']]==[0,1,2],'Complete ordered response population required')
    require(not out.exists() and not out.with_name(out.name+'.intent.json').exists(),'Existing response output/intent retained')
    gate._pair(ctx['binding'])
    if ctx['split']=='test':gate._exact(_seal(ctx['input']),ctx['parent_ids'],'sealed parents immediately before response load')
    gate._write(out.with_name(out.name+'.intent.json'),dict(status='S2_RESPONSE_LATENT_ATTEMPT_STARTED',
        **{k:_common(ctx)[k] for k in ('input_binding','protocol','producer_sources','split','role','count')}))
    out.mkdir(parents=True,exist_ok=False)
    rows=[];all_models=[];completed=0;location={};start=time.monotonic()
    try:
        for pool_data in ctx['pools']:
            pool=pool_data['pool'];folder=out/f'pool_{pool}';folder.mkdir(exist_ok=False)
            require(len(pool_data['cases'])==n and [c['seed'] for c in pool_data['cases']]==ctx['parent_ids'],
                    'Complete response case population required')
            arrays=dict(free=np.empty((2,2,n,5,192),np.float32),truth=np.empty((n,5,6),np.float64),
                seeds=np.asarray(ctx['parent_ids'],np.int64),prefix_actions=np.empty((n,10,2),np.float32),
                selected_actions=np.empty((n,25,2),np.float32),selected_index=np.empty(n,np.int64))
            models=[]
            with (folder/'cases.jsonl').open('x') as journal:
                for mi,(spec,normalizer) in enumerate(zip(pool_data['models'],pool_data['normalizers'],strict=True)):
                    oi,ci=divmod(mi,2);objective,condition=runtime.OBJECTIVES[oi],CONDITIONS[ci]
                    location=dict(pool=pool,objective=objective,condition=condition,stage='response model loading')
                    if ctx['split']=='calibration':
                        handle=runtime.load_calibration_response_model(pool,objective,condition,
                            runtime_contract=ctx['runtime_contract']['path'],runtime_contract_sha256=ctx['runtime_contract']['sha256'])
                    else:
                        b=ctx['input']
                        handle=runtime.load_test_response_model(pool,objective,condition,
                            prediction_lock=b['prediction_lock']['path'],prediction_lock_sha256=b['prediction_lock']['sha256'],
                            response_contract=b['response_contract']['path'],response_contract_sha256=b['response_contract']['sha256'],
                            expected_bindings=b['expected_bindings'])
                    evidence=_model_evidence(handle,ctx,spec,normalizer);models.append(evidence)
                    partial=folder/(objective+'_'+condition);partial.mkdir(exist_ok=False)
                    for i,case in enumerate(pool_data['cases']):
                        location=dict(pool=pool,objective=objective,condition=condition,index=i,stage='response native free rollout')
                        inputs=cache.case_inputs(case)
                        truth=cache.physical_truth(inputs['states'])
                        saved_raw=dict(truth=truth,prefix_actions=inputs['prefix_actions'],
                                       selected_actions=inputs['selected_actions'],selected_index=inputs['selected_index'])
                        for key,value in saved_raw.items():
                            if mi==0:arrays[key][i]=value
                            else:gate._equal(arrays[key][i],value,'same response raw '+key+' across T0/T1 and objectives')
                        result=runtime.native_rollouts(handle,parent_id=case['seed'],
                            **{k:inputs[k] for k in ('history_pixels','goal_pixels','prefix_actions','population_actions','selected_index')})
                        token=_native(result,evidence);arrays['free'][oi,ci,i]=token
                        partial_path=partial/f'case_{i:03d}.npz';_save(partial_path,dict(free=token,**saved_raw))
                        json.dump(dict(index=i,parent_id=case['seed'],objective=objective,condition=condition,
                            arrays=pair(partial_path),checks=result['checks'],model_binding_sha256=result['model_binding_sha256']),
                            journal,sort_keys=True,allow_nan=False)
                        journal.write('\n');journal.flush();os.fsync(journal.fileno());completed+=1
                    del handle;gc.collect()
            require(len(models)==4 and all(np.isfinite(v).all() for v in arrays.values()),'Incomplete finite fixed response pool')
            _save(folder/'data.npz',arrays)
            report=dict(status=POOL_STATUS,**_common(ctx),pool=pool,group=2*pool,policy_index=8*pool+1,
                arrays=pair(folder/'data.npz'),array_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
                cases=pool_data['cases'],models=models,journal=pair(folder/'cases.jsonl'))
            gate._write(folder/'report.json',report)
            with (folder/'DONE').open('x') as stream:stream.write('complete response latent pool; decoding remains separate\n')
            rows.append(dict(pool=pool,report=pair(folder/'report.json')));all_models.extend(models)
        require(completed==12*n and len(all_models)==12,'Incomplete fixed response population')
        gate._pair(ctx['binding'])
        report=dict(status=STATUS,**_common(ctx),pools=rows,models=all_models,complete_pools=3,
            complete_model_strata=12,completed_cases=completed,elapsed_seconds=time.monotonic()-start,
            scope='Complete fixed response free latents and original response-policy physical truth. '
                  'Native identity/endpoint/frozen-state assertions are producer runtime evidence, not independent model replay. '
                  'No q_g outputs, error values, scalar response, donor reassignment or model selection.')
        gate._write(out/'report.json',report)
        with (out/'DONE').open('x') as stream:stream.write('complete response latents; no scalar response computed\n')
        return report
    except Exception as exc:
        gate._write(out/'failure.json',dict(status='BLOCKED_S2_RESPONSE_LATENT_POPULATION',input_binding=ctx['binding'],
            location=location,completed_cases=completed,completed_pools=rows,error_type=type(exc).__name__,error=str(exc),
            scope='Partial successful arrays retained. No skipped/replaced cases, accepted whole population, or overwrite.'))
        raise


def verify_completed_population(report_pair, *, protocol_pair, expected_split, expected_raw_acceptance, expected_seal=None):
    """Saved/source/raw consistency only; never load a model or calculate errors."""
    report=gate._json(report_pair)
    gate._exact(report.get('status'),STATUS,'complete response latent population')
    gate._exact(report.get('protocol'),protocol_pair,'externally expected response protocol')
    gate._exact(report.get('split'),expected_split,'externally expected response split')
    gate._exact(report.get('raw_acceptance'),expected_raw_acceptance,'externally expected full raw population')
    if expected_split=='test':
        require(expected_seal is not None,'Actual expected outer orchestration seal required for test verification')
        gate._exact(report.get('orchestration_seal'),expected_seal,'externally expected outer response seal')
    else:require(expected_split=='calibration' and expected_seal is None,'Calibration verification cannot accept a test seal')
    bp=report['input_binding'];ctx=load_context(bp['path'],bp['sha256'],require_unused_output=False)
    gate._exact(report_pair['path'],str(ctx['output']/'report.json'),'canonical completed response population report')
    common=_common(ctx)
    for key,value in common.items():gate._exact(report.get(key),value,'response population/'+key)
    for key,value in dict(complete_pools=3,complete_model_strata=12,completed_cases=12*ctx['count']).items():
        gate._exact(report.get(key),value,'complete response count/'+key)
    rows=report['pools'];require(len(rows)==3 and [r['pool'] for r in rows]==[0,1,2],'All complete response pool reports required')
    pools=[];models=[];n=ctx['count']
    for data,row in zip(ctx['pools'],rows,strict=True):
        pool=data['pool'];gate._exact(row['report']['path'],str(ctx['output']/f'pool_{pool}'/'report.json'),'canonical response pool report')
        saved=gate._json(row['report'])
        expected=dict(status=POOL_STATUS,**common,pool=pool,group=2*pool,policy_index=8*pool+1,cases=data['cases'])
        for key,value in expected.items():gate._exact(saved.get(key),value,'saved response pool/'+key)
        require(len(saved['models'])==4,'Four fixed models required in every response pool')
        for model,spec,norm in zip(saved['models'],data['models'],data['normalizers'],strict=True):
            for key,value in dict(spec=spec,spec_sha256=runtime.value_sha(spec),normalization_sha256=norm,access_receipt=_access(ctx)).items():
                gate._exact(model.get(key),value,'saved response model evidence/'+key)
            require(_hex(model.get('model_state_sha256')),'Missing producer model-state digest')
        models.extend(saved['models'])
        arrays=gate._arrays(saved['arrays'])
        shapes=dict(free=((2,2,n,5,192),np.float32),truth=((n,5,6),np.float64),seeds=((n,),np.int64),
            prefix_actions=((n,10,2),np.float32),selected_actions=((n,25,2),np.float32),selected_index=((n,),np.int64))
        gate._exact(set(arrays),set(shapes),'free-only complete response archive')
        for key,(shape,dtype) in shapes.items():gate._array(arrays[key],shape,np.dtype(dtype),'response/'+key)
        gate._exact(saved['array_schema'],{k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},'response array schema')
        gate._equal(arrays['seeds'],np.asarray(ctx['parent_ids'],np.int64),'complete response parent order')
        for i,case in enumerate(data['cases']):
            inputs=cache.case_inputs(case)
            gate._equal(arrays['truth'][i],cache.physical_truth(inputs['states']),'raw response physical truth')
            for key in ('prefix_actions','selected_actions','selected_index'):
                gate._equal(arrays[key][i],inputs[key],'original response '+key)
        gate._pair(saved['journal'])
        pools.append(dict(pool=pool,report=saved,report_pair=row['report'],arrays=arrays))
    gate._exact(report['models'],models,'all twelve top-level/per-pool model access evidence')
    gate._pair(report_pair)
    return dict(report=report,context=ctx,pools=pools,parent_ids=ctx['parent_ids'],
        scope='Authenticated source/runtime/input/model access metadata and saved raw truth/control consistency. '
              'Free latent values/native parity remain producer evidence; no independent model inference or q_g scoring.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('calibration','sealed-test'))
    parser.add_argument('--binding',required=True);parser.add_argument('--binding-sha256',required=True)
    args=parser.parse_args();ctx=load_context(args.binding,args.binding_sha256)
    require(ctx['split']==('calibration' if args.mode=='calibration' else 'test'),'CLI path and actual split differ')
    report=produce(ctx);print(json.dumps({k:report[k] for k in ('status','split','count','completed_cases')}))


if __name__=='__main__':main()
