"""DRAFT two-stage orchestration for S2 scalar regression artifacts.

No model loader, response generator, seed/tolerance default or scientific freeze.
The future upstream input/source/access schemas are described in the runbook.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import regression as production
import independent_regression as independent

HERE = Path(__file__).resolve().parent
CANONICAL_STRATA = [f'pool{p}/{o}' for p in range(3) for o in ('decoded_teacher','physical_labels')]
ROLES = {'calibration':('calibration_regression','calibration',256), 'test_probe':('probe_test','test',512)}
require = independent.require


def descriptor(path):
    path = Path(path).resolve()
    return dict(path=str(path),sha256=independent.file_sha(path))


def document(pair,name):
    path=independent.checked_file(pair,name)
    require(path.suffix == '.json',name+': metadata JSON required')
    return independent.read_json(path)


def write_json(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False,
                  default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (_ for _ in ()).throw(TypeError()))
        f.write('\n')


def exact_fields(value,keys,name):
    require(isinstance(value,dict) and set(value)==set(keys),name+': incomplete or unexpected fields')


def check_rosters(rosters):
    expected={f'{split}_{role}_ids':n for split,n in [('calibration',256),('test',512)] for role in ['recipient','donor']}
    exact_fields(rosters,expected,'Expected parent rosters')
    all_ids=[]
    for name,n in expected.items():
        row=rosters[name]
        require(isinstance(row,list) and len(row)==n and all(type(v) is int and 0<=v<2**63 for v in row)
                and len(set(row))==n,name+': invalid ordered parent roster')
        all_ids.extend(row)
    require(len(set(all_ids))==len(all_ids),'Four parent roles must be disjoint')


def load_context(binding_pair):
    """Metadata only; no scalar or response NPZ opened by this function."""
    b=document(binding_pair,'Input binding')
    exact_fields(b,['status','protocol','sources','calibration','test_probe','data_isolation',
                    'response_contract','expected_parent_ids'],'Input binding')
    require(b['status']=='S2_REGRESSION_INPUT_BINDINGS_ACCEPTED','Unaccepted regression input binding')
    protocol=document(b['protocol'],'Protocol')
    require(protocol.get('status')=='S2_SCIENTIFIC_PROTOCOL_FROZEN','A future actual frozen protocol is required')
    spec=protocol['regression']
    exact_fields(spec,['calibration_count','test_count','stratum_ids','raw_columns','ordinary_basis_columns',
                       'ridge_lambda','production_verification_tolerances','comparison_tolerances','statistical_settings'],
                 'Regression protocol')
    require(spec['calibration_count']==256 and spec['test_count']==512 and spec['stratum_ids']==CANONICAL_STRATA and
            spec['raw_columns']==list(production.RAW_COLUMNS) and spec['ordinary_basis_columns']==list(production.BASIS_COLUMNS)
            and spec['ridge_lambda']==.01,'Changed fixed regression population, feature order or estimator')
    for key in ['production_verification_tolerances','comparison_tolerances']:
        independent.explicit_tolerances(spec[key],key)
    settings=spec['statistical_settings']
    require(settings==independent.settings(settings['bootstrap_seed']),'Changed or missing explicit bootstrap settings')
    sources=document(b['sources'],'Sources')
    require(sources.get('status')=='S2_REGRESSION_SOURCES_FROZEN' and
            sources.get('protocol_sha256')==b['protocol']['sha256'],'Source/protocol mismatch')
    required={'finalize_regression.py','regression.py','independent_regression.py'}
    exact_fields(sources['files'],required,'Scalar source closure')
    for name in sorted(required):
        p=independent.checked_file(sources['files'][name],'Source '+name)
        require(p.resolve()==HERE/name,'Bound source does not match executing module: '+name)
    rosters=b['expected_parent_ids'];check_rosters(rosters)
    isolation=document(b['data_isolation'],'Data isolation')
    require(isolation.get('status')=='PASS_COMPLETE_S2_DATA_ISOLATION' and
            isolation.get('protocol_sha256')==b['protocol']['sha256'] and
            isolation.get('sources_sha256')==b['sources']['sha256'] and
            isolation.get('expected_parent_ids')==rosters,'Incomplete or unrelated accepted data isolation')
    response_contract=document(b['response_contract'],'Predeclared response contract')
    require(response_contract.get('status')=='S2_TEST_RESPONSE_CONTRACT_FROZEN' and
            response_contract.get('protocol_sha256')==b['protocol']['sha256'] and
            response_contract.get('role')=='test_response' and response_contract.get('count')==512,
            'Unfrozen or wrong-role predeclared response contract')
    # Authenticate only metadata here. Never traverse its runtime/models or response files.
    receipts={}
    for key,(role,split,n) in ROLES.items():
        row=document(b[key],key+' scalar receipt')
        expected=dict(status='PASS_COMPLETE_S2_SCALAR_INPUTS',role=role,split=split,count=n,
             stratum_ids=spec['stratum_ids'],protocol_sha256=b['protocol']['sha256'],
             sources_sha256=b['sources']['sha256'],complete_population_accepted=True)
        require(all(row.get(k)==v for k,v in expected.items()) and row.get('complete_population_accepted') is True,
                key+': wrong role/source or incomplete scalar population')
        upstream=document(row['upstream_acceptance'],key+' upstream acceptance')
        expected.update(status='PASS_COMPLETE_S2_SCALAR_POPULATION',
             recipient_ids=rosters[split+'_recipient_ids'],donor_ids=rosters[split+'_donor_ids'])
        require(all(upstream.get(k)==v for k,v in expected.items()) and upstream.get('complete_population_accepted') is True,
                key+': wrong upstream role/source/population/ordering')
        # The externally accepted upstream receipt binds the exact generated scalar bytes.
        require(upstream.get('scalar_arrays')==row['arrays'],key+': upstream scalar-array binding mismatch')
        receipts[key]=row
    expected_bindings=dict(protocol_sha256=b['protocol']['sha256'],calibration_inputs_sha256=b['calibration']['sha256'],
         test_probe_inputs_sha256=b['test_probe']['sha256'],data_isolation_sha256=b['data_isolation']['sha256'],
         response_contract_sha256=b['response_contract']['sha256'])
    return dict(binding=b,binding_pair=binding_pair,spec=spec,receipts=receipts,expected_bindings=expected_bindings)


def load_scalar(ctx,key):
    _,split,n=ROLES[key]
    pair=ctx['receipts'][key]['arrays']
    path=independent.checked_file(pair,key+' arrays')
    names={'ordinary','signed_g','recipient_ids','donor_ids'}|({'response'} if key=='calibration' else set())
    # Inspect the archive member roster before decoding any payload. An injected
    # test-response member is rejected without opening its array bytes.
    with np.load(path,allow_pickle=False) as archive:
        require(len(archive.files)==len(names) and set(archive.files)==names,key+' NPZ: incomplete or unexpected fields')
        a={name:archive[name] for name in sorted(names)}
    independent.finite64(a['ordinary'],(n,6,4),key+' ordinary')
    independent.finite64(a['signed_g'],(n,6),key+' signed G')
    require(np.all(a['ordinary']>=0),key+': ordinary errors/dose must be nonnegative')
    if key=='calibration':independent.finite64(a['response'],(256,6),'calibration response')
    for role in ['recipient','donor']:
        ids=independent.ids(a[role+'_ids'],n,key+' '+role+' IDs')
        require(ids.tolist()==ctx['binding']['expected_parent_ids'][split+'_'+role+'_ids'],key+': parent order mismatch')
    return a


def seal(binding_pair,output):
    out=Path(output).resolve();require(not out.exists(),'Output directory already exists; no overwrite/refit')
    ctx=load_context(binding_pair)
    c,p=load_scalar(ctx,'calibration'),load_scalar(ctx,'test_probe')
    spec=ctx['spec'];tol=spec['production_verification_tolerances'];settings=spec['statistical_settings']
    fit=production.fit_calibration(c['ordinary'],c['signed_g'],c['response'],recipient_ids=c['recipient_ids'],
         donor_ids=c['donor_ids'],stratum_ids=spec['stratum_ids'],verification_atol=tol['atol'],verification_rtol=tol['rtol'])
    predictions=production.predict_test(fit,p['ordinary'],p['signed_g'],recipient_ids=p['recipient_ids'],
         donor_ids=p['donor_ids'],stratum_ids=spec['stratum_ids'])
    # No response loader or path exists in this stage's signature.
    lock=production.seal_test_predictions(out,fit,predictions,bindings=ctx['expected_bindings'],
         bootstrap_seed=settings['bootstrap_seed'],bit_generator=settings['bit_generator'],quantile_method=settings['quantile_method'])
    receipt=dict(status='SEALED_COMPLETE_S2_REGRESSION_PREDICTIONS',input_binding=binding_pair,prediction_lock=lock,
         sources_sha256=ctx['binding']['sources']['sha256'],protocol_sha256=ctx['binding']['protocol']['sha256'],
         stratum_ids=spec['stratum_ids'],calibration_count=256,test_count=512,regression_models=2,
         test_response_opened_by_this_stage=False,source_sha256=independent.file_sha(__file__))
    path=out/'ORCHESTRATION_SEAL.json';write_json(path,receipt)
    return descriptor(path)


def verify_seal(ctx,pair):
    sealed=document(pair,'Orchestration seal')
    require(sealed.get('status')=='SEALED_COMPLETE_S2_REGRESSION_PREDICTIONS' and
            sealed.get('input_binding')==ctx['binding_pair'] and sealed.get('regression_models')==2 and
            sealed.get('calibration_count')==256 and sealed.get('test_count')==512 and
            sealed.get('sources_sha256')==ctx['binding']['sources']['sha256'] and
            sealed.get('protocol_sha256')==ctx['binding']['protocol']['sha256'] and
            sealed.get('stratum_ids')==ctx['spec']['stratum_ids'] and
            sealed.get('test_response_opened_by_this_stage') is False and
            sealed.get('source_sha256')==independent.file_sha(__file__),'Unrelated/incomplete orchestration seal')
    lock_pair=sealed['prediction_lock']
    independent.checked_file(lock_pair,'Prediction lock')
    lock,pred,meta=production.verify_prediction_seal(lock_pair['path'],lock_pair['sha256'],expected_bindings=ctx['expected_bindings'])
    require(lock['statistical_settings']==ctx['spec']['statistical_settings'] and
            lock['stratum_ids']==ctx['spec']['stratum_ids'],'Sealed statistical configuration/order changed')
    for role in ['recipient','donor']:
        require(pred['test_'+role+'_ids'].tolist()==ctx['binding']['expected_parent_ids']['test_'+role+'_ids'],
                'Sealed parent roster mismatch')
    # Rebind saved raw regression inputs to the accepted scalar population, not
    # merely to self-consistent prediction/calibration content hashes.
    calibration=independent.load_npz(Path(lock_pair['path']).parent/'calibration.npz')
    for key,record,prefix in [('calibration',calibration,'calibration'),('test_probe',pred,'test')]:
        original=load_scalar(ctx,key)
        fields={'ordinary':prefix+'_ordinary','signed_g':prefix+'_signed_g',
                'recipient_ids':prefix+'_recipient_ids','donor_ids':prefix+'_donor_ids'}
        if key=='calibration':fields['response']='calibration_response'
        for raw,saved in fields.items():
            require(np.array_equal(original[raw],record[saved]),'Sealed inputs differ from accepted scalar population: '+saved)
    return lock_pair


def evaluate(binding_pair,seal_pair,response_pair,access_pair,output):
    out=Path(output).resolve();require(not out.exists(),'Output directory already exists; no overwrite/reselection')
    ctx=load_context(binding_pair)
    lock_pair=verify_seal(ctx,seal_pair)
    # The response descriptor is separate CLI input, and is not opened until the seal verifies.
    access=document(access_pair,'Accepted test-response access')
    expected=dict(status='PASS_COMPLETE_S2_TEST_RESPONSE_ACCESS',protocol_sha256=ctx['binding']['protocol']['sha256'],
         response_contract_sha256=ctx['binding']['response_contract']['sha256'],
         prediction_lock_sha256=lock_pair['sha256'],response_receipt_sha256=response_pair['sha256'],
         count=512,stratum_ids=ctx['spec']['stratum_ids'],
         recipient_ids=ctx['binding']['expected_parent_ids']['test_recipient_ids'],
         donor_ids=ctx['binding']['expected_parent_ids']['test_donor_ids'],prediction_seal_verified_before_response_inference=True)
    require(all(access.get(k)==v for k,v in expected.items()) and
            access.get('prediction_seal_verified_before_response_inference') is True,'Response access/contract/population binding mismatch')
    response=document(response_pair,'Actual held-out response receipt')
    expected_response=dict(status='ACCEPTED_S2_HELDOUT_RESPONSE',count=512,
         protocol_sha256=ctx['binding']['protocol']['sha256'],response_contract_sha256=ctx['binding']['response_contract']['sha256'],
         prediction_lock_sha256=lock_pair['sha256'],stratum_ids=ctx['spec']['stratum_ids'])
    require(all(response.get(k)==v for k,v in expected_response.items()),'Actual response cross-binding mismatch')
    stats=production.evaluate_response_after_seal(lock_pair['path'],lock_pair['sha256'],
         expected_bindings=ctx['expected_bindings'],response_loader=lambda:dict(receipt_path=response_pair['path'],receipt_sha256=response_pair['sha256']))
    out.mkdir(parents=True,exist_ok=False)
    statistics=out/'production_statistics.json';write_json(statistics,stats)
    spec=ctx['spec']
    verification_binding=dict(schema='s2_independent_regression_binding_v1_draft',prediction_lock=lock_pair,
         response_receipt=response_pair,production_statistics=descriptor(statistics),expected_bindings=ctx['expected_bindings'],
         expected_stratum_ids=spec['stratum_ids'],expected_parent_ids=ctx['binding']['expected_parent_ids'],
         expected_statistical_settings=spec['statistical_settings'],comparison_tolerances=spec['comparison_tolerances'],
         production_verification_tolerances=spec['production_verification_tolerances'])
    verifier_binding=out/'independent_binding.json';write_json(verifier_binding,verification_binding)
    cmd=[sys.executable,str(HERE/'independent_regression.py'),'--binding',str(verifier_binding),
         '--binding-sha256',independent.file_sha(verifier_binding),'--output',str(out/'independent')]
    run=subprocess.run(cmd,capture_output=True,text=True,check=False)
    (out/'independent_stdout.txt').write_text(run.stdout)
    (out/'independent_stderr.txt').write_text(run.stderr)
    require(run.returncode==0,'Independent verifier rejected the complete result; outputs retained without acceptance')
    report_pair=descriptor(out/'independent/report.json');report=document(report_pair,'Independent report')
    require(report.get('status')=='PASS_S2_INDEPENDENT_REGRESSION_DRAFT' and
            report.get('binding_sha256')==independent.file_sha(verifier_binding) and
            report.get('verifier_source_sha256')==independent.file_sha(HERE/'independent_regression.py') and
            report.get('prediction_lock_sha256')==lock_pair['sha256'] and
            report.get('production_statistics_sha256')==independent.file_sha(statistics),'Independent completion receipt mismatch')
    independent.checked_file(dict(path=str(out/'independent/recomputed.npz'),sha256=report['recomputed_arrays_sha256']),
                             'Independent recomputed arrays')
    receipt=dict(status='PASS_COMPLETE_S2_REGRESSION_AND_INDEPENDENT_STATISTICS',input_binding=binding_pair,
         orchestration_seal=seal_pair,prediction_lock=lock_pair,response_receipt=response_pair,response_access_acceptance=access_pair,
         production_statistics=descriptor(statistics),independent_binding=descriptor(verifier_binding),independent_report=report_pair,
         count=512,stratum_ids=spec['stratum_ids'],source_sha256=independent.file_sha(__file__),
         scientific_protocol_frozen_by_this_module=False,
         scope='Saved scalar arrays and externally accepted provenance; no model replay, response generation or proof of external prior nonexposure.')
    final=out/'ACCEPTANCE.json';write_json(final,receipt)
    return descriptor(final)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='stage',required=True)
    for name in ['seal','evaluate']:
        c=sub.add_parser(name)
        c.add_argument('--inputs',required=True);c.add_argument('--inputs-sha256',required=True);c.add_argument('--output',required=True)
        if name=='evaluate':
            for option in ['seal','response-receipt','response-access']:
                c.add_argument('--'+option,required=True);c.add_argument('--'+option+'-sha256',required=True)
    args=p.parse_args()
    binding=dict(path=args.inputs,sha256=args.inputs_sha256)
    if args.stage=='seal':result=seal(binding,args.output)
    else:result=evaluate(binding,dict(path=args.seal,sha256=args.seal_sha256),
         dict(path=args.response_receipt,sha256=args.response_receipt_sha256),
         dict(path=args.response_access,sha256=args.response_access_sha256),args.output)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
