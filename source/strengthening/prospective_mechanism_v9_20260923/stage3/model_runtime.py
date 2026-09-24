"""Draft genuine S3 C-model roots, observed donor encoding and fixed32 suffixes.

No Torch/model/data load at import. No S2 public role is reused or fabricated.
Actual execution requires a frozen S3 protocol, complete S2 receipt and genuine
complete512-recipient +512-donor S3 admissions. This is latent arithmetic only;
whole-family projection admission and downstream outcome sealing remain external.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

PHASE=Path(__file__).resolve().parents[1]
ROOT=PHASE.parents[1]
OBJECTIVES=('decoded_teacher','physical_labels')
CONDITIONS=('T0','T1')
RUNTIME_STATUS='S3_MODEL_RUNTIME_CONTRACT_FROZEN'
ADMISSION_STATUS='S3_FIXED_COMMON_PREFIX_CONTEXTS_ACCEPTED'
MANIFEST_STATUS='S3_COMPLETE_COMMON_PREFIX_ROLE_INPUT_MANIFEST'
S2_STATUS='PASS_COMPLETE_S2_REGRESSION_AND_INDEPENDENT_STATISTICS'
LEGACY_PATH=PHASE/'stage2/model_runtime.py'
_LEGACY=None
CASE_SCHEMA={
 'history_pixels':((3,224,224,3),'uint8'),'goal_pixels':((224,224,3),'uint8'),
 'current_pixels':((224,224,3),'uint8'),'prefix':((10,2),'float32'),
 'executed_actions':((5,2),'float32'),'suffix_actions':((32,20,2),'float32'),
 'source_first5_population':((300,5,2),'float32'),'source_population_actions':((300,25,2),'float32'),
 'source_candidate_indices':((32,),'int64'),
 **{k:((),'int64') for k in ['index','seed','reference_route','source_selected_index','source_selected_iteration']}}


def require(ok,message):
    if not ok:raise ValueError(message)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def array_sha(value):
    a=np.asarray(value)
    h=hashlib.sha256(json.dumps([list(a.shape),a.dtype.str],separators=(',',':')).encode())
    h.update(a.tobytes(order='C'));return h.hexdigest()


def bound_path(record):
    require(isinstance(record,dict) and set(record)=={'path','sha256'},'Exact path/SHA pair required')
    p=Path(record['path']);digest=record['sha256']
    require(p.is_absolute() and isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
            'Absolute path and exact SHA256 required')
    require(p.is_file() and file_sha(p)==digest,'Changed bound S3 artifact: '+str(p))
    return p.resolve()


def document(record):
    return json.loads(bound_path(record).read_text())


def legacy():
    global _LEGACY
    if _LEGACY is None:
        spec=importlib.util.spec_from_file_location('_s3_original_C_model_constructor',LEGACY_PATH)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        _LEGACY=module
    require(Path(_LEGACY.__file__).resolve()==LEGACY_PATH.resolve(),'Wrong low-level model constructor')
    return _LEGACY


def describe_fixed_model(pool,objective,condition):
    """Existing audited metadata only; no checkpoint deserialization."""
    return legacy().describe_fixed_model(pool,objective,condition)


def validate_protocol(p,s2_pair):
    require(p.get('status')=='S3_SCIENTIFIC_PROTOCOL_FROZEN' and p.get('scientific_protocol_frozen') is True and
        p.get('stage')=='S3_matched_post_prefix_decision','S3 draft is not execution authorization')
    population=p['population'];shared=p['shared_prefix_decision']
    expected=dict(recipient_count=512,donor_count=512,pools=[0,1,2],groups=[0,2,4],objectives=list(OBJECTIVES),
        conditions=list(CONDITIONS),models=12,model_pairs=6,branches=['free','actual','donor','reset'])
    require(all(population.get(k)==v for k,v in expected.items()),'Fixed S3 model/population roster changed')
    expected=dict(candidate_count=32,native_root_candidate_count=300,common_executed_actions=5,historical_actions=10,
        history_tokens=3,suffix_actions=20,scoring_horizon=25,suffix_prediction_horizons=[10,15,20,25],
        reference_policy_rows=[0,8,16],legacy_reference_route_ids=[0,32,64],same_observed_image_for_all_suffixes=True)
    require(all(shared.get(k)==v for k,v in expected.items()),'Fixed shared-prefix arithmetic changed')
    prior=p['prior_stage_gate']
    require(prior.get('required_status')==S2_STATUS and prior.get('actual_receipt')==s2_pair and
            prior.get('generation_inference_and_effect_access_before_gate') is False,'Actual S2 completion gate absent')
    require(document(s2_pair).get('status')==S2_STATUS,'S2 has not completed')


def validate_admission(receipt,manifest,*,role,protocol_sha):
    require(role in ('recipient','donor'),'Wrong genuine S3 role')
    expected=dict(status=ADMISSION_STATUS,protocol_sha256=protocol_sha,role=role,count=512)
    require(all(receipt.get(k)==v for k,v in expected.items()),'Wrong S3 role/count/protocol admission')
    ids=receipt.get('parent_ids',[])
    require(len(ids)==512 and len(set(ids))==512 and all(type(x) is int and x>=0 for x in ids),
            'Incomplete/duplicate S3 parent roster')
    expected=dict(status=MANIFEST_STATUS,protocol_sha256=protocol_sha,role=role,count=512)
    require(all(manifest.get(k)==v for k,v in expected.items()),'Wrong S3 common-prefix input manifest')
    require(manifest.get('parent_ids')==ids,'Manifest and admission parent order differ')
    routes=manifest.get('routes',[])
    require(len(routes)==3 and {r['pool'] for r in routes}=={0,1,2},'All three accepted input pools required')
    cases={}
    for route in routes:
        pool=route['pool']
        require(type(pool) is int and route.get('group')==2*pool and route.get('policy_index')==8*pool and
            route.get('legacy_reference_route')==32*pool,'Wrong pool/reference-policy mapping')
        rows=route.get('cases',[])
        require(len(rows)==512 and [r['index'] for r in rows]==list(range(512)) and
                [r['seed'] for r in rows]==ids,'Missing/reordered complete S3 input case roster')
        for row in rows:
            # Only scorer-input files are carried into model admission. Complete
            # physical outcome verification belongs to the separate producer gate.
            require(set(row)=={'index','seed','scorer_input'},'Outcome/provenance files must not enter runtime case rows')
            rec=row['scorer_input'];p=Path(rec.get('path',''))
            require(set(rec)=={'path','sha256'} and p.is_absolute() and isinstance(rec['sha256'],str) and
                len(rec['sha256'])==64 and all(c in '0123456789abcdef' for c in rec['sha256']), 'Unbound input case')
            cases[pool,row['index']]=copy.deepcopy(row)
    return dict(role=role,count=512,parent_ids=tuple(ids),cases=cases,input_manifest=receipt['input_manifest'])


def runtime_admission(contract_pair,role):
    contract=document(contract_pair)
    require(contract.get('status')==RUNTIME_STATUS and contract.get('role')==role and
            contract.get('source_root')==str(ROOT),'Genuine frozen S3 runtime contract required')
    protocol=document(contract['protocol']);validate_protocol(protocol,contract['s2_completion'])
    admissions=contract.get('admissions',{})
    require(set(admissions)=={'recipient','donor'},'Both complete S3 role admissions required before model access')
    admitted={};immutable=[contract_pair,contract['protocol'],contract['s2_completion']]
    for which,pair in admissions.items():
        receipt=document(pair);manifest=document(receipt['input_manifest'])
        admitted[which]=validate_admission(receipt,manifest,role=which,protocol_sha=contract['protocol']['sha256'])
        require(receipt.get('s2_completion')==contract['s2_completion'],'S3 admission lacks same actual S2 gate')
        immutable.extend([pair,receipt['input_manifest']])
    require(not set(admitted['recipient']['parent_ids']) & set(admitted['donor']['parent_ids']), 'S3 roles overlap')
    old=legacy();files=contract.get('files',{})
    required=[Path(__file__),LEGACY_PATH,*old.METADATA.values(),
        ROOT/'strengthening/adapters/ac_rollout.py',ROOT/'scripts/visual/factorial_model.py',ROOT/'scripts/visual/lewm_adapter.py',
        ROOT/'scripts/visual/adaptation_freeze.py',ROOT/'scripts/visual/evaluation_precision.py',
        ROOT/'scripts/visual/image_planner_cost.py',ROOT/'scripts/visual/feedback_suffix_rollout.py',
        ROOT/'scripts/visual/score_feedback_ranking.py',ROOT/'scripts/visual/run_adaptation_checkpoint_gate.py',
        old.BASE/'releases/visual-v1/official/jepa.py',old.BASE/'releases/visual-v1/official/module.py']
    resolved={}
    for name,digest in files.items():
        p=Path(name);p=p if p.is_absolute() else ROOT/p
        require(p.suffix not in ('.pt','.npz','.npy','.pkl','.pickle'),'Source closure cannot smuggle tensor files')
        p=p.resolve();require(str(p) not in resolved,'Aliased runtime source')
        bound_path(dict(path=str(p),sha256=digest));resolved[str(p)]=digest
    require(all(str(p.resolve()) in resolved for p in required),'Missing complete model/native/suffix source closure')
    immutable.extend(dict(path=p,sha256=d) for p,d in resolved.items())
    models=[describe_fixed_model(pool,obj,condition) for pool in range(3) for obj in OBJECTIVES for condition in CONDITIONS]
    require(contract.get('models')==models,'All twelve pre-existing C models must match exactly')
    # All complete role metadata is accepted before opening one raw file/model.
    admission=admitted[role]
    admission['immutable_inputs']=immutable
    return contract,admission,models


def _load(pool,objective,condition,role,contract_pair,device):
    legacy()._identity(pool,objective,condition,CONDITIONS)
    c,admission,models=runtime_admission(contract_pair,role)
    spec=next(x for x in models if (x['pool'],x['objective'],x['condition'])==(pool,objective,condition))
    access=dict(role='s3_'+role,runtime_contract=contract_pair,input_admission=c['admissions'][role],
        both_role_admissions=c['admissions'],protocol=c['protocol'],s2_completion=c['s2_completion'],
        permitted_operation='recipient_roots_and_suffixes' if role=='recipient' else 'encode_own_executed_prefix_observation_only')
    return legacy()._load_bound_model(spec,admission,access,device)


def load_recipient_model(pool,objective,condition,*,runtime_contract,runtime_contract_sha256,device='cuda'):
    return _load(pool,objective,condition,'recipient',dict(path=str(Path(runtime_contract).resolve()),sha256=runtime_contract_sha256),device)


def load_donor_encoder(pool,*,runtime_contract,runtime_contract_sha256,device='cuda'):
    """No objective/condition selector: fixed shared T0 image encoder only."""
    return _load(pool,'decoded_teacher','T0','donor',dict(path=str(Path(runtime_contract).resolve()),sha256=runtime_contract_sha256),device)


def _handle(handle,role):
    require(handle.access_receipt.get('role')=='s3_'+role and handle.admission.get('role')==role and
        handle.admission.get('count')==512,'Wrong genuine S3 runtime role')
    if role=='donor':
        require(handle.spec['condition']=='T0' and handle.spec['objective']=='decoded_teacher' and
            handle.access_receipt.get('permitted_operation')=='encode_own_executed_prefix_observation_only', 'Donor is encoder-only')
    else:
        require(handle.access_receipt.get('permitted_operation')=='recipient_roots_and_suffixes','Recipient operation absent')
    require(value_sha(handle.normalization)==handle.normalization_sha256,'Action normalization changed')
    for r in handle.admission['immutable_inputs']:bound_path(r)


def validate_case(arrays,*,pool,index,seed,role):
    require(role in ('recipient','donor'),'Wrong S3 input role')
    schema={k:v for k,v in CASE_SCHEMA.items() if role=='recipient' or k not in ('suffix_actions','source_candidate_indices')}
    require(set(arrays)==set(schema),'Missing/extra raw fields; outcomes are forbidden')
    for k,(shape,dtype) in schema.items():
        x=np.asarray(arrays[k]);require(x.shape==shape and x.dtype==np.dtype(dtype) and np.isfinite(x).all(), 'Bad scorer-input field: '+k)
    require(int(arrays['index'])==index and int(arrays['seed'])==seed and int(arrays['reference_route'])==32*pool,
            'Scorer-input parent/pool differs')
    selected=int(arrays['source_selected_index']);iteration=int(arrays['source_selected_iteration'])
    require(0<=selected<300 and 0<=iteration<30,'Wrong fixed selected index/iteration')
    pop=arrays['source_population_actions']
    require(np.array_equal(arrays['source_first5_population'],pop[:,:5]) and
        np.array_equal(arrays['executed_actions'],pop[selected,:5]),'Executed prefix is not the selected native300 root')
    if role=='donor':return
    suffix=arrays['suffix_actions'];ix=arrays['source_candidate_indices']
    require(ix[0]==selected and ix[1]==-1 and len(set(ix[ix>=0].tolist()))==31 and
            np.all((ix[2:]>=0)&(ix[2:]<300)) and selected not in ix[2:],'Wrong fixed32 source candidates')
    require(np.array_equal(suffix[0],pop[selected,5:]) and not np.count_nonzero(suffix[1]) and
            np.array_equal(suffix[2:],pop[ix[2:],5:]),'Suffix actions differ from fixed source proposals')


def load_case(handle,*,case_index,input_pair,role):
    _handle(handle,role)
    require(type(case_index) is int and 0<=case_index<512,'Invalid full-population case index')
    row=handle.admission['cases'][handle.spec['pool'],case_index]
    require(row['scorer_input']==input_pair,'Input file is outside accepted S3 route')
    with np.load(bound_path(input_pair),allow_pickle=False) as z:
        require(len(z.files)==len(set(z.files)),'Repeated NPZ members')
        arrays=dict(z)
    validate_case(arrays,pool=handle.spec['pool'],index=case_index,seed=row['seed'],role=role)
    return arrays


def _native_dependencies():
    torch,_,_,verify,precision,state_hash,_=legacy()._torch_dependencies()
    from image_planner_cost import ImagePlannerCost
    from feedback_suffix_rollout import rollout_suffixes
    return torch,verify,precision,state_hash,ImagePlannerCost,rollout_suffixes


def _precision(torch,precision):
    precision()
    require(torch.get_default_dtype()==torch.float32 and not torch.is_autocast_enabled(),
            'S3 requires FP32 without active autocast')


def _guard_model(handle,verify,state_hash):
    verify(handle.model,handle.boundary)
    require(state_hash(handle.model)==handle.model_sha256 and value_sha(handle.normalization)==handle.normalization_sha256,
            'C model/encoder/action normalization changed')


def _encode_current(torch,handle,pixel):
    device=next(handle.model.parameters()).device
    mean=torch.tensor([.485,.456,.406],device=device,dtype=torch.float32)[None,None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device,dtype=torch.float32)[None,None,:,None,None]
    x=torch.as_tensor(pixel,device=device).permute(2,0,1)[None,None].float()/255.
    return handle.model.encode({'pixels':(x-mean)/std})['emb'][0,0]


def _arrays(values,shapes):
    output={}
    for k,shape in shapes.items():
        a=values[k].detach().cpu().numpy().copy()
        require(a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all(),'Invalid FP32 native output: '+k)
        a.setflags(write=False);output[k]=a
    return output


def _evidence(handle,input_pair,arrays):
    return dict(source_sha256=file_sha(__file__),input=input_pair,model_spec_sha256=value_sha(handle.spec),
        model_state_sha256=handle.model_sha256,normalization_sha256=handle.normalization_sha256,
        access_receipt=copy.deepcopy(handle.access_receipt),array_sha256={k:array_sha(v) for k,v in arrays.items()})


def native_recipient_root(handle,*,case_index,input_pair):
    case=load_case(handle,case_index=case_index,input_pair=input_pair,role='recipient')
    torch,verify,precision,state_hash,Planner,_=_native_dependencies();_precision(torch,precision);_guard_model(handle,verify,state_hash)
    device=next(handle.model.parameters()).device;before={k:array_sha(v) for k,v in case.items()}
    with torch.inference_mode():
        cost=Planner(handle.model,case['history_pixels'],case['goal_pixels'],case['prefix'],handle.normalization,None,'latent')
        actions=torch.as_tensor(case['source_population_actions'],device=device)
        normalized=cost.normalized_actions(actions);saved=normalized.clone();initial=cost.initial.clone()
        encoded=handle.model.action_encoder(normalized)
        history=cost.initial[None].expand(300,-1,-1).clone()
        roots=handle.model.predict(history,encoded[:,:3])[:,-1]
        repeated=handle.model.predict(history,encoded[:,:3])[:,-1]
        torch.testing.assert_close(roots,repeated,rtol=0,atol=0)
        observed=_encode_current(torch,handle,case['current_pixels'])
        selected=int(case['source_selected_index'])
        torch.testing.assert_close(cost.initial,initial,rtol=0,atol=0)
        torch.testing.assert_close(cost.normalized_actions(actions),saved,rtol=0,atol=0)
        arrays=_arrays(dict(predicted=roots[selected],observed=observed,initial_history=cost.initial,goal_token=cost.goal),
            dict(predicted=(192,),observed=(192,),initial_history=(3,192),goal_token=(192,)))
    require(before=={k:array_sha(v) for k,v in case.items()},'Native root mutated input arrays')
    _guard_model(handle,verify,state_hash)
    return dict(status='S3_NATIVE300_SHARED_PREFIX_ROOT_COMPUTED',arrays=arrays,evidence=_evidence(handle,input_pair,arrays),
        checks=dict(native_population=300,selected_executed_prefix_exact=True,repeated_root_exact=True,
                    one_observed_token=True,model_and_actions_unchanged=True),
        scope='Selected first prediction under fixed native300 action-encoder batch; no suffix outcomes/readout/candidate choice.')


def encode_donor_current(handle,*,case_index,input_pair):
    case=load_case(handle,case_index=case_index,input_pair=input_pair,role='donor')
    torch,verify,precision,state_hash,_,_=_native_dependencies();_precision(torch,precision);_guard_model(handle,verify,state_hash)
    with torch.inference_mode():
        arrays=_arrays(dict(observed=_encode_current(torch,handle,case['current_pixels'])),dict(observed=(192,)))
    _guard_model(handle,verify,state_hash)
    return dict(status='S3_DONOR_CURRENT_OBSERVATION_ENCODED',arrays=arrays,evidence=_evidence(handle,input_pair,arrays),
        checks=dict(own_executed_prefix=True,one_observed_token=True,dynamics_or_action_encoder_called=False),
        scope='Own donor current image only; global donor assignment and shared encoder equality are checked by population producer.')


def validate_root_record(handle,input_pair,record):
    require(record.get('status')=='S3_NATIVE300_SHARED_PREFIX_ROOT_COMPUTED','Accepted native root record required')
    e=record['evidence'];arrays=record['arrays']
    require(e==_evidence(handle,input_pair,arrays),'Root/model/input/source identity drift')
    require(record.get('checks')==dict(native_population=300,selected_executed_prefix_exact=True,repeated_root_exact=True,
        one_observed_token=True,model_and_actions_unchanged=True),'Incomplete native root checks')
    shapes=dict(predicted=(192,),observed=(192,),initial_history=(3,192),goal_token=(192,))
    require(set(arrays)==set(shapes),'Different root fields')
    for k,shape in shapes.items():
        x=arrays[k];require(x.shape==shape and x.dtype==np.float32 and np.isfinite(x).all(),'Bad root array: '+k)


def suffix_rollouts(handle,*,case_index,input_pair,root_record,replacements):
    """Four latent branches, same32 futures; corrected roots need external family gate."""
    case=load_case(handle,case_index=case_index,input_pair=input_pair,role='recipient')
    validate_root_record(handle,input_pair,root_record)
    require(set(replacements)=={'actual','donor'} and all(np.shape(v)==(192,) and np.asarray(v).dtype==np.float32 and
        np.isfinite(v).all() for v in replacements.values()),'Two shared FP32 correction roots required, never candidate-specific')
    torch,verify,precision,state_hash,_,rollout=_native_dependencies();_precision(torch,precision);_guard_model(handle,verify,state_hash)
    device=next(handle.model.parameters()).device;roots=root_record['arrays']
    before={k:array_sha(v) for k,v in case.items()};r_before={k:array_sha(v) for k,v in replacements.items()}
    with torch.inference_mode():
        initial=torch.as_tensor(np.array(roots['initial_history']),device=device)
        # The two blocks immediately preceding the decision, NOT prefix[:10].
        past=torch.as_tensor(np.concatenate([case['prefix'][-5:],case['executed_actions']]),device=device)
        suffix=torch.as_tensor(case['suffix_actions'],device=device)
        mean=torch.tensor(handle.normalization['mean'],device=device,dtype=torch.float32)
        std=torch.tensor(handle.normalization['std'],device=device,dtype=torch.float32)
        feedback={'free':roots['predicted'],'actual':replacements['actual'],'donor':replacements['donor'],'reset':roots['observed']}
        values={k:rollout(handle.model,initial,torch.as_tensor(np.array(z),device=device),past,suffix,mean,std) for k,z in feedback.items()}
        identity=rollout(handle.model,initial,torch.as_tensor(np.array(roots['predicted']),device=device),past,suffix,mean,std)
        torch.testing.assert_close(identity,values['free'],rtol=0,atol=0)
        arrays=_arrays(values,{k:(4,32,192) for k in feedback})
    require(before=={k:array_sha(v) for k,v in case.items()} and r_before=={k:array_sha(v) for k,v in replacements.items()},
            'Suffix propagation mutated inputs')
    validate_root_record(handle,input_pair,root_record);_guard_model(handle,verify,state_hash)
    return dict(status='S3_FIXED32_ALL_BRANCH_LATENT_SUFFIXES_COMPUTED',arrays=arrays,evidence=_evidence(handle,input_pair,arrays),
        root_evidence=copy.deepcopy(root_record['evidence']),replacement_sha256=r_before,
        checks=dict(candidate_count=32,aligned_past_actions=10,one_feedback_per_branch=True,free_identity_exact=True,
                    model_and_normalization_unchanged=True),horizons=[10,15,20,25],
        native300_vs_fixed32_bitwise_equality_claimed=False,
        scope='Latent fixed32 propagation only; complete1536-family acceptance and head/cost/outcome seals remain external.')
