"""Prespecified first-two-goal end-to-end QP inputs; no future effects enter assignments."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha,atomic_json,require_role,namespace_seed
from projection_shard import run_shard
from evaluation_lock import (add_protocol_arguments,evaluation_context,partition,
    donor_assignment,check_evaluation_parent,require_output)


def checked_npz(path,expected):
    assert sha(path)==expected
    return dict(np.load(path))


def main():
    p=argparse.ArgumentParser()
    for k in ['recipient','donor','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--branch',choices=['AC','B'],required=True);p.add_argument('--head');p.add_argument('--group',type=int)
    add_protocol_arguments(p,partitioned=True);a=p.parse_args()
    formal=bool(a.protocol_lock or a.protocol_sha256)
    role=('confirmation_A_C' if a.branch=='AC' else 'confirmation_B') if formal else ('diagnostic_development_A_C' if a.branch=='AC' else 'diagnostic_development_B')
    ctx=evaluation_context(a,role,ROOT,__file__);spec=ctx.spec
    specpath=ROOT/'strengthening/configs/diagnostic_development_checks.json'
    g,shard,cases=partition(a,ctx,a.branch)
    for key,path in [('cache',a.recipient),('donor_cache',a.donor),('QP',a.output)]:require_output(path,ctx,a.branch+'_'+key)
    rp=Path(a.recipient);dp=Path(a.donor);out=Path(a.output)
    if a.branch=='AC':rp=rp/f'group_{g}';dp=dp/f'group_{g}';out=out/f'group_{g}'
    if shard is not None:out=out/f'shard_{shard:03d}'
    rr=json.loads((rp/'report.json').read_text());dr=json.loads((dp/'report.json').read_text())
    require_role(rr,{role})
    donor_role=('donor_bank_A_C' if a.branch=='AC' else 'donor_bank_B') if formal else ('donor_development_A_C' if a.branch=='AC' else 'donor_development_B')
    require_role(dr,{donor_role});check_evaluation_parent(rr,ctx);check_evaluation_parent(dr,ctx)
    assert (rp/'DONE').exists() and (dp/'DONE').exists()
    out.mkdir(parents=True,exist_ok=True)
    binding=dict(**ctx.binding,recipient_report_sha256=sha(rp/'report.json'),donor_report_sha256=sha(dp/'report.json'),
        spec_sha256=ctx.binding.get('scientific_design_sha256',sha(specpath)),source_sha256=sha(__file__),
        role=role if formal else 'closedloop_numerical_development',branch=a.branch,group=g,shard=shard,cases=cases)
    families=[];descriptors=[];assignments={}
    def append(kind,stream,case,model_keys,head_sets,heads,tokens,actual,donor):
        f=dict(tokens=[],guidance=[],head_sets=[],reference_heads=[],member_ids=[])
        for constraint,hs in head_sets:
            for key in model_keys:
                for source,guide in [('actual',actual),('donor',donor)]:
                    f['tokens'].append(tokens[key][stream,case,0]);f['guidance'].append(guide)
                    f['head_sets'].append(hs);f['reference_heads'].append(heads[0]);f['member_ids'].append(f'{constraint}/{key}/{source}')
        families.append(f);descriptors.append(dict(kind=kind,stream=stream,case=case,member_ids=f['member_ids']))
    if a.branch=='AC':
        heads=[checked_npz(rp/(k+'.npz'),rr['head_sha256'][k]) for k in ['head_A','head_B']]
        real=checked_npz(rp/'observed.npz',rr['observed_sha256']);don=checked_npz(dp/'observed.npz',dr['observed_sha256'])
        assert not set(real['seeds'])&set(don['seeds'])
        for k in ['mean','scale','target_mean','target_scale']:
            for h in heads:np.testing.assert_array_equal(h[k],heads[0][k])
        tokens={r['model']['name']:checked_npz(rp/r['file'],r['file_sha256'])['free_tokens'] for r in rr['models']}
        binding['head_sha256']=rr['head_sha256'];all_A=['A_'+q for q in spec['A_models_per_group']]
        for stream in spec['AC_streams']:
            if formal:assignment=np.asarray(donor_assignment(ctx,'AC',g,stream),dtype=int)
            else:
                ns=spec['AC_donor_seed_format'].format(group=g,stream=stream)
                assignment=np.random.default_rng(namespace_seed(spec['root_seed'],ns)).permutation(ctx.count)
            assert sorted(assignment.tolist())==list(range(ctx.count))
            assignments[str(stream)]=assignment.tolist()
            for i in cases:
                actual=real['observed_tokens'][stream,i,0];donor=don['observed_tokens'][stream,assignment[i],0]
                append('A_core',stream,i,all_A,[('single_A',heads[:1])],heads,tokens,actual,donor)
                if g in spec['dual_groups']:
                    append('A_dual',stream,i,all_A,[('single_A',heads[:1]),('dual_A_B',heads)],heads,tokens,actual,donor)
                if g in spec['C_groups']:
                    for q in spec['C_objectives']:
                        keys=[f'pool{g//2}_{q}_{t}' for t in spec['C_conditions']]
                        append('C_'+q,stream,i,keys,[('single_A',heads[:1])],heads,tokens,actual,donor)
    else:
        assert a.head and sha(a.head)==rr['head_sha256']==dr['head_sha256']
        head=dict(np.load(a.head));binding['head_sha256']=sha(a.head)
        assignment=np.asarray(donor_assignment(ctx,'B',0,0),dtype=int) if formal else np.random.default_rng(namespace_seed(spec['root_seed'],spec['B_donor_seed_namespace'])).permutation(ctx.count)
        assert sorted(assignment.tolist())==list(range(ctx.count))
        assignments['0']=assignment.tolist();assert not {r['seed'] for r in rr['rows']}&{r['seed'] for r in dr['rows']}
        for i in cases:
            rec=rr['rows'][i];don=dr['rows'][int(assignment[i])]
            r=checked_npz(rp/rec['file'],rec['file_sha256']);d=checked_npz(dp/don['file'],don['file_sha256'])
            token=r['free_tokens'][3,:,:384].reshape(75264)
            families.append(dict(tokens=[token]*2,guidance=[r['observed_visual'][0],d['observed_visual'][0]],
                head_sets=[[head]]*2,reference_heads=[head]*2,member_ids=['single_A/native/actual','single_A/native/donor']))
            descriptors.append(dict(kind='B_core',stream=0,case=i,member_ids=families[-1]['member_ids']))
    design=dict(binding=binding,assignments=assignments,families=descriptors)
    if (out/'design.json').exists():assert json.loads((out/'design.json').read_text())==design
    else:atomic_json(out/'design.json',design)
    result=run_shard(out/'qp',dict(**binding,design_sha256=sha(out/'design.json')),families)
    atomic_json(out/'report.json',dict(**ctx.binding,status='PASS_CLOSEDLOOP_QP',role=role,branch=a.branch,group=g,shard=shard,cases=cases,
        design_sha256=sha(out/'design.json'),qp_report_sha256=sha(out/'qp/report.json'),family_count=len(families),
        qp_count=sum(len(f['tokens']) for f in families),result=result,source_sha256=sha(__file__),
        scope='Complete prespecified goal partition; donor permutation fixed over the full independent bank before any effects. Numerical failure stops this shard; no case removal or fabricated zero displacement.'))
    (out/'DONE').write_text('qp_accepted\n')


if __name__=='__main__':main()
