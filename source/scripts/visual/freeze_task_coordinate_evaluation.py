"""Freeze all48 controls/conditions and96 routes before the new goal bank exists."""
import argparse,copy,json,os
from pathlib import Path
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('fits','prior-fits','cache','original-plan','protocol','adapters','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();new=[];prior={};cache_entries={}
    for i in range(12):
        cr=json.loads((Path(a.cache)/f'job_{i}'/'report.json').read_text());cache_entries[i]=cr['entry']
    for task in range(24):
        d=Path(a.fits).resolve()/f'job_{task}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text());index=2*(task//4)+(task%4==3)
        assert r['index']==task and r['model_index']==index and r['objective']==('latent','decoded_teacher','physical_labels','native_state')[task%4]
        assert r['entry']==cache_entries[index] and r['protocol_sha256']==sha(a.protocol) and r['updates']==2100
        assert ac['status']=='PASS_TASK_COORDINATE_FORMAL_FIT_AND_CPU_PREDICTIONS' and ac['report_sha256']==sha(d/'report.json') and ac['source_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_formal_fit.py'))
        assert sha(d/'last_weights.pt')==r['weights_sha256'];new.append((d,r,ac))
    for index in range(12):
        d=Path(a.prior_fits).resolve()/f'job_{2*index+1}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
        assert r['model_index']==index and r['stream']=='planner' and r['entry']==cache_entries[index] and r['updates']==2100
        assert ac['status']=='PASS_FORMAL_FIT_FROZEN_STATE_AND_CPU_PREDICTIONS' and ac['report_sha256']==sha(d/'report.json') and ac['source_sha256']==sha(Path(__file__).with_name('accept_adaptation_formal_fit.py'))
        assert sha(d/'last_weights.pt')==r['weights_sha256'];prior[index]=(d,r,ac)
    root=Path(a.adapters).resolve();root.mkdir(parents=True,exist_ok=False);models=[];routes=[];bindings=[]
    for group in range(6):
        specifications=[(2*group,'original',None),(2*group,'clipped_latent',prior[2*group]),(2*group,'unit_latent',new[4*group]),(2*group,'unit_decoded_teacher',new[4*group+1]),(2*group,'unit_physical_labels',new[4*group+2]),(2*group+1,'original',None),(2*group+1,'clipped_state',prior[2*group+1]),(2*group+1,'unit_state',new[4*group+3])]
        for index,condition,fit in specifications:
            entry=copy.deepcopy(cache_entries[index]);td=Path(entry['training_path']);tr=json.loads((td/'summary.json').read_text())
            assert sha(td/'summary.json')==entry['training_summary_sha256'] and sha(td/'last_weights.pt')==entry['weights_sha256']
            mi=len(models)
            if fit:
                d,r,ac=fit;adapter=root/f'model_{mi}';adapter.mkdir()
                summary=dict(artifact_type='task_coordinate_planner_compatibility_adapter',seed=tr['seed'],normalization=tr['normalization'],target_normalization=entry['target_normalization'],mode=entry['mode'],config_sha256=tr['config_sha256'],data_manifest_sha256=entry['data_manifest_sha256'],fit_report=str(d/'report.json'),fit_report_sha256=sha(d/'report.json'),fit_acceptance_sha256=sha(d/'acceptance.json'),condition=condition,continuation_updates=2100,scope='Original data manifest/seed identify initialization and normalization provenance. Actual continuation objective/data/control are recorded in the bound fit report; no original-only training claim.')
                (adapter/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');os.symlink(d/'last_weights.pt',adapter/'last_weights.pt')
                entry.update(training_path=str(adapter),training_summary_sha256=sha(adapter/'summary.json'),weights_sha256=r['weights_sha256'],acceptance_sha256=sha(d/'acceptance.json'),updates=23100,selected_update=23100,initial_updates=21000,continuation_updates=2100);entry.pop('auxiliary_weights_sha256',None)
                bindings.append(dict(model_index=mi,condition=condition,fit_report=str(d/'report.json'),fit_report_sha256=sha(d/'report.json'),fit_acceptance_sha256=sha(d/'acceptance.json'),adapter_summary_sha256=sha(adapter/'summary.json')))
            for name in ('endpoint_head','goal_head'):
                if entry.get(name):assert sha(entry[name]['path'])==entry[name]['sha256']
            entry.update(adaptation_condition=condition,original_model_index=index,comparison_group=group);models.append(entry)
            for algorithm in ('random','cem'):routes.append(dict(model_index=mi,algorithm=algorithm,parameterization='full',adaptation_condition=condition,original_model_index=index,comparison_group=group))
    assert len(models)==48 and len(routes)==96 and len(bindings)==36
    original=json.loads(Path(a.original_plan).read_text());plan=dict(layout='pusht_nonlinear_pose',models=models,routes=routes,bindings=bindings,bank_manifest_sha256=None,original_provenance={k:v for k,v in original.items() if k not in ('models','routes','bank_manifest_sha256')},scope='Complete fixed task-coordinate/label comparison with original and clipped controls.48 conditions96routes,128 separately reserved fresh goals,9000 candidates/search. No filtering by validation.')
    plan['task_coordinate_evaluation']=dict(status='FROZEN_MODELS_PENDING_BANK',seed_start=1351001,max_seeds=2048,cases=128,protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),original_plan_sha256=sha(a.original_plan))
    with Path(a.output).open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print('FROZEN48_MODELS96_ROUTES',flush=True)

if __name__=='__main__':main()
