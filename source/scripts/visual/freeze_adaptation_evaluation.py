"""Freeze all36 original/continued models and72 routes before generating evaluation."""
import argparse,copy,json,os
from pathlib import Path
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('fits','cache','original-plan','adapters','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();original=json.loads(Path(a.original_plan).read_text());models=[];routes=[];bindings=[]
    # Validate the complete24-model acceptance set before creating adapters.
    accepted=[]
    for task in range(24):
        d=Path(a.fits).resolve()/f'job_{task}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
        assert r['index']==task and r['model_index']==task//2 and r['stream']==('expert','planner')[task%2]
        assert ac['status']=='PASS_FORMAL_FIT_FROZEN_STATE_AND_CPU_PREDICTIONS' and ac['report_sha256']==sha(d/'report.json')
        assert ac['source_sha256']==sha(Path(__file__).with_name('accept_adaptation_formal_fit.py'))
        assert sha(d/'last_weights.pt')==r['weights_sha256'] and r['updates']==2100
        accepted.append((d,r,ac))
    root=Path(a.adapters).resolve();root.mkdir(parents=True,exist_ok=False)
    for index in range(12):
        cache=Path(a.cache)/f'job_{index}';cr=json.loads((cache/'report.json').read_text());e=cr['entry'];td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
        assert sha(td/'summary.json')==e['training_summary_sha256'] and sha(td/'last_weights.pt')==e['weights_sha256']
        for condition in ('original','expert','planner'):
            entry=copy.deepcopy(e)
            if condition!='original':
                task=2*index+(condition=='planner');d,r,ac=accepted[task];assert r['entry']==e and r['cache_report_sha256']==sha(cache/'report.json')
                adapter=root/f'job_{task}';adapter.mkdir()
                summary=dict(artifact_type='frozen_adaptation_planner_compatibility_adapter',seed=tr['seed'],normalization=tr['normalization'],target_normalization=e['target_normalization'],mode=e['mode'],config_sha256=tr['config_sha256'],data_manifest_sha256=e['data_manifest_sha256'],scope='Data manifest and seed identify original initialization/normalization provenance, not a claim of original-only continuation training. Actual continuation inputs and2100updates are recorded in the bound fit report.',fit_report=str(d/'report.json'),fit_report_sha256=sha(d/'report.json'),fit_acceptance_sha256=sha(d/'acceptance.json'),condition=condition,continuation_updates=2100)
                (adapter/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');os.symlink(d/'last_weights.pt',adapter/'last_weights.pt')
                entry.update(training_path=str(adapter),training_summary_sha256=sha(adapter/'summary.json'),weights_sha256=r['weights_sha256'],acceptance_sha256=sha(d/'acceptance.json'),updates=23100,selected_update=23100,initial_updates=21000,continuation_updates=2100)
                entry.pop('auxiliary_weights_sha256',None)
                bindings.append(dict(task=task,fit_report_sha256=sha(d/'report.json'),fit_acceptance_sha256=sha(d/'acceptance.json'),adapter_summary_sha256=sha(adapter/'summary.json')))
            for name in ('endpoint_head','goal_head'):
                if entry.get(name):assert sha(entry[name]['path'])==entry[name]['sha256']
            entry.update(adaptation_condition=condition,original_model_index=index)
            model_index=len(models);models.append(entry)
            for algorithm in ('random','cem'):routes.append(dict(model_index=model_index,algorithm=algorithm,parameterization='full',adaptation_condition=condition,original_model_index=index))
    assert len(models)==36 and len(routes)==72 and len(bindings)==24
    plan=copy.deepcopy(original);plan.update(models=models,routes=routes,bank_manifest_sha256=None,scope='Fresh fixed128-goal PushT adaptation comparison:12 starting models ×original/expert/planner ×random/CEM9000. Frozen before evaluation context generation. All conditions retained.')
    plan['original_collection']=plan.pop('adaptation_formal',None)
    plan['original_bindings']=plan.pop('bindings',None)
    plan['bindings']=bindings
    plan['adaptation_evaluation']=dict(status='FROZEN_MODELS_PENDING_BANK',seed_start=1321001,max_seeds=2048,cases=128,fit_bindings=bindings,source_sha256=sha(__file__),original_plan_sha256=sha(a.original_plan))
    with Path(a.output).open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print('FROZEN36_MODELS72_ROUTES',flush=True)

if __name__=='__main__':main()
