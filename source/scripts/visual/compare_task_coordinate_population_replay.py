"""Compare amended-verification runs against every available original case."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('original','updated','plan','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());assert len(plan['routes'])==96
    keys=['parameters','costs','tokens','mean','std','goal_tokens','selected_actions','selected_states','boundary','terminal_pixels','final_distribution_mean']
    rows=[];changes=[];compared=0
    for i in range(96):
        old=Path(a.original)/f'job_{i}';new=Path(a.updated)/f'job_{i}';ac=json.loads((new/'acceptance.json').read_text());nr=json.loads((new/'summary.json').read_text());nm=json.loads((new/'artifact_manifest.json').read_text())
        assert ac['model_free_simulator_replay'] and ac['population_replay_exact_all_cases'] and ac['cases']==128 and nr['hashes']['model_manifest']==sha(a.plan)
        assert ac['verifier_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_population_planner.py'))
        om=json.loads((old/'artifact_manifest.json').read_text()) if (old/'artifact_manifest.json').exists() else None
        if i!=31:assert om is not None
        found=[]
        for j in range(128):
            name=f'case_{j:03d}_predictions.npz';op=old/name;npth=new/name
            if not op.exists():continue
            assert sha(npth)==nm[name]
            if om is not None:assert sha(op)==om[name]
            with np.load(op) as oz,np.load(npth) as nz:
                different=[]
                for k in keys:
                    if not np.array_equal(oz[k],nz[k]):
                        different.append(k)
                        difference=float(np.max(abs(oz[k].astype(float)-nz[k].astype(float)))) if oz[k].shape==nz[k].shape else None
                        changes.append(dict(route=i,case=j,field=k,max_absolute_difference=difference))
            found.append(dict(case=j,original_sha256=sha(op),updated_sha256=nm[name],different_fields=different));compared+=1
        assert len(found)==(22 if i==31 else 128)
        rows.append(dict(route=i,cases=found,updated_acceptance_sha256=sha(new/'acceptance.json')))
    result=dict(status='IDENTICAL_AVAILABLE_ORIGINAL_SEARCH_AND_OUTCOMES' if not changes else 'DIFFERENCES_REQUIRE_REVIEW',compared_cases=compared,changes=changes,rows=rows,plan_sha256=sha(a.plan),source_sha256=sha(__file__),scope='Exact array comparison against95 original complete routes plus22 completed cases of failed route31. New remaining106 cases of route31 have no original successful output. Numerical reproducibility comparison, not independent goal confirmation. Updated verification-only arrays are intentionally excluded.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],compared,len(changes),flush=True)

if __name__=='__main__':main()
