"""Full canonical score numerical audit, without reading physical outcome labels."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=[]
    for actor in range(6):
        for interface in ['pose_encoded','state']:
            d=Path(a.input)/f'actor_{actor}'/interface;r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
            assert r['status']=='EXTRACTED_REQUIRES_ACCEPTANCE' and ac['status']=='PASS' and ac['cases']==128 and ac['numerical_policy']=='canonical_single_action_v1'
            assert ac['report_sha256']==sha(d/'report.json') and ac['scores_sha256']==sha(d/'scores.npz');z=dict(np.load(d/'scores.npz'))
            cases=[]
            for row in r['cases']:
                key=f"case_{row['case']}";s=z[key+'_scores'];b=z[key+'_batch2_scores'];old=np.asarray(row['original_scores']);self_score=s[actor//2]
                token=z[key+'_tokens'][actor//2];original=z[key+'_original_tokens'];goal=z[key+'_goals'][actor//2];oldgoal=z[key+'_original_goals']
                cases.append(dict(case=row['case'],seed=row['seed'],batch_score_violations=int((~np.isclose(s,b,rtol=2e-5,atol=2e-5)).sum()),original_score_violations=int((~np.isclose(self_score,old,rtol=2e-5,atol=2e-5)).sum()),batch_max_absolute_score_difference=float(abs(s-b).max()),original_max_absolute_score_difference=float(abs(self_score-old).max()),original_token_max_absolute_difference=float(abs(token-original).max()),original_goal_max_absolute_difference=float(abs(goal-oldgoal).max()),original_token_violations=int((~np.isclose(token,original,rtol=2e-5,atol=2e-5)).sum()),original_goal_violations=int((~np.isclose(goal,oldgoal,rtol=2e-5,atol=2e-5)).sum()),ensemble_choice_changed=bool((s.mean(0)[1]<s.mean(0)[0])!=(b.mean(0)[1]<b.mean(0)[0]))))
            rows.append(dict(actor=actor,interface=interface,report_sha256=sha(d/'report.json'),acceptance_sha256=sha(d/'acceptance.json'),cases=cases,batch_score_violations=sum(v['batch_score_violations'] for v in cases),original_score_violations=sum(v['original_score_violations'] for v in cases),ensemble_choices_changed=sum(v['ensemble_choice_changed'] for v in cases)))
    result=dict(status='COMPLETE12_CANONICAL_NUMERICAL_AUDITS',rows=rows,source_sha256=sha(__file__),scope='All128 consumed goals, six actors, two interfaces. No actual costs/success labels read. Counts measure previous-equivalence violations; canonical CPU acceptance is distinct from original-search or batch2 equivalence.')
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    for interface in ['pose_encoded','state']:
        g=[r for r in rows if r['interface']==interface]
        print(interface,{k:sum(r[k] for r in g) for k in ['batch_score_violations','original_score_violations','ensemble_choices_changed']})


if __name__=='__main__':main()
