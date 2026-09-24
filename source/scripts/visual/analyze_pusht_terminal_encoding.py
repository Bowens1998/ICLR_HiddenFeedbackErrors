"""Accepted paired terminal-image decoding and hindsight pair choices."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_pose_selected_endpoints import error,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);assert not out.exists();rows=[];pairs=[];bindings=[]
    full=json.loads(Path('outputs/maintrack/pusht_pose_planning_summary.json').read_text())
    for actor in range(6):
        d=Path(a.input)/f'actor_{actor}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
        assert not r['engineering'] and not ac['engineering'] and ac['status']=='PASS' and ac['actor']==r['actor']==actor
        assert ac['report_sha256']==sha(d/'report.json') and r['summary_sha256']==sha('outputs/maintrack/pusht_pose_planning_summary.json')
        assert r['source_sha256']==sha(Path(__file__).with_name('extract_pusht_terminal_encoding.py')) and ac['source_sha256']==sha(Path(__file__).with_name('accept_pusht_terminal_encoding.py'))
        assert r['protocol_sha256']==sha('docs/maintrack/PUSHT_TERMINAL_ENCODING_PROTOCOL.md')
        loaded={}
        for v,av in zip(r['rows'],ac['rows']):
            assert v['route']==av['route'] and v['file_sha256']==av['file_sha256']==sha(d/v['file']) and av['cases']==128
            z=dict(np.load(d/v['file']));np.testing.assert_array_equal(z['seed'],full['goal_seeds']);loaded[v['route']]=z
            metrics={}
            for name,key in [('real_image','real_pose'),('imagined','estimated_endpoint')]:
                components,ok=error(z[key],z['true_endpoint']);metrics[name]=dict(position_mse=float(components[:,0].mean()),weighted_angle_mse=float(components[:,1].mean()),precision=float(ok.mean()))
            rows.append(dict(actor=actor,route=v['route'],interface=v['interface'],algorithm=v['algorithm'],metrics=metrics))
        for offset,interface in [(2,'pose_encoded'),(4,'pose_predicted'),(6,'state')]:
            routes=[actor*8+offset,actor*8+offset+1];zs=[loaded[j] for j in routes]
            np.testing.assert_array_equal(zs[0]['estimated_goal'],zs[1]['estimated_goal'])
            actual=np.stack([full['rows'][j]['cost'] for j in routes],1);success=np.stack([full['rows'][j]['success'] for j in routes],1);policies={}
            for name,key in [('real_image','real_pose'),('uncorrected','estimated_endpoint'),('true_endpoint','true_endpoint')]:
                scores=np.stack([error(z[key],z['estimated_goal'])[0].sum(1) for z in zs],1);choice=(scores[:,1]<scores[:,0]).astype(int)
                cost=actual[np.arange(128),choice];policies[name]=dict(cost=float(cost.mean()),success=float(success[np.arange(128),choice].mean()),regret=float((cost-actual.min(1)).mean()),cem_fraction=float(choice.mean()))
            pairs.append(dict(actor=actor,interface=interface,policies=policies))
        bindings.append(dict(actor=actor,report_sha256=sha(d/'report.json'),acceptance_sha256=sha(d/'acceptance.json')))
    assert len(rows)==36 and len(pairs)==18
    groups=[]
    for interface in ['pose_encoded','pose_predicted','state']:
        metric={alg:{name:{key:float(np.mean([r['metrics'][name][key] for r in rows if r['interface']==interface and r['algorithm']==alg])) for key in ['position_mse','weighted_angle_mse','precision']} for name in ['real_image','imagined']} for alg in ['random','cem']}
        pol={name:{key:float(np.mean([r['policies'][name][key] for r in pairs if r['interface']==interface])) for key in ['cost','success','regret','cem_fraction']} for name in ['real_image','uncorrected','true_endpoint']}
        groups.append(dict(interface=interface,metrics=metric,policies=pol))
    result=dict(status='COMPLETE36_TERMINAL_ENCODING_ROUTES',rows=rows,pairs=pairs,groups=groups,bindings=bindings,source_sha256=sha(__file__),scope='Consumed goals, real terminal images are privileged hindsight; different frozen image and imagined readout roles retained, not unique transition-cause identification or deployable policy.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for g in groups:print(g)


if __name__=='__main__':main()
