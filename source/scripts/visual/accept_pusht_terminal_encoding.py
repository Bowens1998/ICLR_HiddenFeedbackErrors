"""Independent CPU arithmetic and original archive identity acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_pose_selected_endpoints import error,sha
from nonlinear_pose_cost import numpy_pose


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--selected',required=True);p.add_argument('--summary',required=True);a=p.parse_args()
    d=Path(a.input);r=json.loads((d/'report.json').read_text());assert r['status']=='EXTRACTED_REQUIRES_ACCEPTANCE'
    full=json.loads(Path(a.summary).read_text());assert r['summary_sha256']==sha(a.summary) and r['plan_sha256']==full['plan_sha256']
    sd=Path(a.selected)/f"job_{r['actor']}";assert r['selected_report_sha256']==sha(sd/'report.json');sr=json.loads((sd/'report.json').read_text())
    assert [row['route'] for row in r['rows']]==list(range(r['actor']*8+2,r['actor']*8+8))
    n=2 if r['engineering'] else 128;accepted=[]
    for row in r['rows']:
        route=row['route'];original=next(v for v in sr['rows'] if v['route_index']==route);binding=next(v for v in full['bindings'] if v['index']==route)
        for k in ['summary_sha256','acceptance_sha256']:assert row[k]==original[k]==binding[k]
        assert row['selected_sha256']==original['file_sha256']==sha(sd/original['file']) and row['file_sha256']==sha(d/row['file'])
        z=dict(np.load(d/row['file']));old=dict(np.load(sd/original['file']))
        for k,v in old.items():np.testing.assert_array_equal(z[k],v[:n])
        assert len(row['cases'])==n and row['interface']==original['score'] and row['algorithm']==original['algorithm']
        for i,c in enumerate(row['cases']):
            assert c['index']==i and c['seed']==int(z['seed'][i]) and c['archive_sha256']==original['bindings'][i]['archive_sha256'] and c['bank_sha256']==original['bindings'][i]['bank_case_sha256']
        e=row['entry']
        if row['head_file']:
            assert sha(d/row['head_file'])==row['head_sha256']==e['goal_head']['sha256'];h=dict(np.load(d/row['head_file']));pred=numpy_pose(z['real_tokens'],h)
        else:pred=z['real_tokens'].astype(float)*np.array(e['target_normalization']['std'])+np.array(e['target_normalization']['mean'])
        np.testing.assert_allclose(pred,z['real_pose'],rtol=1e-10,atol=1e-8)
        assert np.isfinite(pred).all() and pred.shape==(n,6)
        actual,_=error(z['true_endpoint'],z['true_goal']);np.testing.assert_allclose(actual.sum(1),full['rows'][route]['cost'][:n],rtol=1e-9,atol=1e-7)
        accepted.append(dict(route=route,cases=n,file_sha256=row['file_sha256']))
    ac=dict(status='PASS',actor=r['actor'],engineering=r['engineering'],rows=accepted,report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),scope='CPU decoding and inherited original labels/bindings; neural encoding repeat and goal checks performed by extractor, not independently rerun on CPU.')
    with (d/'acceptance.json').open('x') as f:json.dump(ac,f,indent=2);f.write('\n')
    print('PASS',r['actor'],n)


if __name__=='__main__':main()
