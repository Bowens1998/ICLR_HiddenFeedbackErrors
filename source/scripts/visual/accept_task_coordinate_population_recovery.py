"""Local metadata recovery audit; never substitute for full remote search replay."""
import argparse
import json
from pathlib import Path
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for key in ('runs','plan','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());assert len(plan['routes'])==96
    rows=[];missing=[]
    for i,route in enumerate(plan['routes']):
        d=Path(a.runs)/f'job_{i}'
        if not (d/'acceptance.json').exists():missing.append(i);continue
        r=json.loads((d/'summary.json').read_text());ac=json.loads((d/'acceptance.json').read_text());am=json.loads((d/'artifact_manifest.json').read_text());source=json.loads((d/'source/manifest.json').read_text())
        entry=plan['models'][route['model_index']]
        assert (d/'COMPLETE').exists() and am['summary.json']==sha(d/'summary.json')
        assert r['route_index']==i and r['model_index']==route['model_index'] and r['algorithm']==route['algorithm']
        assert r['hashes']['model_manifest']==sha(a.plan) and r['hashes']['bank_manifest']==plan['bank_manifest_sha256'] and r['hashes']['weights']==entry['weights_sha256']
        assert len(r['cases'])==ac['cases']==128 and ac['candidate_scores_per_case']==9000 and ac['model_free_simulator_replay']
        assert ac['population_replay_exact_all_cases'] and r['verification_policy']=='exact_original_population_replay_with_single_action_diagnostic'
        assert ac['verification_protocol_sha256']==r['verification_protocol_sha256']==sha(Path(__file__).resolve().parents[2]/'docs/maintrack/TASK_COORDINATE_POPULATION_REPLAY_PROTOCOL.md')
        assert ac['single_original_tolerance_failures']==sum(not c['single_matches_original_tolerance'] for c in r['cases'])
        assert all(c['population_replay_exact'] and c['additional_verification_candidates']==301 and c['additional_verification_forwards']==2 for c in r['cases'])
        assert ac['verifier_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_population_planner.py'))
        assert ac['scores_CEM_updates_choices_actions_metrics_hashes']=='independently reconstructed'
        assert source['run_task_coordinate_population_planner.py']==sha(Path(__file__).with_name('run_task_coordinate_population_planner.py'))
        assert {f'case_{j:03d}_predictions.npz' for j in range(128)}<=set(am)
        assert [c['index'] for c in r['cases']]==list(range(128))
        assert all(c['scored_candidates']==9000 for c in r['cases'])
        rows.append(dict(route=i,summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json'),artifact_manifest_sha256=sha(d/'artifact_manifest.json'),source_manifest_sha256=sha(d/'source/manifest.json')))
    assert a.allow_partial or not missing
    assert rows
    result=dict(status='PASS_COMPLETE96_POPULATION_ROUTE_METADATA' if not missing else 'PARTIAL_POPULATION_ROUTE_METADATA_ONLY',rows=rows,missing=missing,plan_sha256=sha(a.plan),source_sha256=sha(__file__),scope='Recovered summary/artifact/source-manifest bindings checked against frozen plan and current planner/verifier source. Raw candidate arrays and full source snapshot remain remote and are not reread by this checker. No scientific aggregate from partial routes.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],len(rows))

if __name__=='__main__':main()
