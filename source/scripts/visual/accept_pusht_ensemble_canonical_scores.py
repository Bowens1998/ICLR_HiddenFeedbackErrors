"""Independent saved-array/CPU-head acceptance; neural rollouts are not replayed."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def head(x,h):
    x=(np.asarray(x,dtype=float)-h['mean'])/h['scale']
    for i in (0,2,4):
        x=np.matmul(x,h[f'{i}.weight'].T)+h[f'{i}.bias']
        if i!=4:x=np.maximum(x,0)
    return x*h['target_scale']+h['target_mean']


def physical_score(ep,gp):
    delta=np.arctan2(ep[...,4],ep[...,5])-np.arctan2(gp[4],gp[5])
    wrapped=(delta+np.pi)%(2*np.pi)-np.pi
    return ((ep[...,2:4]-gp[2:4])**2).sum(-1)+900*wrapped**2


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args()
    folder=Path(a.input);r=json.loads((folder/'report.json').read_text());root=Path(__file__).resolve().parents[2]
    assert r['status'] in ['ENGINEERING_COMPLETE_REQUIRES_ACCEPTANCE','EXTRACTED_REQUIRES_ACCEPTANCE']
    count=2 if r['status'].startswith('ENGINEERING') else 128;assert len(r['cases'])==count
    assert sha(folder/'scores.npz')==r['scores_sha256']
    for name,digest in r['source_sha256'].items():assert sha(root/name)==digest
    assert sha(root/'docs/maintrack/ENSEMBLE_DECISION_RISK_PROTOCOL.md')==r['protocol_sha256']
    assert r['numerical_policy']=='canonical_single_action_v1' and sha(root/'docs/maintrack/PUSHT_ENSEMBLE_NUMERICAL_PROTOCOL.md')==r['numerical_protocol_sha256']
    z=dict(np.load(folder/'scores.npz',allow_pickle=False));max_error=0.;heads=[]
    assert [e['replica'] for e in r['members']]==[0,1,2]
    for e in r['members']:
        assert sha(Path(e['training_path'])/'last_weights.pt')==e['weights_sha256']
        if r['interface']=='pose_encoded':
            pair=[]
            for role in ['endpoint_head','goal_head']:
                path=Path(e[role]['path']);assert sha(path)==e[role]['sha256'];pair.append(dict(np.load(path,allow_pickle=False)))
            heads.append(pair)
        else:heads.append(None)
    for row in r['cases']:
        key=f"case_{row['case']}";scores=z[key+'_scores'];tokens=z[key+'_tokens'];goals=z[key+'_goals'];actions=z[key+'_actions']
        assert scores.shape==(3,2) and np.isfinite(scores).all() and (scores>=0).all()
        assert actions.shape==(2,25,2) and np.isfinite(actions).all() and abs(actions).max()<=np.float32(.35)
        for m,e in enumerate(r['members']):
            if heads[m] is not None:ep=head(tokens[m],heads[m][0]);gp=head(goals[m],heads[m][1])
            else:
                norm=e['target_normalization'];mean=np.asarray(norm['mean'],dtype=np.float32);std=np.asarray(norm['std'],dtype=np.float32)
                ep=(tokens[m]*std+mean).astype(float);gp=(goals[m]*std+mean).astype(float)
            reference=physical_score(ep,gp)
            np.testing.assert_allclose(scores[m],reference,rtol=2e-5,atol=2e-5)
            max_error=max(max_error,float(abs(scores[m]-reference).max()))
        assert row['original_score_violations']==int((~np.isclose(scores[r['actor']//2],row['original_scores'],rtol=2e-5,atol=2e-5)).sum())
        assert row['batch_score_violations']==int((~np.isclose(scores,z[key+'_batch2_scores'],rtol=2e-5,atol=2e-5)).sum())
    output=dict(status='PASS',numerical_policy=r['numerical_policy'],cases=count,actor=r['actor'],interface=r['interface'],max_cpu_score_difference=max_error,report_sha256=sha(folder/'report.json'),scores_sha256=sha(folder/'scores.npz'),verifier_sha256=sha(__file__),scope='Canonical single-action scores independently reconstructed from saved tokens. Original-search and batch2 disagreements explicitly retained; not a pass of old equivalence checks. Native/repeated neural checks remain extractor checks; no independent neural replay or outcome evidence.')
    (folder/'acceptance.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))


if __name__=='__main__':main()
