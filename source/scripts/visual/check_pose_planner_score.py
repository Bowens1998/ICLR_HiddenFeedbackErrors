"""Check fitted PushT scores on native image rollouts before planning runs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from image_planner_cost import ImagePlannerCost
from nonlinear_pose_cost import NonlinearPoseCost,numpy_pose,numpy_cost
from evaluation_precision import configure_evaluation_precision


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','training','heads','features','bank','official','config','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
    m=json.loads(Path(a.manifest).read_text());assert m['layout']=='action_auxiliary'
    rep,arm=a.index//2,['transformer_jepa','gru_jepa'][a.index%2]
    entries=[e for e in m['models'] if (e['replica'],e['arm'],e['mode'],e['checkpoint'])==(rep,arm,'none','last')]
    assert len(entries)==1;e=entries[0];train=Path(a.training)/e['training_path']
    assert sha(train/'last_weights.pt')==e['weights_sha256'] and sha(train/'summary.json')==e['training_summary_sha256']
    tr=json.loads((train/'summary.json').read_text());assert sha(a.config)==tr['config_sha256']
    headroot=Path(a.heads)/f'job_{a.index}';hr=json.loads((headroot/'report.json').read_text());ha=json.loads((headroot/'acceptance.json').read_text())
    assert hr['protocol']=='pusht_nonlinear_pose_fit_v1' and not hr['engineering'] and ha['status']=='PASS'
    assert hr['index']==a.index and ha['report_sha256']==sha(headroot/'report.json')
    assert ha['verifier_sha256']==sha(Path(__file__).with_name('accept_pose_readouts.py'))
    features=Path(a.features)/f'job_{a.index}';fr=json.loads((features/'report.json').read_text())
    assert sha(features/'report.json')==hr['input_report_sha256']
    assert sha(features/'acceptance.json')==hr['input_acceptance_sha256']
    assert fr['weights_sha256']==e['weights_sha256'] and fr['manifest_sha256']==sha(a.manifest)
    assert fr['training_summary_sha256']==e['training_summary_sha256']
    heads={}
    for row in hr['rows']:
        assert row['updates']==2000
        f=headroot/row['role']/'weights.npz';assert sha(f)==row['files_sha256']['weights.npz']
        with np.load(f) as z:heads[row['role']]={k:z[k] for k in z.files}
    assert set(heads)=={'encoded','predicted'}
    bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text());assert sha(bank/'manifest.json')==m['bank_manifest_sha256']
    precision=configure_evaluation_precision();torch.set_num_threads(2)
    model=make_model(a.official,a.config,arm,tr['seed']);model.load_state_dict(torch.load(train/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval()
    rows=[]
    with torch.inference_mode():
        for item in bm['cases'][:2]:
            f=bank/f"case_{item['index']:03d}.npz";assert sha(f)==item['sha256'];z=np.load(f)
            latent=ImagePlannerCost(model,z['history_pixels'],z['goal_pixels'],z['prefix'],tr['normalization'],None,'latent')
            generator=torch.Generator(device='cuda').manual_seed(955001+int(z['seed']))
            actions=(.2*torch.randn(300,25,2,device='cuda',generator=generator)).clamp(-.35,.35);actions[0]=0
            latent.verify_native(actions)
            _,original=latent(actions)
            for role in ['encoded','predicted']:
                scorer=NonlinearPoseCost(latent,heads[role],heads['encoded']);cost,tokens=scorer(actions)
                torch.testing.assert_close(tokens,original,rtol=0,atol=0)
                ep=numpy_pose(tokens.cpu().numpy(),heads[role]);gp=numpy_pose(latent.goal.cpu().numpy(),heads['encoded'])
                rebuilt=numpy_cost(ep,gp);np.testing.assert_allclose(cost.cpu().numpy(),rebuilt,rtol=1e-10,atol=1e-8)
                rows.append(dict(case_index=item['index'],role=role,candidates=300,max_abs_score=float(np.max(np.abs(cost.cpu().numpy()-rebuilt)))))
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False)
    result=dict(status='PASS',index=a.index,replica=rep,arm=arm,rows=rows,precision=precision,gpu=torch.cuda.get_device_name(),
                manifest_sha256=sha(a.manifest),bank_manifest_sha256=sha(bank/'manifest.json'),weights_sha256=e['weights_sha256'],head_report_sha256=sha(headroot/'report.json'),head_acceptance_sha256=sha(headroot/'acceptance.json'),
                source_sha256={name:sha(Path(__file__).with_name(name)) for name in ['check_pose_planner_score.py','nonlinear_pose_cost.py','image_planner_cost.py','factorial_model.py','lewm_adapter.py','evaluation_precision.py']},
                scope='Two previously examined development goals, 300 bounded actions each, two fitted scores, unchanged native tokens and NumPy reconstruction. No search or simulator outcome claim.')
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
