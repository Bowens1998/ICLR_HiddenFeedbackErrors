"""Accepted selected outcomes: imagined/real-image/goal errors and same-image freeze."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from adaptation_freeze import configure_dynamics_only,verify_frozen
from evaluation_precision import configure_evaluation_precision
from extract_pose_selected_endpoints import sha,decode,pose,components


def main():
    p=argparse.ArgumentParser()
    for k in ('plan','runs','bank','official','config','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(12),required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text())
    assert plan['adaptation_evaluation']['status']=='READY_COMPLETE_FRESH_EVALUATION' and len(bm['cases'])==128 and sha(bank/'manifest.json')==plan['bank_manifest_sha256']
    original_entry=plan['models'][a.index*3];td=Path(original_entry['training_path']);tr=json.loads((td/'summary.json').read_text())
    assert sha(td/'summary.json')==original_entry['training_summary_sha256'] and sha(td/'last_weights.pt')==original_entry['weights_sha256'] and sha(a.config)==tr['config_sha256']
    configure_evaluation_precision();torch.set_num_threads(2)
    weights=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True)
    reference=make_model(a.official,a.config,original_entry['arm'],tr['seed']);reference.load_state_dict(weights,strict=True);reference=reference.cuda().eval()
    model=make_model(a.official,a.config,original_entry['arm'],tr['seed']);model.load_state_dict(weights,strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);rows=[];start=time.monotonic()
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,1,3,1,1)
    def encode(m,pixels):
        assert pixels.shape==(224,224,3) and pixels.dtype==np.uint8
        image=torch.as_tensor(pixels.copy(),device='cuda').permute(2,0,1)[None,None].float()/255
        return m.encode({'pixels':(image-mean)/std})['emb'][0,0]
    with torch.inference_mode():
        for route_index in range(a.index*6,a.index*6+6):
            route=plan['routes'][route_index];entry=plan['models'][route['model_index']];assert entry['original_model_index']==a.index
            wd=Path(entry['training_path']);assert sha(wd/'last_weights.pt')==entry['weights_sha256']
            model.load_state_dict(torch.load(wd/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);verify_frozen(model,boundary)
            d=Path(a.runs)/f'job_{route_index}';r=json.loads((d/'summary.json').read_text());ac=json.loads((d/'acceptance.json').read_text());am=json.loads((d/'artifact_manifest.json').read_text())
            assert ac['model_free_simulator_replay'] and ac['cases']==128 and r['route_index']==route_index
            assert r['hashes']['model_manifest']==sha(a.plan) and r['hashes']['weights']==entry['weights_sha256'] and am['summary.json']==sha(d/'summary.json')
            head=None
            if entry['score']!='state':
                assert entry['endpoint_head']==entry['goal_head'];hp=Path(entry['goal_head']['path']);assert sha(hp)==entry['goal_head']['sha256'];head=dict(np.load(hp));shutil.copyfile(hp,out/f'head_{route_index}.npz')
            def physical(token):
                if head is not None:return decode(token,head)
                return token.astype(float)*np.asarray(entry['target_normalization']['std'])+np.asarray(entry['target_normalization']['mean'])
            values={k:[] for k in ('seed','imagined_tokens','real_tokens','goal_tokens','imagined_pose','real_pose','goal_pose','true_endpoint','true_goal','realized_cost','predicted_cost','success')};cases=[]
            for i,(c,b) in enumerate(zip(r['cases'],bm['cases'])):
                assert c['index']==b['index']==i and c['seed']==b['seed'];bp=bank/f'case_{i:03d}.npz';fp=d/f'case_{i:03d}_predictions.npz'
                assert sha(bp)==b['sha256'] and sha(fp)==am[fp.name]
                with np.load(bp) as truth,np.load(fp) as z:
                    real=encode(model,z['terminal_pixels']);original_real=encode(reference,z['terminal_pixels']);torch.testing.assert_close(real,original_real,rtol=0,atol=0)
                    goal=encode(model,truth['goal_pixels']);original_goal=encode(reference,truth['goal_pixels']);torch.testing.assert_close(goal,original_goal,rtol=0,atol=0)
                    goal=goal.cpu().numpy();np.testing.assert_allclose(goal,z['goal_tokens'],rtol=2e-5,atol=2e-5)
                    imagined=z['tokens'][c['selected_iteration'],c['selected_candidate']];real=real.cpu().numpy();ep=physical(imagined);gp=physical(z['goal_tokens']);true_ep=pose(z['selected_states'][-1]);true_goal=pose(truth['goal_state'])
                    np.testing.assert_allclose(components(ep,gp).sum(),c['predicted_cost'],rtol=2e-5,atol=2e-3)
                    np.testing.assert_allclose(components(true_ep,true_goal).sum(),c['realized_cost'],rtol=1e-10,atol=1e-7)
                    record=dict(seed=c['seed'],imagined_tokens=imagined,real_tokens=real,goal_tokens=z['goal_tokens'],imagined_pose=ep,real_pose=physical(real),goal_pose=gp,true_endpoint=true_ep,true_goal=true_goal,realized_cost=c['realized_cost'],predicted_cost=c['predicted_cost'],success=c['success'])
                    for k,v in record.items():values[k].append(v)
                cases.append(dict(index=i,seed=c['seed'],archive_sha256=am[fp.name],bank_sha256=b['sha256']))
            assert len(cases)==128;name=f'route_{route_index}.npz';np.savez_compressed(out/name,**{k:np.asarray(v) for k,v in values.items()})
            rows.append(dict(route=route_index,entry=entry,file=name,sha256=sha(out/name),head_file=f'head_{route_index}.npz' if head is not None else None,head_sha256=sha(out/f'head_{route_index}.npz') if head is not None else None,cases=cases,summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json')))
    report=dict(status='EXTRACTED_REQUIRES_NUMERIC_ACCEPTANCE',index=a.index,rows=rows,plan_sha256=sha(a.plan),source_sha256=sha(__file__),same_image_checks=1536,max_same_image_difference=0.,elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),scope='All6 routes128goals for one starting model. Original and adapted image encoders compared exactly on each same terminal/goal image; frozen tensors independently checked. Goal encoding also matches stored planner tokens within2e-5. Privileged real-terminal diagnostics, not a deployable policy; changes in visited-state errors remain possible despite identical perception.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('EXTRACTED',a.index,flush=True)

if __name__=='__main__':main()
