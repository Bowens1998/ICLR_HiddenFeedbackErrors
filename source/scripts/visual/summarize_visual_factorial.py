"""Summarize only fully accepted four-arm training and both checkpoint/condition routes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ARMS=['transformer_jepa','gru_jepa','transformer_state','gru_state']
def main():
    p=argparse.ArgumentParser()
    for key in ['training','inview','stress','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();training=Path(a.training);accept=json.loads((training/'acceptance.json').read_text());assert accept['accepted_arms']==ARMS and accept['epochs']==100
    result={'scope':'one dataset and optimizer seed; matched steps, unequal capacity; state arms receive extra labels and a physical task-cost interface','training':{},'conditions':{}}
    lines=['# Matched-data visual architecture / target feasibility results','',result['scope'],'',
           '| Arm | Parameters | Selected epoch | Final prediction loss, validation | Median training clips/s |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        run=training/arm;r=json.loads((run/'summary.json').read_text());assert len(r['epochs'])==100 and r['max_steps']==0
        for file,h in json.loads((run/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((run/file).read_bytes()).hexdigest()==h
        speed=float(np.median([e['train']['sequences']/e['train']['seconds'] for e in r['epochs'][1:]]));result['training'][arm]={'parameters':r['parameters'],'best_epoch':r['best_epoch'],'median_training_clips_per_second':speed,'wall_seconds':r['wall_seconds'],'epochs':r['epochs'],'seed':r['seed'],'data_manifest_sha256':r['data_manifest_sha256']}
        lines.append(f'| {arm} | {r["parameters"]} | {r["best_epoch"]} | {r["epochs"][-1]["validation"]["prediction_loss"]:.5f} | {speed:.1f} |')
    lines+=['','Prediction losses have different targets across objectives; their magnitudes are not an algorithm-ranking metric.','']
    banks=[]
    for condition,folder in [('inview',a.inview),('stress',a.stress)]:
        rows={};bank_hash=None;identity=None
        lines += [f'## {condition}','', '| Arm / checkpoint | Block rank ρ | Native block cost | Zero-action block cost | Native agent + block cost |','|---|---:|---:|---:|---:|']
        for arm in ARMS:
            for checkpoint in ['best','last']:
                run=Path(folder)/arm/checkpoint;s=json.loads((run/'summary.json').read_text());accepted=json.loads((run/'acceptance.json').read_text());assert accepted['accepted_cases']==32 and accepted['arm']==arm and accepted['checkpoint']==checkpoint
                assert s['precision']['matmul_allow_tf32'] is False and s['precision']['cudnn_allow_tf32'] is False and s['precision']['matmul_precision']=='highest'
                for file,h in json.loads((run/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((run/file).read_bytes()).hexdigest()==h
                assert s['hashes']['training_summary']==hashlib.sha256((training/arm/'summary.json').read_bytes()).hexdigest()
                assert s['hashes']['weights']==hashlib.sha256((training/arm/f'{checkpoint}_weights.pt').read_bytes()).hexdigest()
                if bank_hash is None:bank_hash=s['hashes']['bank_manifest'];identity=[(r['index'],r['seed']) for r in s['cases']]
                else:assert bank_hash==s['hashes']['bank_manifest'] and identity==[(r['index'],r['seed']) for r in s['cases']]
                rows[arm+'_'+checkpoint]={'aggregate':s['aggregate'],'cases':s['cases']};b=s['aggregate']['block_pose'];c=s['aggregate']['agent_and_block_pose'];rho=b['native']['spearman'];fmt='—' if rho is None else f'{rho:.3f}'
                lines.append(f'| {arm} / {checkpoint} | {fmt} | {b["native"]["selected_cost"]:.2f} | {b["zero_action"]["selected_cost"]:.2f} | {c["native"]["selected_cost"]:.2f} |')
        banks.append(bank_hash);result['conditions'][condition]={'bank_manifest_sha256':bank_hash,'routes':rows};lines.append('')
    assert banks[0]!=banks[1]
    lines+=['Both checkpoints were prespecified. No task-based selection among them is performed. In-view and stress conditions change both visibility conditioning and candidate support, so their contrast is not a pure causal visibility intervention.','',
            'This feasibility study does not supply equal-observation pixel baselines, multiple independent training datasets, tuning parity across architecture families or a main-track novelty claim.']
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
