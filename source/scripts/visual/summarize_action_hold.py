"""Independent paired-data QA and physical prediction-error reconstruction."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for k in ['bank','runs','selection','output']:p.add_argument('--'+k,required=True)
a=p.parse_args();bank=Path(a.bank);runs=Path(a.runs);frozen=json.loads(Path(a.selection).read_text());assert frozen['heldout_indices']==[1,3,4]
manifest={c:json.loads((bank/c/'manifest.json').read_text()) for c in ['held','varied']};assert all(len(v['cases'])==64 for v in manifest.values());pairs=json.loads((bank/'paired_manifest.json').read_text());assert len(pairs['pairs'])==64
max_mean_error=0.;variation=[];truth={c:[] for c in manifest}
for i in range(64):
 pair={}
 for c in manifest:
  item=manifest[c]['cases'][i];path=bank/c/f'case_{i:03d}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];z=np.load(path);pair[c]={k:z[k] for k in z.files};assert int(z['seed'])==pairs['pairs'][i]['seed']==item['seed'];truth[c].append(pair[c]);assert z['actions'].shape==(17,25,2) and z['future_states'].shape==(17,5,7);assert np.isfinite(z['future_states']).all();assert np.max(abs(z['actions']))<=np.float32(.35)
 for key in ['history_pixels','history_states','prefix','goal_pixels','goal_state','goal_actions']:np.testing.assert_array_equal(pair['held'][key],pair['varied'][key])
 for c in pair:
  assert not pair[c]['actions'][0].any();np.testing.assert_array_equal(pair[c]['future_states'][:,-1],pair[c]['terminal_states'])
 for key in ['terminal_pixels','terminal_states','future_states']:np.testing.assert_array_equal(pair['held'][key][0],pair['varied'][key][0])
 ha=pair['held']['actions'].astype(float).reshape(17,5,5,2);va=pair['varied']['actions'].astype(float).reshape(17,5,5,2);d=float(np.max(abs(ha.mean(2)-va.mean(2))));assert d<=5e-8;max_mean_error=max(max_mean_error,d)
 assert np.max(abs(ha-ha[:,:,:1,:]))==0;variation.append(float(np.sqrt(np.mean((va[1:]-va[1:].mean(2,keepdims=True))**2))))
rows=[];draws=np.random.default_rng(954001).integers(0,64,(10000,64))
for index in [1,3,4]:
 errors={};secondary={}
 for c in manifest:
  run=runs/c/f'job_{index}';r=json.loads((run/'summary.json').read_text());assert (run/'acceptance.json').exists();assert r['hashes']['weights']==frozen['all_validation_candidates'][index]['weights_sha256'];assert r['hashes']['bank_manifest']==hashlib.sha256((bank/c/'manifest.json').read_bytes()).hexdigest()
  for name,h in json.loads((run/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h
  mn=np.array(r['target_normalization']['mean']);sd=np.array(r['target_normalization']['std']);values=[]
  for i in range(64):
   z=np.load(run/f'case_{i:03d}_predictions.npz');pred=z['predicted_future_tokens'].astype(float);assert pred.shape==(17,5,6);np.testing.assert_array_equal(pred[:,-1],z['predicted_tokens']);actual=truth[c][i]['future_states'];decoded=pred*sd+mn;delta=np.angle(np.exp(1j*(np.arctan2(decoded[...,4],decoded[...,5])-actual[...,4])));err=np.sum((decoded[...,2:4]-actual[...,2:4])**2,-1)+900*delta**2;assert np.isfinite(err).all();values.append(err[1:].mean(0))
  errors[c]=np.array(values);secondary[c]=r['aggregate']['block_pose']
 delta=errors['varied'][:,-1]-errors['held'][:,-1];rows.append({'index':index,'held_horizon_mean_error':errors['held'].mean(0).tolist(),'varied_horizon_mean_error':errors['varied'].mean(0).tolist(),'primary_varied_minus_held':float(delta.mean()),'paired_95_percentile_interval':np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),'per_context_horizon_errors':{k:v.tolist() for k,v in errors.items()},'secondary_task_metrics':secondary})
result={'scope':'frozen privileged state models; paired action timing changes actual physical trajectories; conditional feasibility sample, one training seed','cases':64,'candidate_pairs_per_case_excluding_zero':16,'horizons':[5,10,15,20,25],'data_qa':{'max_group_mean_difference':max_mean_error,'mean_varied_within_group_rms':float(np.mean(variation)),'history_goal_zero_replay_saved_arrays_and_hashes':'accepted'},'models':rows}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Paired action timing: physical prediction errors','',result['scope'],'','| Frozen model index | Held terminal error | Varied terminal error | Varied minus held | Paired 95% interval |','|---|---:|---:|---:|---|']
for r in rows:lines.append(f"| {r['index']} | {r['held_horizon_mean_error'][-1]:.2f} | {r['varied_horizon_mean_error'][-1]:.2f} | {r['primary_varied_minus_held']:.2f} | {r['paired_95_percentile_interval']} |")
out.with_suffix('.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
