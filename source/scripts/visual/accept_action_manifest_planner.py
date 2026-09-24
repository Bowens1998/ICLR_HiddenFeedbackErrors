"""Bind auxiliary mode identity, then use the existing physical trace verifier."""
import argparse,json,runpy,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--model-manifest',required=True);a,_=p.parse_known_args()
r=json.loads((Path(a.run)/'summary.json').read_text());f=json.loads(Path(a.model_manifest).read_text())
assert f['layout']=='action_auxiliary'
entry=f['models'][f['routes'][r['route_index']]['model_index']]
assert r['mode']==entry['mode'] and r['score_space']=='latent' and r['target_normalization'] is None
assert r['hashes']['bank_manifest']==f['bank_manifest_sha256']
runpy.run_path(str(Path(__file__).with_name('accept_manifest_planner.py')),run_name='__main__')

af=Path(a.run)/"acceptance.json"
accepted=json.loads(af.read_text());accepted["mode_binding_verifier_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();accepted["mode"]=r["mode"]
af.write_text(json.dumps(accepted,indent=2)+"\n")
