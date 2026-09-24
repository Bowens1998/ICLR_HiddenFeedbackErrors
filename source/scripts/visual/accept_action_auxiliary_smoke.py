"""Check integration coverage and retained artifacts; no performance claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
r=json.loads((root/'summary.json').read_text());assert (root/'COMPLETE').exists()
expected={(arm,mode) for arm in ['transformer_jepa','gru_jepa'] for mode in ['none','inverse','inverse_goal']}
assert len(r['rows'])==6 and {(x['arm'],x['mode']) for x in r['rows']}==expected
for name,h in r['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
for arm in ['transformer_jepa','gru_jepa']:
    rows=[x for x in r['rows'] if x['arm']==arm]
    assert all(x['initial_hashes']==rows[0]['initial_hashes'] for x in rows)
    for row in rows:
        assert row['steps']==8 and row['batch']==16 and len(row['losses'])==len(row['seconds_per_step'])==8
        assert np.isfinite(row['losses']).all() and min(row['seconds_per_step'])>0
        assert row['peak_allocated_bytes']>0
        np.testing.assert_equal(row['warm_median_seconds'],np.median(row['seconds_per_step'][2:]))
        file=root/f"{arm}_{row['mode']}.pt";assert hashlib.sha256(file.read_bytes()).hexdigest()==row['weights_sha256']
result={'accepted_cells':6,'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'complete engineering coverage, paired initialization, finite optimization and retained checkpoint hashes; no scientific performance acceptance'}
(root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
