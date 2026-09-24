"""Verify training metadata identities before matching wide-state and scaling results."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for key in ['wide-training','scaling-training','wide-manifests','scaling-manifest','output']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args();manifest=json.loads(Path(a.scaling_manifest).read_text());rows=[]
    for replica in range(3):
        for arch in ['transformer','gru']:
            relative=f'replica_{replica}/n256/u21000'
            wp=Path(a.wide_training)/relative/f'{arch}_latent_state/summary.json'
            wide=json.loads(wp.read_text())
            assert sha(wp)==json.loads((wp.parent/'artifact_manifest.json').read_text())['summary.json']
            for checkpoint in ['best','last']:
                wm=json.loads((Path(a.wide_manifests)/f'replica_{replica}_{checkpoint}.json').read_text())
                entry=next(m for m in wm['models'] if m['arm']==wide['arm'])
                assert entry['training_summary_sha256']==sha(wp)
            for target in ['state','jepa']:
                bp=Path(a.scaling_training)/relative/f'{arch}_{target}/summary.json'
                baseline=json.loads(bp.read_text())
                assert sha(bp)==json.loads((bp.parent/'artifact_manifest.json').read_text())['summary.json']
                for checkpoint in ['best','last']:
                    entry=next(m for m in manifest['models'] if m['training_path']==f'{relative}/{arch}_{target}' and m['checkpoint']==checkpoint)
                    assert entry['training_summary_sha256']==sha(bp)
                for key in ['data_manifest_sha256','config_sha256','clip_counts','seed','normalization','target_normalization','expected_updates','completed_updates','validation_every']:
                    assert wide[key]==baseline[key],(replica,arch,target,key)
                assert wide['completed_updates']==21000 and wide['seed']==3072
                shared={k:wide['initial_hashes'][k]==baseline['initial_hashes'][k] for k in ['encoder','action_encoder','temporal_core']}
                assert all(shared.values())
                rows.append(dict(replica=replica,architecture=arch,baseline_target=target,
                    wide_summary_sha256=sha(wp),baseline_summary_sha256=sha(bp),
                    data_manifest_sha256=wide['data_manifest_sha256'],reported_shared_initialization_matches=shared,
                    wide_parameters=wide['parameters'],baseline_parameters=baseline['parameters']))
    output=dict(comparisons=rows,scope='Frozen training summary and metadata equality, not a new replay of initial weights. Same dataset, clip counts, seed, normalization and update schedule; reported shared module initialization hashes match. Wide head and recurrent representation, parameter counts, supervision/objective and checkpoint selection can still differ. No planning performance claim.')
    Path(a.output).write_text(json.dumps(output,indent=2)+'\n')
    print('VERIFIED_MATCHED_TRAINING_METADATA',len(rows))


if __name__=='__main__':main()
