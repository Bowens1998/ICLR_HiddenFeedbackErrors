"""Bind matched training identities before producing wide-state contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
from matched_wide_effects import compare


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


p=argparse.ArgumentParser()
for key in ['scaling-summary','wide-summary','training-audit','scaling-manifest','wide-manifests','output']:p.add_argument('--'+key,required=True)
a=p.parse_args();sp=Path(a.scaling_summary);wp=Path(a.wide_summary);ap=Path(a.training_audit);mp=Path(a.scaling_manifest)
scaling=json.loads(sp.read_text());wide=json.loads(wp.read_text());audit=json.loads(ap.read_text());manifest=json.loads(mp.read_text())
assert scaling['layout']=='scaling' and scaling['manifest_sha256']==sha(mp) and scaling['cases']==128
assert len(audit['comparisons'])==12
seen=set()
for pair in audit['comparisons']:
    replica,arch,target=pair['replica'],pair['architecture'],pair['baseline_target'];seen.add((replica,arch,target))
    for checkpoint in ['best','last']:
        entry=next(m for m in manifest['models'] if (m['replica'],m['arm'],m['checkpoint'],m['episodes'],m['updates'])==(replica,arch+'_'+target,checkpoint,256,21000))
        assert pair['baseline_summary_sha256']==entry['training_summary_sha256'] and pair['data_manifest_sha256']==entry['data_manifest_sha256']
        wm=Path(a.wide_manifests)/f'replica_{replica}_{checkpoint}.json';wide_entry=next(m for m in json.loads(wm.read_text())['models'] if m['arm']==arch+'_latent_state')
        assert pair['wide_summary_sha256']==wide_entry['training_summary_sha256']
        for row in wide['rows']:
            if (row['replica'],row['arm'],row['checkpoint'])==(replica,arch+'_latent_state',checkpoint):
                assert next(x for x in wide['provenance'] if x['route']==row['route'])['manifest_sha256']==sha(wm)
assert len(seen)==12
result=compare(scaling,wide);result.update(scaling_summary_sha256=sha(sp),wide_summary_sha256=sha(wp),matched_training_audit_sha256=sha(ap))
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print('MATCHED',len(result['contrasts']),'contrasts',len(result['aggregates']),'across-pool cells')
