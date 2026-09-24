"""Apply prespecified effects only to a complete verified scaling summary."""
import argparse,json,hashlib
from pathlib import Path
from scaling_effects import analyze
p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
f=Path(a.summary);r=json.loads(f.read_text());assert r['layout']=='scaling' and r['cases']==128
result=analyze(r['tables']);result.update(summary_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),manifest_sha256=r['manifest_sha256'],bank_manifest_sha256=r['bank_manifest_sha256'])
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print('ANALYZED',len(result['aggregates']),'cells and',len(result['effects']),'paired effects')
