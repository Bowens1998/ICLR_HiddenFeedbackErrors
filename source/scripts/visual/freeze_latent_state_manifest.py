"""Freeze all accepted state-head checkpoints by the prespecified validation rule."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',choices=['best','last'],default='best');a=p.parse_args();root=Path(a.training)
acc=json.loads((root/'acceptance.json').read_text());arms=['transformer_latent_state','gru_latent_state'];assert acc['accepted_arms']==arms
models=[]
for arm in arms:
 r=json.loads((root/arm/'summary.json').read_text());assert (root/arm/'COMPLETE').exists()
 models.append({'arm':arm,'checkpoint':a.checkpoint,'weights_sha256':hashlib.sha256((root/arm/(a.checkpoint+'_weights.pt')).read_bytes()).hexdigest(),'training_summary_sha256':hashlib.sha256((root/arm/'summary.json').read_bytes()).hexdigest(),'updates':r['completed_updates'],'scope':'extra state labels; 192-dimensional recurrent token with shared state head'})
with Path(a.output).open('x') as f:json.dump({'models':models,'selection':'all accepted arms; checkpoint type fixed before task evaluation; no task-based seed selection'},f,indent=2);f.write('\n')
