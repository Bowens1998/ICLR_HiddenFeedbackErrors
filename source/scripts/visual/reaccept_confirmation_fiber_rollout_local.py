"""Local independent decoding; retain canonical remote statistics and report rounding."""
import argparse,json,subprocess,sys,tempfile,shutil
from pathlib import Path
from adaptation_streams import sha
from reaccept_task_coordinate_horizon_local import differences


def main():
    p=argparse.ArgumentParser()
    for k in ('runs','plan','protocol','horizon','fibers','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();rows=[]
    for g in range(6):
        d=Path(a.runs)/f'job_{g}'
        with tempfile.TemporaryDirectory(prefix='jrs-fiber-reaccept-') as td:
            t=Path(td)
            for fp in d.iterdir():
                if fp.name not in ('acceptance.json','local_acceptance.json'):(t/fp.name).symlink_to(fp.resolve())
            cmd=[sys.executable,str(Path(__file__).with_name('accept_confirmation_fiber_rollout.py')),'--run',str(t)]
            for name in ('plan','protocol','horizon','fibers'):cmd+=['--'+name,getattr(a,name)]
            subprocess.run(cmd,check=True);local=json.loads((t/'acceptance.json').read_text());remote=json.loads((d/'acceptance.json').read_text());diff=differences(local,remote)
            assert all(('/metrics/' in key and ('/position_mse/' in key or '/angle_mse/' in key)) or key.endswith('/max_initial_normalized_readout_error') for key,delta in diff),diff[:10]
            shutil.copyfile(t/'acceptance.json',d/'local_acceptance.json')
        rows.append(dict(group=g,canonical_acceptance_sha256=sha(d/'acceptance.json'),local_acceptance_sha256=sha(d/'local_acceptance.json'),report_sha256=sha(d/'report.json'),different_scalars=len(diff),max_position_mse_difference=max((v for k,v in diff if '/position_mse/' in k),default=0),max_angle_mse_difference=max((v for k,v in diff if '/angle_mse/' in k),default=0),max_readout_error_difference=max((v for k,v in diff if k.endswith('/max_initial_normalized_readout_error')),default=0),precision_token_metrics_and_bindings_exact=True))
    result=dict(status='PASS6_LOCAL_CONFIRMATION_FIBER_ROLLOUT_REACCEPTANCES',rows=rows,source_sha256=sha(__file__),scope='Unchanged independent decoder and readout-preservation tolerances. Cross-platform FP64 metric/readout rounding separately reported. Exact saved-array free trajectories, correction/reset identities and bindings checked.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
