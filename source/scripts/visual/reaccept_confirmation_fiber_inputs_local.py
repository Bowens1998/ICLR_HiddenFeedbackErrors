"""Local numerical checks of both full and externally guided confirmation inputs."""
import argparse,json,subprocess,sys,tempfile,shutil
from pathlib import Path
from adaptation_streams import sha
from reaccept_task_coordinate_horizon_local import differences


def main():
    p=argparse.ArgumentParser()
    for k in ('full','inputs','horizon','donor-horizon','protocol','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();rows=[]
    for g in range(6):
        for tag,base,verifier,arguments in [('full',a.full,'accept_readout_fiber_rollout_inputs.py',['--inputs',a.horizon]),('paired',a.inputs,'accept_confirmation_fiber_inputs.py',['--horizon',a.horizon,'--full',a.full,'--donor-horizon',a.donor_horizon])]:
            d=Path(base)/f'job_{g}'
            with tempfile.TemporaryDirectory(prefix='jrs-confirm-input-check-') as td:
                t=Path(td)
                for fp in d.iterdir():
                    if fp.name not in ('acceptance.json','local_acceptance.json'):(t/fp.name).symlink_to(fp.resolve())
                result=subprocess.run([sys.executable,str(Path(__file__).with_name(verifier)),'--run',str(t),'--protocol',a.protocol,*arguments],capture_output=True,text=True)
                if result.returncode:print(result.stdout,result.stderr);raise RuntimeError((g,tag,result.returncode))
                local=json.loads((t/'acceptance.json').read_text());remote=json.loads((d/'acceptance.json').read_text());diff=differences(local,remote);assert all(delta<1e-8 for key,delta in diff),diff[:5];shutil.copyfile(t/'acceptance.json',d/'local_acceptance.json')
            rows.append(dict(group=g,input_kind=tag,canonical_sha256=sha(d/'acceptance.json'),local_sha256=sha(d/'local_acceptance.json'),different_scalars=len(diff),max_rounding=max((v for k,v in diff),default=0)));print('PASS',g,tag,flush=True)
    result=dict(status='PASS12_LOCAL_CONFIRMATION_INPUT_RECONSTRUCTIONS',rows=rows,source_sha256=sha(__file__),scope='Unchanged original full-projection and paired-donor validators; all tokens, readouts, donor identities and matching tolerances checked. Diagnostic scalar rounding recorded separately.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
