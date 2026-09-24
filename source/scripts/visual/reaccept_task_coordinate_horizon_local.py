"""Re-run independent decoding locally, retaining canonical remote acceptance."""
import argparse,json,subprocess,sys,tempfile,shutil
from pathlib import Path
from adaptation_streams import sha

def differences(x,y,path=''):
    if isinstance(x,dict):
        assert x.keys()==y.keys()
        return sum((differences(x[k],y[k],path+'/'+k) for k in x),[])
    if isinstance(x,list):
        assert len(x)==len(y)
        return sum((differences(a,b,path+'/'+str(i)) for i,(a,b) in enumerate(zip(x,y))),[])
    if x==y:return []
    assert isinstance(x,(int,float)) and isinstance(y,(int,float)),(path,x,y)
    return [(path,abs(x-y))]

def main():
    p=argparse.ArgumentParser()
    for k in ('runs','plan','protocol','matched','frames','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();rows=[]
    for g in range(6):
        d=Path(a.runs)/f'job_{g}';r=json.loads((d/'report.json').read_text())
        for b in r['trajectory_bindings']:
            fp=Path(a.frames)/f"job_{b['route']}"/'report.json';assert sha(fp)==b['report_sha256'];fr=json.loads(fp.read_text());assert fr['source_sha256']==sha(Path(__file__).with_name('extract_task_coordinate_horizon_frames.py')) and fr['protocol_sha256']==sha(a.protocol)
            ref=next(x for x in r['reference_bindings'] if x['route']==b['route']);assert fr['summary_sha256']==ref['summary_sha256'] and fr['acceptance_sha256']==ref['acceptance_sha256']
            for source,case in zip(fr['cases'],ref['cases']):assert source['index']==case['case'] and source['source_archive_sha256']==case['archive_sha256']
        with tempfile.TemporaryDirectory(prefix='jrs-horizon-reaccept-') as td:
            temp=Path(td)
            for fp in d.iterdir():
                if fp.name not in ('acceptance.json','local_acceptance.json'): (temp/fp.name).symlink_to(fp.resolve())
            subprocess.run([sys.executable,str(Path(__file__).with_name('accept_task_coordinate_horizon.py')),'--run',str(temp),'--plan',a.plan,'--protocol',a.protocol,'--matched',a.matched],check=True)
            local=json.loads((temp/'acceptance.json').read_text());canonical=json.loads((d/'acceptance.json').read_text());diff=differences(local,canonical)
            # The verifier independently reconstructs poses within its unchanged tolerance.
            # Metrics use those reconstructed poses, so BLAS last bits can affect MSE.
            # This recovery report is not an additional cross-platform metric gate.
            assert all('/metrics/' in key and ('/position_mse/' in key or '/angle_mse/' in key) for key,delta in diff),diff[:10]
            shutil.copyfile(temp/'acceptance.json',d/'local_acceptance.json')
        rows.append(dict(group=g,report_sha256=sha(d/'report.json'),canonical_acceptance_sha256=sha(d/'acceptance.json'),local_acceptance_sha256=sha(d/'local_acceptance.json'),different_scalars=len(diff),max_position_mse_difference=max((delta for key,delta in diff if '/position_mse/' in key),default=0),max_angular_difference=max((delta for key,delta in diff if '/angle_mse/' in key),default=0),all_other_fields_exact=True))
    result=dict(status='PASS6_LOCAL_HORIZON_NUMPY_REACCEPTANCES',rows=rows,source_sha256=sha(__file__),scope='Local saved-array reconstruction and exact endpoint/first-step/common-input anchors. Frame report hashes and source archive bindings checked; intermediate raw images remain on HPG and are not resimulated locally. Position/angle MSE platform differences reported separately, no change to decoder acceptance tolerances.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
