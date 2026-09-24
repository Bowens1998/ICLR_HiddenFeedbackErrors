"""Exclude every discovered prior PushT bank's seed interval and realized cases."""
import argparse,hashlib,json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['config','data-root','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();cfg=json.loads(Path(a.config).read_text());start=cfg['evaluation_seed_start'];stop=start+cfg['evaluation_max_seeds'];rows=[]
 for f in sorted(Path(a.data_root).rglob('manifest.json')):
  if 'pusht' not in str(f).lower():continue
  r=json.loads(f.read_text());cases=r.get('cases',[])
  if not isinstance(cases,list):continue
  seeds=[v['seed'] for v in cases if isinstance(v,dict) and isinstance(v.get('seed'),int)]
  seeds += [v['seed'] for v in r.get('rejected_seeds',[]) if isinstance(v,dict) and isinstance(v.get('seed'),int)]
  interval=None
  if isinstance(r.get('seed_start'),int) and isinstance(r.get('max_seeds'),int):
   lo=r['seed_start'];hi=lo+r['max_seeds'];assert stop<=lo or start>=hi,(str(f),lo,hi);interval=[lo,hi]
  assert not any(start<=s<stop for s in seeds),str(f)
  if seeds or interval:rows.append(dict(path=str(f),sha256=sha(f),interval_half_open=interval,recorded_seeds=len(set(seeds))))
 assert rows,'No existing goal banks found: missing inventory is not proof of disjointness'
 result=dict(status='PASS',new_interval_half_open=[start,stop],config_sha256=sha(a.config),banks=rows,source_sha256=sha(__file__),scope='All discovered PushT manifest files under specified data root; prior declared ranges and accepted/rejected case seeds. No claim of independence beyond these inventoried banks.')
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('PASS',len(rows),'bank manifests')


if __name__=='__main__':main()
