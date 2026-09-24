"""Accept complete coverage and reverify actual inserted tokens before rollout."""
import argparse
from pathlib import Path
import numpy as np
from s1_common import atomic_json, checked_json, load_protocol, namespace_seed, sha
from s1_projection import verify_token, MEMBERS
from project_s1 import read_cache, validate_qualification


def main():
    p = argparse.ArgumentParser()
    for key in ('protocol', 'bindings', 'selected-lock', 'qualification', 'a-bindings', 'recipient-cache', 'donor-cache'):
        p.add_argument('--'+key, required=True)
        p.add_argument('--'+key+'-sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    cfg = load_protocol(a.protocol, a.protocol_sha256)
    bindings = checked_json(a.bindings, a.bindings_sha256)
    selected = checked_json(a.selected_lock, a.selected_lock_sha256)
    qual = checked_json(a.qualification, a.qualification_sha256)
    validate_qualification(selected, qual, a.protocol_sha256, a.selected_lock_sha256, a.a_bindings_sha256)
    references = checked_json(a.a_bindings, a.a_bindings_sha256)
    group, stream, kind = [bindings[k] for k in ('group', 'stream', 'kind')]
    if group not in cfg['groups'] or stream not in cfg['scores']['streams'] or kind not in ['development','confirmation']:
        raise ValueError('Unknown population')
    count = cfg['scores']['goal_count'] if kind == 'confirmation' else cfg['scores']['development_goal_count']
    _, free, recipient_seeds = read_cache(a.recipient_cache, a.recipient_cache_sha256, a.protocol_sha256,
                           kind, 'recipient', group, stream, count)
    _, _, donor_seeds = read_cache(a.donor_cache, a.donor_cache_sha256, a.protocol_sha256, kind, 'donor', group, stream, count)
    if set(recipient_seeds.tolist()) & set(donor_seeds.tolist()):raise ValueError('Recipient/donor parent overlap')
    ref = next(r for r in references['groups'] if r['group']==group)['head_A']
    c = next(r for r in selected['heads'] if r['group']==group and r['head_role']=='C')
    if sha(ref['path']) != ref['sha256'] or sha(c['checkpoint']) != c['checkpoint_sha256']:
        raise ValueError('Head identity changed')
    with np.load(ref['path'],allow_pickle=False) as f: ha=dict(f)
    with np.load(c['checkpoint'],allow_pickle=False) as f: hc=dict(f)
    perm = np.random.default_rng(namespace_seed(cfg['root_seed'],cfg['donor_assignment']['seed_namespace'].format(
        kind=kind,stream=stream))).permutation(count)
    covered=np.zeros(count,bool); rows=[]; qualification=set(); max_residual=0.
    for row in sorted(bindings['shards'],key=lambda x:x['start']):
        report=checked_json(row['report'],row['report_sha256'])
        expected=dict(status='ACCEPTED_COMPLETE_SHARD',protocol_sha256=a.protocol_sha256,
            selected_lock_sha256=a.selected_lock_sha256,a_bindings_sha256=a.a_bindings_sha256,
            qualification_sha256=a.qualification_sha256,
            recipient_cache_sha256=a.recipient_cache_sha256,donor_cache_sha256=a.donor_cache_sha256,
            head_A_sha256=ref['sha256'],head_C_sha256=c['checkpoint_sha256'],
            group=group,stream=stream,kind=kind,count=count,start=row['start'],stop=row['stop'])
        if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('Shard identity mismatch')
        start,stop=row['start'],row['stop']; n=stop-start
        if not 0<=start<stop<=count or covered[start:stop].any():raise ValueError('Overlap or invalid shard range')
        arr=report['arrays']
        if sha(arr['path'])!=arr['sha256']:raise ValueError('Changed projection payload')
        with np.load(arr['path'],allow_pickle=False) as f:
            repl=f['replacements']; direction=f['directions']; norm=f['common_norm']; ix=f['goal_indices']; donor=f['donor_indices']
        if repl.shape!=(2,2,n,2,192) or repl.dtype!=np.float32 or direction.shape!=repl.shape or direction.dtype!=np.float64:
            raise ValueError('Wrong complete-family payload')
        if norm.shape!=(n,) or not np.isfinite(norm).all() or (norm<0).any() or not np.isfinite(direction).all():
            raise ValueError('Invalid matched norm/direction')
        np.testing.assert_array_equal(ix,np.arange(start,stop,dtype=np.int64))
        np.testing.assert_array_equal(donor,perm[start:stop])
        if len(report['matching_reports'])!=n:raise ValueError('Missing family receipts')
        for j,goal in enumerate(range(start,stop)):
            r=report['matching_reports'][j]
            if r['goal']!=goal or r['status']!='ACCEPTED_S1_EIGHT_MEMBER_FAMILY' or r['family_size']!=8:
                raise ValueError('Wrong family receipt')
            if (r['members']!=MEMBERS or len(r['solvers'])!=8 or r['effective_norm']!=norm[j] or
                    not all(s['status'].lower()=='solved' and s['head_count']==(1 if k<4 else 2)
                            for k,s in enumerate(r['solvers']))):
                raise ValueError('Norm or solver receipt mismatch')
            if (r['recipient_seed']!=int(recipient_seeds[goal]) or r['donor_index']!=int(perm[goal]) or
                    r['donor_seed']!=int(donor_seeds[perm[goal]])):
                raise ValueError('Donor/recipient identity changed')
            member_axes=[(cfg['objectives'].index(m['objective']),cfg['scores']['constraints'].index(m['condition']),
                          cfg['projection']['sources'].index(m['source'])) for m in MEMBERS]
            native=np.array([np.linalg.norm(direction[oi,ci,j,si]) for oi,ci,si in member_axes])
            np.testing.assert_allclose(native,r['native_norms'],rtol=1e-12,atol=1e-12)
            target=float(native.min());shrink=r['shrink_factor']
            if (shrink not in cfg['projection']['shrink_factors'] or r['unshrunk_common_norm']!=target or
                    norm[j]!=target*shrink or r['legitimate_zero_norm']!=(target==0)):
                raise ValueError('Shared dose differs from frozen eight-direction rule')
            prior=cfg['projection']['shrink_factors'][:cfg['projection']['shrink_factors'].index(shrink)]
            if len(r['attempts'])!=len(prior)+1 or [x['shrink'] for x in r['attempts']]!=prior+[shrink]:
                raise ValueError('Incomplete or altered numerical backoff history')
            for earlier in prior:
                earlier_checks=[]
                for native_norm,(oi,ci,si) in zip(native,member_axes):
                    alpha=0. if native_norm==0 else earlier*target/native_norm
                    candidate=(free[oi,goal].astype(np.float64)+alpha*direction[oi,ci,j,si]*ha['scale']).astype(np.float32)
                    hs=[ha] if ci==0 else [ha,hc]
                    earlier_checks.append(verify_token(free[oi,goal],candidate,hs,ha,expected_norm=earlier*target)['accepted'])
                if all(earlier_checks):raise ValueError('A passing earlier shrink was skipped')
            for native_norm,(oi,ci,si) in zip(native,member_axes):
                alpha=0. if native_norm==0 else shrink*target/native_norm
                reconstructed=(free[oi,goal].astype(np.float64)+alpha*direction[oi,ci,j,si]*ha['scale']).astype(np.float32)
                np.testing.assert_array_equal(reconstructed,repl[oi,ci,j,si])
            for oi in range(2):
                for ci,heads in enumerate(([ha],[ha,hc])):
                    for si in range(2):
                        check=verify_token(free[oi,goal],repl[oi,ci,j,si],heads,ha,expected_norm=norm[j])
                        if not check['accepted']:raise ValueError('Independent full-function/norm recheck failed')
                        max_residual=max(max_residual,max(x['normalized_output_deviation'] for x in check['heads']))
        covered[start:stop]=True;qualification.add(report['qualification_sha256'])
        rows.append(dict(start=start,stop=stop,report=row['report'],report_sha256=row['report_sha256'],
                         arrays=arr['path'],arrays_sha256=arr['sha256']))
    if not covered.all() or len(qualification)!=1:raise ValueError('Missing population or inconsistent qualification')
    if Path(a.output).exists():raise ValueError('Never overwrite accepted projection locks')
    atomic_json(a.output,dict(status='ALL_S1_QP_SHARDS_ACCEPTED',protocol_sha256=a.protocol_sha256,
        kind=kind,group=group,stream=stream,count=count,selected_lock_sha256=a.selected_lock_sha256,
        qualification_sha256=next(iter(qualification)),recipient_cache_sha256=a.recipient_cache_sha256,
        donor_cache_sha256=a.donor_cache_sha256,shards=rows,maximum_rechecked_output_deviation=max_residual,
        bindings_sha256=a.bindings_sha256,source_sha256=sha(__file__),all_goals_covered=True))


if __name__=='__main__':main()
