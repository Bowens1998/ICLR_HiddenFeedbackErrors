"""Freeze the complete four-fit roster before any C/D qualification."""
import argparse
from pathlib import Path
from s1_common import atomic_json, checked_json, load_protocol, sha


def freeze(cfg, protocol_sha, bindings, bindings_sha, output):
    if Path(output).exists(): raise ValueError('Selected-head lock already exists; never overwrite')
    expected={(g,h) for g in cfg['groups'] for h in cfg['head_roles']}
    rows=bindings['fits']
    if len(rows)!=4 or {(r['group'],r['head_role']) for r in rows}!=expected:
        raise ValueError('All four unique fits are required before qualification')
    selected=[]; views=set()
    for row in sorted(rows,key=lambda r:(r['group'],r['head_role'])):
        folder=Path(row['path']); report=checked_json(folder/'report.json',row['report_sha256'])
        if not (folder/'DONE').exists() or report.get('status')!='PASS_S1_FIXED_HEAD_FIT':
            raise ValueError('Incomplete fixed fit')
        for key,value in [('protocol_sha256',protocol_sha),('group',row['group']),('head_role',row['head_role']),('updates',8000)]:
            if report.get(key)!=value: raise ValueError('Fit identity/configuration mismatch')
        if [r['step'] for r in report['history']]!=list(range(250,8001,250)):
            raise ValueError('Incomplete frozen validation schedule')
        best=min(report['history'],key=lambda r:(r['balanced_six_normalized_validation_mse'],r['step']))
        if report['selected']['step']!=best['step'] or report['selected']['balanced_six_normalized_validation_mse']!=best['balanced_six_normalized_validation_mse']:
            raise ValueError('Validation-only checkpoint rule changed')
        cp=folder/'selected.npz'; fixture=folder/'forward_verification.npz'
        if sha(cp)!=report['selected']['sha256'] or sha(fixture)!=report['forward_fixture_sha256']:
            raise ValueError('Frozen head or forward fixture changed')
        checked_json(Path(report['views'])/'report.json',report['views_report_sha256'])
        views.add((report['views'],report['views_report_sha256']))
        selected.append(dict(group=row['group'],head_role=row['head_role'],checkpoint=str(cp.resolve()),
            checkpoint_sha256=sha(cp),fit_report=str((folder/'report.json').resolve()),fit_report_sha256=row['report_sha256'],
            views=report['views'],views_report_sha256=report['views_report_sha256'],selected_step=best['step'],
            forward_fixture=str(fixture.resolve()),forward_fixture_sha256=sha(fixture)))
    if len(views)!=1: raise ValueError('Four fits must share the same frozen C/D view split')
    atomic_json(output,dict(status='ALL_FOUR_S1_HEADS_FROZEN_BEFORE_QUALIFICATION',protocol_sha256=protocol_sha,
        fit_bindings_sha256=bindings_sha,heads=selected,formal_authorized=False,source_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser()
    for k in ['protocol','protocol-sha256','fit-bindings','fit-bindings-sha256','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args(); cfg=load_protocol(a.protocol,a.protocol_sha256)
    freeze(cfg,a.protocol_sha256,checked_json(a.fit_bindings,a.fit_bindings_sha256),a.fit_bindings_sha256,a.output)


if __name__=='__main__':main()
