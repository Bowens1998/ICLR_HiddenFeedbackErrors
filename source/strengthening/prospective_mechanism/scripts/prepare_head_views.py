"""Input-only C/D whole-parent views of accepted immutable v8 observed caches."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from s1_common import (GROUPS, OLD_ROLES, DOMAINS, HEAD_ROLES, PARENT_COUNTS, SOURCE_CACHE_PROTOCOL,
    atomic_json, checked_json, load_protocol, namespace_seed, sha, source_hashes, view_role)


def parent_assignment(parents, domain, old_role, root_seed, namespace):
    ids = sorted(set(map(int, parents)))
    if not ids or len(ids) % 2:
        raise ValueError('Complete even parent roster required')
    def key(p):
        ns = namespace.format(domain=domain, old_role=old_role, parent_id=p)
        return (namespace_seed(root_seed, ns), p)
    ordered = sorted(ids, key=key)
    half = len(ordered) // 2
    return {'C': sorted(ordered[:half]), 'D': sorted(ordered[half:])}


def partition_indices(parent_ids, pixel_ids, assignment):
    parent_ids, pixel_ids = np.asarray(parent_ids), np.asarray(pixel_ids)
    if parent_ids.ndim != 1 or pixel_ids.shape != parent_ids.shape or parent_ids.dtype.kind not in 'iu':
        raise ValueError('Invalid parent/pixel identity arrays')
    if set(assignment['C']) & set(assignment['D']) or set(parent_ids.tolist()) != set(assignment['C'] + assignment['D']):
        raise ValueError('Overlapping or incomplete parent assignment')
    rows = {h: np.flatnonzero(np.isin(parent_ids, assignment[h])).astype(np.int64) for h in HEAD_ROLES}
    if not len(rows['C']) or not len(rows['D']) or set(pixel_ids[rows['C']]) & set(pixel_ids[rows['D']]):
        raise ValueError('Empty half or C/D pixel overlap')
    if not np.array_equal(np.sort(np.r_[rows['C'], rows['D']]), np.arange(len(parent_ids))):
        raise ValueError('Source row coverage changed')
    return rows


def audit_exposures(exposures):
    """Across both encoders and domains, forbid any new-head/old-role sharing."""
    conflicts=[]; permitted=[]; checked=0
    fields=['group','head_role','old_role','domain','rows','parents']
    for i,left in enumerate(exposures):
        for right in exposures[i+1:]:
            overlap=left['pixels'] & right['pixels']; checked+=1
            if not overlap:continue
            row=dict(left={k:left[k] for k in fields},right={k:right[k] for k in fields},
                     overlap_count=len(overlap))
            if left['head_role']!=right['head_role'] or left['old_role']!=right['old_role']:
                conflicts.append(dict(**row,example_pixel_hashes=sorted(overlap)[:8]))
            else:permitted.append(row)
    return dict(status='PASS_GLOBAL_PIXEL_ISOLATION' if not conflicts else 'FAIL_GLOBAL_PIXEL_ISOLATION',
        exposure_count=len(exposures),compared_pairs=checked,forbidden_overlap_pairs=conflicts,
        permitted_same_head_same_old_role_overlap_pairs=permitted,
        exposure_counts=[{k:e[k] for k in fields} for e in exposures])


def prepare(cfg, protocol_sha, lock, lock_sha, output):
    by_group = {r['group']: r for r in lock['groups']}
    if not all(g in by_group for g in GROUPS) or lock.get('protocol_sha256') != SOURCE_CACHE_PROTOCOL:
        raise ValueError('Missing groups or wrong accepted cache protocol')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    assignments, views = {}, {}
    identities = {g: {} for g in GROUPS}
    exposures = []
    for g in GROUPS:
        row = by_group[g]; cache = Path(row['cache'])
        report = checked_json(cache / 'report.json', row['cache_report_sha256'])
        if report.get('status') != 'PASS_G_EVAL_OBSERVED_CACHE' or report.get('group') != g or report.get('protocol_sha256') != SOURCE_CACHE_PROTOCOL:
            raise ValueError('Unaccepted source cache')
        for old_role in OLD_ROLES:
            for domain in DOMAINS:
                stem = f'{old_role}_{domain}'
                p = cache / (stem + '.npz'); mp = cache / (stem + '.json')
                if sha(p) != row['arrays'][p.name] or report['files'][p.name] != row['arrays'][p.name]:
                    raise ValueError('Immutable source array changed')
                meta = checked_json(mp, report['metadata_files'][mp.name])
                expected = dict(role='g_eval_' + old_role, domain=domain, group=g,
                    protocol_sha256=SOURCE_CACHE_PROTOCOL, arrays_sha256=sha(p))
                if any(meta.get(k) != v for k, v in expected.items()):
                    raise ValueError('Wrong old cache role')
                with np.load(p, allow_pickle=False) as z:
                    parents, pixels = z['parent_ids'], z['pixel_sha256']
                    if z['observed'].shape != (len(parents), 192) or z['labels'].shape != (len(parents), 6):
                        raise ValueError('Invalid observed cache dimensions')
                parent_list = sorted(np.unique(parents).tolist())
                if len(parent_list) != PARENT_COUNTS[domain][old_role] or parent_list != report['parent_ids'][old_role + '/' + domain]:
                    raise ValueError('Parent population changed')
                key = domain + '/' + old_role
                assignment = parent_assignment(parent_list, domain, old_role, cfg['root_seed'], cfg['split_seed_namespace'])
                if key in assignments and assignments[key] != assignment:
                    raise ValueError('Architecture groups disagree on the shared parent split')
                assignments[key] = assignment
                indices = partition_indices(parents, pixels, assignment)
                identities[g][key] = dict(parents=set(parent_list), pixels=set(pixels.tolist()))
                for h in HEAD_ROLES:
                    folder = out / f'group_{g}' / h; folder.mkdir(parents=True, exist_ok=True)
                    ix = indices[h]; ip = folder / (stem + '_rows.npy')
                    exposures.append(dict(group=g,head_role=h,old_role=old_role,domain=domain,
                        pixels=set(pixels[ix].tolist()),rows=len(ix),parents=len(assignment[h])))
                    with ip.open('xb') as f: np.save(f, ix, allow_pickle=False)
                    metadata = dict(role=view_role(h, old_role), head_role=h, old_role=old_role,
                        group=g, domain=domain, protocol_sha256=protocol_sha,
                        source_cache_lock_sha256=lock_sha, source_cache_report_sha256=row['cache_report_sha256'],
                        source_arrays=str(p.resolve()), source_arrays_sha256=sha(p),
                        source_metadata=str(mp.resolve()), source_metadata_sha256=sha(mp),
                        input_manifest_sha256=meta['input_manifest_sha256'],
                        encoder_checkpoint_sha256=meta['encoder_checkpoint_sha256'],
                        row_indices=str(ip.relative_to(out)), row_indices_sha256=sha(ip),
                        rows=len(ix), parent_ids=assignment[h], parents=len(assignment[h]),
                        pixel_array_sha256=hashlib.sha256(np.ascontiguousarray(pixels[ix]).tobytes()).hexdigest(),
                        policy='Exact retained v8 rows only; no alias restoration or relabeling')
                    dest = folder / (stem + '.json'); atomic_json(dest, metadata)
                    views[f'group_{g}/{h}/{stem}'] = dict(metadata=str(dest.relative_to(out)), metadata_sha256=sha(dest))
    for g in GROUPS:
        items = list(identities[g].items())
        for i, (left, a) in enumerate(items):
            for right, b in items[i + 1:]:
                if left.split('/')[0] == right.split('/')[0] and a['parents'] & b['parents']:
                    raise ValueError('Old-role parent overlap')
    pixel_audit = audit_exposures(exposures)
    if pixel_audit['forbidden_overlap_pairs']:
        atomic_json(out/'INPUT_ISOLATION_FAILED.json',pixel_audit)
        raise ValueError('Global cross-head/original-role pixel exposure; no fitting authorized')
    atomic_json(out / 'assignments.json', dict(protocol_sha256=protocol_sha, root_seed=cfg['root_seed'],
        namespace=cfg['split_seed_namespace'], assignments=assignments,
        rule='Sort derived uint32 from first4 SHA256(root:namespace) bytes, then integer parent id; first half C, remaining half D; shared across groups'))
    atomic_json(out / 'report.json', dict(status='PASS_S1_DISJOINT_PARENT_VIEWS', protocol_sha256=protocol_sha,
        source_cache_lock_sha256=lock_sha, assignments_sha256=sha(out / 'assignments.json'),
        views=views, global_pixel_audit=pixel_audit, source_sha256=source_hashes(), prepare_source_sha256=sha(__file__),
        scope='Read-only views of accepted observed arrays. No head predictions, fit, intervention arrays or effects accessed.'))
    (out / 'DONE').write_text('complete_disjoint_views\n')


def main():
    p = argparse.ArgumentParser()
    for k in ['protocol','protocol-sha256','cache-lock','cache-lock-sha256','output']: p.add_argument('--'+k, required=True)
    a = p.parse_args(); cfg = load_protocol(a.protocol, a.protocol_sha256)
    prepare(cfg, a.protocol_sha256, checked_json(a.cache_lock, a.cache_lock_sha256), a.cache_lock_sha256, a.output)


if __name__ == '__main__': main()
