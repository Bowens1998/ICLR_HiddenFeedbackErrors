"""Independent synthetic rejection tests; no fitted heads or scientific results."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
import numpy as np

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / 'scripts'))
from project_s1 import validate_qualification, read_cache
from s1_common import sha


def valid_gate():
    selected = dict(status='ALL_FOUR_S1_HEADS_FROZEN_BEFORE_QUALIFICATION', protocol_sha256='protocol', heads=[])
    qualification = dict(status='COMPLETE_S1_QUALIFICATION_PASS', all_gates_passed=True,
        all_four_completed=True, selected_lock_sha256='selected', protocol_sha256='protocol',
        a_bindings_sha256='A', rows=[])
    for g in [0, 1]:
        for role in ['C', 'D']:
            row = dict(group=g, head_role=role, checkpoint_sha256=f'synthetic-{g}-{role}')
            selected['heads'].append(copy.deepcopy(row))
            row['gate'] = dict(passed=True, checks={d: {m: dict(passed=True, ratio=1.1)
                for m in ['six_normalized_mse', 'block_position_mse', 'agent_position_mse', 'wrapped_angle_mse']}
                for d in ['expert', 'planner']})
            qualification['rows'].append(row)
    return selected, qualification


class QualificationAdmission(unittest.TestCase):
    def test_complete_roster_and_exact_boundary_pass(self):
        s, q = valid_gate(); validate_qualification(s, q, 'protocol', 'selected', 'A')

    def test_every_incomplete_or_false_gate_rejects(self):
        mutations = [
            lambda s, q: q.update(all_gates_passed=False),
            lambda s, q: q.update(all_four_completed=False),
            lambda s, q: q.update(selected_lock_sha256='other'),
            lambda s, q: q.update(a_bindings_sha256='other'),
            lambda s, q: q['rows'].pop(),
            lambda s, q: s['heads'].pop(),
            lambda s, q: s['heads'].__setitem__(3, copy.deepcopy(s['heads'][2])),
            lambda s, q: q['rows'][3].update(checkpoint_sha256='wrong'),
            lambda s, q: q['rows'][3]['gate']['checks'].pop('expert'),
            lambda s, q: q['rows'][3]['gate']['checks']['expert'].pop('wrapped_angle_mse'),
            lambda s, q: q['rows'][3]['gate']['checks']['expert']['wrapped_angle_mse'].update(ratio=float('nan')),
            lambda s, q: q['rows'][3]['gate']['checks']['expert']['wrapped_angle_mse'].update(ratio=1.1000001),
            lambda s, q: q['rows'][3]['gate']['checks']['expert']['wrapped_angle_mse'].update(passed=False),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(corruption=i):
                s, q = valid_gate(); mutate(s, q)
                with self.assertRaises(ValueError): validate_qualification(s, q, 'protocol', 'selected', 'A')


class CacheAdmission(unittest.TestCase):
    def fixture(self, directory, role='recipient', edit=None):
        arrays = dict(observed=np.zeros((2, 5, 192), np.float32), seeds=np.array([31, 32], np.int64),
            truth=np.array([object()], dtype=object), predictions_D=np.array([object()], dtype=object))
        if role == 'recipient': arrays['free'] = np.zeros((2, 2, 5, 192), np.float32)
        
        report = dict(status='PASS_S1_OBSERVED_AND_FREE_CACHE', protocol_sha256='protocol',
            kind='development', role=role, group=0, stream=0, count=2,
            objectives=['decoded_teacher', 'physical_labels'] if role == 'recipient' else ['decoded_teacher'],
            frozen_tensors_unchanged=True, native_endpoint_and_identity_replacement_exact=role=='recipient')
        if edit is not None: edit(arrays, report)
        path = Path(directory) / 'arrays.npz'; np.savez(path, **arrays)
        report['arrays'] = dict(path=str(path), sha256=sha(path))
        rp = Path(directory) / 'report.json'; rp.write_text(json.dumps(report))
        return rp, sha(rp), 'protocol', 'development', role, 0, 0, 2

    def test_neither_truth_nor_D_is_opened(self):
        # Object arrays intentionally cannot be read with allow_pickle=False.
        # Only token/seed keys required by each role may be accessed.
        for role in ['recipient', 'donor']:
            with tempfile.TemporaryDirectory() as d:
                obs, free, seeds = read_cache(*self.fixture(d, role))
                self.assertEqual(obs.shape, (2, 192)); self.assertEqual(seeds.tolist(), [31, 32])
                if role == 'donor': self.assertIsNone(free)

    def test_wrong_shape_dtype_or_objective_order_rejects(self):
        edits = [
            lambda a, r: a.update(observed=np.zeros((2, 6, 192), np.float32)),
            lambda a, r: a.update(free=np.zeros((2, 2, 6, 192), np.float32)),
            lambda a, r: a.update(observed=a['observed'].astype(np.float64)),
            lambda a, r: a.update(free=a['free'].astype(np.float64)),
            lambda a, r: a.update(seeds=a['seeds'].astype(np.float64)),
            lambda a, r: a.update(seeds=np.array([31, 31], np.int64)),
            lambda a, r: r.update(objectives=['physical_labels', 'decoded_teacher']),
        ]
        for i, edit in enumerate(edits):
            with self.subTest(corruption=i), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError): read_cache(*self.fixture(d, edit=edit))




class FullShardCLI(unittest.TestCase):
    """Complete 64-goal CLI admission fixtures; no QP-optimality certification."""
    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value)); return dict(path=str(path), sha256=sha(path))

    def run_fixture(self, mutation=None):
        import subprocess
        from s1_projection import MEMBERS
        from s1_common import namespace_seed
        with tempfile.TemporaryDirectory(prefix='s1_projection_guard_') as d:
            root = Path(d); cfg_path = PHASE / 'protocol/DESIGN.lock.json'
            cfg = json.loads(cfg_path.read_text()); protocol_sha = sha(cfg_path); n = 64
            norm = np.ones(192, np.float64)
            head = dict(mean=np.zeros(192), scale=norm, target_mean=np.zeros(6), target_scale=np.ones(6),
                **{'0.weight': np.zeros((2, 192), np.float32), '0.bias': np.ones(2, np.float32),
                   '2.weight': np.zeros((2, 2), np.float32), '2.bias': np.ones(2, np.float32),
                   '4.weight': np.zeros((6, 2), np.float32), '4.bias': np.zeros(6, np.float32)})
            hp = root/'constant_head.npz'; np.savez(hp, **head)
            ab = self.write_json(root/'A.json', dict(groups=[dict(group=g, head_A=dict(path=str(hp), sha256=sha(hp))) for g in [0, 1]]))
            selected, qual = valid_gate(); selected['protocol_sha256'] = protocol_sha
            for h in selected['heads']:
                h['checkpoint'] = str(hp) if h['head_role']=='C' else '/D_MUST_NOT_BE_OPENED'
                h['checkpoint_sha256'] = sha(hp) if h['head_role']=='C' else f"unopened-D-{h['group']}"
            sb = self.write_json(root/'selected.json', selected)
            qual.update(protocol_sha256=protocol_sha, selected_lock_sha256=sb['sha256'], a_bindings_sha256=ab['sha256'])
            for q, h in zip(qual['rows'], selected['heads']): q['checkpoint_sha256'] = h['checkpoint_sha256']
            qb = self.write_json(root/'qualification.json', qual)
            caches = {}
            for role, seed_start in [('recipient',1000),('donor',2000)]:
                arrays = dict(observed=np.zeros((n,5,192),np.float32), seeds=np.arange(seed_start,seed_start+n,dtype=np.int64),
                    truth=np.array([object()],object), predictions_D=np.array([object()],object))
                arrays['observed'][...,0] = 1 if role=='recipient' else 2
                if role=='recipient': arrays['free']=np.zeros((2,n,5,192),np.float32)
                ap=root/f'{role}.npz'; np.savez(ap,**arrays)
                cr=dict(status='PASS_S1_OBSERVED_AND_FREE_CACHE',protocol_sha256=protocol_sha,kind='development',role=role,
                    group=0,stream=0,count=n,objectives=cfg['objectives'] if role=='recipient' else cfg['objectives'][:1],
                    frozen_tensors_unchanged=True,native_endpoint_and_identity_replacement_exact=role=='recipient',
                    arrays=dict(path=str(ap),sha256=sha(ap)))
                caches[role]=self.write_json(root/f'{role}.json',cr)
            perm=np.random.default_rng(namespace_seed(cfg['root_seed'],'v9_s1_donor/development/stream_0')).permutation(n)
            direction=np.zeros((2,2,n,2,192),np.float64); direction[:,:,:,0,0]=1; direction[:,:,:,1,0]=2
            repl=np.zeros((2,2,n,2,192),np.float32); repl[...,0]=1
            payload=dict(replacements=repl,directions=direction,common_norm=np.ones(n),
                goal_indices=np.arange(n,dtype=np.int64),donor_indices=perm.astype(np.int64))
            identity=dict(status='ACCEPTED_COMPLETE_SHARD',protocol_sha256=protocol_sha,group=0,stream=0,kind='development',count=n,
                start=0,stop=n,selected_lock_sha256=sb['sha256'],a_bindings_sha256=ab['sha256'],qualification_sha256=qb['sha256'],
                recipient_cache_sha256=caches['recipient']['sha256'],donor_cache_sha256=caches['donor']['sha256'],
                head_A_sha256=sha(hp),head_C_sha256=sha(hp))
            receipts=[]
            for i in range(n):
                receipts.append(dict(goal=i,status='ACCEPTED_S1_EIGHT_MEMBER_FAMILY',family_size=8,members=copy.deepcopy(MEMBERS),
                    solvers=[dict(status='solved',head_count=1 if j<4 else 2) for j in range(8)],effective_norm=1.,
                    recipient_seed=1000+i,donor_index=int(perm[i]),donor_seed=2000+int(perm[i]),
                    native_norms=[1.,2.,1.,2.,1.,2.,1.,2.],unshrunk_common_norm=1.,shrink_factor=1.,
                    legitimate_zero_norm=False,attempts=[dict(shrink=1.)]))
            report=dict(**identity,matching_reports=receipts)
            if mutation: mutation(payload, report)
            pp=root/'projection.npz'; np.savez(pp,**payload)
            report['arrays']=dict(path=str(pp),sha256=sha(pp))
            rb=self.write_json(root/'shard.json',report)
            bb=self.write_json(root/'bindings.json',dict(group=0,stream=0,kind='development',shards=[dict(start=0,stop=n,report=rb['path'],report_sha256=rb['sha256'])]))
            args=[sys.executable,str(PHASE/'scripts/freeze_qp_shards.py')]
            bindings={'protocol':dict(path=str(cfg_path),sha256=protocol_sha),'bindings':bb,'selected-lock':sb,
                'qualification':qb,'a-bindings':ab,'recipient-cache':caches['recipient'],'donor-cache':caches['donor']}
            for name,b in bindings.items(): args += ['--'+name,b['path'],'--'+name+'-sha256',b['sha256']]
            out=root/'accepted.json'; args += ['--output',str(out)]
            run=subprocess.run(args,capture_output=True,text=True)
            return run.returncode, run.stdout+run.stderr, json.loads(out.read_text()) if out.exists() else None

    def test_complete_64_goal_synthetic_shard_passes_without_D(self):
        code, text, receipt = self.run_fixture()
        self.assertEqual(code, 0, text); self.assertTrue(receipt['all_goals_covered']); self.assertEqual(receipt['count'],64)

    def test_solver_roster_ray_common_min_donor_and_shrink_corruption_rejected(self):
        def skip_first(payload, report):
            row=report['matching_reports'][0]; row.update(effective_norm=.5,shrink_factor=.5,attempts=[dict(shrink=1.),dict(shrink=.5)])
            payload['common_norm'][0]=.5;payload['replacements'][:,:,0,:,0]=.5
        cases=[
            ('empty solvers',lambda p,r:r['matching_reports'][0].update(solvers=[]),'Norm or solver'),
            ('seven solvers',lambda p,r:r['matching_reports'][0]['solvers'].pop(),'Norm or solver'),
            ('wrong member order',lambda p,r:r['matching_reports'][0]['members'].reverse(),'Norm or solver'),
            ('wrong solver head count',lambda p,r:r['matching_reports'][0]['solvers'][4].update(head_count=1),'Norm or solver'),
            ('raw direction mismatch',lambda p,r:p['directions'].__setitem__((0,0,0,0,0),1.5),'Not equal'),
            ('wrong common minimum',lambda p,r:r['matching_reports'][0].update(unshrunk_common_norm=2.),'Shared dose'),
            ('off-ray feasible replacement',lambda p,r:(p['replacements'].__setitem__((0,0,0,0,0),0.),p['replacements'].__setitem__((0,0,0,0,1),1.)),'not equal'),
            ('wrong donor receipt',lambda p,r:r['matching_reports'][0].update(donor_seed=99999),'Donor/recipient'),
            ('skip passing earlier factor',skip_first,'A passing earlier shrink was skipped'),
        ]
        for name,mutation,reason in cases:
            with self.subTest(corruption=name):
                code,text,receipt=self.run_fixture(mutation)
                self.assertNotEqual(code,0);self.assertIsNone(receipt);self.assertIn(reason.lower(),text.lower())


if __name__ == '__main__': unittest.main(verbosity=2)
