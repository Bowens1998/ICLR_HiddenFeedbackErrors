"""Synthetic S3 adapter checks; injected directions are not OSQP evidence."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import projection_eight as p


def fixture():
    head = dict(mean=np.zeros(192), scale=np.ones(192), target_mean=np.zeros(6),
        target_scale=np.arange(1., 7.), **{'0.weight':np.zeros((2,192)), '0.bias':np.zeros(2),
        '2.weight':np.eye(2), '2.bias':np.ones(2) * .25,
        '4.weight':np.zeros((6,2)), '4.bias':np.zeros(6)})
    head['0.weight'][0,0] = 1.; head['0.weight'][1,2] = 1.
    head['4.weight'][0,0] = 1.; head['4.weight'][1,1] = .5
    anchors = {c:{} for c in p.CONDITIONS}
    for j,(c,o) in enumerate((c,o) for c in p.CONDITIONS for o in p.OBJECTIVES):
        z = np.zeros(192, np.float32); z[0] = (1., -1., 2., -2.)[j]
        z[2] = 2.; z[3] = 10 * (j + 1); anchors[c][o] = z
    guides = dict(actual=np.ones(192,np.float32), donor=-np.ones(192,np.float32))
    directions = []
    for i in range(8):
        v = np.zeros(192,np.float64); v[1] = (8-i) * (1. if i % 2 == 0 else -1.)
        directions.append(v)
    return head,anchors,guides,directions


def solver_for(directions, fail_at=()):
    calls = []
    def solve(heads,anchor,guide,reference_head=None):
        i=len(calls);calls.append(dict(anchor=anchor.copy(),guide=guide.copy(),head_count=len(heads)))
        if len(heads)!=1 or heads[0] is not reference_head:raise AssertionError('A-only shared metric')
        if i in fail_at:raise RuntimeError('Synthetic failure '+str(i))
        return SimpleNamespace(delta=directions[i].copy(),solver=dict(status='solved',head_count=1))
    return solve,calls


def construct(head=None,anchors=None,guides=None,directions=None,goal=1,donor=0,pool=2):
    h,z,g,d=fixture();h=h if head is None else head;z=z if anchors is None else anchors
    g=g if guides is None else guides;d=d if directions is None else directions
    solve,calls=solver_for(d)
    result=p.project_matched_eight_family(z,g,h,pool=pool,goal_index=goal,donor_index=donor,projector=solve)
    return h,z,g,result,calls


def accept(h,z,g,result,goal=1,pool=2,assignment=None):
    if assignment is None:assignment=np.array([1,0,2],np.int64)
    return p.accept_matched_eight_family(z,g,h,*result,pool=pool,goal_index=goal,donor_assignment=assignment)


class EightFamilyTests(unittest.TestCase):
    def test_all_four_own_anchors_and_eighth_norm_control_entire_family(self):
        h,z,g,result,calls=construct();r,d,report=result
        self.assertEqual(len(calls),8);self.assertEqual(r.shape,(2,2,2,192))
        self.assertEqual(report['native_norms'],list(map(float,range(8,0,-1))))
        self.assertEqual(report['effective_norm'],1.)
        self.assertEqual(report['members'],[list(x) for x in p.MEMBERS])
        for i,(c,o,s) in enumerate(p.MEMBERS):
            np.testing.assert_array_equal(calls[i]['anchor'],z[c][o])
            np.testing.assert_array_equal(calls[i]['guide'],g[s])
            ci,oi,si=p.CONDITIONS.index(c),p.OBJECTIVES.index(o),p.SOURCES.index(s)
            delta=r[ci,oi,si].astype(float)-z[c][o]
            self.assertEqual(delta[1],1. if s=='actual' else -1.)
            self.assertEqual(np.linalg.norm(delta),1.)
            self.assertEqual(r[ci,oi,si,3],z[c][o][3])
        self.assertEqual(accept(h,z,g,result)['family_size'],8)

    def test_nonunit_normalizer_shared_across_all_models_and_each_actual_donor_pair(self):
        h,z,g,d=fixture();h['scale'][1]=7.5;h['mean'][1]=-12.
        h,z,g,result,_=construct(head=h,anchors=z,guides=g,directions=d)
        for ci,c in enumerate(p.CONDITIONS):
            for oi,o in enumerate(p.OBJECTIVES):
                norms=[np.linalg.norm((result[0][ci,oi,si].astype(float)-z[c][o])/h['scale']) for si in range(2)]
                np.testing.assert_array_equal(norms,[1.,1.])
                np.testing.assert_array_equal(result[0][ci,oi,:,1],[7.5,-7.5])
        self.assertEqual(accept(h,z,g,result)['effective_norm_squared'],1.)

    def test_distinct_activation_regions_are_checked_at_the_own_anchor(self):
        h,z,g,d=fixture()
        # Inactive x0 may move farther negative; the positive-anchor readout
        # would change. A shared-anchor shortcut would reject valid members.
        for index in (2,3,6,7):
            magnitude=np.linalg.norm(d[index]);d[index][:]=0.;d[index][0]=-magnitude
        h,z,g,result,_=construct(head=h,anchors=z,guides=g,directions=d)
        for ci,c in enumerate(p.CONDITIONS):
            np.testing.assert_array_equal(result[0][ci,1,:,0],np.full(2,z[c]['physical_labels'][0]-1))
        self.assertTrue(all(x['accepted'] for x in accept(h,z,g,result)['checks']))

    def test_one_zero_direction_preserves_every_anchor_and_all_solves(self):
        h,z,g,d=fixture();d[6][:]=0.
        h,z,g,result,calls=construct(head=h,anchors=z,guides=g,directions=d)
        self.assertEqual(len(calls),8);self.assertTrue(result[2]['legitimate_zero_norm'])
        for ci,c in enumerate(p.CONDITIONS):
            for oi,o in enumerate(p.OBJECTIVES):
                for si in range(2):np.testing.assert_array_equal(result[0][ci,oi,si],z[c][o])
        self.assertEqual(accept(h,z,g,result)['effective_norm'],0.)

    def test_multiple_solver_failures_retained_all_eight_attempted_no_zero(self):
        h,z,g,d=fixture();solve,calls=solver_for(d,fail_at=(1,6))
        with self.assertRaises(p.EightFamilyFailure) as caught:
            p.project_matched_eight_family(z,g,h,pool=0,goal_index=0,donor_index=0,projector=solve)
        detail=caught.exception.detail;self.assertEqual(len(calls),8)
        self.assertEqual(detail['successful_directions'],6)
        self.assertEqual([x['member'] for x in detail['failures']],[list(p.MEMBERS[i]) for i in (1,6)])
        self.assertNotIn('effective_norm',detail)

    def test_one_member_fp32_failure_shrinks_all_models_and_sources(self):
        h,z,g,d=fixture();d[0][0]=32e-6
        h,z,g,result,_=construct(head=h,anchors=z,guides=g,directions=d)
        report=result[2];self.assertLess(report['shrink_factor'],1.)
        self.assertEqual(len(report['attempts']),p.SHRINK_FACTORS.index(report['shrink_factor'])+1)
        self.assertTrue(all(not all(v['accepted'] for v in a['checks']) for a in report['attempts'][:-1]))
        accepted=accept(h,z,g,result)
        for check in accepted['checks']:self.assertAlmostEqual(check['actual_norm'],report['effective_norm'],places=6)

    def test_region_crossing_rejected_even_when_output_is_constant(self):
        h,z,g,d=fixture();h['4.weight'][:]=0.
        # T1 physical-label anchor is -2. Crossing zero changes only region.
        d[6][:]=0.;d[6][0]=4.
        d[7][:]=0.;d[7][0]=4.
        for i in range(6):d[i][1]=4.
        h,z,g,result,_=construct(head=h,anchors=z,guides=g,directions=d)
        self.assertEqual(result[2]['shrink_factor'],.5)
        self.assertFalse(result[2]['attempts'][0]['checks'][6]['accepted'])
        self.assertEqual(result[2]['attempts'][0]['checks'][6]['heads'][0]['normalized_output_deviation'],0.)
        self.assertEqual(accept(h,z,g,result)['shrink_factor'],.5)

    def test_no_accepted_backoff_blocks_whole_family(self):
        h,z,g,d=fixture();d[0][0]=8.
        solve,calls=solver_for(d)
        with self.assertRaises(p.EightFamilyFailure) as caught:
            p.project_matched_eight_family(z,g,h,pool=0,goal_index=0,donor_index=0,projector=solve)
        self.assertEqual(len(calls),8);self.assertEqual(caught.exception.detail['stage'],'verification')
        self.assertEqual(len(caught.exception.detail['attempts']),9)

    def test_global_donor_permutation_pool_and_indices(self):
        h,z,g,result,_=construct(goal=7,donor=1,pool=1)
        assignment=np.array([0,7,2,3,4,5,6,1],np.int64)
        self.assertEqual(accept(h,z,g,result,goal=7,pool=1,assignment=assignment)['donor_index'],1)
        for bad in (np.arange(8,dtype=np.int64),np.zeros(8,np.int64),assignment.astype(np.int32)):
            with self.assertRaises(ValueError):accept(h,z,g,result,goal=7,pool=1,assignment=bad)
        with self.assertRaises(ValueError):accept(h,z,g,result,goal=7,pool=0,assignment=assignment)

    def test_malformed_roster_tokens_head_and_candidate_feedback_rejected(self):
        for mode in ('condition','objective','guide','per_candidate','dtype','nonfinite','scale','pool','goal'):
            h,z,g,d=fixture();pool=0;goal=1
            if mode=='condition':z.pop('T1')
            if mode=='objective':z['T1'].pop('physical_labels')
            if mode=='guide':g['q_g']=np.zeros(192,np.float32)
            if mode=='per_candidate':g['actual']=np.zeros((32,192),np.float32)
            if mode=='dtype':z['T1']['decoded_teacher']=z['T1']['decoded_teacher'].astype(float)
            if mode=='nonfinite':z['T1']['decoded_teacher'][1]=np.nan
            if mode=='scale':h['scale'][0]=0.
            if mode=='pool':pool=True
            if mode=='goal':goal=-1
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                p.project_matched_eight_family(z,g,h,pool=pool,goal_index=goal,donor_index=0)

    def test_saved_wrong_anchor_axis_shape_and_ray_rejected(self):
        h,z,g,result,_=construct()
        for mode in ('anchor','swap','truncated','direction','dtype','nan'):
            damaged=copy.deepcopy(result)
            if mode=='anchor':damaged[0][1,0,0]=damaged[0][0,0,0]
            if mode=='swap':damaged=(damaged[0].swapaxes(0,1).copy(),damaged[1],damaged[2])
            if mode=='truncated':damaged=(damaged[0][:1],damaged[1],damaged[2])
            if mode=='direction':damaged[1][1,1,1,4]=.1
            if mode=='dtype':damaged=(damaged[0].astype(float),damaged[1],damaged[2])
            if mode=='nan':damaged[1][0,0,0,4]=np.nan
            with self.subTest(mode=mode),self.assertRaises(ValueError):accept(h,z,g,damaged)

    def test_receipt_corruption_and_later_backoff_rejected(self):
        h,z,g,result,_=construct()
        mutations=[lambda x:x.update(family_size=4),lambda x:x['members'].reverse(),
            lambda x:x.update(effective_norm=2.),lambda x:x['native_norms'].__setitem__(7,2.),
            lambda x:x['solvers'][7].update(head_count=2),lambda x:x['solvers'].pop(),
            lambda x:x['attempts'][0]['checks'].pop(),lambda x:x.update(attempts=[]),
            lambda x:x.update(legitimate_zero_norm=True),lambda x:x['defaults'].update(tolerance=1.),
            lambda x:x['attempts'].append(dict(shrink=.5,checks=copy.deepcopy(x['attempts'][0]['checks'])))]
        for i,change in enumerate(mutations):
            damaged=copy.deepcopy(result);change(damaged[2])
            with self.subTest(i=i),self.assertRaises(ValueError):accept(h,z,g,damaged)

    def test_input_identity_changes_rejected(self):
        h,z,g,result,_=construct()
        for mode in ('T1anchor','guide','normalizer'):
            hh,zz,gg=copy.deepcopy((h,z,g))
            if mode=='T1anchor':zz['T1']['physical_labels'][8]=.25
            if mode=='guide':gg['donor'][8]=.25
            if mode=='normalizer':hh['target_scale'][5]+=1.
            with self.subTest(mode=mode),self.assertRaises(ValueError):accept(hh,zz,gg,result)


if __name__=='__main__':unittest.main()
