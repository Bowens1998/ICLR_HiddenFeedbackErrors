"""Synthetic saved-input/schema corruption checks, no real data or simulation."""
import copy
import unittest
import numpy as np
import accept_raw_inputs as a


def fixture():
    seed=200
    bank=dict(seed=np.asarray(seed),prefix=np.zeros((10,2),np.float32),
        history_pixels=np.zeros((3,224,224,3),np.uint8),history_states=np.zeros((3,7)),
        goal_pixels=np.zeros((224,224,3),np.uint8),goal_state=np.array([0.,0.,50.,0.,0.,0.,0.]),
        terminal_pixels=np.zeros((32,224,224,3),np.uint8),terminal_states=np.zeros((32,7)),
        actions=np.zeros((32,25,2),np.float32),goal_actions=np.zeros((25,2),np.float32),
        contacts=np.zeros(32,np.int64),terminations=np.zeros(32,np.int64))
    trace=np.ones((30,300),np.float64);trace[0,0]=0
    actions=dict(seed=np.asarray(seed),population_actions=np.zeros((300,25,2),np.float32),
        selected_actions=np.zeros((25,2),np.float32),selected_index=np.asarray(0),selected_iteration=np.asarray(0),
        population_costs=trace[0].copy(),cost_trace=trace,selected_token=np.zeros(192,np.float32))
    physics=dict(seed=np.asarray(seed),actions=np.zeros((35,2),np.float32),states=np.zeros((36,7)),
        pixels=np.zeros((8,224,224,3),np.uint8),contacts=np.zeros(35,bool),terminations=np.zeros(35,bool),boundary=np.zeros(35,bool))
    ar=dict(selected_index=0,selected_iteration=0,population_replay_exact=True,scored_candidates=9000,
            torch_seed=int(np.random.SeedSequence([seed,955001]).generate_state(1)[0]))
    pr=dict(boundary_steps=0,all_two_replays_exact=True)
    return bank,actions,physics,ar,pr


class RawAcceptance(unittest.TestCase):
    def setUp(self): self.b,self.a,self.p,self.ar,self.pr=fixture()
    def verify(self):return a.validate_case_arrays(self.b,self.a,self.p,seed=200,action_row=self.ar,physics_row=self.pr)
    def test_saved_triple_complete_reference_parity(self):
        self.assertEqual(self.verify(),dict(selected_index=0,selected_iteration=0))
        a.validate_bank_arrays(self.b,dict(seed=200,all_shapes_visible_every_step=True,initial_goal_block_distance=50.))
    def test_parent_drift(self):
        for z in (self.b,self.a,self.p):
            old=z['seed'];z['seed']=np.asarray(201)
            with self.assertRaises(ValueError):self.verify()
            z['seed']=old
    def test_selected_actions_are_from_native_population(self):
        self.a['selected_actions'][3,0]=.1
        with self.assertRaises(AssertionError):self.verify()
    def test_history_and_selected_controls_link_to_physics(self):
        self.p['actions'][30,0]=.2
        with self.assertRaises(AssertionError):self.verify()
    def test_physical_initial_context_replay_is_exact(self):
        for key,index in [('states',(5,2)),('pixels',(1,0,0,0))]:
            self.p[key][index]=1
            with self.assertRaises(AssertionError):self.verify()
            self.p[key][index]=0
    def test_whole_9000_search_selection_and_ties(self):
        self.a['cost_trace'][2,1]=-1
        with self.assertRaises(ValueError):self.verify()
        self.a['cost_trace'][2,1]=0 # Equal later cost does not replace the earlier argmin.
        self.verify()
    def test_selected_population_costs_are_actual_search_population(self):
        self.a['population_costs'][1]=3
        with self.assertRaises(AssertionError):self.verify()
    def test_full_population_and_fp32_input_precision(self):
        original=self.a['population_actions']
        for v in (original[:299],original.astype(np.float64)):
            self.a['population_actions']=v
            with self.assertRaises(ValueError):self.verify()
    def test_receipts_cannot_replace_failed_replay_or_bad_seed(self):
        for target,key,value in [(self.ar,'population_replay_exact',False),(self.ar,'scored_candidates',300),
                                 (self.pr,'all_two_replays_exact',False),(self.ar,'torch_seed',1)]:
            old=target[key];target[key]=value
            with self.assertRaises(ValueError):self.verify()
            target[key]=old
    def test_pose_cost_precision_and_physical_state_type(self):
        self.a['cost_trace']=self.a['cost_trace'].astype(np.float32)
        with self.assertRaises(ValueError):self.verify()
        self.a['cost_trace']=self.a['cost_trace'].astype(np.float64)
        self.p['states']=self.p['states'].astype(np.int64)
        with self.assertRaises(ValueError):self.verify()
    def test_nonfinite_any_future_blocks_population(self):
        self.p['states'][35,2]=np.nan
        with self.assertRaises(ValueError):self.verify()
    def test_goal_input_admission_is_not_softened(self):
        self.b['goal_state'][2]=39.
        with self.assertRaises(ValueError):a.validate_bank_arrays(self.b,dict(seed=200,all_shapes_visible_every_step=True,initial_goal_block_distance=39.))
    def test_six_role_names_do_not_authorize_donor_response(self):
        names={a.runtime_role(r,s) for r in a.raw.ROLES for s in (('donor_observed',) if r.endswith('donor') else ('probe','response'))}
        self.assertEqual(names,{'probe_calibration','calibration_response','probe_test','test_response','probe_donor_calibration','probe_donor_test'})
        with self.assertRaises(ValueError):a.runtime_role('test_donor','response')
        with self.assertRaises(ValueError):a.runtime_role('test_donor','probe')
        with self.assertRaises(ValueError):a.runtime_role('test_recipient','donor_observed')


def content_fixture():
    historical={k+'/'+r for k in ('development','confirmation') for r in ('recipient','donor')}
    d=dict(status='PASS_S2_RETAINED_PIXEL_ISOLATION',protocol_sha256='p',sources_sha256='s',
        seed_lineage=dict(status='PASS_S2_ALL_RECORDED_ATTEMPTS_DISJOINT',protocol_sha256='p',sources_sha256='s'),
        head_overlap_hashes={r:[] for r in a.raw.ROLES},s1_overlap_hashes={r:{k:[] for k in historical} for r in a.raw.ROLES},
        new_role_pairs=[dict(left=x,right=y,overlap=[]) for i,x in enumerate(a.raw.ROLES) for y in a.raw.ROLES[i+1:]],
        head_caches=[dict(group=g,path=f'/synthetic/{g}/{i}') for g in (0,1,2,4) for i in range(6)],
        input_files_sha256={'/synthetic/content':'x'},populations={k:{} for k in ['S1/'+s for s in historical]})
    d['populations'].update({'S2/'+r:dict(total_frames=a.raw.COUNTS[r]*(36+8*len(a.raw.routes(r))),unique_pixels=1) for r in a.raw.ROLES})
    return d


class ContentReceipt(unittest.TestCase):
    def test_complete_scope(self):a.validate_content(content_fixture(),'p','s')
    def test_rehashed_semantic_corruptions_cannot_hide_partial_scope(self):
        for change in [lambda d:d['head_caches'].pop(),lambda d:d['new_role_pairs'].pop(),
                       lambda d:d['head_overlap_hashes']['test_recipient'].append('alias'),
                       lambda d:d['s1_overlap_hashes']['test_donor'].pop('development/donor'),
                       lambda d:d['populations']['S2/test_donor'].__setitem__('total_frames',1),
                       lambda d:d.__setitem__('sources_sha256','wrong')]:
            d=content_fixture();change(d)
            with self.assertRaises(ValueError):a.validate_content(d,'p','s')


if __name__=='__main__':unittest.main()
