"""Synthetic complete-population guards; no real S2 data, heads or execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import accept_probe_population as gate
import projection_four as four


def fixture(split='calibration'):
    n = 256 if split == 'calibration' else 512
    head = dict(mean=np.zeros(192), scale=2.**(np.arange(192) % 3),
                target_mean=np.arange(6, dtype=float), target_scale=np.arange(1, 7, dtype=float))
    head.update({'0.weight': np.eye(2, 192), '0.bias': np.ones(2)*10,
                 '2.weight': np.eye(2), '2.bias': np.ones(2),
                 '4.weight': np.ones((6, 2)), '4.bias': np.zeros(6)})
    # Distinct objective, recipient, horizon and coordinate values detect axis errors.
    free = (np.arange(2)[:, None, None, None]/2 + np.arange(n)[None, :, None, None]/64 +
            np.arange(5)[None, None, :, None]/8 + np.arange(192)[None, None, None, :]/1024).astype(np.float32)
    observed = (free[0] + np.float32(2)).copy()
    history = free + np.float32(.25); history[:, :, 0] = free[:, :, 0]
    reset = np.broadcast_to(observed[None], free.shape).copy()
    reset[:, :, 1:] += np.array([.125, .25], np.float32)[:, None, None, None]
    recipient = dict(free=free, observed_history=history, reset=reset, observed=observed,
        truth=np.arange(n*5*6, dtype=np.float64).reshape(n, 5, 6)/16,
        prefix_actions=np.zeros((n, 10, 2), np.float32), selected_actions=np.zeros((n, 25, 2), np.float32),
        selected_index=np.arange(n, dtype=np.int64)%300, seeds=np.arange(1000, 1000+n, dtype=np.int64))
    donor = dict(observed5=(observed[:, 0]+np.float32(4)).copy(), seeds=np.arange(10000, 10000+n, dtype=np.int64))
    assignment = np.random.Generator(np.random.PCG64(123)).permutation(n).astype(np.int64)
    replacements = np.zeros((n, 2, 2, 192), np.float32); directions = np.zeros((n, 2, 2, 192), np.float64)
    families = []
    for goal in range(n):
        zero = goal % 17 == 0; norms = [0.] * 4 if zero else [.25, .5, .75, 1.]
        predicted = {o: free[i, goal, 0] for i, o in enumerate(four.OBJECTIVES)}
        guides = dict(actual=observed[goal, 0], donor=donor['observed5'][assignment[goal]])
        checks = []
        for j, (objective, source) in enumerate(four.MEMBERS):
            oi, si = divmod(j, 2); directions[goal, oi, si, j+2] = norms[j]
            value = predicted[objective].astype(np.float64).copy()
            if not zero:
                value[j+2] += .25 * head['scale'][j+2]
            replacements[goal, oi, si] = value.astype(np.float32)
            # Analytical oracle: only coordinates ignored by both ReLU layers change.
            # All first/second preactivations stay positive; normalized and physical
            # output differences are exactly zero; each realized norm is .25 or0.
            checks.append(dict(accepted=True, finite=True, norm_ok=True, actual_norm=norms[0], expected_norm=norms[0],
                heads=[dict(normalized_output_deviation=0., region_violation=0., physical_output_max_deviation=0.)]))
        families.append(dict(status=four.STATUS, draft_engineering_only=True, model_role='T0', constraint='g_A',
            family_size=4, members=[list(x) for x in four.MEMBERS], goal_index=goal, donor_index=int(assignment[goal]),
            **four._bindings(predicted, guides, four._head(head)), native_norms=norms,
            unshrunk_common_norm=norms[0], effective_norm=norms[0], shrink_factor=1., legitimate_zero_norm=zero,
            attempts=[dict(shrink=1., checks=checks)], solvers=[dict(status='solved', head_count=1) for _ in range(4)],
            defaults=dict(shrink_factors=list(four.SHRINK_FACTORS), tolerance=four.TOLERANCE,
                          eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, receipt_atol=four.RECEIPT_ATOL,
                          physical_receipt_atol='receipt_atol*max(1,max_target_scale)')))
    tokens = np.stack([free, free+np.float32(.5), free+np.float32(.75), reset], axis=2)
    tokens[:, :, 1:3, 0] = replacements.transpose(1, 0, 2, 3)
    for goal in range(0, n, 17):
        tokens[:, goal, 1] = free[:, goal]; tokens[:, goal, 2] = free[:, goal]
    rollout = dict(tokens=tokens, identity=free.copy(), observed_history=history.copy(),
        native_endpoint=free[:, :, -1].copy(), inserted=replacements.transpose(1, 0, 2, 3).copy(), seeds=recipient['seeds'].copy())
    return dict(recipient=recipient, donor=donor, projection=dict(replacements=replacements, directions=directions),
                families=families, rollout=rollout, head_A=head, assignment=assignment, split=split)


def json_file(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return dict(path=str(path), sha256=gate.sha(path))


class ProbePopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = fixture()

    def test_complete256_and512_with_zero_families(self):
        for values in (self.cal, fixture('test')):
            result = gate.validate_pool(**values)
            n = len(values['families'])
            self.assertEqual(result['families'], n); self.assertEqual(result['directions'], 4*n)
            self.assertEqual(result['zero_families'], len(range(0, n, 17)))
            self.assertEqual(result['maximum_normalized_output_deviation'], 0.)
            self.assertEqual(result['maximum_region_violation'], 0.)
            self.assertEqual(result['maximum_norm_residual'], 0.)

    def test_no_producer_is_called_by_gate(self):
        with mock.patch.object(four, 'project_t0_four_family', side_effect=AssertionError('producer forbidden')):
            self.assertEqual(gate.validate_pool(**self.cal)['directions'], 1024)

    def test_missing_population_or_family_and_extra_readout_arrays(self):
        for kind, key in [('recipient', 'truth'), ('donor', 'observed5'), ('rollout', 'identity')]:
            values = dict(self.cal); part = dict(values[kind]); part.pop(key); values[kind] = part
            with self.assertRaises(ValueError): gate.validate_pool(**values)
        values = dict(self.cal, families=self.cal['families'][:-1])
        with self.assertRaises(ValueError): gate.validate_pool(**values)
        values = dict(self.cal, recipient=dict(self.cal['recipient'], q_g_prediction=np.zeros(1)))
        with self.assertRaises(ValueError): gate.validate_pool(**values)

    def test_wrong_dtype_size_or_nonfinite_any_late_cell(self):
        for kind, key, value in [
            ('rollout','tokens', self.cal['rollout']['tokens'][:, :-1]),
            ('rollout','tokens', self.cal['rollout']['tokens'].astype(np.float64)),
            ('donor','seeds', self.cal['donor']['seeds'].astype(np.int32)),
            ('projection','directions', self.cal['projection']['directions'].astype(np.float32))]:
            values = dict(self.cal); values[kind] = dict(values[kind]); values[kind][key] = value
            with self.assertRaises(ValueError): gate.validate_pool(**values)
        values = dict(self.cal, rollout=dict(self.cal['rollout']))
        values['rollout']['tokens'] = values['rollout']['tokens'].copy(); values['rollout']['tokens'][1,-1,2,-1,-1] = np.nan
        with self.assertRaises(ValueError): gate.validate_pool(**values)

    def test_distinct_objective_branch_horizon_axis_corruptions(self):
        variants = [self.cal['rollout']['tokens'][::-1], self.cal['rollout']['tokens'][:, :, [1,0,2,3]],
                    self.cal['rollout']['tokens'][:, :, :, ::-1]]
        for tokens in variants:
            values = dict(self.cal, rollout=dict(self.cal['rollout'], tokens=tokens))
            with self.assertRaises(ValueError): gate.validate_pool(**values)

    def test_exact_endpoint_identity_history_insert_and_reset_guards(self):
        for key in ('identity', 'observed_history', 'native_endpoint', 'inserted'):
            values = dict(self.cal, rollout=dict(self.cal['rollout']))
            values['rollout'][key] = values['rollout'][key].copy()
            values['rollout'][key].flat[-1] += np.float32(.125)
            with self.subTest(key=key), self.assertRaises(ValueError): gate.validate_pool(**values)
        values = dict(self.cal, rollout=dict(self.cal['rollout']))
        values['rollout']['tokens'] = values['rollout']['tokens'].copy()
        values['rollout']['tokens'][1,-1,3,0,-1] += np.float32(.125)
        with self.assertRaises(ValueError): gate.validate_pool(**values)

    def test_zero_family_cannot_hide_changed_future(self):
        values = dict(self.cal, rollout=dict(self.cal['rollout']))
        values['rollout']['tokens'] = values['rollout']['tokens'].copy()
        values['rollout']['tokens'][1,0,2,-1,-1] += np.float32(.125)
        with self.assertRaisesRegex(ValueError, 'zero-dose'): gate.validate_pool(**values)

    def test_global_permutation_exact_seed_order_and_parent_isolation(self):
        gate.validate_assignment(self.cal['assignment'], split='calibration', seed=123)
        with self.assertRaises(ValueError): gate.validate_assignment(self.cal['assignment'], split='calibration', seed=124)
        repeated = self.cal['assignment'].copy(); repeated[0] = repeated[1]
        with self.assertRaises(ValueError): gate.validate_assignment(repeated, split='calibration', seed=123)
        values = dict(self.cal, donor=dict(self.cal['donor'], seeds=self.cal['recipient']['seeds'].copy()))
        with self.assertRaises(ValueError): gate.validate_pool(**values)
        values = dict(self.cal, rollout=dict(self.cal['rollout'], seeds=self.cal['rollout']['seeds'][::-1]))
        with self.assertRaises(ValueError): gate.validate_pool(**values)

    def test_each_report_source_member_solver_minimum_backoff_is_checked(self):
        mutations = [lambda r:r.update(donor_index=999), lambda r:r.update(model_role='T1'),
            lambda r:r['members'].reverse(), lambda r:r['solvers'][3].update(status='failed'),
            lambda r:r.update(unshrunk_common_norm=.5), lambda r:r.update(effective_norm=.5),
            lambda r:r.update(shrink_factor=.5), lambda r:r['attempts'].append(copy.deepcopy(r['attempts'][0])),
            lambda r:r['attempts'][0]['checks'][3]['heads'][0].update(normalized_output_deviation=.1)]
        for mutation in mutations:
            families = list(self.cal['families']); families[1] = copy.deepcopy(families[1]); mutation(families[1])
            with self.assertRaises(ValueError): gate.validate_pool(**dict(self.cal, families=families))

    def test_direction_and_actual_guide_tampering_rejected(self):
        values = dict(self.cal, projection=dict(self.cal['projection']))
        values['projection']['directions'] = values['projection']['directions'].copy()
        values['projection']['directions'][1,1,1,0] += .1
        with self.assertRaises(ValueError): gate.validate_pool(**values)
        values = dict(self.cal, donor=dict(self.cal['donor']))
        values['donor']['observed5'] = values['donor']['observed5'].copy()
        values['donor']['observed5'][self.cal['assignment'][1],0] += np.float32(.125)
        with self.assertRaises(ValueError): gate.validate_pool(**values)

    def test_physical_pose_is_agent_block_then_sin_cos_fp64(self):
        states = np.array([[9,8,7,6,np.pi/2,100,200], [1,2,3,4,0,300,400]], np.float32)
        truth = gate._pose(states)
        self.assertEqual(truth.dtype, np.float64)
        np.testing.assert_array_equal(truth[:, :4], states[:, :4].astype(np.float64))
        self.assertEqual(truth[0,4], np.sin(np.float64(states[0,4])))
        np.testing.assert_array_equal(truth[1], [1.,2.,3.,4.,0.,1.])

    def test_saved_raw_truth_control_and_selected_index_linkage(self):
        # One raw triple unit fixture; the public gate separately enforces full N.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def save(name, **arrays):
                path=root/name;np.savez(path,**arrays)
                return dict(path=str(path),sha256=gate.sha(path))
            states=np.arange(36*7,dtype=np.float64).reshape(36,7)/16
            prefix=np.zeros((10,2),np.float32);pop=np.zeros((300,25,2),np.float32);pop[7]=.125
            bank=save('bank.npz',prefix=prefix,history_states=states[[0,5,10]],seed=np.int64(123))
            actions=save('actions.npz',population_actions=pop,selected_actions=pop[7],selected_index=np.int64(7),
                         selected_iteration=np.int64(2),seed=np.int64(123))
            physics=save('physics.npz',states=states,actions=np.concatenate([prefix,pop[7]]),seed=np.int64(123))
            original=dict(index=0,seed=123,bank=bank,actions=actions,physics=physics,selected_index=7,selected_iteration=2)
            report=dict(pool=0,count=1,cases=[dict(original,parent_id=123)])
            manifest=dict(routes=[dict(pool=p,policy_index=8*p,cases=[original]) for p in range(3)])
            arrays=dict(seeds=np.array([123],np.int64),prefix_actions=prefix[None],selected_actions=pop[7][None],
                        selected_index=np.array([7],np.int64),truth=gate._pose(states[[15,20,25,30,35]])[None])
            gate._raw_cases(report,{'parent_ids':[123]},manifest,arrays,donor=False)
            for key in ('truth','selected_actions','prefix_actions','selected_index'):
                changed=dict(arrays);changed[key]=changed[key].copy();changed[key].flat[-1]+=1
                with self.subTest(key=key),self.assertRaises(ValueError):
                    gate._raw_cases(report,{'parent_ids':[123]},manifest,changed,donor=False)
            report['cases'][0]['seed']=124
            with self.assertRaises(ValueError):gate._raw_cases(report,{'parent_ids':[123]},manifest,arrays,donor=False)

    def test_bound_files_changed_hash_and_exclusive_output_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pair = json_file(root/'input.json', {'synthetic': True})
            self.assertEqual(gate._json(pair), {'synthetic': True})
            (root/'input.json').write_text('{}')
            with self.assertRaises(ValueError): gate._json(pair)
            output = root/'accepted.json'; output.write_text('retained')
            with mock.patch.object(gate, '_json') as read:
                with self.assertRaises(ValueError): gate.accept_population('/unused', '0'*64, output)
                read.assert_not_called()
            output.unlink(); output.with_name(output.name+'.intent.json').write_text('retained')
            with mock.patch.object(gate, '_json') as read:
                with self.assertRaises(ValueError): gate.accept_population('/unused', '0'*64, output)
                read.assert_not_called()

    def test_frozen_protocol_and_exact_top_fields_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); protocol = json_file(root/'protocol.json', {'status': 'DRAFT'})
            binding = dict(status='S2_PROBE_POPULATION_BINDING_FROZEN',split='calibration', protocol=protocol,
                sources={},raw_acceptance={},caches=[],projections=[],rollouts=[],donor_assignment={})
            with self.assertRaisesRegex(ValueError,'scientific freeze'): gate._configuration(binding)
            with self.assertRaises(ValueError): gate._configuration(dict(binding, q_g={}))
            with self.assertRaises(ValueError): gate._configuration(dict(binding, split='development'))

    def test_whole_population_missing_routes_blocks_before_any_qp_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); binding = dict(split='calibration',caches=[],projections=[],rollouts=[])
            pair = json_file(root/'binding.json', binding)
            with mock.patch.object(gate,'_configuration',return_value=('calibration',256,{}, {}, {})), \
                 mock.patch.object(gate,'_raw_admission',return_value=({}, {}, {})), \
                 mock.patch.object(gate,'validate_pool') as verify:
                with self.assertRaisesRegex(ValueError,'six'): gate.accept_population(pair['path'],pair['sha256'],root/'out.json')
                verify.assert_not_called(); self.assertFalse((root/'out.json').exists())

    def test_rehashed_model_evidence_and_normalizer_receipts_rejected(self):
        # File bindings are real temporary hashes. No model/array is loaded;
        # only saved cache metadata checks are exercised in this fixture.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); empty = json_file(root/'empty.json', {})
            summary = json_file(root/'summary.json', {'normalization': {'mean':[1.,2.],'std':[3.,4.]}})
            models = [dict(pool=p,group=2*p,objective=o,condition='T0',summary=summary) for p in range(3) for o in gate.OBJECTIVES]
            contract = json_file(root/'runtime.json', dict(status='S2_MODEL_RUNTIME_CONTRACT_FROZEN',role='probe_calibration',
                protocol=empty,input_admission=empty,models=models))
            cs = json_file(root/'cache_sources.json', {'runtime_contracts': {'probe_calibration': contract}})
            sources = dict(cache_sources=cs, raw_sources=empty)
            admission = dict(parent_ids=self.cal['recipient']['seeds'].tolist(), input_manifest=empty)
            admissions = {'probe_calibration': (empty,admission,{})}
            cache_models = [dict(objective=s['objective'],spec=s,spec_sha256=gate.model_runtime.value_sha(s),
                normalization_sha256=gate.model_runtime.value_sha({'mean':[1.,2.],'std':[3.,4.]}),model_state_sha256='0'*64,
                access_receipt=dict(role='probe_calibration',runtime_contract_sha256=contract['sha256'],input_admission=empty)) for s in models[:2]]
            report = dict(status='PASS_COMPLETE_S2_LATENT_CACHE',role='calibration_recipient',split='calibration',pool=0,group=0,
                policy_index=0,count=256,protocol_sha256=empty['sha256'],cache_sources_sha256=cs['sha256'],raw_sources_sha256=empty['sha256'],
                raw_population_acceptance=empty,input_admission=empty,input_manifest=empty,content_lineage=empty,axes=gate.CACHE_AXES,
                whole_population_retained=True,response_models_deserialized=False,readout_weights_deserialized=False,
                readout_outputs_computed=False,donor_model_predictions_generated=False,native300_identity_and_endpoint_exact=True,
                observed_shared_encoder_exact=True,frozen_model_state_unchanged=True,action_normalization_unchanged=True,
                donor_encoder_only=False,source_sha256=gate.sha(gate.HERE/'cache_population.py'),runtime_contract=contract,
                models=cache_models,arrays=empty,array_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in self.cal['recipient'].items()})
            args = (dict(protocol=empty,raw_acceptance=empty),sources,dict(content_lineage=empty),admissions)
            with mock.patch.object(gate.model_runtime,'describe_fixed_model',side_effect=lambda p,o,c:next(s for s in models if s['pool']==p and s['objective']==o)), \
                 mock.patch.object(gate,'_arrays',return_value=self.cal['recipient']),mock.patch.object(gate,'_raw_cases'):
                pair=json_file(root/'cache.json',report)
                self.assertEqual(gate._cache(pair,*args,split='calibration',pool=0,donor=False)[0]['count'],256)
                mutations=[lambda r:r['models'][1].update(spec_sha256='0'*64),
                    lambda r:r['models'][1].update(normalization_sha256='0'*64),
                    lambda r:r['models'][1].update(objective='decoded_teacher'),
                    lambda r:r['models'][1]['access_receipt'].update(input_admission=summary),
                    lambda r:r.update(readout_outputs_computed=True),lambda r:r.update(source_sha256='0'*64)]
                for mutation in mutations:
                    bad=copy.deepcopy(report);mutation(bad);pair=json_file(root/'cache.json',bad)
                    with self.assertRaises(ValueError):gate._cache(pair,*args,split='calibration',pool=0,donor=False)

    def test_complete_synthetic_three_pool_file_gate_and_report_source_binding(self):
        # Earlier raw/source/runtime admission is explicitly mocked. Actual
        # synthetic NPZ hashing, three-pool assembly, dense family checks and
        # exclusive final publication run normally. This is not a real receipt.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);empty=json_file(root/'empty.json',{})
            def npz(name, arrays):
                path=root/name;np.savez_compressed(path,**arrays)
                return dict(path=str(path),sha256=gate.sha(path))
            head=npz('synthetic_head.npz',self.cal['head_A'])
            assignment=npz('assignment.npz',dict(donor_assignment=self.cal['assignment'],
                recipient_ids=self.cal['recipient']['seeds'],donor_ids=self.cal['donor']['seeds']))
            projections=npz('directions.npz',self.cal['projection']);rollouts=npz('rollouts.npz',self.cal['rollout'])
            sources=dict(heads_A=[dict(pool=p,group=2*p,file=head) for p in range(3)],
                producers=dict(projection=empty,rollout=empty))
            models=[dict(spec={'head_A_sha256':head['sha256']}) for _ in range(2)]
            recipient_report=dict(models=models,arrays=empty);donor_report=dict(models=models[:1],arrays=empty)
            binding=dict(split='calibration',protocol=empty,sources=empty,raw_acceptance=empty,
                         caches=[],projections=[],rollouts=[],donor_assignment=assignment)
            for pool in range(3):
                rp=json_file(root/f'cache_{pool}_recipient.json',{'synthetic':'recipient','pool':pool})
                dp=json_file(root/f'cache_{pool}_donor.json',{'synthetic':'donor','pool':pool})
                binding['caches'] += [dict(pool=pool,role='recipient',report=rp),dict(pool=pool,role='donor',report=dp)]
                common=dict(split='calibration',pool=pool,group=2*pool,count=256,protocol_sha256=empty['sha256'],
                    probe_sources_sha256=empty['sha256'],recipient_cache=rp,donor_cache=dp,donor_assignment=assignment,
                    head_A=head,model_role='T0',source_sha256=empty['sha256'])
                pp=json_file(root/f'projection_{pool}.json',dict(**common,status='PASS_COMPLETE_DRAFT_S2_T0_FOUR_FAMILY_PROJECTION',
                    axes=gate.PROJECTION_AXES,arrays=projections,families=self.cal['families']))
                rr=json_file(root/f'rollout_{pool}.json',dict(**common,status='PASS_COMPLETE_S2_T0_CORRECTED_PROBE_ROLLOUT',
                    projection=pp,axes=gate.ROLLOUT_AXES,models=models,native_population=300,history_tokens=3,prefix_steps=10,arrays=rollouts))
                binding['projections'].append(dict(pool=pool,report=pp));binding['rollouts'].append(dict(pool=pool,report=rr))
            def cache(*args,**kwargs):
                return (donor_report,self.cal['donor']) if kwargs['donor'] else (recipient_report,self.cal['recipient'])
            parents=dict(calibration_recipient=self.cal['recipient']['seeds'].tolist(),calibration_donor=self.cal['donor']['seeds'].tolist())
            with mock.patch.object(gate,'_configuration',return_value=('calibration',256,{}, {'donor_assignment_seeds':{'calibration':123}}, sources)), \
                 mock.patch.object(gate,'_raw_admission',return_value=({},parents,{})), mock.patch.object(gate,'_cache',side_effect=cache):
                pair=json_file(root/'binding.json',binding)
                result=gate.accept_population(pair['path'],pair['sha256'],root/'accepted.json')
                self.assertEqual(result['complete_families'],768);self.assertEqual(result['complete_directions'],3072)
                self.assertEqual(result['complete_objective_strata'],6);self.assertFalse(result['q_g_outputs_computed'])
                wrong=json.loads((root/'rollout_2.json').read_text());wrong['source_sha256']='0'*64
                binding['rollouts'][2]['report']=json_file(root/'rollout_2.json',wrong)
                pair=json_file(root/'bad_binding.json',binding)
                with self.assertRaisesRegex(ValueError,'source_sha256'):
                    gate.accept_population(pair['path'],pair['sha256'],root/'blocked.json')
                self.assertFalse((root/'blocked.json').exists());self.assertTrue((root/'blocked.json.intent.json').exists())


if __name__ == '__main__':
    unittest.main()
