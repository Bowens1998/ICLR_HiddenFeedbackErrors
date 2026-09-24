"""Synthetic/schema/source-bridge tests. No actual simulation, arrays or Torch."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import raw_inputs as r


def design():
    return dict(status='S2_SCIENTIFIC_PROTOCOL_FROZEN', raw_inputs=dict(native_population=300,
        relative_horizons=[5,10,15,20,25], source_base='/synthetic/source', output_root='/synthetic/output',
        reference_policy_indices=dict(recipient=[0,1,8,9,16,17], donor=[0,8,16]),
        banks=[dict(role=role,count=r.COUNTS[role],seed_start=10000*(i+1),max_seeds=2000) for i,role in enumerate(r.ROLES)]))


def generated(count=256):
    # Explicitly synthetic metadata. One rejected seed is retained with reason.
    return dict(seed_start=10,max_seeds=1000,cases=[dict(index=i,seed=11+i) for i in range(count)],
                rejected_seeds=[dict(seed=10,reason='synthetic rejection')])


class RawInputTests(unittest.TestCase):
    def test_import_is_lazy_and_reads_no_torch(self):
        code="import sys;sys.path.insert(0,"+repr(str(Path(r.__file__).parent))+");import raw_inputs;assert 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_exact_four_banks_and_native_plan(self):
        raw=r.validate_design(design())
        self.assertEqual([b['count'] for b in raw['banks']],[256,256,512,512])
        self.assertEqual(sum(len(r.routes(role)) for role in r.ROLES),18)

    def test_no_default_seeds_or_attempt_budget(self):
        for field in ('seed_start','max_seeds'):
            d=design();del d['raw_inputs']['banks'][0][field]
            with self.assertRaises(ValueError):r.validate_design(d)

    def test_no_s1_protocol_or_unfrozen_execution(self):
        for status in ('DRAFT','S1_SCIENTIFIC_PROTOCOL_FROZEN',None):
            d=design();d['status']=status
            with self.assertRaises(ValueError):r.validate_design(d)

    def test_wrong_count_population_horizon_route_rejected(self):
        variants=[]
        for field,value in [('native_population',1),('relative_horizons',[5,25]),('reference_policy_indices',{'recipient':[0,1],'donor':[0]})]:
            d=design();d['raw_inputs'][field]=value;variants.append(d)
        d=design();d['raw_inputs']['banks'][2]['count']=511;variants.append(d)
        for d in variants:
            with self.assertRaises(ValueError):r.validate_design(d)

    def test_duplicate_role_or_overlap_fails(self):
        d=design();d['raw_inputs']['banks'][1]=copy.deepcopy(d['raw_inputs']['banks'][0])
        with self.assertRaises(ValueError):r.validate_design(d)
        d=design();d['raw_inputs']['banks'][1]['seed_start']=10001
        with self.assertRaises(ValueError):r.validate_design(d)

    def test_actual_attempts_exclude_unused_budget(self):
        m=generated(2);m['max_seeds']=100000
        self.assertEqual(r.historical_attempt_seeds(m),{10,11,12})
        del m['max_seeds'];self.assertEqual(r.historical_attempt_seeds(m),{10,11,12})
        m['seed_start']=None;self.assertEqual(r.historical_attempt_seeds(m),{10,11,12})

    def test_missing_attempt_duplicate_and_over_budget_block(self):
        for change in (lambda m:m['rejected_seeds'].clear(),lambda m:m['rejected_seeds'].append({'seed':11}),
                       lambda m:m.update(max_seeds=2)):
            m=generated(2);change(m)
            with self.assertRaises(ValueError):r.historical_attempt_seeds(m)

    def test_derived_attempts_follow_bound_context_parent(self):
        parent=generated(3);child=dict(cases=[dict(seed=11),dict(seed=13)],bindings={'contexts_manifest_sha256':'synthetic-parent'})
        self.assertEqual(r.historical_attempt_seeds(child,{'synthetic-parent':parent}),{10,11,12,13})
        child['cases'][0]['seed']=99
        with self.assertRaises(ValueError):r.historical_attempt_seeds(child,{'synthetic-parent':parent})

    def test_unresolved_or_cyclic_attempt_parent_blocks(self):
        child=dict(cases=[dict(seed=11)],bindings={'contexts_manifest_sha256':'self'})
        with self.assertRaisesRegex(ValueError,'UNRESOLVED'):r.historical_attempt_seeds(child)
        with self.assertRaisesRegex(ValueError,'UNRESOLVED'):r.historical_attempt_seeds(child,{'self':child})
        with self.assertRaisesRegex(ValueError,'UNRESOLVED'):r.historical_attempt_seeds({'cases':[{'seed':11}]})

    def test_existing_metadata_attempt_semantics_preserved(self):
        # Prior local metadata only; no remote bank or scientific array read.
        path=r.PHASE/'audits/S1_REMOTE_BANK_LINEAGE_AND_RESOURCES.json'
        audit=json.loads(path.read_text())['bank_manifests'];actual=set();covered=set()
        for entry in audit:
            actual.update(entry['seeds']+entry['rejected_seeds'])
            values=set(entry['seeds']+entry['rejected_seeds'])
            if len(values)==max(values)-min(values)+1:
                manifest=dict(cases=[{'seed':x} for x in entry['seeds']],rejected_seeds=[{'seed':x} for x in entry['rejected_seeds']],
                              seed_start=entry['seed_start'],max_seeds=entry['max_seeds'])
                covered.update(r.historical_attempt_seeds(manifest))
        self.assertEqual(len(audit),105);self.assertEqual(len(actual),29624);self.assertEqual(covered,actual)

    def test_new_generated_complete_indices_seeds_and_reasons(self):
        m=generated();s=dict(count=256,seed_start=10,max_seeds=1000)
        r._validate_generated_manifest(m,s)
        for mutation in (lambda m:m['cases'].pop(),lambda m:m['cases'][0].update(index=2),
                         lambda m:m['rejected_seeds'][0].update(reason=''),lambda m:m['cases'][0].update(seed=12)):
            altered=copy.deepcopy(m);mutation(altered)
            with self.assertRaises(ValueError):r._validate_generated_manifest(altered,s)

    def test_recipient_and_donor_policy_indices_are_original_not_reindexed(self):
        policies=dict(rows=[dict(index=i,entry={'adaptation_condition':'original','arm':'transformer_jepa'}) for i in range(18)])
        for role in r.ROLES:
            ctx=dict(role=role,sources={'reference_policies':{}})
            with mock.patch.object(r,'_read_pair',return_value=policies):
                for index in r.routes(role):self.assertEqual(r._reference(ctx,index)['index'],index)
                for index in set(range(18))-set(r.routes(role)):
                    with self.assertRaises(ValueError):r._reference(ctx,index)
        with self.assertRaises(ValueError):r.routes('recipient')

    def test_changed_original_policy_identity_blocks(self):
        p=dict(rows=[dict(index=0,entry={'adaptation_condition':'T0','arm':'transformer_jepa'})])
        with mock.patch.object(r,'_read_pair',return_value=p),self.assertRaises(ValueError):
            r._reference(dict(role='calibration_recipient',sources={'reference_policies':{}}),0)

    def test_real_bridge_retains_identical_code_and_only_three_global_changes(self):
        spec=importlib.util.spec_from_file_location('_synthetic_bridge_source_inspection',r.KERNEL)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        before=module.__dict__.copy()
        for name in ('s1_plan','s1_replay'):
            original=getattr(module,name);bridged=r.bridge_function(module,name)
            self.assertIs(bridged.__code__,original.__code__)
            changed={k for k in before if bridged.__globals__.get(k) is not before[k]}
            self.assertEqual(changed,{'s1_bank','s1_path','s1_reference'})
            self.assertEqual(bridged.__globals__['__file__'],str(r.KERNEL))
            self.assertNotIn('s1_context',original.__code__.co_names)
            for key in ('s1_sha','s1_json','s1_write_json','s1_npz'):self.assertIs(bridged.__globals__[key],before[key])
        self.assertTrue(all(module.__dict__[k] is v for k,v in before.items()))
        self.assertNotIn('torch',module.__dict__)

    def test_bridge_helper_identities_and_hashes_complete(self):
        bindings=r.bridge_identity()
        self.assertEqual(set(bindings),{'s1_bank','s1_path','s1_reference'})
        for row in bindings.values():
            self.assertEqual(row['wrapper_source_sha256'],r.sha(r.__file__))
            self.assertEqual(len(row['function_source_sha256']),64)
        with self.assertRaises(ValueError):r.bridge_function(SimpleNamespace(),'s1_context')

    def test_existing_output_is_never_resumed_or_changed(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);out=base/'existing';out.mkdir();(out/'partial').write_text('retained')
            ctx=dict(artifacts=base,role='test_donor',binding={})
            with self.assertRaises(ValueError):r._intent(ctx,'plan',out,16)
            self.assertEqual((out/'partial').read_text(),'retained');self.assertFalse((base/'intents').exists())

    def test_prior_intent_blocks_even_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);ctx=dict(artifacts=base,role='test_donor',binding={})
            path=r._intent(ctx,'replay',base/'missing',16);before=path.read_bytes()
            with self.assertRaises(FileExistsError):r._intent(ctx,'replay',base/'missing',16)
            self.assertEqual(path.read_bytes(),before)

    def test_partial_failed_kernel_is_retained_and_second_attempt_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);ctx=dict(artifacts=base,role='test_donor',binding={},spec={'count':512})
            out=base/'actions/test_donor/route_16'
            def failed(*args):out.mkdir(parents=True);(out/'partial').write_text('retained');raise RuntimeError('synthetic failure')
            with mock.patch.object(r,'_reference'),mock.patch.object(r,'_bank'),mock.patch.object(r,'_kernel'),mock.patch.object(r,'bridge_function',return_value=failed) as bridge:
                with self.assertRaises(RuntimeError):r.run_reference(ctx,16,'plan')
                with self.assertRaises(ValueError):r.run_reference(ctx,16,'plan')
                self.assertEqual(bridge.call_count,1);self.assertEqual((out/'partial').read_text(),'retained')

    def test_all_prior_physics_routes_have_unique_exact_paths(self):
        bank=Path('/synthetic/artifacts/confirmation/banks/donor')
        records=[dict(index=i,path=str(bank.parent.parent/'physics/donor'/f'route_{i}'/'report.json'),sha256='synthetic') for i in range(8)]
        self.assertEqual(len(r._prior_physics_paths(bank,records)),8)
        records[1]['path']=records[0]['path']
        with self.assertRaises(ValueError):r._prior_physics_paths(bank,records)
        records[1]['path']='/synthetic/wrong/route_1/report.json'
        with self.assertRaises(ValueError):r._prior_physics_paths(bank,records)

    def test_content_covers_four_roles_four_s1_banks_and_head_aliases(self):
        new={role:{'new-'+role} for role in r.ROLES}
        old={k+'/'+role:{'old-'+k+role} for k in ('development','confirmation') for role in ('recipient','donor')}
        self.assertEqual(r.compare_content(new,old,{'head'})['status'],'PASS_S2_RETAINED_PIXEL_ISOLATION')
        for source in ('head','old','new'):
            changed=copy.deepcopy(new)
            changed[r.ROLES[0]].add('head' if source=='head' else next(iter(old['confirmation/donor'])) if source=='old' else 'new-'+r.ROLES[3])
            self.assertEqual(r.compare_content(changed,old,{'head'})['status'],'BLOCKED_S2_PIXEL_ALIAS')
        self.assertEqual(r.HEAD_GROUPS,(0,1,2,4))

    def test_missing_content_population_cannot_pass(self):
        new={role:set() for role in r.ROLES};old={k+'/'+role:set() for k in ('development','confirmation') for role in ('recipient','donor')}
        altered=dict(new);del altered[r.ROLES[0]]
        with self.assertRaises(ValueError):r.compare_content(altered,old,set())
        old.pop('confirmation/donor')
        with self.assertRaises(ValueError):r.compare_content(new,old,set())

    def test_raw_pixel_hash_uses_exact_bytes_and_authenticates_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'synthetic.npz';np.savez(path,pixels=np.zeros((2,3,3,3),np.uint8))
            hashes,n=r._pixel_hashes(path,r.sha(path),[('pixels',(2,3,3,3))]);self.assertEqual(n,2);self.assertEqual(len(hashes),1)
            with self.assertRaises(ValueError):r._pixel_hashes(path,'0'*64,[('pixels',(2,3,3,3))])
            with self.assertRaises(ValueError):r._pixel_hashes(path,r.sha(path),[('pixels',(1,3,3,3))])


if __name__=='__main__':unittest.main()
