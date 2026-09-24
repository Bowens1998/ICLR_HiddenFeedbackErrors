"""Synthetic metadata/assignment tests; no actual S2 banks or models."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_donor_assignments as p
from test_accept_raw_inputs import content_fixture


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True));return p.pair(path)


def fixture(root):
    root.mkdir(exist_ok=True);banks=[];parents={};bank_values={}
    for i,role in enumerate(p.raw.ROLES):
        n=p.raw.COUNTS[role];ids=list(range(10000*i+1,10000*i+1+n));parents[role]=ids
        folder=root/role;folder.mkdir();manifest={'cases':[{'seed':seed} for seed in ids]};record={'count':n}
        banks.append(dict(bank_role=role,count=n,parent_ids=ids,manifest=write(folder/'manifest.json',manifest),
                          role_metadata=write(folder/'role.json',record)))
        bank_values[role]=(folder,record,manifest)
    protocol=write(root/'protocol.json',{'status':'S2_SCIENTIFIC_PROTOCOL_FROZEN','probe_population':{'donor_assignment_seeds':{'calibration':123,'test':456}}})
    raw=write(root/'raw.json',{});content_doc=content_fixture()
    content_doc.update(protocol_sha256=protocol['sha256'],sources_sha256=raw['sha256'])
    content_doc['seed_lineage'].update(protocol_sha256=protocol['sha256'],sources_sha256=raw['sha256'])
    content=write(root/'content.json',content_doc)
    population=dict(banks=banks,content_lineage=content,admissions=[dict(role='synthetic metadata')])
    accepted=write(root/'accepted.json',population)
    binding=dict(status=p.INPUT_STATUS,protocol=protocol,raw_sources=raw,raw_acceptance=accepted,
        implementation_files={str(path.relative_to(p.gate.ROOT)):p.gate.sha(path) for path in p.SOURCE_FILES},output_root=str(root/'out'))
    bp=write(root/'input.json',binding)
    return dict(binding=bp,input=binding,population=population,parents=parents,bank_values=bank_values)


class AssignmentTests(unittest.TestCase):
    def admit(self,f):
        with (mock.patch.object(p.raw,'validate_design'),
              mock.patch.object(p.gate,'_raw_admission',return_value=(f['population'],f['parents'],{})),
              mock.patch.object(p.raw,'context',side_effect=lambda *args:{'role':args[-1]}) as raw_context,
              mock.patch.object(p.raw,'_bank',side_effect=lambda ctx:f['bank_values'][ctx['role']])):
            ctx=p.load_context(f['binding']['path'],f['binding']['sha256'])
        self.assertEqual([c.args[-1] for c in raw_context.call_args_list],list(p.raw.ROLES))
        return ctx

    def test_both_complete_archives_satisfy_exact_existing_consumers(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=fixture(Path(tmp));ctx=self.admit(f);report=p.produce(ctx)
            self.assertEqual(report['status'],p.STATUS);self.assertEqual(report['complete_splits'],2)
            self.assertEqual([r['split'] for r in report['assignments']],['calibration','test'])
            for row in report['assignments']:
                split=row['split'];arrays=p.gate._arrays(row['arrays']);n=p.gate._size(split)
                self.assertEqual(set(arrays),{'donor_assignment','recipient_ids','donor_ids'})
                self.assertTrue(all(v.dtype==np.int64 and v.shape==(n,) for v in arrays.values()))
                p.gate.validate_assignment(arrays['donor_assignment'],split=split,seed=ctx['seeds'][split])
                np.testing.assert_array_equal(arrays['recipient_ids'],f['parents'][split+'_recipient'])
                np.testing.assert_array_equal(arrays['donor_ids'],f['parents'][split+'_donor'])
                self.assertFalse(np.array_equal(arrays['donor_ids'],arrays['donor_ids'][arrays['donor_assignment']]))
            self.assertTrue((ctx['output']/'DONE').exists());self.assertFalse(report['raw_triples_replayed'])

    def test_existing_output_and_intent_block_reexecution(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=fixture(Path(tmp));ctx=self.admit(f);p.produce(ctx)
            with self.assertRaises(ValueError):p.produce(ctx)
        with tempfile.TemporaryDirectory() as tmp:
            f=fixture(Path(tmp));ctx=self.admit(f);intent=ctx['output'].with_name('out.intent.json');intent.write_text('retain')
            with self.assertRaises(ValueError):p.produce(ctx)
            self.assertEqual(intent.read_text(),'retain');self.assertFalse(ctx['output'].exists())

    def test_changed_source_closure_payload_map_or_binding_blocks_before_admission(self):
        for change in ('source','payload','binding'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);f=fixture(root);b=copy.deepcopy(f['input'])
                if change=='source':b['implementation_files'][next(iter(b['implementation_files']))]='0'*64
                elif change=='payload':b['implementation_files']['forbidden.pt']='0'*64
                else:b['status']='DRAFT'
                pair=write(root/'bad.json',b)
                with mock.patch.object(p.gate,'_raw_admission') as audit,self.assertRaises(ValueError):p.load_context(pair['path'],pair['sha256'])
                audit.assert_not_called()

    def test_original_four_bank_order_and_manifest_ids_required(self):
        for change in ('bank_order','manifest_order'):
            with tempfile.TemporaryDirectory() as tmp:
                f=fixture(Path(tmp))
                if change=='bank_order':f['population']['banks'].reverse()
                else:
                    folder,role,manifest=f['bank_values']['test_donor'];manifest['cases'].reverse()
                    f['population']['banks'][3]['manifest']=write(folder/'manifest.json',manifest)
                with self.assertRaises(ValueError):self.admit(f)
                self.assertFalse((Path(tmp)/'out').exists())

    def test_fixed_uint32_seeds_counts_and_disjoint_ids(self):
        ids=np.arange(256,dtype=np.int64);donors=ids+1000
        for seed in (True,-1,2**32,3.5,None):
            with self.assertRaises(ValueError):p.assignment_arrays('calibration',seed,ids,donors)
        for rec,don in ((ids[:-1],donors),(ids,ids),(ids.astype(np.int32),donors),(ids,np.zeros(256,np.int64))):
            with self.assertRaises(ValueError):p.assignment_arrays('calibration',123,rec,don)
        a=p.assignment_arrays('calibration',123,ids,donors)
        b=p.assignment_arrays('calibration',123,ids,donors)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_missing_or_changed_frozen_seed_cannot_create_artifacts(self):
        for seeds in ({'calibration':123},{'calibration':True,'test':456}):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);f=fixture(root);f['input']['protocol']=write(root/'protocol.json',{'probe_population':{'donor_assignment_seeds':seeds}})
                f['binding']=write(root/'input.json',f['input'])
                with mock.patch.object(p.raw,'validate_design'),mock.patch.object(p.gate,'_raw_admission') as audit,self.assertRaises(ValueError):
                    p.load_context(f['binding']['path'],f['binding']['sha256'])
                audit.assert_not_called();self.assertFalse((root/'out').exists())

    def test_second_archive_failure_preserves_first_and_blocks_partial_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=fixture(Path(tmp));ctx=self.admit(f);read=p.gate._arrays
            def fail_second(pair):
                if Path(pair['path']).name=='test.npz':raise ValueError('synthetic readback failure')
                return read(pair)
            with mock.patch.object(p.gate,'_arrays',side_effect=fail_second),self.assertRaisesRegex(ValueError,'readback'):p.produce(ctx)
            self.assertTrue((ctx['output']/'calibration.npz').exists());self.assertTrue((ctx['output']/'failure.json').exists())
            self.assertFalse((ctx['output']/'report.json').exists());self.assertFalse((ctx['output']/'DONE').exists())
            with self.assertRaises(ValueError):p.produce(ctx)

    def test_real_raw_metadata_and_content_validators_before_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=fixture(root);b=f['input'];population=f['population']
            # Only raw source deployment and generated-bank replay are mocked.
            # Real shared raw-admission/content/hash validators read these
            # synthetic metadata files; no accepted real dataset is claimed.
            population.update(status='PASS_S2_COMPLETE_RAW_INPUT_POPULATION',protocol_sha256=b['protocol']['sha256'],
                sources_sha256=b['raw_sources']['sha256'],raw_sources_sha256=b['raw_sources']['sha256'],
                source_sha256=p.gate.sha(p.HERE/'accept_raw_inputs.py'),bank_count=4,route_count=18,
                saved_triples_verified=6912,models_or_readouts_loaded=False)
            admissions=[]
            for bank in population['banks']:
                role=bank['bank_role'];split=role.split('_')[0];donor=role.endswith('_donor')
                for stream in (('donor_observed',) if donor else ('probe','response')):
                    runtime_role=p.raw_gate.runtime_role(role,stream)
                    manifest=dict(status='S2_COMPLETE_ROLE_STREAM_INPUT_MANIFEST',protocol_sha256=b['protocol']['sha256'],
                        sources_sha256=b['raw_sources']['sha256'],bank_role=role,stream_role=stream,
                        count=bank['count'],parent_ids=bank['parent_ids'])
                    mp=write(root/(runtime_role+'_manifest.json'),manifest)
                    doc=dict(manifest,status='S2_FIXED_CONTEXTS_ACCEPTED',bank_manifest=bank['manifest'],input_manifest=mp)
                    admissions.append(dict(role=runtime_role,**write(root/(runtime_role+'.json'),doc)))
            population['admissions']=admissions;b['raw_acceptance']=write(root/'accepted.json',population)
            f['binding']=write(root/'input.json',b)
            def load():
                with (mock.patch.object(p.raw,'validate_design'),mock.patch.object(p.raw,'context',side_effect=lambda *args:{'role':args[-1]}),
                      mock.patch.object(p.raw,'_bank',side_effect=lambda ctx:f['bank_values'][ctx['role']])):
                    return p.load_context(f['binding']['path'],f['binding']['sha256'])
            self.assertEqual(load()['parents'],f['parents'])
            content=p.gate._json(population['content_lineage']);content['head_caches'].pop()
            population['content_lineage']=write(root/'content.json',content)
            b['raw_acceptance']=write(root/'accepted.json',population);f['binding']=write(root/'input.json',b)
            with self.assertRaisesRegex(ValueError,'historical head exposure'):load()
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
