"""S3 source/interface guards only; metadata fixtures and mocked subprocesses."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import raw_inputs as raw


def protocol_fixture():
    protocol = json.loads((raw.PHASE / 'protocol/S3_PROTOCOL.draft.json').read_text())
    protocol.update(status='S3_SCIENTIFIC_PROTOCOL_FROZEN', scientific_protocol_frozen=True,
                    raw_inputs=dict(source_base='/synthetic/base', output_root='/synthetic/output'))
    return protocol  # Memory-only interface fixture, never written as a source/protocol lock.


def bank_fixture():
    spec = dict(count=512, seed_start=1000, max_seeds=100000)
    cases = [dict(index=i, seed=1000 + i, sha256=f'{i:064x}') for i in range(512)]
    manifest = dict(cases=cases, seed_start=1000, max_seeds=100000, rejected_seeds=[])
    accepted = dict(status='PASS_FULL_REFERENCE_AND_GOAL_REPLAY', manifest_sha256='a' * 64,
        rows=[dict(r, independently_replayed_branches=33) for r in cases])
    return manifest, accepted, spec


class RawTests(unittest.TestCase):
    def test_existing_draft_cannot_authorize_execution(self):
        protocol = protocol_fixture(); raw.validate_design(protocol)
        for field, value in [('status', 'DRAFT_NOT_EXECUTION_AUTHORIZATION'), ('scientific_protocol_frozen', False)]:
            wrong = copy.deepcopy(protocol); wrong[field] = value
            with self.assertRaises(ValueError): raw.validate_design(wrong)

    def test_fixed_native_scope_counts_seeds_and_paths(self):
        for section, field, value in [('population', 'recipient_count', 513),
                ('population', 'max_attempts_per_role', 100001), ('population', 'recipient_seed_start', True),
                ('shared_prefix_decision', 'reference_policy_rows', [0, 8, 17]),
                ('shared_prefix_decision', 'legacy_reference_route_ids', [0, 8, 16]),
                ('shared_prefix_decision', 'native_root_candidate_count', 1),
                ('raw_inputs', 'output_root', 'relative')]:
            with self.subTest(field=field):
                p = protocol_fixture(); p[section][field] = value
                with self.assertRaises(ValueError): raw.validate_design(p)
        p = protocol_fixture(); p['population']['donor_seed_start'] = p['population']['recipient_seed_start'] + 1
        with self.assertRaises(ValueError): raw.validate_design(p)

    def test_bank_acceptor_requires_every_ordered33branch_row(self):
        m, a, s = bank_fixture(); raw.validate_bank(m, a, s, 'a' * 64)
        for kind in ['count', 'duplicate', 'seed', 'hash', 'branches']:
            with self.subTest(kind=kind):
                bad = copy.deepcopy(a)
                if kind == 'count': bad['rows'].pop()
                if kind == 'duplicate': bad['rows'][511] = bad['rows'][0]
                if kind == 'seed': bad['rows'][255]['seed'] += 1
                if kind == 'hash': bad['rows'][0]['sha256'] = 'f' * 64
                if kind == 'branches': bad['rows'][0]['independently_replayed_branches'] = 32
                with self.assertRaises(ValueError): raw.validate_bank(m, bad, s, 'a' * 64)

    def test_actual_rejections_are_required_but_unused_budget_not_exposure(self):
        m, a, s = bank_fixture()
        for r in m['cases']: r['seed'] += 1
        for r in a['rows']: r['seed'] += 1
        m['rejected_seeds'] = [dict(seed=1000, reason='synthetic visibility rejection')]
        raw.validate_bank(m, a, s, 'a' * 64)
        self.assertEqual(len(raw.prior.historical_attempt_seeds(m)), 513)
        self.assertNotIn(90000, raw.prior.historical_attempt_seeds(m))
        for rejection in [[], [dict(seed=1000, reason='')]]:
            bad = copy.deepcopy(m); bad['rejected_seeds'] = rejection
            with self.assertRaises(ValueError): raw.validate_bank(bad, a, s, 'a' * 64)

    def test_three_helper_bridge_preserves_exact_bodies_and_module(self):
        kernel = raw._load(raw.KERNEL, '_synthetic_readonly_kernel_bridge_test')
        before = dict(vars(kernel))
        for name in ('s1_plan', 's1_replay'):
            original = getattr(kernel, name); bridged = raw.bridge_function(kernel, name)
            self.assertIs(bridged.__code__, original.__code__)
            self.assertEqual(bridged.__globals__['__file__'], original.__globals__['__file__'])
            changed = {k for k in original.__globals__ if bridged.__globals__[k] is not original.__globals__[k]}
            self.assertEqual(changed, {'s1_bank', 's1_path', 's1_reference'})
        self.assertEqual(vars(kernel), before)
        self.assertEqual(set(raw.bridge_identity()), {'s1_bank', 's1_path', 's1_reference'})
        with self.assertRaises(ValueError): raw.bridge_function(kernel, 's1_prepare')

    def test_exclusive_intent_blocks_retry_even_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx = dict(artifacts=Path(temp), role='recipient', binding=dict(stage='S3'))
            target = Path(temp) / 'new-output'
            raw._intent(ctx, 'prepare', target)
            with self.assertRaises(FileExistsError): raw._intent(ctx, 'prepare', target)
            existing = Path(temp) / 'existing'; existing.mkdir()
            with self.assertRaises(ValueError): raw._intent(ctx, 'plan', existing, 0)

    def test_generation_failure_preserves_intent_and_never_accepts(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx = dict(artifacts=Path(temp), base=Path('/synthetic/base'), role='donor',
                spec=dict(count=512, seed_start=942872802, max_seeds=100000), binding=dict(stage='S3'))
            with patch.object(raw, 'lineage', return_value={'synthetic_metadata': True}), \
                    patch.object(raw.subprocess, 'run', side_effect=subprocess.CalledProcessError(2, 'synthetic')) as call:
                with self.assertRaises(subprocess.CalledProcessError): raw.prepare(ctx)
            self.assertEqual(call.call_count, 1)
            argv = call.call_args.args[0]
            self.assertIn('512', argv); self.assertIn('100000', argv); self.assertIn('942872802', argv)
            self.assertTrue((Path(temp) / 'intents/donor/prepare.json').is_file())
            self.assertFalse((Path(temp) / 'banks/donor/role.json').exists())


if __name__ == '__main__': unittest.main()
