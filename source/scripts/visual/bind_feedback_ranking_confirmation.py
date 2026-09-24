"""Bind new accepted goals to an unchanged, already verified model roster."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-plan', 'bank', 'protocol', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    assert sha(args.source_plan) == 'f305542c01347c9d9b2035faa7de29c9802d7012000581a656207088e5b129c5'
    plan = json.loads(Path(args.source_plan).read_text())
    assert len(plan['models']) == 48 and len(plan['routes']) == 96
    bank = Path(args.bank)
    manifest = json.loads((bank / 'manifest.json').read_text())
    acceptance = json.loads((bank / 'acceptance.json').read_text())
    assert manifest['seed_start'] == 1381001 and manifest['max_seeds'] == 8192
    assert len(manifest['cases']) == len(acceptance['rows']) == 512
    assert acceptance['status'] == 'PASS_FULL_REFERENCE_AND_GOAL_REPLAY'
    assert acceptance['manifest_sha256'] == sha(bank / 'manifest.json')
    for row, checked in zip(manifest['cases'], acceptance['rows']):
        assert row['index'] == checked['index'] and row['seed'] == checked['seed']
        assert row['sha256'] == checked['sha256'] == sha(bank / f"case_{row['index']:03d}.npz")
        assert checked['independently_replayed_branches'] == 33
    # Freeze the six references and all eighteen scored continuations without
    # requiring unrelated legacy experiment artifacts to remain on disk.
    for group in range(6):
        for slot in (0, 2, 3, 4):
            entry = plan['models'][8 * group + slot]
            path = Path(entry['training_path'])
            assert sha(path / 'last_weights.pt') == entry['weights_sha256']
            assert sha(path / 'summary.json') == entry['training_summary_sha256']
            for role in ('endpoint_head', 'goal_head'):
                assert sha(entry[role]['path']) == entry[role]['sha256']
        assert plan['routes'][16 * group]['model_index'] == 8 * group
    plan['prediction_confirmation_source'] = plan.pop('fiber_confirmation')
    plan['source_plan_sha256'] = sha(args.source_plan)
    plan['bank_manifest_sha256'] = sha(bank / 'manifest.json')
    plan['feedback_ranking_confirmation'] = dict(
        status='BOUND512_FRESH_GOALS_UNCHANGED_MODELS', cases=512, seed_start=1381001,
        protocol_sha256=sha(args.protocol), source_sha256=sha(__file__),
        bank_acceptance_sha256=sha(bank / 'acceptance.json'), reference_routes=[16 * g for g in range(6)])
    with Path(args.output).open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    print('BOUND512_FRESH_GOALS_UNCHANGED_MODELS', flush=True)


if __name__ == '__main__':
    main()
