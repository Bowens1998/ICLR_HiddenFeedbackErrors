"""Draft S2 scalar joins and held-out response admission; no scientific freeze.

This wrapper reuses the reviewed model, readout, feature and regression modules.
It performs no model inference, regression fitting, model choice or resampling.
Actual future source/protocol/input bindings are mandatory; fixtures are not
execution authority. The two entry points preserve the prediction-seal barrier.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import features
import finalize_regression as finalizer
import accept_probe_population as gate
import accept_raw_inputs
import cache_population

ROOT = gate.ROOT
require = finalizer.require
pair = finalizer.descriptor
read = finalizer.document
SOURCE_STATUS = 'S2_SCALAR_ASSEMBLY_SOURCES_FROZEN'
PREPARE_STATUS = 'S2_CALIBRATION_AND_PROBE_ASSEMBLY_BINDING_FROZEN'
TEST_STATUS = 'S2_HELDOUT_SCALAR_ASSEMBLY_BINDING_FROZEN'


def modules():
    # Lazy imports permit source-only tooling while other draft wrappers are
    # being implemented. Actual execution requires both real source bindings.
    return importlib.import_module('score_probe_population'), importlib.import_module('response_population')


def exact_fields(value, expected, label):
    finalizer.exact_fields(value, expected, label)


def write(path, value):
    gate._write(path, value)


def save(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    return pair(path)


def source_context(source_pair, protocol_pair):
    """Authenticate existing source bytes and numeric contracts, not outcomes."""
    protocol = read(protocol_pair, 'Scientific protocol')
    require(protocol.get('status') == 'S2_SCIENTIFIC_PROTOCOL_FROZEN', 'Draft protocol cannot execute scalar assembly')
    source = read(source_pair, 'Scalar assembly source')
    require(source.get('status') == SOURCE_STATUS and source.get('protocol') == protocol_pair,
            'Unfrozen or wrong-protocol scalar assembly source')
    require(source.get('source_root') == str(ROOT), 'Wrong deployed source root')
    needed = [Path(__file__), HERE/'features.py', HERE/'finalize_regression.py', HERE/'regression.py',
        HERE/'independent_regression.py', HERE/'score_probe_population.py', HERE/'response_population.py',
        HERE/'accept_probe_population.py', HERE/'accept_raw_inputs.py', HERE/'cache_population.py']
    actual = {}
    for name, digest in source['files'].items():
        path = Path(name); path = path if path.is_absolute() else ROOT/path
        finalizer.independent.checked_file(dict(path=str(path), sha256=digest), 'Assembly source')
        actual[str(path.resolve())] = digest
    require(all(str(path.resolve()) in actual for path in needed), 'Missing direct scalar assembly source')
    numerical = read(source['numerical_sources'], 'Frozen numerical source')
    require(numerical.get('status') == 'S2_REGRESSION_SOURCES_FROZEN' and
            numerical.get('protocol_sha256') == protocol_pair['sha256'], 'Wrong numerical source/protocol')
    names = {'finalize_regression.py','regression.py','independent_regression.py'}
    exact_fields(numerical['files'], names, 'Exact numerical source modules')
    for name in names:
        require(numerical['files'][name] == pair(HERE/name), 'Wrong executing numerical module')
    for name in ('measurement_sources','response_latent_sources'):
        read(source[name], name)
    spec = protocol['regression']
    require(spec['calibration_count'] == 256 and spec['test_count'] == 512 and
            spec['stratum_ids'] == list(features.STRATA), 'Changed fixed scalar population or stratum order')
    return dict(protocol=protocol, protocol_pair=protocol_pair, sources=source, source_pair=source_pair)


def check_response_contract(ctx, descriptor):
    contract = read(descriptor, 'Predeclared response contract')
    expected = dict(status='S2_TEST_RESPONSE_CONTRACT_FROZEN', role='test_response', count=512,
        protocol_sha256=ctx['protocol_pair']['sha256'], scalar_assembly_sources=ctx['source_pair'],
        response_latent_sources=ctx['sources']['response_latent_sources'],
        measurement_sources=ctx['sources']['measurement_sources'])
    require(all(contract.get(k) == v for k,v in expected.items()), 'Predeclared response computation changed')
    latent_sources = read(ctx['sources']['response_latent_sources'], 'Response producer sources')
    require(contract['runtime_contract'] == latent_sources['runtime_contracts']['test_response'],
            'Response contract/runtime identity mismatch')
    runtime = read(contract['runtime_contract'], 'Response runtime')
    require(runtime.get('protocol') == ctx['protocol_pair'] and runtime.get('role') == 'test_response' and
            runtime.get('status') == 'S2_MODEL_RUNTIME_CONTRACT_FROZEN' and
            contract['input_admission'] == runtime['input_admission'], 'Response runtime/input mismatch')
    return contract


def validate_probe_arrays(data, report, split):
    n = gate._size(split)
    exact_fields(data, ['ordinary','signed_g','recipient_ids','donor_ids'], 'Complete probe scalar archive')
    finalizer.independent.finite64(data['ordinary'], (n,6,4), 'Probe ordinary')
    finalizer.independent.finite64(data['signed_g'], (n,6), 'Probe signed G')
    require(np.all(data['ordinary'] >= 0), 'Ordinary errors and squared dose cannot be negative')
    for role in ('recipient','donor'):
        ids = finalizer.independent.ids(data[role+'_ids'], n, role+' IDs')
        require(report[role+'_ids'] == ids.tolist(), 'Probe scalar ID report/order mismatch')
    require(not np.intersect1d(data['recipient_ids'],data['donor_ids']).size, 'Probe recipient/donor overlap')
    return data


def complete_probe(ctx, descriptor, split):
    scorer, _ = modules()
    checked = scorer.verify_completed_probe(descriptor, protocol_pair=ctx['protocol_pair'],
        measurement_sources_pair=ctx['sources']['measurement_sources'], expected_split=split)
    report, data = checked['report'], checked['arrays']
    require(report.get('status') == 'PASS_COMPLETE_S2_PROBE_SCALARS' and report.get('split') == split and
            report.get('count') == gate._size(split), 'Incomplete scalar probe report')
    validate_probe_arrays(data, report, split)
    return checked


def validate_raw_isolation(raw_pair, ctx, probe_reports, rosters):
    raw = read(raw_pair, 'Accepted complete raw population')
    raw_sources_pair = ctx['sources']['raw_sources']
    read(raw_sources_pair, 'Raw sources')
    cache_population.validate_population(raw, protocol_sha256=ctx['protocol_pair']['sha256'],
        raw_sources_sha256=raw_sources_pair['sha256'])
    require(raw.get('source_sha256') == gate.sha(HERE/'accept_raw_inputs.py') and
            raw.get('bank_count') == 4 and raw.get('route_count') == 18 and
            raw.get('saved_triples_verified') == 6912, 'Incomplete or changed raw-input acceptance')
    content = read(raw['content_lineage'], 'Complete raw content/lineage')
    accept_raw_inputs.validate_content(content, ctx['protocol_pair']['sha256'], raw_sources_pair['sha256'])
    finalizer.check_rosters(rosters)
    for split in ('calibration','test'):
        report = probe_reports[split]
        require(report['raw_acceptance'] == raw_pair, 'Probe populations have different raw ancestry')
        for kind in ('recipient','donor'):
            bank = next(x for x in raw['banks'] if x['bank_role'] == split+'_'+kind)
            ids = rosters[split+'_'+kind+'_ids']
            # Donor IDs deliberately use recipient-assigned order. The probe
            # verifier has reconstructed bank_ids[permutation] exactly once.
            require((ids == bank['parent_ids']) if kind == 'recipient' else
                    (sorted(ids) == sorted(bank['parent_ids'])), 'Raw bank/assigned parent coverage differs')
    return raw


def response_population(ctx, descriptor, split, raw_pair, seal_pair=None):
    _, producer = modules()
    validated = producer.verify_completed_population(descriptor, protocol_pair=ctx['protocol_pair'],
        expected_split=split, expected_raw_acceptance=raw_pair, expected_seal=seal_pair)
    report = validated['report']
    require(report.get('status') == 'PASS_COMPLETE_S2_RESPONSE_LATENT_POPULATION' and
            report.get('producer_sources') == ctx['sources']['response_latent_sources'] and
            report.get('protocol') == ctx['protocol_pair'] and report.get('split') == split and
            report.get('raw_acceptance') == raw_pair, 'Changed or incomplete response production source')
    return validated


def decode_response(ctx, verified, recipient_ids, *, split, probe_acceptance):
    """Called only after complete probe and response admission by either entry."""
    scorer, _ = modules()
    n = gate._size(split)
    require(verified['parent_ids'] == recipient_ids.tolist(), 'Probe/response recipient order differs')
    pools = verified['pools']
    require(len(pools) == 3 and [r['pool'] for r in pools] == [0,1,2], 'All response pools required')
    measurement = scorer.authenticate_measurement_sources(ctx['sources']['measurement_sources'], ctx['protocol_pair'])
    measurement = scorer.authorize_measurement_context(measurement, probe_acceptance)
    poses, truths, checks = [], [], []
    for entry in pools:
        pool = entry['pool']; arrays = entry['arrays']
        tokens = gate._array(arrays['free'], (2,2,n,5,192), np.dtype('float32'), 'Response latent free')
        truth = finalizer.independent.finite64(arrays['truth'], (n,5,6), 'Response physical truth')
        gate._equal(arrays['seeds'], recipient_ids, 'Response raw parent order')
        # The axes change is explicit: objective,condition,N,h,z -> N,obj,cond,h,z.
        result = scorer.decode_pool_tokens(measurement, pool, np.transpose(tokens,(2,0,1,3,4)))
        decoded = finalizer.independent.finite64(result['poses'], (n,2,2,5,6), 'Decoded response poses')
        poses.append(decoded); truths.append(truth); checks.append(result['verification'])
    prediction = np.stack(poses, axis=1); truth = np.stack(truths, axis=1)
    axes = dict(groups=list(features.GROUPS), objectives=list(features.OBJECTIVES),
        horizons=list(features.HORIZONS), pose=list(features.POSE), conditions=list(features.CONDITIONS))
    result = features.response_values(prediction, truth, split=split, axes=axes)
    return dict(prediction=prediction, truth=truth, block_errors=result['block_errors'],
                response=result['response']), checks


def start_output(path, intent):
    out = Path(path)
    require(out.is_absolute() and not out.exists(), 'An unused absolute output directory is required')
    marker = out.with_name(out.name+'.intent.json')
    require(not marker.exists(), 'Prior scalar assembly intent forbids automatic retry')
    write(marker, intent); out.mkdir(parents=True, exist_ok=False)
    return out


def scalar_receipt(out, ctx, split, probe_pair, probe, response, response_evidence):
    data = {name:np.array(value,copy=True) for name,value in probe['arrays'].items()}
    role = 'calibration_regression' if split == 'calibration' else 'probe_test'
    if split == 'calibration':
        data['response'] = finalizer.independent.finite64(response, (256,6), 'Complete calibration Y')
    else:
        require(response is None and response_evidence is None, 'Test probe cannot contain a response')
    ap = save(out/(split+'_scalars.npz'), data)
    fields = dict(role=role, split=split, count=gate._size(split), stratum_ids=list(features.STRATA),
        protocol_sha256=ctx['protocol_pair']['sha256'], sources_sha256=ctx['sources']['numerical_sources']['sha256'],
        complete_population_accepted=True)
    upstream = dict(status='PASS_COMPLETE_S2_SCALAR_POPULATION', **fields, scalar_arrays=ap,
        recipient_ids=data['recipient_ids'].tolist(), donor_ids=data['donor_ids'].tolist(),
        probe_scalar_population=probe_pair, assembly_sources=ctx['source_pair'],
        measurement_sources=ctx['sources']['measurement_sources'], calibration_response=response_evidence,
        source_sha256=gate.sha(__file__), error_units='position_squared', forecast_mse_units='position_to_the_fourth',
        source_scope='Authenticated complete producer evidence and saved-array arithmetic; no independent native model replay.')
    up = out/(split+'_scalar_acceptance.json');write(up,upstream)
    receipt = out/(split+'_scalar_receipt.json')
    write(receipt,dict(status='PASS_COMPLETE_S2_SCALAR_INPUTS',**fields,arrays=ap,upstream_acceptance=pair(up)))
    return pair(receipt)


def prepare(binding_pair):
    b = read(binding_pair, 'Calibration/probe assembly binding')
    exact_fields(b, ['status','protocol','sources','probe_scalars','calibration_response',
        'raw_acceptance','response_contract','output_root'], 'Calibration/probe assembly binding')
    require(b['status'] == PREPARE_STATUS, 'Wrong assembly role')
    ctx = source_context(b['sources'],b['protocol']);check_response_contract(ctx,b['response_contract'])
    exact_fields(b['probe_scalars'], ['calibration','test'], 'Both complete probe scalar splits')
    probes = {split:complete_probe(ctx,b['probe_scalars'][split],split) for split in ('calibration','test')}
    rosters = {split+'_'+role+'_ids':probes[split]['arrays'][role+'_ids'].tolist()
        for split in ('calibration','test') for role in ('recipient','donor')}
    raw = validate_raw_isolation(b['raw_acceptance'],ctx,{s:p['report'] for s,p in probes.items()},rosters)
    response = response_population(ctx,b['calibration_response'],'calibration',b['raw_acceptance'])
    require(response['parent_ids'] == rosters['calibration_recipient_ids'], 'Calibration response population mismatch')
    out = start_output(b['output_root'],dict(status='S2_CALIBRATION_AND_PROBE_ASSEMBLY_STARTED',
        input_binding=binding_pair, source_sha256=gate.sha(__file__), test_response_opened=False))
    decoded, checks = decode_response(ctx,response,probes['calibration']['arrays']['recipient_ids'],split='calibration',
        probe_acceptance=probes['calibration']['report']['probe_acceptance'])
    decoded_pair = save(out/'calibration_response_decoded.npz',decoded)
    response_evidence = dict(latent_population=b['calibration_response'],decoded_arrays=decoded_pair,
        measurement_sources=ctx['sources']['measurement_sources'],decoder_checks=checks)
    inputs = {s:scalar_receipt(out,ctx,s,b['probe_scalars'][s],probes[s],
        decoded['response'] if s == 'calibration' else None,response_evidence if s == 'calibration' else None)
        for s in ('calibration','test')}
    isolation = out/'isolation.json'
    write(isolation,dict(status='PASS_COMPLETE_S2_DATA_ISOLATION',protocol_sha256=b['protocol']['sha256'],
        sources_sha256=ctx['sources']['numerical_sources']['sha256'],expected_parent_ids=rosters,
        raw_acceptance=b['raw_acceptance'],content_lineage=raw['content_lineage'],
        probe_scalar_populations=b['probe_scalars'],assembly_sources=b['sources'],
        scope='Inherited full raw/content acceptance; assigned donor order revalidated by probe verifier. No pretraining-isolation claim.'))
    top = dict(status='S2_REGRESSION_INPUT_BINDINGS_ACCEPTED',protocol=b['protocol'],
        sources=ctx['sources']['numerical_sources'],calibration=inputs['calibration'],test_probe=inputs['test'],
        data_isolation=pair(isolation),response_contract=b['response_contract'],expected_parent_ids=rosters)
    path = out/'REGRESSION_INPUTS.json';write(path,top)
    # Exercise the real downstream consumer, without fitting or loading test Y.
    consumer = finalizer.load_context(pair(path))
    for name in ('calibration','test_probe'):finalizer.load_scalar(consumer,name)
    write(out/'report.json',dict(status='PASS_COMPLETE_S2_REGRESSION_INPUT_ASSEMBLY',input_binding=binding_pair,
        regression_inputs=pair(path),calibration_response=response_evidence,source_sha256=gate.sha(__file__),
        test_response_opened=False,regressors_fitted=False,scientific_protocol_frozen_by_this_module=False))
    (out/'DONE').write_text('Complete fixed calibration and held-out probe scalar inputs\n')
    return pair(out/'report.json')


def complete_test(binding_pair):
    b = read(binding_pair, 'Held-out scalar assembly binding')
    exact_fields(b, ['status','regression_inputs','orchestration_seal','sources','response_latent_population',
        'output_root'], 'Held-out scalar assembly binding')
    require(b['status'] == TEST_STATUS, 'Wrong held-out response assembly role')
    numerical = finalizer.load_context(b['regression_inputs'])
    inner = finalizer.verify_seal(numerical,b['orchestration_seal'])
    # No response receipt/latent payload is opened until this exact seal verifies.
    ctx = source_context(b['sources'],numerical['binding']['protocol'])
    contract_pair = numerical['binding']['response_contract'];check_response_contract(ctx,contract_pair)
    require(ctx['sources']['numerical_sources'] == numerical['binding']['sources'], 'Changed numerical source identity')
    upstream = read(numerical['receipts']['test_probe']['upstream_acceptance'], 'Sealed probe upstream')
    require(upstream['assembly_sources'] == b['sources'], 'Probe and response use different assembly sources')
    probe_pair = upstream['probe_scalar_population'];probe = complete_probe(ctx,probe_pair,'test')
    sealed_probe = finalizer.load_scalar(numerical,'test_probe')
    for key in sealed_probe:gate._equal(sealed_probe[key],probe['arrays'][key],'Sealed probe '+key)
    response = response_population(ctx,b['response_latent_population'],'test',probe['report']['raw_acceptance'],
                                   b['orchestration_seal'])
    report = response['report']
    for name,value in dict(prediction_lock=inner,orchestration_seal=b['orchestration_seal'],
            response_contract=contract_pair,expected_bindings=numerical['expected_bindings']).items():
        require(report.get(name) == value, 'Actual response access has different '+name)
    require(report.get('prediction_seal_verified_before_response_inference') is True,
            'Missing actual seal-first model-access evidence')
    out = start_output(b['output_root'],dict(status='S2_HELDOUT_SCALAR_ASSEMBLY_STARTED',input_binding=binding_pair,
        prediction_lock=inner,source_sha256=gate.sha(__file__)))
    decoded, checks = decode_response(ctx,response,probe['arrays']['recipient_ids'],split='test',
        probe_acceptance=probe['report']['probe_acceptance'])
    dp = save(out/'test_response_decoded.npz',decoded)
    arrays = save(out/'response.npz',dict(response=decoded['response'],
        recipient_ids=probe['arrays']['recipient_ids'],donor_ids=probe['arrays']['donor_ids']))
    receipt = out/'response_receipt.json'
    write(receipt,dict(status='ACCEPTED_S2_HELDOUT_RESPONSE',count=512,stratum_ids=list(features.STRATA),
        protocol_sha256=ctx['protocol_pair']['sha256'],response_contract_sha256=contract_pair['sha256'],
        prediction_lock_sha256=inner['sha256'],arrays=arrays,latent_population=b['response_latent_population'],
        decoded_arrays=dp,decoder_checks=checks,measurement_sources=ctx['sources']['measurement_sources'],
        assembly_sources=b['sources'],source_sha256=gate.sha(__file__)))
    access = out/'response_access.json'
    write(access,dict(status='PASS_COMPLETE_S2_TEST_RESPONSE_ACCESS',protocol_sha256=ctx['protocol_pair']['sha256'],
        response_contract_sha256=contract_pair['sha256'],prediction_lock_sha256=inner['sha256'],
        response_receipt_sha256=gate.sha(receipt),count=512,stratum_ids=list(features.STRATA),
        recipient_ids=probe['arrays']['recipient_ids'].tolist(),donor_ids=probe['arrays']['donor_ids'].tolist(),
        prediction_seal_verified_before_response_inference=True,latent_population=b['response_latent_population'],
        orchestration_seal=b['orchestration_seal'],model_access_evidence=report['models'],
        scope='Verified source-enforced response loader order and complete saved response evidence; no claim of external historical nonexposure.'))
    write(out/'report.json',dict(status='PASS_COMPLETE_S2_HELDOUT_SCALARS_AND_ACCESS',input_binding=binding_pair,
        response_receipt=pair(receipt),response_access=pair(access),prediction_lock=inner,
        source_sha256=gate.sha(__file__),regressors_refitted=False,models_loaded_by_this_module=False))
    (out/'DONE').write_text('Complete sealed held-out response scalars and access evidence\n')
    return pair(out/'report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','test-response'])
    parser.add_argument('--binding',required=True);parser.add_argument('--binding-sha256',required=True)
    args = parser.parse_args();binding = dict(path=args.binding,sha256=args.binding_sha256)
    print(json.dumps((prepare if args.stage == 'prepare' else complete_test)(binding),sort_keys=True))


if __name__ == '__main__':main()
