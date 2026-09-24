"""Draft complete S2 corrected T0 rollout producer; no execution authority.

Consume an authentic complete projection population and independently accept
all four-ray families before any model load. Only the two fixed T0 probe models
per pool are permitted. Native300 arithmetic remains in model_runtime. No q_g,
response models, diagnostic errors, sampling, or scientific freeze occurs here.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import project_population as project
import cache_population as cache

gate, four, runtime = project.gate, project.four, cache.runtime
INPUT_STATUS = 'S2_ROLLOUT_INPUT_BINDING_FROZEN'
POOL_STATUS = 'PASS_COMPLETE_S2_T0_CORRECTED_PROBE_ROLLOUT'
POPULATION_STATUS = 'PASS_COMPLETE_S2_CORRECTED_ROLLOUT_POPULATION'
ADMISSION_STATUS = 'ALL_COMPLETE_CACHES_AND_PROJECTED_FAMILIES_ACCEPTED_BEFORE_MODEL_LOAD'
require, file_pair = project.require, project.file_pair


def validate_projected_pool(data, projected, *, count, assignment):
    """Independent dense family checks, never call the projecting QP solver."""
    arrays = gate._arrays(projected['arrays'])
    gate._exact(set(arrays), {'replacements', 'directions'}, 'complete projected array roster')
    gate._array(arrays['replacements'], (count, 2, 2, 192), np.dtype('float32'), 'projected tokens')
    gate._array(arrays['directions'], (count, 2, 2, 192), np.dtype('float64'), 'native projected rays')
    families = projected['families']
    require(len(families) == count and [r['goal_index'] for r in families] == list(range(count)),
            'Complete ordered four-member family reports required')
    accepted = []
    for goal in range(count):
        predicted = {objective: data['predicted'][i, goal] for i, objective in enumerate(four.OBJECTIVES)}
        guides = dict(actual=data['actual'][goal], donor=data['donor'][assignment[goal]])
        accepted.append(four.accept_t0_four_family(predicted, guides, data['head_A'],
            arrays['replacements'][goal], arrays['directions'][goal], families[goal],
            goal_index=goal, donor_assignment=assignment))
    return arrays, accepted


def load_context(binding_path, binding_sha256):
    """Authenticate actual prior artifacts without inventing downstream reports."""
    pair = dict(path=str(Path(binding_path).resolve()), sha256=binding_sha256)
    binding = gate._json(pair)
    gate._exact(set(binding), {'status', 'projection_population', 'output_root'}, 'rollout input fields')
    gate._exact(binding['status'], INPUT_STATUS, 'rollout input status')
    out = Path(binding['output_root'])
    require(out.is_absolute(), 'Actual absolute rollout output root required')
    require(not out.exists() and not out.with_name(out.name + '.intent.json').exists(),
            'Existing output or intent retained; no retry or overwrite')
    population = gate._json(binding['projection_population'])
    gate._exact(population.get('status'), project.POPULATION_STATUS, 'complete projection population')
    upstream = population['input_binding']
    # This is the authentic previous projection binding. The read-only mode skips
    # only its already-used destination check, not input/source/raw authentication.
    ctx = project.load_context(upstream['path'], upstream['sha256'], require_unused_output=False, retain_caches=True)
    n, split, sources = ctx['count'], ctx['split'], ctx['sources']
    gate._exact(sources['producers']['rollout'], file_pair(__file__), 'actual corrected rollout wrapper')
    expected = dict(input_binding=upstream, protocol_sha256=ctx['input']['protocol']['sha256'],
        probe_sources_sha256=ctx['input']['sources']['sha256'], source_sha256=gate.sha(project.__file__),
        split=split, count=n, pool_count=3, attempted_families=3*n, completed_families=3*n,
        failed_families=0, rows_removed=0, q_g_weights_deserialized=False,
        q_g_outputs_computed=False, response_models_deserialized=False)
    for key, value in expected.items():
        gate._exact(population.get(key), value, 'complete projected population/' + key)
    gate._exact(binding['projection_population']['path'], str(ctx['output'] / 'report.json'),
                'canonical authentic projection population report')
    reports = population['projections']
    require(len(reports) == 3 and [r['pool'] for r in reports] == [0, 1, 2],
            'All three complete projected pools in fixed order required')
    total_zero = 0
    for data, row in zip(ctx['pools'], reports, strict=True):
        pool = data['pool']
        gate._exact(row['status'], project.POOL_STATUS, 'projected pool status')
        gate._exact(row['report']['path'], str(ctx['output'] / f'pool_{pool}' / 'report.json'),
                    'canonical authentic projected pool report')
        projected = gate._json(row['report'])
        common = dict(status=project.POOL_STATUS, split=split, pool=pool, group=2*pool,
            count=n, model_role='T0', protocol_sha256=ctx['input']['protocol']['sha256'],
            probe_sources_sha256=ctx['input']['sources']['sha256'], source_sha256=gate.sha(project.__file__),
            recipient_cache=data['recipient_cache'], donor_cache=data['donor_cache'],
            donor_assignment=ctx['input']['donor_assignment'], head_A=data['head_A_file'],
            axes=gate.PROJECTION_AXES, input_binding=upstream, attempted_families=n,
            completed_families=n, failed_families=0, rows_removed=0,
            q_g_weights_deserialized=False, q_g_outputs_computed=False,
            response_models_deserialized=False, later_horizon_values_used_for_projection=False)
        for key, value in common.items():
            gate._exact(projected.get(key), value, 'projected pool/' + key)
        arrays, accepted = validate_projected_pool(data, projected, count=n, assignment=ctx['assignment'])
        zero = sum(r['effective_norm'] == 0 for r in accepted)
        gate._exact(projected['zero_families'], zero, 'all retained zero families')
        total_zero += zero
        data.update(projection=row['report'], projected_report=projected, projected_arrays=arrays,
                    projection_acceptance=accepted)
    gate._exact(population['zero_families'], total_zero, 'complete retained zero population')
    gate._pair(pair)
    ctx.update(rollout_binding=pair, rollout_input=binding, projection_population=binding['projection_population'],
               projection_output=ctx['output'], output=out, admission_status=ADMISSION_STATUS)
    return ctx


def validate_native_result(result, *, model, cached, objective_index, goal, replacements, zero):
    """Admit actual runtime checks before materializing equal-valued evidence."""
    arrays = result['arrays']
    gate._exact(set(arrays), {'free', 'observed', 'observed_history', 'reset', 'actual', 'donor'},
                'complete native corrected branch roster')
    for key, value in arrays.items():
        gate._array(value, (5, 192), np.dtype('float32'), 'native/' + key)
    for key in ('native_endpoint_exact', 'identity_exact', 'observed_history_first_exact',
                'model_unchanged', 'actions_and_normalization_unchanged'):
        require(result['checks'].get(key) is True, 'Runtime exact-parity/frozen-state assertion missing: ' + key)
    gate._exact(result['model_binding_sha256'], model['spec_sha256'], 'actual fixed T0 model')
    gate._exact(result['access_receipt'], model['access_receipt'], 'actual admitted probe role')
    for key in ('free', 'observed_history', 'reset'):
        gate._equal(arrays[key], cached[key][objective_index, goal], 'rerun fixed cached ' + key)
    gate._equal(arrays['observed'], cached['observed'][goal], 'shared original observed encoding')
    gate._equal(arrays['observed_history'][0], arrays['free'][0], 'first observed-history prediction')
    gate._equal(arrays['reset'][0], arrays['observed'][0], 'first reset insertion')
    for source in four.SOURCES:
        gate._equal(arrays[source][0], replacements[source], 'actual inserted ' + source + ' token')
        if zero:
            gate._equal(arrays[source], arrays['free'], 'retained zero-dose full reroll')
    # Runtime really computes identity reroll and native300 endpoint, and asserts
    # exact equality before returning. These saved equal-valued copies are NOT a
    # second implementation or another independent forward pass.
    return dict(tokens=np.stack([arrays[b] for b in gate.BRANCHES]),
        identity=arrays['free'].copy(), observed_history=arrays['observed_history'].copy(),
        native_endpoint=arrays['free'][-1].copy(),
        inserted=np.stack([replacements[source] for source in four.SOURCES]))


def _save_arrays(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())


def produce(ctx):
    """Fixed full split; failures retain partial artifacts and cannot pass gate."""
    require(ctx.get('admission_status') == ADMISSION_STATUS,
            'Every cache and four-family projection must be accepted before model loading')
    n, split, out = ctx['count'], ctx['split'], Path(ctx['output'])
    require(n == gate._size(split) and [p['pool'] for p in ctx['pools']] == [0, 1, 2],
            'Complete fixed split with three ordered pools required')
    require(not out.exists(), 'Existing rollout/partial output retained; no resume or overwrite')
    intent = out.with_name(out.name + '.intent.json')
    require(not intent.exists(), 'Existing rollout intent blocks retry')
    gate._pair(ctx['rollout_binding']); gate._pair(ctx['projection_population'])
    gate._write(intent, dict(status='S2_COMPLETE_CORRECTED_ROLLOUT_ATTEMPT_STARTED',
        input_binding=ctx['rollout_binding'], projection_population=ctx['projection_population'],
        split=split, count=n, pool_count=3, source_sha256=gate.sha(__file__)))
    out.mkdir(parents=True, exist_ok=False)
    reports = []; completed = 0; started = time.monotonic(); location = {}
    try:
        for data in ctx['pools']:
            pool = data['pool']; folder = out / f'pool_{pool}'; folder.mkdir(exist_ok=False)
            recipient = data['recipient_arrays']; report = data['recipient_report']
            require(len(report['cases']) == n and [c['seed'] for c in report['cases']] == ctx['recipient_ids'].tolist(),
                    'Complete admitted recipient case order required')
            arrays = dict(tokens=np.empty((2,n,4,5,192),np.float32), identity=np.empty((2,n,5,192),np.float32),
                observed_history=np.empty((2,n,5,192),np.float32), native_endpoint=np.empty((2,n,192),np.float32),
                inserted=np.empty((2,n,2,192),np.float32), seeds=ctx['recipient_ids'].copy())
            runtime_pair = report['runtime_contract']; models = []
            info = dict(pool=pool,group=2*pool,parent_ids=ctx['recipient_ids'].tolist(),runtime_role='probe_'+split)
            with (folder / 'cases.jsonl').open('x') as journal:
                for oi, objective in enumerate(four.OBJECTIVES):
                    location = dict(pool=pool, objective=objective, stage='fixed T0 model loading')
                    handle = runtime.load_probe_model(pool, objective, split=split,
                        runtime_contract=runtime_pair['path'],runtime_contract_sha256=runtime_pair['sha256'])
                    model = cache.model_evidence(handle, info, objective)
                    gate._exact(model, report['models'][oi], 'same complete cached T0 model, admission and normalizer')
                    models.append(model)
                    partial = folder / objective; partial.mkdir(exist_ok=False)
                    for goal, evidence in enumerate(report['cases']):
                        location = dict(pool=pool, objective=objective, goal_index=goal, stage='native corrected rollout')
                        inputs = cache.case_inputs(evidence)
                        for key in ('prefix_actions','selected_actions'):
                            gate._equal(inputs[key], recipient[key][goal], 'unchanged raw/cache ' + key)
                        gate._exact(inputs['selected_index'], int(recipient['selected_index'][goal]), 'original selected action member')
                        replacement = {source:data['projected_arrays']['replacements'][goal,oi,si].copy()
                                       for si,source in enumerate(four.SOURCES)}
                        result = runtime.native_rollouts(handle,parent_id=evidence['seed'],replacements=replacement,
                            **{k:inputs[k] for k in ('history_pixels','goal_pixels','prefix_actions',
                                                  'population_actions','selected_index','observed_pixels')})
                        saved = validate_native_result(result,model=model,cached=recipient,objective_index=oi,
                            goal=goal,replacements=replacement,zero=data['projection_acceptance'][goal]['effective_norm']==0)
                        for key,value in saved.items(): arrays[key][oi,goal] = value
                        # Persist each successful case before proceeding. An interrupted
                        # population retains these arrays but no accepted aggregate.
                        partial_path = partial / f'case_{goal:03d}.npz'
                        _save_arrays(partial_path, saved)
                        project._append(journal,dict(status='PASS_S2_NATIVE_CORRECTED_CASE',
                            goal_index=goal,parent_id=evidence['seed'],objective=objective,
                            donor_index=int(ctx['assignment'][goal]), arrays=file_pair(partial_path),
                            checks=result['checks'],model_binding_sha256=result['model_binding_sha256'],
                            identity_native_endpoint_saved_from_checked_free=True))
                        completed += 1
                    del handle; gc.collect()
            # Saved-consistency gate rechecks complete native/cache/raw-derived
            # relationships and every independently accepted projection family.
            accepted = gate.validate_pool(recipient,data['donor_arrays'],data['projected_arrays'],
                data['projected_report']['families'],arrays,data['head_A'],ctx['assignment'],split=split)
            _save_arrays(folder/'arrays.npz',arrays)
            common = dict(split=split,pool=pool,group=2*pool,count=n,model_role='T0',
                protocol_sha256=ctx['input']['protocol']['sha256'],probe_sources_sha256=ctx['input']['sources']['sha256'],
                source_sha256=gate.sha(__file__),recipient_cache=data['recipient_cache'],donor_cache=data['donor_cache'],
                donor_assignment=ctx['input']['donor_assignment'],head_A=data['head_A_file'],projection=data['projection'],
                axes=gate.ROLLOUT_AXES,models=models,native_population=300,history_tokens=3,prefix_steps=10)
            rolled = dict(status=POOL_STATUS,**common,arrays=file_pair(folder/'arrays.npz'),
                array_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
                input_binding=ctx['rollout_binding'],projection_population=ctx['projection_population'],
                input_admission=report['input_admission'],input_manifest=report['input_manifest'],
                runtime_contract=runtime_pair,cases=report['cases'],journal=file_pair(folder/'cases.jsonl'),
                independent_projection_and_saved_consistency=accepted,whole_population_retained=True,rows_removed=0,
                identity_native_endpoint_saved_from_checked_free=True,
                identity_endpoint_provenance='model_runtime computes identity reroll and native300 endpoint and asserts exact '
                    'equality internally; saved identity and native_endpoint are materialized from the checked free values. '
                    'They are not separately returned arrays or independent rerun evidence.',
                frozen_model_state_unchanged=True,action_normalization_unchanged=True,
                q_g_weights_deserialized=False,q_g_outputs_computed=False,response_models_deserialized=False,
                donor_models_loaded=False,diagnostic_errors_computed=False,
                scope='All fixed T0 corrected native300 trajectories and all rows including zero dose. '
                    'Independent saved-family and consistency checks; actual native model inference is reused, '
                    'not independently reimplemented. Future whole-split acceptance remains required before q_g scoring.')
            gate._write(folder/'report.json',rolled)
            with (folder/'DONE').open('x') as stream: stream.write('complete corrected probe pool\n')
            reports.append(dict(pool=pool,report=file_pair(folder/'report.json'),status=POOL_STATUS))
        gate._pair(ctx['rollout_binding']); gate._pair(ctx['projection_population'])
        result = dict(status=POPULATION_STATUS,input_binding=ctx['rollout_binding'],
            projection_population=ctx['projection_population'],protocol_sha256=ctx['input']['protocol']['sha256'],
            probe_sources_sha256=ctx['input']['sources']['sha256'],source_sha256=gate.sha(__file__),
            split=split,count=n,pool_count=3,objective_strata=6,completed_cases=completed,rollouts=reports,
            rows_removed=0,whole_population_retained=True,q_g_weights_deserialized=False,
            q_g_outputs_computed=False,response_models_deserialized=False,diagnostic_errors_computed=False,
            elapsed_seconds=time.monotonic()-started,
            scope='Complete T0 rollout production only. Not whole-split scoring acceptance or response-model access.')
        require(completed == 6*n, 'Incomplete predetermined rollout population')
        gate._write(out/'report.json',result)
        with (out/'DONE').open('x') as stream: stream.write('all three complete corrected probe pools\n')
        return result
    except Exception as exc:
        gate._write(out/'failure.json',dict(status='BLOCKED_S2_CORRECTED_ROLLOUT_POPULATION',
            input_binding=ctx['rollout_binding'],location=location,completed_cases=completed,
            error_type=type(exc).__name__,error=str(exc),completed_pools=reports,rows_removed=0,
            scope='Partial case arrays and receipts retained. No substitute rows, resumed outputs, '
                  'accepted whole population, or scoring authorization.'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True); parser.add_argument('--binding-sha256', required=True)
    args = parser.parse_args()
    result = produce(load_context(args.binding, args.binding_sha256))
    print(json.dumps({k:result[k] for k in ('status','split','count','completed_cases')}))


if __name__ == '__main__': main()
