"""Draft whole-population q_g probe scoring; no execution authorization.

Only an authentic frozen protocol, measurement-source inventory and complete
accepted probe permit q_g deserialization. All saved probe admission checks are
repeated before decoding, without recomputing a simulator or world model. The
two existing v8 NumPy decoders evaluate the same old selected q_g independently.
No response model, response predictions, regression, selection or inference is
implemented here. A passing synthetic test is not an actual S2 result.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import accept_probe_population as gate
import features

ROOT = gate.ROOT
V8 = ROOT/'strengthening/presubmission_v8_20260922/independent_readout'
PRODUCTION_DECODER = V8/'scripts/readout.py'
REFERENCE_DECODER = V8.parent/'verification/readout_reference.py'
SELECTED_KEY = str((V8/'manifests/SELECTED_READOUTS.lock.json').relative_to(ROOT))
ENCODER_KEY = str((V8/'manifests/encoder_bindings.json').relative_to(ROOT))
INPUT_STATUS = 'S2_PROBE_SCORING_INPUTS_FROZEN'
STATUS = 'PASS_COMPLETE_S2_PROBE_SCALARS'
BATCH_SIZE = 512
# Existing v8 readout_reference.py fixture rule, fixed before S2 outputs.
DECODER_SETTINGS = dict(batch_size=BATCH_SIZE, atol=1e-9, rtol=1e-12,
    tolerance_origin='v8 readout_reference.py frozen-forward fixture comparison')
ARCHITECTURE = dict(name='gelu_residual_256_two_blocks_v1', input_dim=192,
    output_dim=6, width=256, blocks=2, residual_factor=0.5,
    gelu_approximation='none', skip_bias=False)
AXES = dict(groups=list(features.GROUPS), objectives=list(features.OBJECTIVES),
    branches=list(features.BRANCHES), horizons=list(features.HORIZONS), pose=list(features.POSE))


def file_pair(path):
    return dict(path=str(Path(path).resolve()), sha256=gate.sha(path))


def require(value, message):
    if not value:
        raise ValueError(message)


def _roster(rows, label):
    require(len(rows) == 3 and all(type(r['pool']) is int and type(r['group']) is int for r in rows)
            and {(r['pool'], r['group']) for r in rows} == {(p, 2*p) for p in range(3)},
            'Exactly three fixed pool/group rows required: '+label)
    return {r['pool']: r for r in rows}


def authenticate_measurement_sources(source_pair, protocol_pair):
    """Authenticate metadata and all head bytes, but do not deserialize q_g.

    Returned context is deliberately unauthorized for decoding until the whole
    probe is checked by authorize_measurement_context. This source inventory is
    separate from the finalizer's exact three-module numerical source lock.
    """
    protocol = gate._json(protocol_pair)
    gate._exact(protocol.get('status'), 'S2_SCIENTIFIC_PROTOCOL_FROZEN', 'S2 protocol freeze')
    sources = gate._json(source_pair)
    gate._exact(sources.get('status'), 'S2_PROBE_MEASUREMENT_SOURCES_FROZEN', 'measurement source freeze')
    gate._exact(sources.get('protocol_sha256'), protocol_pair['sha256'], 'measurement/protocol hash')
    gate._exact(sources.get('decoder'), DECODER_SETTINGS, 'fixed existing v8 decoding settings')
    inventory_pair = {k: protocol['assets_inventory'][k] for k in ('path', 'sha256')}
    gate._exact(sources['assets_inventory'], inventory_pair, 'protocol-bound asset inventory')
    inventory = gate._json(inventory_pair)
    selected = gate._json(sources['selected_readouts'])
    encoder = gate._json(sources['encoder_bindings'])
    old_protocol = gate._json(sources['readout_protocol'])
    gate._exact(sources['selected_readouts']['sha256'], inventory['source_bindings'][SELECTED_KEY], 'original selected q_g lock')
    gate._exact(sources['encoder_bindings']['sha256'], inventory['source_bindings'][ENCODER_KEY], 'original encoder lock')
    gate._exact(selected.get('status'), 'ALL_SIX_READOUT_CHECKPOINTS_FROZEN_BEFORE_QUALIFICATION', 'v8 selected lock status')
    gate._exact(sources['readout_protocol']['sha256'], selected['protocol_sha256'], 'selected architecture protocol')
    gate._exact(old_protocol.get('status'), 'DESIGN_FROZEN_BEFORE_ANY_G_EVAL_FITTING_OR_EFFECTS', 'old design status')
    gate._exact(old_protocol['architecture'], ARCHITECTURE, 'selected v8 architecture')
    require(len(selected['groups']) == 6 and {r['group'] for r in selected['groups']} == set(range(6)),
            'Complete historical selected lock required')
    require(len(encoder['groups']) == 6 and {r['group'] for r in encoder['groups']} == set(range(6)),
            'Complete historical encoder binding required')
    old_rows = {r['group']: r for r in selected['groups']}
    encoder_rows = {r['group']: r for r in encoder['groups']}
    heads = _roster(sources['heads'], 'measurement heads')
    assets = _roster(inventory['readouts'], 'fixed asset inventory')
    probe_sources = gate._json(sources['probe_sources'])
    gate._exact(probe_sources.get('status'), 'S2_PROBE_POPULATION_SOURCES_FROZEN', 'probe sources')
    gate._exact(probe_sources.get('protocol_sha256'), protocol_pair['sha256'], 'probe source protocol')
    for path, digest in probe_sources['files'].items():
        gate._exact(sources['files'].get(path), digest, 'inherited probe source closure')
    paths = set()
    for name, digest in sources['files'].items():
        path = Path(name); path = path if path.is_absolute() else ROOT/path
        gate._pair(dict(path=str(path), sha256=digest)); paths.add(path.resolve())
    required = [Path(__file__), Path(gate.__file__), Path(features.__file__),
                PRODUCTION_DECODER, REFERENCE_DECODER]
    require({p.resolve() for p in required} <= paths, 'Complete scorer/gate/feature/dual-decoder source closure required')
    for pool, h in heads.items():
        gate._exact(set(h), {'pool', 'group', 'file'}, 'q_g head fields')
        group = h['group']; asset = assets[pool]
        gate._pair(h['file'])  # bytes only; no NumPy archive open
        gate._exact(h['file']['sha256'], old_rows[group]['checkpoint_sha256'], 'old selected q_g identity')
        gate._exact(h['file']['sha256'], asset['measurement']['sha256'], 'fixed inventory q_g identity')
        gate._exact(asset['encoder'], encoder_rows[group], 'q_g original encoder provenance')
        gate._exact(encoder_rows[group]['architecture'], 'transformer_jepa', 'fixed Transformer pools')
        gate._exact(asset['normalization_not_recomputed'], True, 'own embedded normalizers')
    return dict(sources=sources, source_pair=source_pair, protocol_pair=protocol_pair,
        protocol=protocol, heads=heads, encoders=encoder_rows, architecture=old_protocol['architecture'],
        probe_gate_verified=False, loaded_heads={}, decoder_checks=[])


def authorize_measurement_context(context, probe_acceptance_pair):
    """Repeat only the saved-array gate and compare its complete actual receipt.

    Temporary verification receipts are cleaned. Original scientific artifacts
    and admissions remain untouched. No q_g weights or outputs are accessed.
    """
    receipt = gate._json(probe_acceptance_pair)
    split = receipt.get('split'); n = gate._size(split)
    expected = dict(status=gate.STATUS, count=n, complete_pools=3, complete_objective_strata=6,
        complete_families=3*n, complete_directions=12*n, axes=gate.ROLLOUT_AXES,
        protocol_sha256=context['protocol_pair']['sha256'],
        sources_sha256=context['sources']['probe_sources']['sha256'], source_sha256=gate.sha(gate.__file__),
        q_g_weights_deserialized=False, q_g_outputs_computed=False, response_model_outputs_opened=False,
        outcome_statistics_computed=False, rows_removed=0, q_g_scoring_gate_passed=True)
    for key, value in expected.items():
        gate._exact(receipt.get(key), value, 'complete probe admission/'+key)
    _roster(receipt['pools'], 'accepted probe pools')
    binding = gate._json(receipt['population_binding'])
    gate._exact(binding['protocol'], context['protocol_pair'], 'probe/scorer protocol pair')
    gate._exact(binding['sources'], context['sources']['probe_sources'], 'probe/scorer source pair')
    with tempfile.TemporaryDirectory(prefix='s2_probe_admission_recheck_') as temp:
        repeated = gate.accept_population(receipt['population_binding']['path'],
            receipt['population_binding']['sha256'], Path(temp)/'receipt.json')
    gate._exact(repeated, receipt, 'complete saved probe revalidation receipt')
    gate._pair(probe_acceptance_pair)
    context.update(probe_gate_verified=True, probe_acceptance=probe_acceptance_pair,
                   admission=receipt, split=split, count=n)
    return context


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _normalizer_digest(head):
    digest = hashlib.sha256()
    for key, size in (('mean',192),('scale',192),('target_mean',6),('target_scale',6)):
        value = np.asarray(head[key], np.float64)
        require(value.shape == (size,) and np.isfinite(value).all(), 'Invalid own q_g '+key)
        if key.endswith('scale'):
            require(np.all(value > 0), 'Nonpositive own q_g '+key)
        digest.update(key.encode()); digest.update(value.astype('<f8').tobytes())
    return digest.hexdigest()


def decode_pool_tokens(authenticated_context, pool, tokens):
    """Return {poses: FP64 physical poses, verification: JSON metadata}.

    Uses the same fixed batch512 for both v8 algebra implementations. No adaptive
    chunk retry or relaxed tolerance exists. Caller determines the task's array
    ordering; probe free and OH calls use exactly the same [2,N,5,192] ordering.
    """
    ctx = authenticated_context
    require(ctx.get('probe_gate_verified') is True, 'Complete saved probe gate required before q_g deserialization')
    require(type(pool) is int and pool in (0,1,2), 'Fixed pool required')
    z = np.asarray(tokens)
    require(z.ndim >= 2 and z.shape[-1] == 192 and z.size > 0 and z.dtype == np.float32
            and np.isfinite(z).all(), 'Nonempty finite FP32 latent tokens required')
    head_pair = ctx['heads'][pool]['file']; gate._pair(head_pair)
    if pool not in ctx['loaded_heads']:
        head = gate._arrays(head_pair)
        ctx['loaded_heads'][pool] = dict(arrays=head, normalizer_sha256=_normalizer_digest(head))
    head = ctx['loaded_heads'][pool]['arrays']
    for path in (PRODUCTION_DECODER, REFERENCE_DECODER):
        entries = ctx['sources']['files']
        digest = entries.get(str(path),entries.get(str(path.relative_to(ROOT))))
        gate._pair(dict(path=str(path),sha256=digest))
    production = _load_module(PRODUCTION_DECODER, '_s2_old_qg_production')
    reference = _load_module(REFERENCE_DECODER, '_s2_old_qg_independent')
    decoded = production.numpy_forward(z, head, ctx['architecture'], batch_size=BATCH_SIZE)
    independent = reference.decode(z, head, batch_size=BATCH_SIZE)
    shape = (*z.shape[:-1],6)
    gate._array(decoded, shape, np.dtype('float64'), 'production physical q_g poses')
    gate._array(independent, shape, np.dtype('float64'), 'independent physical q_g poses')
    np.testing.assert_allclose(decoded, independent, atol=DECODER_SETTINGS['atol'], rtol=DECODER_SETTINGS['rtol'])
    gate._pair(head_pair)
    verification=dict(pool=pool, token_count=z.size//192, output_count=decoded.size,
        maximum_absolute_difference=float(np.max(np.abs(decoded-independent))),
        maximum_tolerance_ratio=float(np.max(np.abs(decoded-independent)/
            (DECODER_SETTINGS['atol']+DECODER_SETTINGS['rtol']*np.abs(independent)))),
        comparison_passed=True,
        head=head_pair,normalizer_sha256=ctx['loaded_heads'][pool]['normalizer_sha256'],
        decoder=DECODER_SETTINGS)
    ctx['decoder_checks'].append(verification)
    return dict(poses=decoded,verification=verification)


def _probe_inputs(context):
    """Assemble authenticated saved tokens/truth; no head deserialization."""
    ctx = context; n = ctx['count']; receipt = ctx['admission']
    rows = _roster(receipt['pools'], 'admitted pool roster')
    assignment = gate._arrays(receipt['donor_assignment'])
    gate._exact(set(assignment), {'donor_assignment','recipient_ids','donor_ids'}, 'assignment fields')
    gate.validate_assignment(assignment['donor_assignment'],split=ctx['split'],
        seed=ctx['protocol']['probe_population']['donor_assignment_seeds'][ctx['split']])
    for key, expected in (('recipient_ids',receipt['parent_ids']),('donor_ids',receipt['donor_ids'])):
        gate._array(assignment[key], (n,), np.dtype('int64'), key)
        gate._equal(assignment[key], np.asarray(expected,np.int64), 'accepted bank-order '+key)
    tokens=[]; histories=[]; truths=[]; norms=[]
    for pool in range(3):
        row = rows[pool]; cache_report = gate._json(row['recipient_cache'])
        rolled = gate._json(row['rollout'])
        gate._exact(cache_report['arrays'], row['truth_array_source'], 'accepted raw truth source')
        cache = gate._arrays(cache_report['arrays']); arrays = gate._arrays(rolled['arrays'])
        z = gate._array(arrays['tokens'],(2,n,4,5,192),np.dtype('float32'),'all probe tokens')
        oh = gate._array(arrays['observed_history'],(2,n,5,192),np.dtype('float32'),'all OH predictions')
        truth = gate._array(cache['truth'],(n,5,6),np.dtype('float64'),'accepted raw physical truth')
        gate._equal(arrays['seeds'],assignment['recipient_ids'],'rollout scalar row order')
        gate._equal(cache['seeds'],assignment['recipient_ids'],'truth scalar row order')
        gate._equal(oh[:,:,0],z[:,:,0,0],'exact OH5/free5 latent identity')
        gate._equal(oh,cache['observed_history'],'accepted predicted OH tokens')
        gate._equal(z[:,:,0],cache['free'],'accepted free tokens')
        gate._equal(z[:,:,3],cache['reset'],'accepted reset tokens')
        for model in cache_report['models']:
            spec = model['spec']
            gate._exact(spec['original']['sha256'],ctx['encoders'][2*pool]['checkpoint_sha256'],
                        'measurement/cache original encoder identity')
        norm = np.asarray(row['effective_common_norm'],np.float64)
        gate._array(norm,(n,),np.dtype('float64'),'accepted post-backoff common norm')
        require(np.all(norm >= 0),'Negative family norm')
        tokens.append(z);histories.append(oh);truths.append(truth);norms.append(norm)
    return dict(tokens=tokens,histories=histories,truth=np.stack(truths,axis=1),
        common_norm=np.stack(norms,axis=1),recipient_ids=assignment['recipient_ids'],
        donor_ids=assignment['donor_ids'][assignment['donor_assignment']])


def load_context(binding_path, binding_sha256, *, require_unused_output=True):
    pair=dict(path=str(Path(binding_path).resolve()),sha256=binding_sha256)
    binding=gate._json(pair)
    gate._exact(set(binding),{'status','split','protocol','sources','probe_acceptance','output_root'},'scoring input fields')
    gate._exact(binding['status'],INPUT_STATUS,'probe scoring input status')
    n=gate._size(binding['split']);out=Path(binding['output_root'])
    require(out.is_absolute(),'Absolute bound scoring output root required')
    if require_unused_output:
        require(not out.exists() and not out.with_name(out.name+'.intent.json').exists(),
                'Existing output/intent retained; no overwrite or implicit retry')
    ctx=authenticate_measurement_sources(binding['sources'],binding['protocol'])
    authorize_measurement_context(ctx,binding['probe_acceptance'])
    gate._exact(ctx['split'],binding['split'],'scorer/probe split')
    gate._exact(ctx['count'],n,'whole split count')
    inputs=_probe_inputs(ctx)
    ctx.update(binding_pair=pair,binding=binding,output=out,inputs=inputs)
    return ctx


def _save(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    return file_pair(path)


def _extract(audit, split):
    gate._exact(set(audit),{'prediction','observed_history_prediction','truth','common_norm',
        'block_errors','observed_history_block_errors'},'complete decoded/error audit keys')
    computed=features.probe_features(audit['prediction'],audit['observed_history_prediction'],
        audit['truth'],audit['common_norm'],split=split,axes=AXES)
    for key in ('block_errors','observed_history_block_errors'):
        gate._equal(audit[key],computed[key],'saved/recomputed '+key)
    return computed


def produce(context):
    ctx=context;out=ctx['output'];n=ctx['count'];split=ctx['split']
    require(not out.exists(),'Output already exists')
    gate._write(out.with_name(out.name+'.intent.json'),dict(status='S2_PROBE_SCORING_STARTED',
        input_binding=ctx['binding_pair'],probe_acceptance=ctx['probe_acceptance'],source_sha256=gate.sha(__file__)))
    out.mkdir()
    try:
        require(ctx.get('probe_gate_verified') is True,'Whole probe gate not verified')
        inputs=ctx['inputs'];prediction=np.empty((n,3,2,4,5,6),np.float64)
        history=np.empty((n,3,2,5,6),np.float64)
        for pool in range(3):
            # Identical complete row/chunk layout for each branch and OH.
            for branch in range(4):
                decoded=decode_pool_tokens(ctx,pool,np.ascontiguousarray(inputs['tokens'][pool][:,:,branch]))['poses']
                ctx['decoder_checks'][-1]['probe_branch']=features.BRANCHES[branch]
                prediction[:,pool,:,branch]=decoded.transpose(1,0,2,3)
            oh=decode_pool_tokens(ctx,pool,np.ascontiguousarray(inputs['histories'][pool]))['poses']
            ctx['decoder_checks'][-1]['probe_branch']='observed_history'
            history[:,pool]=oh.transpose(1,0,2,3)
            gate._equal(history[:,pool,:,0],prediction[:,pool,:,0,0],'exact decoded OH5/free5')
        computed=features.probe_features(prediction,history,inputs['truth'],inputs['common_norm'],split=split,axes=AXES)
        scalar=dict(ordinary=computed['ordinary'],signed_g=computed['signed_g'],
            recipient_ids=inputs['recipient_ids'],donor_ids=inputs['donor_ids'])
        audit=dict(prediction=prediction,observed_history_prediction=history,truth=inputs['truth'],
            common_norm=inputs['common_norm'],block_errors=computed['block_errors'],
            observed_history_block_errors=computed['observed_history_block_errors'])
        arrays=_save(out/'scalars.npz',scalar); audit_pair=_save(out/'decoded_audit.npz',audit)
        heads=[dict(pool=p,group=2*p,file=ctx['heads'][p]['file'],
            normalizer_sha256=ctx['loaded_heads'][p]['normalizer_sha256']) for p in range(3)]
        report=dict(status=STATUS,split=split,count=n,complete_pools=3,complete_objective_strata=6,
            complete_branches=4,complete_horizons=5,axes=AXES,ordinary_columns=list(features.ORDINARY),
            stratum_ids=list(features.STRATA),protocol=ctx['protocol_pair'],protocol_sha256=ctx['protocol_pair']['sha256'],
            sources=ctx['source_pair'],sources_sha256=ctx['source_pair']['sha256'],probe_acceptance=ctx['probe_acceptance'],
            raw_acceptance=ctx['admission']['raw_population_acceptance'],
            input_binding=ctx['binding_pair'],arrays=arrays,decoded_audit=audit_pair,
            recipient_ids=inputs['recipient_ids'].tolist(),donor_ids=inputs['donor_ids'].tolist(),
            donor_order='bank_ids[donor_assignment], applied exactly once',heads=heads,
            decoder=DECODER_SETTINGS,decoder_checks=ctx['decoder_checks'],source_sha256=gate.sha(__file__),
            q_g_weights_deserialized=True,q_g_outputs_computed=True,complete_probe_gate_before_q_g=True,
            response_models_deserialized=False,response_outputs_computed=False,regression_fitted=False,
            inferential_statistics_computed=False,rows_removed=0,
            scope='Complete saved probe gate revalidation; dual v8 FP64 q_g forward comparison; physical block-error scalars. '
                  'No independent simulator, encoder, native dynamics or QP optimality recomputation; no response or regression.')
        gate._pair(ctx['binding_pair']);gate._pair(ctx['probe_acceptance']);gate._pair(ctx['source_pair'])
        gate._write(out/'report.json',report)
        with (out/'DONE').open('x') as stream:stream.write(STATUS+'\n')
        return report
    except Exception as exc:
        gate._write(out/'failure.json',dict(status='BLOCKED_S2_PROBE_SCORING',input_binding=ctx['binding_pair'],
            exception_type=type(exc).__name__,message=str(exc),rows_removed=0))
        raise


def verify_completed_probe(report_pair, *, protocol_pair, measurement_sources_pair, expected_split):
    """Recheck saved provenance and physical scalar arithmetic, without decoding.

    Returns {report, arrays}. This authenticates the producer's full dual-forward
    receipt; it does not independently rerun q_g or native dynamics. Saved raw
    truth/norms/ID order must match the complete accepted probe exactly.
    """
    report=gate._json(report_pair);n=gate._size(expected_split)
    expected=dict(status=STATUS,split=expected_split,count=n,complete_pools=3,complete_objective_strata=6,
        complete_branches=4,complete_horizons=5,axes=AXES,ordinary_columns=list(features.ORDINARY),
        stratum_ids=list(features.STRATA),protocol=protocol_pair,protocol_sha256=protocol_pair['sha256'],
        sources=measurement_sources_pair,sources_sha256=measurement_sources_pair['sha256'],
        source_sha256=gate.sha(__file__),decoder=DECODER_SETTINGS,
        donor_order='bank_ids[donor_assignment], applied exactly once',q_g_weights_deserialized=True,
        q_g_outputs_computed=True,complete_probe_gate_before_q_g=True,response_models_deserialized=False,
        response_outputs_computed=False,regression_fitted=False,inferential_statistics_computed=False,rows_removed=0)
    for key,value in expected.items():gate._exact(report.get(key),value,'completed scalar report/'+key)
    ctx=load_context(report['input_binding']['path'],report['input_binding']['sha256'],require_unused_output=False)
    gate._exact(ctx['protocol_pair'],protocol_pair,'completed/provided protocol')
    gate._exact(ctx['source_pair'],measurement_sources_pair,'completed/provided measurement sources')
    gate._exact(ctx['split'],expected_split,'completed/provided split')
    gate._exact(report['probe_acceptance'],ctx['probe_acceptance'],'complete admission pair')
    gate._exact(report['raw_acceptance'],ctx['admission']['raw_population_acceptance'],'complete raw admission pair')
    gate._exact(Path(report_pair['path']).resolve(),(ctx['output']/'report.json').resolve(),'bound report output')
    for key, filename in (('arrays','scalars.npz'),('decoded_audit','decoded_audit.npz')):
        gate._exact(Path(report[key]['path']).resolve(),(ctx['output']/filename).resolve(),'bound '+key+' output')
    heads=_roster(report['heads'],'scored q_g roster')
    for pool,h in heads.items():
        gate._exact(h['file'],ctx['heads'][pool]['file'],'scored selected q_g identity')
        # The complete gate has already passed; authenticate actual embedded
        # normalizers here without recomputing any q_g prediction.
        gate._exact(h.get('normalizer_sha256'),_normalizer_digest(gate._arrays(h['file'])),
                    'actual selected q_g embedded normalizer receipt')
    checks=report['decoder_checks']
    require(len(checks)==15,'All three pools times five complete dual-forward calls required')
    for i,check in enumerate(checks):
        gate._exact(check.get('pool'),i//5,'dual-forward pool order')
        gate._exact(check.get('token_count'),10*n,'complete dual-forward token count')
        gate._exact(check.get('output_count'),60*n,'complete dual-forward pose count')
        gate._exact(check.get('head'),heads[i//5]['file'],'dual-forward head identity')
        gate._exact(check.get('normalizer_sha256'),heads[i//5]['normalizer_sha256'],'dual-forward own normalizers')
        gate._exact(check.get('decoder'),DECODER_SETTINGS,'dual-forward fixed settings')
        gate._exact(check.get('probe_branch'),(*features.BRANCHES,'observed_history')[i%5],
                    'dual-forward branch call order')
        gate._exact(check.get('comparison_passed'),True,'dual-forward comparison pass')
        ratio=check.get('maximum_tolerance_ratio')
        require(isinstance(ratio,(float,int)) and not isinstance(ratio,bool) and np.isfinite(ratio)
                and 0 <= ratio <= 1,'Dual-forward discrepancy exceeds original v8 tolerance')
        require(np.isfinite(check['maximum_absolute_difference']) and check['maximum_absolute_difference']>=0,
                'Finite dual-forward comparison receipt required')
    scalar=gate._arrays(report['arrays']);audit=gate._arrays(report['decoded_audit'])
    gate._exact(set(scalar),{'ordinary','signed_g','recipient_ids','donor_ids'},'scalar archive fields')
    computed=_extract(audit,expected_split)
    for key in ('ordinary','signed_g'):
        gate._array(scalar[key],computed[key].shape,np.dtype('float64'),key)
        gate._equal(scalar[key],computed[key],'recomputed '+key)
    for key in ('recipient_ids','donor_ids'):
        gate._array(scalar[key],(n,),np.dtype('int64'),key)
        gate._equal(scalar[key],ctx['inputs'][key],'accepted assigned '+key)
        gate._exact(report[key],scalar[key].tolist(),'report/scalar '+key)
    for key in ('truth','common_norm'):
        gate._equal(audit[key],ctx['inputs'][key],'original raw '+key)
    gate._pair(report_pair)
    return dict(report=report,arrays=scalar)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',required=True);parser.add_argument('--binding-sha256',required=True)
    args=parser.parse_args();produce(load_context(args.binding,args.binding_sha256))


if __name__=='__main__':main()
