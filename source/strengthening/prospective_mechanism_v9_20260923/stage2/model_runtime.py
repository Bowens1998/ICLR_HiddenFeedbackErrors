"""DRAFT fixed-C model/runtime adapters with separate probe/response admission.

No CLI or scientific freeze. The future runtime/input/response contracts below
are target schemas, not claims that accepted S2 inputs already exist. Importing
this module reads no files and imports no Torch or model/readout implementation.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
BASE = Path('/external-assets/cluster/projects/jepa-regime-study-maintrack-20260908')
OBJECTIVES = ('decoded_teacher', 'physical_labels')
RUNTIME_STATUS = 'S2_MODEL_RUNTIME_CONTRACT_FROZEN'
INPUT_STATUS = 'S2_FIXED_CONTEXTS_ACCEPTED'
RESPONSE_STATUS = 'S2_TEST_RESPONSE_CONTRACT_FROZEN'
ROLE_INFO = {'probe_calibration': ('calibration_recipient', 256, 'probe'),
             'probe_test': ('test_recipient', 512, 'probe'),
             'probe_donor_calibration': ('calibration_donor', 256, 'donor_observed'),
             'probe_donor_test': ('test_donor', 512, 'donor_observed'),
             'calibration_response': ('calibration_recipient', 256, 'response'),
             'test_response': ('test_recipient', 512, 'response')}
METADATA = {'roster': ROOT / 'strengthening/manifests/development_model_roster.json',
            'policies': ROOT / 'strengthening/configs/AC_reference_policies.json',
            'audit': PHASE / 'audits/REMOTE_ASSET_METADATA.json',
            'old_sources': PHASE / 'manifests/DEVELOPMENT_INPUT_SOURCES.lock.json'}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''): h.update(part)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked_json(path, checksum):
    if (not isinstance(checksum, str) or len(checksum) != 64 or
            any(c not in '0123456789abcdef' for c in checksum) or file_sha(path) != checksum):
        raise ValueError('Missing or changed actual metadata binding')
    return json.loads(Path(path).read_text())


def _identity(pool, objective, condition, allowed):
    if isinstance(pool, bool) or pool not in (0, 1, 2) or objective not in OBJECTIVES or condition not in allowed:
        raise ValueError('Outside the fixed pool/objective/model-role roster')


def describe_fixed_model(pool, objective, condition):
    """Read existing audited metadata only; never deserialize a checkpoint."""
    _identity(pool, objective, condition, ('T0', 'T1'))
    docs = {key: json.loads(path.read_text()) for key, path in METADATA.items()}
    roster, policies, audit, old = (docs[k] for k in ('roster', 'policies', 'audit', 'old_sources'))
    roster_name = 'strengthening/manifests/development_model_roster.json'
    policies_name = 'strengthening/configs/AC_reference_policies.json'
    if audit['local_source_bindings'][roster_name] != file_sha(METADATA['roster']) or old['files'][policies_name] != file_sha(METADATA['policies']):
        raise ValueError('Previously audited model/policy metadata changed')
    selected = [r for r in roster['C_models'] if (r['pool'], r['objective'], r['condition']) == (pool, objective, condition)]
    if len(selected) != 1: raise ValueError('Missing or duplicate audited fixed-C checkpoint')
    selected = selected[0]
    original = policies['rows'][8 * pool]['entry']
    a_group = next(g for g in roster['A_groups'] if g['group'] == 2 * pool)
    initial = next(m for m in a_group['models'] if m['objective'] == objective)
    report_path = str(Path(selected['checkpoint']).with_name('report.json'))
    for path, checksum in ((selected['checkpoint'], selected['checkpoint_sha256']), (report_path, selected['report_sha256'])):
        hits = [f for f in audit['files'] if f['path'] == path]
        if len(hits) != 1 or not hits[0]['exists'] or hits[0].get('sha256') != checksum or hits[0].get('frozen_hash_matches') is not True:
            raise ValueError('Missing prior live ownership/hash audit for C checkpoint/report')
    meta = next(f for f in audit['files'] if f['path'] == report_path)['training_metadata']
    if any(meta.get(k) != v for k, v in dict(status='PASS_FIXED_C_CONTINUATION', updates=2100,
            pool=pool, group=2 * pool, objective=objective, condition=condition,
            initial_A_checkpoint_sha256=initial['checkpoint_sha256']).items()):
        raise ValueError('Fixed-C training ancestry changed')
    if selected['head_sha256'] != a_group['head_A']['sha256'] or original['arm'] != 'transformer_jepa':
        raise ValueError('Wrong selected construction head or architecture lineage')
    config = str(BASE / 'assets/pusht-v1/models/config.json')
    spec = dict(pool=pool, group=2 * pool, objective=objective, condition=condition,
        architecture='transformer_jepa', seed=original['seed'],
        original=dict(path=str(Path(original['training_path']) / 'last_weights.pt'), sha256=original['weights_sha256']),
        summary=dict(path=str(Path(original['training_path']) / 'summary.json'), sha256=original['training_summary_sha256']),
        checkpoint=dict(path=selected['checkpoint'], sha256=selected['checkpoint_sha256']),
        report=dict(path=report_path, sha256=selected['report_sha256']),
        initial_A_checkpoint_sha256=initial['checkpoint_sha256'], head_A_sha256=selected['head_sha256'],
        config=dict(path=config, sha256=old['files'][config]),
        official=str(BASE / 'releases/visual-v1/official'),
        metadata_sha256={key: file_sha(path) for key, path in METADATA.items()})
    return spec


def validate_input_admission(receipt, *, role, protocol_sha256):
    """Target input-admission schema; actual S2 paths/seeds remain unresolved."""
    if role not in ROLE_INFO: raise ValueError('Unknown runtime role')
    bank_role, count, stream_role = ROLE_INFO[role]
    expected = dict(status=INPUT_STATUS, protocol_sha256=protocol_sha256, bank_role=bank_role,
                    count=count, stream_role=stream_role)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong independently accepted split/count/stream role')
    ids = receipt.get('parent_ids', [])
    if (len(ids) != count or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in ids) or
            len(set(ids)) != count):
        raise ValueError('Incomplete or duplicate accepted parent roster')
    for field in ('bank_manifest', 'input_manifest'):
        row = receipt.get(field, {})
        if (not Path(row.get('path', '')).is_absolute() or len(row.get('sha256', '')) != 64 or
                any(c not in '0123456789abcdef' for c in row['sha256'])):
            raise ValueError('Actual bank/input manifest hash pair is required')
    return dict(role=role, bank_role=bank_role, count=count, stream_role=stream_role,
                parent_ids=tuple(ids), bank_manifest=receipt['bank_manifest'], input_manifest=receipt['input_manifest'])


def _runtime_admission(path, checksum, role):
    contract = checked_json(path, checksum)
    if contract.get('status') != RUNTIME_STATUS or contract.get('role') != role or contract.get('source_root') != str(ROOT):
        raise ValueError('Unfrozen or wrong-role runtime contract')
    protocol = contract['protocol']
    protocol_doc = checked_json(protocol['path'], protocol['sha256'])
    if protocol_doc.get('status') != 'S2_SCIENTIFIC_PROTOCOL_FROZEN':
        raise ValueError('S2 scientific protocol is not frozen; draft module is not authorization')
    files = contract['files']
    required = [Path(__file__), Path(__file__).with_name('regression.py'), *METADATA.values(),
        ROOT / 'strengthening/adapters/ac_rollout.py',
        ROOT / 'scripts/visual/factorial_model.py', ROOT / 'scripts/visual/lewm_adapter.py',
        ROOT / 'scripts/visual/adaptation_freeze.py',
        ROOT / 'scripts/visual/image_planner_cost.py', ROOT / 'scripts/visual/evaluation_precision.py',
        ROOT / 'scripts/visual/score_feedback_ranking.py', ROOT / 'scripts/visual/run_adaptation_checkpoint_gate.py']
    resolved = {}
    for name, digest in files.items():
        source = Path(name); source = source if source.is_absolute() else ROOT / source
        if source.suffix in ('.pt', '.npz', '.npy', '.pkl', '.pickle'):
            raise ValueError('Runtime source closure cannot smuggle model/readout arrays')
        if file_sha(source) != digest: raise ValueError('Changed runtime/source metadata: ' + str(source))
        resolved[str(source.resolve())] = digest
    if any(str(p.resolve()) not in resolved for p in required):
        raise ValueError('Missing required direct runtime source/metadata; complete closure must be frozen externally')
    for name in ('jepa.py', 'module.py'):
        if str((BASE / 'releases/visual-v1/official' / name).resolve()) not in resolved:
            raise ValueError('Official model construction source is unbound')
    allowed = ('T0',) if role.startswith('probe_') else ('T0', 'T1')
    expected_models = [describe_fixed_model(pool, objective, condition)
        for pool in range(3) for objective in OBJECTIVES for condition in allowed]
    if contract.get('models') != expected_models:
        raise ValueError('Runtime model allowlist must match the entire fixed role roster exactly')
    record = contract['input_admission']; raw = checked_json(record['path'], record['sha256'])
    admission = validate_input_admission(raw, role=role, protocol_sha256=protocol['sha256'])
    for key in ('bank_manifest', 'input_manifest'):
        if file_sha(admission[key]['path']) != admission[key]['sha256']:
            raise ValueError('Accepted context/input manifest changed')
    return contract, admission, expected_models


def _normalization(value):
    if not {'mean', 'std'} <= set(value): raise ValueError('Missing actual action normalization')
    mean, std = np.asarray(value['mean'], float), np.asarray(value['std'], float)
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid actual action normalization')
    return copy.deepcopy(value)


@dataclass
class RuntimeModel:
    model: object
    boundary: Mapping
    model_sha256: str
    normalization: Mapping
    normalization_sha256: str
    spec: Mapping
    admission: Mapping
    access_receipt: Mapping


def _torch_dependencies():
    import torch
    for folder in (ROOT / 'scripts/visual', ROOT / 'strengthening/adapters'):
        if str(folder) not in sys.path: sys.path.insert(0, str(folder))
    from factorial_model import make_model
    from adaptation_freeze import configure_dynamics_only, verify_frozen
    from evaluation_precision import configure_evaluation_precision
    from score_feedback_ranking import state_hash
    from run_adaptation_checkpoint_gate import tensor_digest
    return torch, make_model, configure_dynamics_only, verify_frozen, configure_evaluation_precision, state_hash, tensor_digest


def _load_bound_model(spec, admission, access, device):
    # Authenticate metadata and bytes before deserializing either original or C checkpoint.
    summary = checked_json(spec['summary']['path'], spec['summary']['sha256'])
    report = checked_json(spec['report']['path'], spec['report']['sha256'])
    expected = dict(status='PASS_FIXED_C_CONTINUATION', pool=spec['pool'], group=spec['group'],
        objective=spec['objective'], condition=spec['condition'], updates=2100,
        weights_sha256=spec['checkpoint']['sha256'], initial_A_checkpoint_sha256=spec['initial_A_checkpoint_sha256'],
        head_A_sha256=spec['head_A_sha256'])
    if any(report.get(k) != v for k, v in expected.items()): raise ValueError('C training report identity changed')
    if summary['config_sha256'] != spec['config']['sha256'] or summary['seed'] != spec['seed']:
        raise ValueError('Original construction/configuration changed')
    for key in ('original', 'checkpoint', 'config'):
        if file_sha(spec[key]['path']) != spec[key]['sha256']: raise ValueError('Changed model/config bytes')
    normalization = _normalization(summary['normalization'])
    torch, make_model, configure, verify, precision, state_hash, tensor_digest = _torch_dependencies()
    precision(); torch.set_num_threads(4)
    model = make_model(Path(spec['official']), Path(spec['config']['path']), spec['architecture'], spec['seed'])
    model.load_state_dict(torch.load(spec['original']['path'], map_location='cpu', weights_only=True), strict=True)
    model = model.to(device); boundary = configure(model)
    model.load_state_dict(torch.load(spec['checkpoint']['path'], map_location='cpu', weights_only=True), strict=True)
    model.eval(); verify(model, boundary)
    if tensor_digest(boundary['frozen']) != report['frozen_sha256']:
        raise ValueError('C encoder/projector frozen boundary differs from actual training lineage')
    # C's trained action_encoder is retained by strict loading; it is not replaced by original weights.
    return RuntimeModel(model, boundary, state_hash(model), normalization, value_sha(normalization),
                        copy.deepcopy(spec), admission, access)


def load_probe_model(pool, objective, *, runtime_contract, runtime_contract_sha256, split, device='cuda'):
    """Strict probe API has no condition selector, response model or readout input."""
    _identity(pool, objective, 'T0', ('T0',))
    if split not in ('calibration', 'test'): raise ValueError('Unknown fixed probe split')
    role = 'probe_' + split
    contract, admission, models = _runtime_admission(runtime_contract, runtime_contract_sha256, role)
    spec = next(s for s in models if (s['pool'], s['objective'], s['condition']) == (pool, objective, 'T0'))
    access = dict(role=role, runtime_contract_sha256=runtime_contract_sha256, input_admission=contract['input_admission'])
    return _load_bound_model(spec, admission, access, device)


def load_calibration_response_model(pool, objective, condition, *, runtime_contract, runtime_contract_sha256, device='cuda'):
    """Separate256-parent calibration admission; it cannot accept test512 roles."""
    _identity(pool, objective, condition, ('T0', 'T1'))
    contract, admission, models = _runtime_admission(runtime_contract, runtime_contract_sha256, 'calibration_response')
    spec = next(s for s in models if (s['pool'], s['objective'], s['condition']) == (pool, objective, condition))
    return _load_bound_model(spec, admission, dict(role='calibration_response',
        runtime_contract_sha256=runtime_contract_sha256, input_admission=contract['input_admission']), device)


def load_donor_encoder(pool, objective, *, runtime_contract, runtime_contract_sha256, split, device='cuda'):
    """Separate donor admission; only observed-image encoding is exposed afterward."""
    _identity(pool, objective, 'T0', ('T0',))
    if split not in ('calibration', 'test'): raise ValueError('Unknown fixed donor split')
    role = 'probe_donor_' + split
    contract, admission, models = _runtime_admission(runtime_contract, runtime_contract_sha256, role)
    spec = next(s for s in models if (s['pool'], s['objective'], s['condition']) == (pool, objective, 'T0'))
    return _load_bound_model(spec, admission, dict(role=role, runtime_contract_sha256=runtime_contract_sha256,
        input_admission=contract['input_admission'], permitted_operation='encode_donor_observations_only'), device)


def load_test_response_model(pool, objective, condition, *, prediction_lock, prediction_lock_sha256,
        response_contract, response_contract_sha256, expected_bindings, device='cuda'):
    """Verify/reconstruct sealed512×6 predictions before any response model load."""
    _identity(pool, objective, condition, ('T0', 'T1'))
    from regression import verify_prediction_seal
    lock, predictions, _ = verify_prediction_seal(prediction_lock, prediction_lock_sha256,
                                                 expected_bindings=expected_bindings)
    if lock['bindings']['response_contract_sha256'] != response_contract_sha256:
        raise ValueError('Response contract was not bound in the prediction seal')
    response = checked_json(response_contract, response_contract_sha256)
    if (response.get('status') != RESPONSE_STATUS or response.get('protocol_sha256') != lock['bindings']['protocol_sha256'] or
            response.get('role') != 'test_response' or response.get('count') != 512):
        raise ValueError('Unfrozen or wrong-role held-out response contract')
    runtime = response['runtime_contract']
    contract, admission, models = _runtime_admission(runtime['path'], runtime['sha256'], 'test_response')
    if contract['protocol']['sha256'] != response['protocol_sha256'] or response['input_admission'] != contract['input_admission']:
        raise ValueError('Response source/input contract mismatch')
    if list(admission['parent_ids']) != predictions['test_recipient_ids'].tolist():
        raise ValueError('Response parent order differs from sealed predictions')
    spec = next(s for s in models if (s['pool'], s['objective'], s['condition']) == (pool, objective, condition))
    return _load_bound_model(spec, admission, dict(role='test_response', runtime_contract_sha256=runtime['sha256'],
        prediction_lock_sha256=prediction_lock_sha256, response_contract_sha256=response_contract_sha256,
        input_admission=contract['input_admission']), device)


def validate_native_inputs(history_pixels, goal_pixels, prefix_actions, population_actions, selected_index, observed_pixels=None):
    shapes = [(history_pixels, (3, 224, 224, 3)), (goal_pixels, (224, 224, 3))]
    if observed_pixels is not None: shapes.append((observed_pixels, (5, 224, 224, 3)))
    for value, shape in shapes:
        if np.asarray(value).dtype != np.uint8 or np.shape(value) != shape: raise ValueError('Wrong native image shape/dtype')
    for value, shape in ((prefix_actions, (10, 2)), (population_actions, (300, 25, 2))):
        if np.asarray(value).dtype != np.float32 or np.shape(value) != shape or not np.isfinite(value).all():
            raise ValueError('Full native FP32 action population/prefix required')
    if isinstance(selected_index, (bool, np.bool_)) or not isinstance(selected_index, (int, np.integer)) or not 0 <= int(selected_index) < 300:
        raise ValueError('Invalid selected member of the unchanged300 population')


def _assert_handle(handle, parent_id, stream_role):
    if parent_id not in handle.admission['parent_ids'] or handle.admission['stream_role'] != stream_role:
        raise ValueError('Context is outside admitted split/stream')
    if value_sha(handle.normalization) != handle.normalization_sha256: raise ValueError('Action normalization changed')
    if stream_role == 'probe' and (not handle.access_receipt['role'].startswith('probe_') or handle.spec['condition'] != 'T0'):
        raise ValueError('Probe propagation cannot use a response model')
    if stream_role == 'response' and handle.access_receipt['role'] not in ('calibration_response', 'test_response'):
        raise ValueError('Response propagation lacks explicit access admission')


def encode_donor_observations(handle, *, parent_id, observed_pixels):
    """Encode all five accepted donor frames; no actions, readout or rollout API."""
    if (handle.access_receipt['role'] not in ('probe_donor_calibration', 'probe_donor_test') or
            handle.access_receipt.get('permitted_operation') != 'encode_donor_observations_only' or
            handle.spec['condition'] != 'T0'):
        raise ValueError('Donor encoder requires its own T0-only admission')
    _assert_handle(handle, parent_id, 'donor_observed')
    if np.asarray(observed_pixels).dtype != np.uint8 or np.shape(observed_pixels) != (5, 224, 224, 3):
        raise ValueError('Exactly five accepted uint8 donor frames are required')
    torch, _, _, verify, precision, state_hash, _ = _torch_dependencies()
    precision(); verify(handle.model, handle.boundary)
    if state_hash(handle.model) != handle.model_sha256: raise ValueError('Donor model changed before encoding')
    device = next(handle.model.parameters()).device; before = np.array(observed_pixels, copy=True)
    with torch.inference_mode():
        mean = torch.tensor([.485, .456, .406], device=device)[None, None, :, None, None]
        std = torch.tensor([.229, .224, .225], device=device)[None, None, :, None, None]
        encoded = []
        for pixel in observed_pixels:
            image = torch.as_tensor(pixel, device=device).permute(2, 0, 1)[None, None].float() / 255.
            encoded.append(handle.model.encode({'pixels': (image - mean) / std})['emb'][0, 0])
        observed = torch.stack(encoded)
        if observed.shape != (5, 192) or not torch.isfinite(observed).all(): raise ValueError('Incomplete donor observed tokens')
    np.testing.assert_array_equal(before, observed_pixels); verify(handle.model, handle.boundary)
    if state_hash(handle.model) != handle.model_sha256 or value_sha(handle.normalization) != handle.normalization_sha256:
        raise ValueError('Donor model/normalization changed during encoding')
    result = observed.detach().cpu().numpy().copy(); result.setflags(write=False)
    return dict(observed=result, model_binding_sha256=value_sha(handle.spec), access_receipt=copy.deepcopy(handle.access_receipt),
        scope='Accepted donor observed frames only; no action encoder, dynamics prediction, physical labels, readout or response.')


def native_rollouts(handle, *, parent_id, history_pixels, goal_pixels, prefix_actions, population_actions,
                    selected_index, observed_pixels=None, replacements=None):
    """Return latent tokens only, retaining native300 arithmetic and fixed state.

    Probe requires observed pixels. Optional replacements contain exactly the
    admitted actual/donor tokens for this objective. The caller must bind their
    independently accepted four-family receipt and the raw case file identity;
    this thin arithmetic adapter does not replace that population admission.
    """
    if handle.access_receipt['role'].startswith('probe_donor_'):
        raise ValueError('Donor admission authorizes observed encoding only, never rollout')
    role = 'probe' if handle.access_receipt['role'].startswith('probe_') else 'response'
    _assert_handle(handle, parent_id, role)
    if role == 'response' and (observed_pixels is not None or replacements is not None):
        raise ValueError('Response API accepts free rollout only')
    if role == 'probe' and observed_pixels is None: raise ValueError('Complete observed-history probe tokens required')
    validate_native_inputs(history_pixels, goal_pixels, prefix_actions, population_actions, selected_index, observed_pixels)
    if replacements is not None:
        if set(replacements) != {'actual', 'donor'} or any(np.asarray(v).dtype != np.float32 or np.shape(v) != (192,) or not np.isfinite(v).all() for v in replacements.values()):
            raise ValueError('Both accepted FP32 actual/donor replacements are required')
    torch, _, _, verify, precision, state_hash, _ = _torch_dependencies()
    from image_planner_cost import ImagePlannerCost
    from ac_rollout import rollout_population
    precision(); verify(handle.model, handle.boundary)
    if state_hash(handle.model) != handle.model_sha256: raise ValueError('Loaded C model state changed before inference')
    device = next(handle.model.parameters()).device
    originals = [np.array(v, copy=True) for v in (history_pixels, goal_pixels, prefix_actions, population_actions)]
    with torch.inference_mode():
        cost = ImagePlannerCost(handle.model, history_pixels, goal_pixels, prefix_actions,
                                handle.normalization, None, 'latent')
        actions = torch.as_tensor(population_actions, device=device)
        normalized = cost.normalized_actions(actions); normalized_before = normalized.clone()
        initial_before = cost.initial.clone(); encoded = handle.model.action_encoder(normalized)
        free = rollout_population(handle.model, cost.initial, encoded, int(selected_index))
        _, endpoint = cost(actions); torch.testing.assert_close(free[-1], endpoint[selected_index], rtol=0, atol=0)
        identity = rollout_population(handle.model, cost.initial, encoded, int(selected_index), free[0].clone())
        torch.testing.assert_close(identity, free, rtol=0, atol=0)
        output = {'free': free}
        if role == 'probe':
            mean = torch.tensor([.485, .456, .406], device=device)[None, None, :, None, None]
            std = torch.tensor([.229, .224, .225], device=device)[None, None, :, None, None]
            observed = []
            for pixel in observed_pixels:
                image = torch.as_tensor(pixel, device=device).permute(2, 0, 1)[None, None].float() / 255.
                observed.append(handle.model.encode({'pixels': (image - mean) / std})['emb'][0, 0])
            observed = torch.stack(observed)
            teacher = rollout_population(handle.model, cost.initial, encoded, int(selected_index), observed=observed)
            torch.testing.assert_close(teacher[0], free[0], rtol=0, atol=0)
            reset = rollout_population(handle.model, cost.initial, encoded, int(selected_index), observed[0])
            torch.testing.assert_close(reset[0], observed[0], rtol=0, atol=0)
            output.update(observed=observed, observed_history=teacher, reset=reset)
            for name, replacement in (replacements or {}).items():
                token = torch.as_tensor(replacement, device=device)
                corrected = rollout_population(handle.model, cost.initial, encoded, int(selected_index), token)
                torch.testing.assert_close(corrected[0], token, rtol=0, atol=0); output[name] = corrected
        torch.testing.assert_close(cost.initial, initial_before, rtol=0, atol=0)
        torch.testing.assert_close(cost.normalized_actions(actions), normalized_before, rtol=0, atol=0)
        if not all(torch.isfinite(v).all() for v in output.values()): raise ValueError('Nonfinite complete native rollout')
    for before, after in zip(originals, (history_pixels, goal_pixels, prefix_actions, population_actions)):
        np.testing.assert_array_equal(before, after)
    verify(handle.model, handle.boundary)
    if state_hash(handle.model) != handle.model_sha256 or value_sha(handle.normalization) != handle.normalization_sha256:
        raise ValueError('Model/action normalization changed during inference')
    arrays = {k: v.detach().cpu().numpy().copy() for k, v in output.items()}
    for a in arrays.values(): a.setflags(write=False)
    return dict(arrays=arrays, checks=dict(native_endpoint_exact=True, identity_exact=True,
        observed_history_first_exact=role == 'probe', model_unchanged=True, actions_and_normalization_unchanged=True),
        model_binding_sha256=value_sha(handle.spec), access_receipt=copy.deepcopy(handle.access_receipt),
        scope='Latent native300 predictions only; no readout, physical labels, outcome score or model choice. Raw input and four-family provenance require the external fixed population gate.')
