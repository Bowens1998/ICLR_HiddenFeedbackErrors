"""Draft metadata adapter for the two already specified S2 donor permutations.

No scientific freeze, model/readout loading, raw-triple replay or job submission.
Actual complete input/protocol/source receipts are mandatory before output.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import accept_probe_population as gate
import raw_inputs as raw
import accept_raw_inputs as raw_gate

INPUT_STATUS='S2_DONOR_ASSIGNMENT_INPUT_BINDING_FROZEN'
STATUS='PASS_COMPLETE_S2_DONOR_ASSIGNMENTS'
SOURCE_FILES=(Path(__file__),Path(gate.__file__),Path(raw.__file__),HERE/'accept_raw_inputs.py',
    HERE/'model_runtime.py',HERE/'projection_four.py',gate.PHASE/'scripts/s1_projection.py',
    gate.PHASE/'scripts/s1_common.py',gate.PHASE/'scripts/s1_readout.py',
    gate.ROOT/'strengthening/adapters/verifier.py')


def require(value,message):
    if not value:raise ValueError(message)


def pair(path):return dict(path=str(Path(path).resolve()),sha256=gate.sha(path))


def load_context(binding_path,binding_sha256):
    bp=dict(path=str(Path(binding_path).resolve()),sha256=binding_sha256);b=gate._json(bp)
    gate._exact(set(b),{'status','protocol','raw_sources','raw_acceptance','implementation_files','output_root'},'assignment input fields')
    gate._exact(b['status'],INPUT_STATUS,'assignment input status')
    out=Path(b['output_root']);require(out.is_absolute(),'Actual absolute assignment output root required')
    require(not out.exists() and not out.with_name(out.name+'.intent.json').exists(),'Existing assignment output/intent retained')
    expected={str(path.relative_to(gate.ROOT)):gate.sha(path) for path in SOURCE_FILES}
    # This is a small Python implementation closure, not a model runtime or raw
    # payload source inventory. Inherited raw files remain in their own lock.
    gate._exact(b['implementation_files'],expected,'actual helper/reused-validator source identities')
    protocol=gate._json(b['protocol']);raw.validate_design(protocol)
    seeds=protocol['probe_population']['donor_assignment_seeds']
    gate._exact(set(seeds),{'calibration','test'},'both frozen donor seeds')
    for seed in seeds.values():
        require(type(seed) is int and 0<=seed<2**32,'Frozen uint32 donor seed required; no inferred default')
    population,parents,admissions=gate._raw_admission(
        dict(protocol=b['protocol'],raw_acceptance=b['raw_acceptance']),dict(raw_sources=b['raw_sources']),'calibration',256)
    raw_gate.validate_content(gate._json(population['content_lineage']),b['protocol']['sha256'],b['raw_sources']['sha256'])
    banks=population['banks']
    gate._exact([bank['bank_role'] for bank in banks],list(raw.ROLES),'canonical ordered four-bank roster')
    # Reauthenticate actual source/content/bank identities; inherit the accepted
    # complete raw-triple/replay evidence instead of replaying every payload.
    for bank in banks:
        role=bank['bank_role']
        ctx=raw.context(b['protocol']['path'],b['protocol']['sha256'],b['raw_sources']['path'],b['raw_sources']['sha256'],role)
        folder,role_record,manifest=raw._bank(ctx)
        gate._exact(bank['manifest'],pair(folder/'manifest.json'),'canonical original bank manifest')
        gate._exact(bank['role_metadata'],pair(folder/'role.json'),'canonical original bank role metadata')
        gate._exact([row['seed'] for row in manifest['cases']],parents[role],'original bank parent order')
        gate._exact(role_record['count'],raw.COUNTS[role],'complete original bank count')
    gate._pair(bp)
    return dict(binding=bp,input=b,output=out,seeds=seeds,parents=parents,banks=banks,
        content_lineage=population['content_lineage'],admissions=population['admissions'],
        admitted='COMPLETE_ORIGINAL_RAW_BANK_ORDER_AUTHENTICATED')


def assignment_arrays(split,seed,recipient_ids,donor_ids):
    n=gate._size(split)
    require(type(seed) is int and 0<=seed<2**32,'Frozen uint32 donor seed required')
    arrays=dict(donor_assignment=np.random.Generator(np.random.PCG64(seed)).permutation(n).astype(np.int64),
                recipient_ids=np.asarray(recipient_ids),donor_ids=np.asarray(donor_ids))
    for key in ('recipient_ids','donor_ids'):
        a=gate._array(arrays[key],(n,),np.dtype('int64'),key)
        require(len(np.unique(a))==n and np.all((0<=a)&(a<2**32)),'Complete unique uint32 parent bank required')
    require(not np.intersect1d(arrays['recipient_ids'],arrays['donor_ids']).size,'Recipient/donor bank parents overlap')
    gate.validate_assignment(arrays['donor_assignment'],split=split,seed=seed)
    return arrays


def produce(ctx):
    require(ctx.get('admitted')=='COMPLETE_ORIGINAL_RAW_BANK_ORDER_AUTHENTICATED','Complete original bank admission required')
    out=Path(ctx['output']);intent=out.with_name(out.name+'.intent.json')
    require(not out.exists() and not intent.exists(),'Existing assignment output/intent retained; no overwrite or retry')
    gate._pair(ctx['binding'])
    arrays={split:assignment_arrays(split,ctx['seeds'][split],np.asarray(ctx['parents'][split+'_recipient'],np.int64),
                                   np.asarray(ctx['parents'][split+'_donor'],np.int64)) for split in ('calibration','test')}
    gate._write(intent,dict(status='S2_DONOR_ASSIGNMENT_WRITE_STARTED',input_binding=ctx['binding'],source_sha256=gate.sha(__file__)))
    out.mkdir(parents=True,exist_ok=False);rows=[]
    try:
        for split in ('calibration','test'):
            dest=out/(split+'.npz')
            with dest.open('xb') as stream:
                np.savez_compressed(stream,**arrays[split]);stream.flush();os.fsync(stream.fileno())
            archive=pair(dest);saved=gate._arrays(archive)
            gate._exact(set(saved),{'donor_assignment','recipient_ids','donor_ids'},'exact consumer archive fields')
            for key,value in arrays[split].items():gate._equal(saved[key],value,'saved assignment '+key)
            gate.validate_assignment(saved['donor_assignment'],split=split,seed=ctx['seeds'][split])
            rows.append(dict(split=split,count=gate._size(split),seed=ctx['seeds'][split],bit_generator='PCG64',arrays=archive,
                recipient_bank=next(b for b in ctx['banks'] if b['bank_role']==split+'_recipient'),
                donor_bank=next(b for b in ctx['banks'] if b['bank_role']==split+'_donor'),
                array_schema={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in saved.items()}))
        gate._pair(ctx['binding'])
        b=ctx['input'];report=dict(status=STATUS,input_binding=ctx['binding'],protocol=b['protocol'],raw_sources=b['raw_sources'],
            raw_acceptance=b['raw_acceptance'],content_lineage=ctx['content_lineage'],banks=ctx['banks'],
            role_admissions=ctx['admissions'],implementation_files=b['implementation_files'],source_sha256=gate.sha(__file__),
            assignments=rows,donor_id_order='original bank order; consumers apply donor_assignment exactly once',
            complete_splits=2,rows_removed=0,models_or_readouts_deserialized=False,raw_triples_replayed=False,
            scope='Original protocol/source/content/bank/admission metadata reauthenticated; complete previously accepted raw '
                  'numerics/replay inherited. Fixed PCG64 donor permutations and bank-order IDs only; no scientific freeze.')
        gate._write(out/'report.json',report)
        with (out/'DONE').open('x') as stream:stream.write('both complete donor assignment archives\n')
        return report
    except Exception as exc:
        gate._write(out/'failure.json',dict(status='BLOCKED_S2_DONOR_ASSIGNMENTS',input_binding=ctx['binding'],
            completed_assignments=rows,error_type=type(exc).__name__,error=str(exc),
            scope='Partial original output retained; no retry, overwrite, sample replacement or partial population admission.'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',required=True);parser.add_argument('--binding-sha256',required=True)
    args=parser.parse_args();produce(load_context(args.binding,args.binding_sha256))


if __name__=='__main__':main()
