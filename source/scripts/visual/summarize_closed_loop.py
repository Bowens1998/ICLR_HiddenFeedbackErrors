"""Compare only accepted, matched scenario/budget closed-loop runs."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--require-strict-fp32',action='store_true');a=p.parse_args();models={};reference=None
    lines=['# Closed-loop development results','',
           'Single training dataset/seed for trained arms; released weights are a different-budget reference. All original cases retained.','',
           '| Policy | Terminal block cost | Terminal block success | Initial block success | Ever block success | Mean boundary-exit steps |','|---|---:|---:|---:|---:|---:|']
    for folder in map(Path,a.runs):
        s=json.loads((folder/'summary.json').read_text());n=json.loads((folder/'numeric_acceptance.json').read_text());r=json.loads((folder/'replay_acceptance.json').read_text())
        if a.require_strict_fp32 and s['policy']=='model':assert s['precision']['cudnn_allow_tf32'] is False and s['precision']['matmul_allow_tf32'] is False and s['precision']['matmul_precision']=='highest'
        assert len(s['cases'])==n['accepted_cases']==r['accepted_cases'];identity=(s['hashes']['bank_manifest'],s['replans'],[(x['index'],x['seed']) for x in s['cases']])
        if reference is None:reference=identity;initial=[x['initial_block_cost'] for x in s['cases']]
        else:assert identity==reference;np.testing.assert_allclose(initial,[x['initial_block_cost'] for x in s['cases']],rtol=0,atol=1e-7)
        key=s['model']+('_'+s['checkpoint'] if s['checkpoint'] else '');assert key not in models
        cases=s['cases'];extra={'new_terminal_successes':sum(x['terminal_block_success'] and not x['initial_block_success'] for x in cases),'initially_unsuccessful_cases':sum(not x['initial_block_success'] for x in cases)}
        models[key]={'aggregate':s['aggregate'],'additional_counts':extra,'cases':cases,'source_run':str(folder),'hashes':s['hashes'],'precision':s.get('precision'),'wall_seconds':s['wall_seconds']}
        d=s['aggregate'];lines.append(f'| {key} | {d["terminal_block_cost"]:.2f} | {d["terminal_block_success"]:.1%} | {d["initial_block_success"]:.1%} | {d["ever_block_success"]:.1%} | {d["boundary_exit_steps"]:.2f} |')
    result={'scope':'matched scenarios and execution budget; descriptive development results, not across-training-seed inference','case_count':len(reference[2]),'replans':reference[1],'models':models}
    lines+=['',f'Each policy: {len(reference[2])} scenarios, {reference[1]*5} executed primitive steps, 64 candidates per replan and 25-step lookahead.',
            'Numeric reconstruction and fresh simulator replay passed for every included run. Boundary conditioning applies to initial contexts/goals, not later closed-loop trajectories.']
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
